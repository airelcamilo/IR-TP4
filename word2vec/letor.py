import numpy as np
import lightgbm as lgb
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from bsbi import BSBIIndex
from compression import VBEPostings
from util import TrainDataset, TextPreprocessing

class LETOR:
    def __init__(self, qrels_path, BSBI_instance, mode):
        self.qrels_path = qrels_path
        self.BSBI_instance = BSBI_instance
        self.NUM_LATENT_TOPICS = 200
        self.lsi_model = {}
        self.dictionary = Dictionary()
        self.ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)
        self.mode = mode
        self.train()

    def build_lsi_model(self, documents):
        """
        Membuat bag-of-words corpus dan Latent Semantic Indexing
        dari kumpulan dokumen.

        Parameters
        ----------
        documents: dict[str, str]
            Dictionary doc id dan term di dokumen
        """
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
        self.lsi_model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS)

    def build_tfidf_model(self, documents):
        """
        Membuat bag-of-words corpus dan score TF-IDF
        dari kumpulan dokumen.

        Parameters
        ----------
        documents: dict[str, str]
            Dictionary doc id dan term di dokumen
        """
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
        self.tfidf_model = TfidfModel(bow_corpus, smartirs='lfn')

    def vector_rep(self, text):
        """
        Representasi vector dari sebuah dokumen atau query
        
        Parameters
        ----------
        text: list[str]
            Term dari text (dokumen atau query)
        """
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS
    
    def features(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi concat(vector(query), vector(document)) + informasi lain

        Informasi lain: Cosine Distance & Jaccard Similarity antara query dan doc

        Parameters
        ----------
        query: List[str]
            Term dari query
        doc: List[str]
            Term dari dokumen
        """
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def get_tfidf(self, query, doc):
        query_term_ids = [term_id for (term_id, _) in self.tfidf_model[self.dictionary.doc2bow(query)]]
        tfidf = 0
        for (term_id, score) in self.tfidf_model[self.dictionary.doc2bow(doc)]:
            if term_id in query_term_ids:
                tfidf += score
        return tfidf

    def features2(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi TF-IDF + Cosine Distance + Jaccard Similarity antara query dan doc

        Parameters
        ----------
        query: List[str]
            Term dari query
        doc: List[str]
            Term dari dokumen
        """
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        tfidf = self.get_tfidf(query, doc)
        return [tfidf] + [jaccard] + [cosine_dist]
    
    def split_dataset(self, dataset):
        """
        Split dataset menjadi X (representasi gabungan query dan dokumen)
        dan Y (label relevance)

        Parameters
        ----------
        dataset: List[tuple(query, doc, rel)]
            Dataset
        """
        X, Y = [], []
        for (query, doc, rel) in dataset:
            if self.mode == 0:
                X.append(self.features(query, doc))
            elif self.mode == 1:
                X.append(self.features2(query, doc))
            Y.append(rel)
        X = np.array(X)
        Y = np.array(Y)
        return (X, Y)
    
    def train(self):
        """
        Training model LightGBM LambdaMART Model
        """
        train_dataset = TrainDataset(self.qrels_path)
        train_dataset.create()
        self.build_lsi_model(train_dataset.documents)
        if self.mode == 1:
            self.build_tfidf_model(train_dataset.documents)

        X_train, Y_train = self.split_dataset(train_dataset.dataset)
        self.ranker.fit(X_train, Y_train, 
                        group=train_dataset.group_qid_count)
        
    def retrieve_reranking(self, query, k=10):
        """
        Melakukan re-ranking terhadap retrieval model TF-IDF k=100
        untuk memperbaiki kualitas SERP yang dihasilkan sebelumnya

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        """
        X = []
        docs_path = []
        scores_docs_path = self.BSBI_instance.retrieve_tfidf(query, k=100)
        for _, doc_path in scores_docs_path:
            with open(doc_path, 'rb') as file:
                doc = str(file.readline().decode())
                d_terms = TextPreprocessing.get_terms(doc)
                q_terms = TextPreprocessing.get_terms(query)
                docs_path.append(doc_path)
                if self.mode == 0:
                    X.append(self.features(q_terms, d_terms))
                elif self.mode == 1:
                    X.append(self.features2(q_terms, d_terms))
        if len(X) == 0:
            return []
        X = np.array(X)
        scores = self.ranker.predict(X)
        score_doc_path = [ (score, doc_path) for score, doc_path in zip(scores, docs_path) ]
        return sorted(score_doc_path, key=lambda x: x[0], reverse=True)[:k]
    

if __name__ == "__main__":
    BSBI_instance = BSBIIndex(data_dir='collections',
                            postings_encoding=VBEPostings,
                            output_dir='index')
    mode = 1
    LETOR_instance = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=mode)
    """
    LETOR agak berbeda dengan yang terdapat di Google Colab karena ditambahkan proses
    text preprocessing terlebih dahulu, seperti saat retrieve dari index.
    mode: int
        Mode fitur yang digunakan untuk reranking
        0: LSI query + LSI dokumen + Cosine Distance + Jaccard Similarity (seperti Google Colab)
        1: TF-IDF + Cosine Distance + Jaccard Similarity
    """

    queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri"]
    print("BSBI TF-IDF")
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
            print(f"{doc:30} {score:>.3f}")
        print()

    print(f"BSBI TF-IDF with LETOR Mode {mode}")
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in LETOR_instance.retrieve_reranking(query, k=10):
            print(f"{doc:30} {score:>.3f}")
        print()
