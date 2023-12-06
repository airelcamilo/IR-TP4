import numpy as np
import lightgbm as lgb
from gensim.models import LsiModel, TfidfModel, OkapiBM25Model, Word2Vec, FastText
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

    def get_tfidf(self, query, doc):
        query_term_ids = [term_id for (term_id, _) in self.dictionary.doc2bow(query)]
        tfidf = 0
        for (term_id, score) in self.tfidf_model[self.dictionary.doc2bow(doc)]:
            if term_id in query_term_ids:
                tfidf += score
        return tfidf

    def build_bm25_model(self, documents):
        """
        Membuat bag-of-words corpus dan score BM25
        dari kumpulan dokumen.

        Parameters
        ----------
        documents: dict[str, str]
            Dictionary doc id dan term di dokumen
        """
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
        self.bm25_model = OkapiBM25Model(bow_corpus)

    def get_bm25(self, query, doc):
        query_term_ids = [term_id for (term_id, _) in self.dictionary.doc2bow(query)]
        bm25 = 0
        for (term_id, score) in self.bm25_model[self.dictionary.doc2bow(doc)]:
            if term_id in query_term_ids:
                bm25 += score
        return bm25
    
    def build_word2vec_model(self, documents, sg):
        """
        Membuat bag-of-words corpus dan model Word2Vec
        dari kumpulan dokumen.

        Parameters
        ----------
        documents: dict[str, str]
            Dictionary doc id dan term di dokumen
        sg: int
            0: CBOW; 1: Skip-Gram
        """
        sentences = [doc for doc in documents.values()]
        self.word2vec_model = Word2Vec(sentences, sg=sg, window=5, vector_size=self.NUM_LATENT_TOPICS)

    def build_fasttext_model(self, documents, sg):
        """
        Membuat bag-of-words corpus dan model FastText
        dari kumpulan dokumen.

        Parameters
        ----------
        documents: dict[str, str]
            Dictionary doc id dan term di dokumen
        sg: int
            0: FastText CBOW; 1: FastText Skip-Gram
        """
        sentences = [doc for doc in documents.values()]
        self.fasttext_model = FastText(sentences, sg=sg, window=5, vector_size=self.NUM_LATENT_TOPICS)

    def word2vec_vector_rep(self, text):
        tokens = [token for token in text if token in self.word2vec_model.wv]
        vector = np.zeros(200)
        for token in tokens:
            vector += self.word2vec_model.wv[token]
        return vector / len(tokens)
    
    def fasttext_vector_rep(self, text):
        tokens = [token for token in text if token in self.fasttext_model.wv]
        vector = np.zeros(200)
        for token in tokens:
            vector += self.fasttext_model.wv[token]
        return vector / len(tokens)

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
    
    def features0(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi concat(vector(query), vector(document)) + informasi lain

        Informasi lain: Cosine Distance LSI & Jaccard Similarity antara query dan doc

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

    def features1(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi TF-IDF + Cosine Distance LSI + Jaccard Similarity antara query dan doc

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
    
    def features2(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi BM25 + Cosine Distance LSI + Jaccard Similarity antara query dan doc

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
        bm25 = self.get_bm25(query, doc)
        return [bm25] + [jaccard] + [cosine_dist]
    
    def features34(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi BM25 + Cosine Distance CBOW + CBOW query + CBOW dokumen
        or BM25 + Cosine Distance Skip-Gram + Skip-Gram query + Skip-Gram dokumen

        Parameters
        ----------
        query: List[str]
            Term dari query
        doc: List[str]
            Term dari dokumen
        """
        v_q = self.word2vec_vector_rep(query)
        v_d = self.word2vec_vector_rep(doc)
        cosine_dist = cosine(v_q, v_d)
        bm25 = self.get_bm25(query, doc)
        return [bm25] + [cosine_dist] + v_q.tolist() + v_d.tolist()
    
    def features56(self, query, doc):
        """
        Representasi vector dari gabungan dokumen dan query
        Berisi BM25 + Cosine Distance FastText CBOW + FastText CBOW query + 
        FastText CBOW dokumen or BM25 + Cosine Distance FastText Skip-Gram + 
        FastText Skip-Gram query + FastText Skip-Gram dokumen

        Parameters
        ----------
        query: List[str]
            Term dari query
        doc: List[str]
            Term dari dokumen
        """
        v_q = self.fasttext_vector_rep(query)
        v_d = self.fasttext_vector_rep(doc)
        cosine_dist = cosine(v_q, v_d)
        bm25 = self.get_bm25(query, doc)
        return [bm25] + [cosine_dist] + v_q.tolist() + v_d.tolist()
    
    def append_features(self, X, query, doc):
        """
        Append features to array X

        Parameters
        ----------
        X: List[float]
            Empty list
        query: List[str]
            Term dari query
        doc: List[str]
            Term dari dokumen
        """
        if self.mode == 0:
            X.append(self.features0(query, doc))
        elif self.mode == 1:
            X.append(self.features1(query, doc))
        elif self.mode == 2:
            X.append(self.features2(query, doc))
        elif self.mode in [3, 4]:
            X.append(self.features34(query, doc))
        elif self.mode in [5, 6]:
            X.append(self.features56(query, doc))
        return X
    
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
            token = TextPreprocessing.stem_tokens(doc)
            d_terms = TextPreprocessing.remove_stop_words(token)
            token = TextPreprocessing.stem_tokens(query)
            q_terms = TextPreprocessing.remove_stop_words(token)
            X = self.append_features(X, q_terms, d_terms)
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
        if self.mode in [0, 1, 2]:
            self.build_lsi_model(train_dataset.documents)
        if self.mode == 1:
            self.build_tfidf_model(train_dataset.documents)
        elif self.mode in [2, 3, 4, 5, 6]:
            self.build_bm25_model(train_dataset.documents)
        if self.mode == 3:
            self.build_word2vec_model(train_dataset.documents, 0)
        elif self.mode == 4:
            self.build_word2vec_model(train_dataset.documents, 1)
        if self.mode == 5:
            self.build_fasttext_model(train_dataset.documents, 0)
        elif self.mode == 6:
            self.build_fasttext_model(train_dataset.documents, 1)

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
        if self.mode in [0, 1]:
            scores_docs_path = self.BSBI_instance.retrieve_tfidf(query, k=100)
        else:
            scores_docs_path = self.BSBI_instance.retrieve_bm25(query, k=100)
        for _, doc_path in scores_docs_path:
            with open(doc_path, 'rb') as file:
                doc = str(file.readline().decode())
                d_terms = TextPreprocessing.get_terms(doc)
                q_terms = TextPreprocessing.get_terms(query)
                docs_path.append(doc_path)
                X = self.append_features(X, q_terms, d_terms)
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
    mode = 5
    LETOR_instance = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=mode)
    """
    LETOR agak berbeda dengan yang terdapat di Google Colab karena ditambahkan proses
    text preprocessing terlebih dahulu, seperti saat retrieve dari index.
    mode: int
        Mode fitur yang digunakan untuk reranking
        0: LSI query + LSI dokumen + Cosine Distance + Jaccard Similarity (seperti Google Colab)
        1: TF-IDF + Cosine Distance LSI + Jaccard Similarity
        2: BM25 + Cosine Distance LSI + Jaccard Similarity
        3: BM25 + CBOW
        4: BM25 + Skip-Gram
        5: BM25 + FastText CBOW
        6: BM25 + FastText Skip-Gram
    """

    queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri"]
    print(f"BSBI with LETOR Mode {mode}")
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in LETOR_instance.retrieve_reranking(query, k=10):
            print(f"{doc:30} {score:>.3f}")
        print()
