import numpy as np
import os
import random
import lightgbm as lgb
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from bsbi import BSBIIndex
from compression import VBEPostings

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import re

class TextPreprocessing:
    tokenizer_pattern = r'\w+'
    stemmer = MPStemmer()
    stop_words = set(StopWordRemoverFactory().get_stop_words())

    @staticmethod
    def tokenization(text):
        """
        Mengubah huruf di text menjadi huruf kecil, membersihkan text 
        dari tanda baca, dan tokenisasi text

        Parameters
        ----------
        text: str
            Text
        """
        text = text.translate(str.maketrans("", "", string.punctuation)).lower()
        tokens = re.findall(TextPreprocessing.tokenizer_pattern, text)
        return tokens
    
    @staticmethod
    def stem_tokens(tokens):
        """
        Melakukan stemming pada setiap token

        Parameters
        ----------
        tokens: List[str]
            List of token
        """
        stemmed_tokens = [
            TextPreprocessing.stemmer.stem(token) if token else ''
            for token in tokens
        ]
        stemmed_tokens_without_empty_string = [
            token
            for token in stemmed_tokens
            if not ((token == '') or (token == None))
        ]
        return stemmed_tokens_without_empty_string
    
    @staticmethod
    def remove_stop_words(tokens):
        """
        Menghapus stop words dalam tokens

        Parameters
        ----------
        tokens: List[str]
            List of token
        """
        tokens_without_stop_words = [
            token
            for token in tokens
            if token not in TextPreprocessing.stop_words
        ]
        return tokens_without_stop_words
    
    @staticmethod
    def get_terms(text):
        """
        Mendapatkan list of terms dari sebuah text

        Parameters
        ----------
        text: str
            Text dokumen atau query
        """
        token = TextPreprocessing.tokenization(text)
        token = TextPreprocessing.stem_tokens(token)
        token = TextPreprocessing.remove_stop_words(token)
        return token


class TrainDataset:
    def __init__(self, qrels_path):
        self.qrels_path = qrels_path
        self.queries = {} 
        self.documents = {}
        self.q_docs_rel = {}
        self.group_qid_count = []
        self.dataset = []

    def parse_queries(self):
        """
        Parse q_id dan content dari train_queries.txt ke dalam dictionary
        """
        with open(os.path.join(self.qrels_path, 'train_queries.txt')) as file:
            for line in file:
                q_id, *content = line.strip(" \n").split(" ")
                terms = TextPreprocessing.get_terms(" ".join(content))
                self.queries[q_id] = terms

    def parse_docs(self):
        """
        Parse doc_id dan content dari train_docs.txt ke dalam dictionary
        """
        with open(os.path.join(self.qrels_path, 'train_docs.txt'), encoding='utf-8') as file:
            for line in file:
                doc_id, *content = line.strip(" \n").split(" ")
                terms = TextPreprocessing.get_terms(" ".join(content))
                self.documents[doc_id] = terms

    def parse_qrels(self):
        """
        Parse q_id, doc_id, dan rel dari train_qrels.txt ke dalam dictionary
        """
        with open(os.path.join(self.qrels_path, 'train_qrels.txt')) as file:
            for line in file:
                q_id, doc_id, rel = line.strip(" \n").split(" ")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

    def create(self):
        """
        Membuat dataset untuk LambdaMart model dengan format
        [(query_text, doc_text, rel), ...]

        Relevance level:
        3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)
        """
        NUM_NEGATIVES = 1

        self.parse_queries()
        self.parse_docs()
        self.parse_qrels()
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # Tambah satu negative (random sampling dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))


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
