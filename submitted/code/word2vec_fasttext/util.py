from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import re
import os
import random

class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        try:
            i = self.str_to_id[s]
            return i
        except:
            i =  len(self)
            self.str_to_id[s] = i
            self.id_to_str.append(s)
            return i
    
    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        s = self.id_to_str[i]
        return s

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        """
        if type(key) is str:
            return self.__get_id(key)
        elif type(key) is int:
            return self.__get_str(key)


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    intersect_list = []
    index1, index2 = 0, 0

    while index1 < len(posts_tfs1) and index2 < len(posts_tfs2):
        if posts_tfs1[index1][0] == posts_tfs2[index2][0]:
            intersect_list.append((posts_tfs1[index1][0], posts_tfs1[index1][1]+posts_tfs2[index2][1]))
            index1 += 1
            index2 += 1
        elif posts_tfs1[index1][0] < posts_tfs2[index2][0]:
            intersect_list.append(posts_tfs1[index1])
            index1 += 1
        else:
            intersect_list.append(posts_tfs2[index2])
            index2 += 1

    while index1 < len(posts_tfs1):
        intersect_list.append(posts_tfs1[index1])
        index1 += 1
    
    while index2 < len(posts_tfs2):
        intersect_list.append(posts_tfs2[index2])
        index2 += 1
    return intersect_list

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

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname]
            for docname in docs] == [0, 1, 2], "docs_id salah"
    assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"
