import re
import os
from bsbi import BSBIIndex
from compression import VBEPostings
from collections import defaultdict
from tqdm import tqdm
import math

from letor import LETOR

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i + 1)
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    num_relevant_at_k = sum(ranking[:k])
    return num_relevant_at_k / k


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += prec(ranking, i) * ranking[pos]

    num_relevant = sum(ranking)
    if num_relevant == 0:
        return 0
    return score / num_relevant

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


# >>>>> EVALUASI !

def get_scores(LETOR_instance, query, qrels, qid, k, rbp_scores_letor, dcg_scores_letor, ap_scores_letor):
    """
    Evaluasi with LETOR
    """
    ranking_letor = []
    for (score, doc) in LETOR_instance.retrieve_reranking(query, k=k):
        did = int(os.path.splitext(os.path.basename(doc))[0])
        if (did in qrels[qid]):
            ranking_letor.append(1)
        else:
            ranking_letor.append(0)
    rbp_scores_letor.append(rbp(ranking_letor))
    dcg_scores_letor.append(dcg(ranking_letor))
    ap_scores_letor.append(ap(ranking_letor))

def print_scores(f, rbp_scores_letor, dcg_scores_letor, ap_scores_letor, mode):
    print(F"Hasil evaluasi with LETOR Mode {mode}", file=f)
    print("RBP score = {:.2f}".format(sum(rbp_scores_letor) / len(rbp_scores_letor)), file=f)
    print("DCG score = {:.2f}".format(sum(dcg_scores_letor) / len(dcg_scores_letor)), file=f)
    print("AP score  = {:.2f}".format(sum(ap_scores_letor) / len(ap_scores_letor)), file=f)
    print()

def eval_retrieval(qrels, f, query_file="qrels-folder/test_queries.txt", k=1000):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    LETOR_instance_1 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=0)
    LETOR_instance_2 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=1)
    LETOR_instance_3 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=2)
    LETOR_instance_4 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=3)
    LETOR_instance_5 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=4)
    LETOR_instance_6 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=5)
    LETOR_instance_7 = LETOR(qrels_path='qrels-folder', BSBI_instance=BSBI_instance, mode=6)
    data_type = 'Test' if query_file == "qrels-folder/test_queries.txt" else 'Validation'

    with open(query_file) as file:
        rbp_scores_letor_1, dcg_scores_letor_1, ap_scores_letor_1 = [], [], []
        rbp_scores_letor_2, dcg_scores_letor_2, ap_scores_letor_2 = [], [], []
        rbp_scores_letor_3, dcg_scores_letor_3, ap_scores_letor_3 = [], [], []
        rbp_scores_letor_4, dcg_scores_letor_4, ap_scores_letor_4 = [], [], []
        rbp_scores_letor_5, dcg_scores_letor_5, ap_scores_letor_5 = [], [], []
        rbp_scores_letor_6, dcg_scores_letor_6, ap_scores_letor_6 = [], [], []
        rbp_scores_letor_7, dcg_scores_letor_7, ap_scores_letor_7 = [], [], []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            get_scores(LETOR_instance_1, query, qrels, qid, k, rbp_scores_letor_1, dcg_scores_letor_1, ap_scores_letor_1)
            get_scores(LETOR_instance_2, query, qrels, qid, k, rbp_scores_letor_2, dcg_scores_letor_2, ap_scores_letor_2)
            get_scores(LETOR_instance_3, query, qrels, qid, k, rbp_scores_letor_3, dcg_scores_letor_3, ap_scores_letor_3)
            get_scores(LETOR_instance_4, query, qrels, qid, k, rbp_scores_letor_4, dcg_scores_letor_4, ap_scores_letor_4)
            get_scores(LETOR_instance_5, query, qrels, qid, k, rbp_scores_letor_5, dcg_scores_letor_5, ap_scores_letor_5)
            get_scores(LETOR_instance_6, query, qrels, qid, k, rbp_scores_letor_6, dcg_scores_letor_6, ap_scores_letor_6)
            get_scores(LETOR_instance_7, query, qrels, qid, k, rbp_scores_letor_7, dcg_scores_letor_7, ap_scores_letor_7)

    print(f"Hasil evaluasi {data_type} Set", file=f)
    print_scores(f, rbp_scores_letor_1, dcg_scores_letor_1, ap_scores_letor_1, 0)
    print_scores(f, rbp_scores_letor_2, dcg_scores_letor_2, ap_scores_letor_2, 1)
    print_scores(f, rbp_scores_letor_3, dcg_scores_letor_3, ap_scores_letor_3, 2)
    print_scores(f, rbp_scores_letor_4, dcg_scores_letor_4, ap_scores_letor_4, 3)
    print_scores(f, rbp_scores_letor_5, dcg_scores_letor_5, ap_scores_letor_5, 4)
    print_scores(f, rbp_scores_letor_6, dcg_scores_letor_6, ap_scores_letor_6, 5)
    print_scores(f, rbp_scores_letor_7, dcg_scores_letor_7, ap_scores_letor_7, 6)


if __name__ == '__main__':
    with open('hasil-evaluasi1.txt', 'w') as f:
        # # Evaluasi Validation set
        # qrels = load_qrels(qrel_file='qrels-folder/val_qrels.txt')
        # eval_retrieval(qrels, f, query_file='qrels-folder/val_queries.txt')

        # Evaluasi Test set
        qrels = load_qrels()
        eval_retrieval(qrels, f)
