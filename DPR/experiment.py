import re
import os
from collections import defaultdict
from tqdm import tqdm
import math

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
def load_qrels(qrel_file="qrels-folder-for-dpr/test_qrels.txt"):
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


# >>>>> EVALUASI
def eval_retrieval(qrels, query_file="qrels-folder-for-dpr/test_queries.txt", k=5):
    """
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-5 documents
    """
    data_type = 'Test' if query_file == "qrels-folder-for-dpr/test_queries.txt" else 'Validation'

    with open(query_file) as file:
        rbp_scores_rr, dcg_scores_rr, ap_scores_rr = [], [], []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi Reader-Relevance Score
            """
            ranking_reader_relevance = []
            results = dpr.search(query)

            # Normalisasi min-max pada kolom 'reader_relevance'
            max_relevance = max(data['scores']['reader_relevance'] for data in results)
            min_relevance = min(data['scores']['reader_relevance'] for data in results)

            for data in results:
                data['scores']['normalized_reader_relevance'] = (data['scores']['reader_relevance'] - min_relevance) / (max_relevance - min_relevance)

            for result in results:
                score = result['scores']['normalized_reader_relevance']
                doc = int(result['document']['title'])

                if (doc in qrels[qid]):
                    ranking_reader_relevance.append(1)
                else:
                    ranking_reader_relevance.append(0)

            rbp_scores_rr.append(rbp(ranking_reader_relevance))
            dcg_scores_rr.append(dcg(ranking_reader_relevance))
            ap_scores_rr.append(ap(ranking_reader_relevance))


    print(f"Hasil evaluasi {data_type} Set")
    print("Hasil evaluasi Reader Relevance Score")
    print("RBP score = {:.2f}".format(sum(rbp_scores_rr) / len(rbp_scores_rr)))
    print("DCG score = {:.2f}".format(sum(dcg_scores_rr) / len(dcg_scores_rr)))
    print("AP score  = {:.2f}".format(sum(ap_scores_rr) / len(ap_scores_rr)))
    print()

if __name__ == '__main__':
    with open('hasil-evaluasi.txt', 'w') as f:
        # Evaluasi Validation set
        qrels = load_qrels(qrel_file='qrels-folder-for-dpr/val_qrels.txt')
        eval_retrieval(qrels, query_file='qrels-folder-for-dpr/val_queries.txt')

        # Evaluasi Test set
        qrels = load_qrels()
        eval_retrieval(qrels)