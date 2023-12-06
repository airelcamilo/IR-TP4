# from dpr import DensePassageRetriever
# from document import DPRDocument

dpr_docs = [DPRDocument(**doc) for doc in documents]
dpr = DensePassageRetriever(dpr_docs)