from dpr import DensePassageRetriever
from document import DPRDocument
from converter import documents

dpr_docs = [DPRDocument(**doc) for doc in documents]
dpr = DensePassageRetriever(dpr_docs)