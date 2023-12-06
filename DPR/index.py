import re
import logging
import torch
import faiss
from tqdm import tqdm
from typing import List
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from chunker import DocumentChunker
from document import DPRDocument
import os
import json

class DPRIndex(DocumentChunker):

    '''
    Class for indexing and searching documents, using a combination of
    vectors produced by DPR and keyword matching from Elastic TF-IDF. As a
    subclass of DocumentChunker, this class automatically handles document
    chunking as well.
    '''

    INDEX_NAME = 'dense-passage-retrieval'
    D = 768
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base')
    context_model = DPRContextEncoder.from_pretrained(
        'facebook/dpr-ctx_encoder-single-nq-base', return_dict=True)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base')
    question_model = DPRQuestionEncoder.from_pretrained(
        'facebook/dpr-question_encoder-single-nq-base', return_dict=True)

    def __init__(self, documents: List[DPRDocument]):
        super(DocumentChunker).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.reader_model = self.reader_model.cuda()
        self.faiss_index = faiss.IndexFlatIP(self.D)
        self._setup_json_index()
        self._build_json_index(documents)

    def _setup_json_index(self):
        '''Sets up the JSON Index. Deletes old ones if needed.'''
        json_filename = 'index_data.json'
        if os.path.exists(json_filename):
            logging.warning(f'Deleting old JSON index file: {json_filename}.')
            os.remove(json_filename)

    def _build_json_index(self, documents):
        '''
        Initializes the data structure to keep track of which chunks
        correspond to which documents.
        '''
        self.documents = documents
        self.doc_bodies = [doc.body for doc in self.documents]
        self.chunks = []
        self.chunk_index = {}  # {chunk: document}
        self.inverse_chunk_index = {}  # {document: [chunks]}
        chunk_counter = 0
        for doc_counter, doc_body in tqdm(enumerate(self.doc_bodies),
                                          total=len(self.doc_bodies)):
            self.inverse_chunk_index[doc_counter] = []
            chunked_docs = self.chunk_document(doc_body)
            self.chunks.extend(chunked_docs)
            for chunked_doc in chunked_docs:
                chunk_embedding = self.embed_context(chunked_doc)
                self.faiss_index.add(chunk_embedding)
                # Save chunk data to JSON file
                with open('index_data.json', 'a') as json_file:
                    json.dump({'chunk_id': chunk_counter, 'chunk': chunked_doc, 'doc_id': doc_counter}, json_file)
                    json_file.write('\n')  # Add newline for multiple entries
                self.chunk_index[chunk_counter] = doc_counter
                self.inverse_chunk_index[doc_counter].append(chunk_counter)
                chunk_counter += 1
        self.total_docs = len(self.documents)
        self.total_chunks = len(self.chunks)

    def embed_question(self, question: str):
        '''Embed the question in vector space with the question encoder.'''
        input_ids = self.question_tokenizer(
            question, return_tensors='pt')['input_ids']
        embeddings = self.question_model(
            input_ids).pooler_output.detach().numpy()
        return embeddings

    def embed_context(self, context: str):
        '''Embed the context (doc) in vector space with the question encoder.'''
        input_ids = self.context_tokenizer(
            context, return_tensors='pt')['input_ids']
        embeddings = self.context_model(
            input_ids).pooler_output.detach().numpy()
        return embeddings

    def search_dense_index(self, question: str, k: int = 5):
        '''
        Search the vector index by encoding the question and then performing
        nearest neighbor on the FAISS index of context vectors.

        Args:
            question (str):
                The natural language question, e.g., `who is bill gates?`
            k (int):
                The number of documents to return from the index.
        '''
        if k > self.total_chunks:
            k = self.total_chunks
        question_embedding = self.embed_question(question)
        dists, chunk_ids = self.faiss_index.search(question_embedding, k=k)
        dists, chunk_ids = list(dists[0]), list(chunk_ids[0])
        dists = list(map(float, dists))  # For Flask
        structured_response = []
        for dist, chunk_id in zip(dists, chunk_ids):
            chunk = self.chunks[chunk_id]
            document_id = self.chunk_index[chunk_id]
            document = self.documents[document_id]
            blob = {
                'document': document,
                'document_id': document_id,
                'chunk': chunk,
                'chunk_id': int(chunk_id),  # For Flask
                'faiss_dist': dist
            }
            structured_response.append(blob)
        return structured_response

    def search_sparse_index(self, query):
        with open('index_data.json', 'r') as json_file:
            data = [json.loads(line) for line in json_file]
        
        # Perform search in-memory
        results = [entry for entry in data if query in entry['chunk']]

        return results

    def _merge_results(self, sparse_results, dense_results):
        '''Merges the results of sparse and dense retrieval.'''
        results_index = {}
        for sparse_result in sparse_results:
            id, score = sparse_result['chunk_id'], sparse_result['elastic_score']
            id = int(id)
            results_index[id] = {'elastic_score': score}
        for dense_result in dense_results:
            id, score = dense_result['chunk_id'], dense_result['faiss_dist']
            if id in results_index:
                results_index[id]['faiss_dist'] = score
            else:
                results_index[id] = {'faiss_dist': score}
        results = []
        for chunk_id, scores in results_index.items():
            document_id = self.chunk_index[chunk_id]
            document = self.documents[document_id]
            chunk = self.chunks[chunk_id]
            doc_profile = document.to_dict()
            result = {
                'chunk_id': chunk_id,
                'chunk': chunk,
                'document_id': document_id,
                'document': doc_profile,
                'scores': scores
            }
            results.append(result)
        return results

    def search_dual_index(self, query: str):
        '''Search both the sparse and dense indices and merge the results.'''
        sparse_result = self.search_sparse_index(query)
        dense_result = self.search_dense_index(query)
        merged_results = self._merge_results(sparse_result, dense_result)
        return merged_results
