from __future__ import annotations

from embeddings.embedding import APIEmbedder
from qdrant_client import QdrantClient
import bm25s
import numpy as np
from utils.utils import cosine_similarity
from abc import ABC, abstractmethod
from typing import Any, List
from vectordb.vector_store import VectorStore
from pathlib import Path
import json


class Retriever(ABC):
    def __init__(self, vectordb: VectorStore):
        self.vector_store = vectordb

    @abstractmethod
    def retrieve(self, texts, top_k=5) -> list[Any] | None:
        pass


class SimpleRetriever(Retriever):
    def __init__(self, vectordb: VectorStore):
        super().__init__(vectordb)

    def retrieve(self, texts, top_k=5) -> list[Any] | None:
        results = self.vector_store.search(texts, top_k=top_k)

        return results


class HybridRetriever(Retriever):
    def __init__(self, vectordb: VectorStore, corpus_path: str):
        super().__init__(vectordb)
        self.corpus_path = corpus_path
        self._build_corpus()

    def retrieve(self, texts, top_k=5) -> list[Any] | None:
        # Step 1: Retrieve using the vector store
        vector_list = self.vector_store.search(texts, top_k=top_k)

        # Step 2: Retrieve using the BM25 retriever
        tokenized_query = bm25s.tokenize(texts)
        bm25_list, scores = self.bm25_retriever.retrieve(tokenized_query, k=top_k)

        # Extract the first element to get the list of retrieved documents
        bm25_list = bm25_list[0]

        # Step 3: Combine the results using RRF
        ids = self._rrf(vector_list, bm25_list, top_k=top_k)

        return [doc for doc in self.corpus if doc["id"] in ids]

    def _build_corpus(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            corpus = [
                {
                    "id": doc["id"],
                    "section": doc["section"],
                    "question": doc["question"],
                    "answer": doc["answer"],
                }
                for doc in docs
            ]

        self.corpus = corpus
        self.bm25_retriever = bm25s.BM25(corpus=corpus)
        tokenized_data = bm25s.tokenize(
            [f"{doc["question"]} {doc["answer"]}" for doc in corpus]
        )
        self.bm25_retriever.index(tokenized_data)

    def _rrf(self, vector_list, bm25_list, top_k=5, K=60):
        # Implement Reciprocal Rank Fusion (RRF) to combine results
        rrf_scores = {}

        for lst in [vector_list, bm25_list]:
            for rank, item in enumerate(lst, start=1):
                if item["id"] not in rrf_scores:
                    rrf_scores[item["id"]] = 0

                rrf_scores[item["id"]] += 1 / (K + rank)

        # Sort by combined score and return top_k results
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [item[0] for item in sorted_results[:top_k]]


class QdrantRetriever:
    def __init__(
        self,
        embedder: APIEmbedder,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vietnam_law_corpus",
    ):
        self.qdrant = QdrantClient(qdrant_url)
        self.collection_name = collection_name
        self.embedder = embedder  # Tiêm module Embedder vào đây

    def retrieve(self, texts, top_k=5) -> list[Any] | None:
        query_embeddings = self.embedder.encode(texts)

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embeddings,
            limit=top_k,
        )

        return search_result


class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.retriever = bm25s.BM25(corpus=corpus)
        self.tokenized_data = bm25s.tokenize(corpus)
        self.retriever.index(self.tokenized_data)

    def retrieve(self, query: str, top_k: int = 5) -> list[Any] | None:
        """
        Retrieves the top k relevant documents for a given query using the BM25 algorithm.

        This function tokenizes the input query and uses a pre-indexed BM25 retriever to
        search through a collection of documents. It returns the indices of the top k documents
        that are most relevant to the query.

        Args:
            query (str): The search query for which documents need to be retrieved.
            top_k (int): The number of top relevant documents to retrieve. Default is 5.

        Returns:
            List[int]: A list of indices corresponding to the top k relevant documents
            within the corpus.
        """
        tokenized_query = bm25s.tokenize(query)
        # Use the 'BM25_RETRIEVER' to retrieve documents and their scores based on the tokenized query
        # Retrieve the top 'k' documents
        results, scores = self.retriever.retrieve(tokenized_query, k=top_k)

        # Extract the first element from 'results' to get the list of retrieved documents
        results = results[0]

        # Convert the retrieved documents into their corresponding indices in the results list
        top_k_indices = [self.corpus.index(doc) for doc in results]

        return top_k_indices


class SemanticRetriever:
    def __init__(
        self, embedder, embeddings, similarity_func: callable = cosine_similarity
    ):
        self.embedder = embedder
        self.embeddings = embeddings
        self.similarity_func = similarity_func

    def retrieve(self, query, top_k=5):
        """
        Retrieves the top k relevant documents for a given query using semantic search and cosine similarity.

        This function generates an embedding for the input query and compares it against pre-computed document
        embeddings using cosine similarity. The indices of the top k most similar documents are returned.

        Args:
            query (str): The search query for which relevant documents need to be retrieved.
            top_k (int): The number of top relevant documents to retrieve. Default value is 5.

        Returns:
            List[int]: A list of indices corresponding to the top k most relevant documents in the corpus.
        """
        # Generate the embedding for the query using the pre-trained model
        query_embedding = self.embedder.encode(query)

        # Calculate the cosine similarity scores between the query embedding and the pre-computed document embeddings
        similarity_scores = self.similarity_func(query_embedding, self.embeddings)

        # Sort the similarity scores in descending order and get the indices
        similarity_indices = np.argsort(-similarity_scores)

        # Select the indices of the top k documents as a numpy array
        top_k_indices_array = similarity_indices[:top_k]

        # Cast them to int
        top_k_indices = [int(x) for x in top_k_indices_array]

        return top_k_indices
