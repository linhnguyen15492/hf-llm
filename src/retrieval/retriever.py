from __future__ import annotations

from embeddings.embedding import APIEmbedder
from qdrant_client import QdrantClient
import bm25s
import numpy as np
from utils.utils import cosine_similarity
from abc import ABC, abstractmethod
from typing import Any, List
from vectordb.vector_store import VectorStore


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
    def __init__(self, vectordb: VectorStore, bm25: BM25Retriever):
        super().__init__(vectordb)
        self.bm25 = bm25

    def retrieve(self, texts, top_k=5) -> list[Any] | None:
        # Step 1: Retrieve using the vector store
        vector_indices = self.vector_store.search(texts, top_k=top_k)

        # Step 2: Retrieve using the embedder
        bm25_indices = self.bm25.retrieve(texts, top_k=top_k)

        return combined_results


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
