from embeddings.embedder import APIEmbedder
from qdrant_client import QdrantClient
import bm25s
import numpy as np
from utils.utils import cosine_similarity


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

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.encode(query)[0]

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        )

        return search_result


class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.retriever = bm25s.BM25(corpus=corpus)
        self.tokenized_data = bm25s.tokenize(corpus)
        self.retriever.index(self.tokenized_data)

    def retrieve(
            self,
            query: str, top_k: int = 5
    ):
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
    def __init__(self, embedder, embeddings, similarity_func: callable = cosine_similarity):
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
