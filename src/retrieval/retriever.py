from embeddings.embedder import APIEmbedder
from qdrant_client import QdrantClient


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
