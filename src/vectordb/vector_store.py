import json
import chromadb
from chromadb import EmbeddingFunction
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings.embedding import Embedding
from abc import ABC, abstractmethod
from typing import Any, List


class VectorStore(ABC):
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = None  # Đối tượng client sẽ được khởi tạo trong các lớp con
        self.collection = (
            None  # Đối tượng collection sẽ được khởi tạo trong các lớp con
        )

    def search(self, texts, top_k=5) -> list[Any] | None:
        """
        Tìm kiếm các vector gần nhất trong cơ sở dữ liệu vector.
        :param texts: Danh sách các truy vấn (dạng string).
        :param top_k: Số lượng kết quả trả về.
        :return: Danh sách các điểm gần nhất.
        """
        pass


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding: Embedding,
        url: str = "http://localhost:6333",
        collection_name="vietnam_law_corpus",
        vector_size: int = 1536,
    ):
        super().__init__(collection_name=collection_name)
        self.client = QdrantClient(url=url)
        self.vector_size = vector_size

        # Tạo Collection trong Qdrant nếu chưa tồn tại
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            print(
                f"Collection '{self.collection_name}' đã được tạo thành công trong Qdrant."
            )
        else:
            print(f"Collection '{self.collection_name}' đã tồn tại trong Qdrant.")

    def process_and_chunk_law_json(self, json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks_to_insert = []
        global_id = 0

        # Duyệt qua từng văn bản luật trong file JSON
        for law_item in data:
            law_id = law_item.get("law_id", "UNKNOWN_LAW")
            articles = law_item.get("content", [])

            for article in articles:
                aid = article.get("aid", "UNKNOWN_ARTICLE")
                raw_content = article.get("content_Article", "").strip()

                if not raw_content:
                    continue

                # CHIẾN LƯỢC: Làm giàu ngữ cảnh bằng cách gắn Metadata trực tiếp vào nội dung text
                # Giúp mô hình embedding nhận diện được ngữ cảnh thuộc Điều nào, Luật nào khi tính toán vector
                enriched_content = (
                    f"[Law ID: {law_id}] [Article ID: {aid}]\n{raw_content}"
                )

                # Lưu trữ thông tin chunk kèm metadata sạch để nạp vào DB
                chunks_to_insert.append(
                    {
                        "id": global_id,
                        "text": enriched_content,
                        "metadata": {
                            "law_id": law_id,
                            "article_id": aid,
                            "raw_text": raw_content,  # Giữ lại text gốc để trả về cho LLM sau này
                        },
                    }
                )
                global_id += 1

        return chunks_to_insert

    def insert_chunks_to_vector_db(self, chunks):
        print(f"Bắt đầu tạo embedding và nạp {len(chunks)} chunks vào Qdrant...")

        points = []
        # Gom batch để xử lý embedding nhanh hơn thay vì chạy từng câu (tối ưu hóa tốc độ)
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts_to_embed, batch_size=32)

        assert len(embeddings) == len(texts_to_embed), (
            f"Expected {len(texts_to_embed)} embeddings, " f"got {len(embeddings)}"
        )

        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=embeddings[i],
                    payload={
                        "text": chunk["text"],
                        **chunk[
                            "metadata"
                        ],  # Giải nén các trường metadata (law_id, article_id, raw_text)
                    },
                )
            )

        # Push dữ liệu lên Qdrant theo từng cụm (upsert)
        self.client.upsert(collection_name=self.collection_name, points=points)
        print("Nạp dữ liệu hoàn tất thành công!")

    def search(self, texts, top_k=5):
        """
        Tìm kiếm các vector gần nhất trong cơ sở dữ liệu vector.
        :param texts: Danh sách các truy vấn (dạng string).
        :param top_k: Số lượng kết quả trả về.
        :return: Danh sách các điểm gần nhất.
        """
        query_embeddings = self.embedder.encode(texts)

        search_results = []
        for query_embedding in query_embeddings:
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )
            search_results.append(result)

        return search_results


class ChromaVectorStore(VectorStore):
    def __init__(
        self, persistent_path: str, embedder: EmbeddingFunction, collection_name: str
    ):
        super().__init__(collection_name)
        self.client = chromadb.PersistentClient(path=persistent_path)

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedder,
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedder,
            )

    def search(self, texts, top_k=5) -> list[Any] | None:
        """
        Tìm kiếm các vector gần nhất trong cơ sở dữ liệu vector.
        :param texts: Danh sách các truy vấn (dạng string).
        :param top_k: Số lượng kết quả trả về.
        :return: Danh sách các điểm gần nhất.
        """

        results = self.collection.query(query_texts=texts, n_results=top_k).get(
            "documents"
        )

        if not results:
            return None

        documents = []
        for result in results:
            [documents.append(r) for r in result]

        return documents
