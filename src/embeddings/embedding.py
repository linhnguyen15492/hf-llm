import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
from openai import OpenAI
import os
from typing import List, Union
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from functools import lru_cache


class Embedding(ABC):
    def __init__(self, model: str):
        self.model_name = model


class Embedder:
    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled


class APIEmbedder:
    def __init__(
        self, api_url: str = None, api_key: str = None, model_name: str = None
    ):
        """
        Khởi tạo Embedder kết nối qua API.
        :param api_url: Đường dẫn Endpoint của API (Ví dụ: http://192.168.1.x:8000/v1/embeddings)
        :param api_key: Mã bảo mật API nếu Server yêu cầu xác thực
        :param model_name: Tên mô hình đang cấu hình trên Server API
        """
        self.api_url = api_url or os.environ.get("OPENAI_URL")
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

    def encode(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Hàm mã hóa danh sách văn bản thành các Vector qua API.
        Thiết kế gom cụm (Batching) để tối ưu đường truyền mạng và giảm tải cho Server API.
        """
        all_embeddings = []

        # Chia nhỏ danh sách chuỗi văn bản thành từng cụm (Batch) trước khi gửi qua mạng
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch_texts, model=self.model_name
                )

                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)

            except Exception as e:
                print(
                    f"[API Embedder Network Error] Thất bại khi kết nối tới Server Embedding: {e}"
                )
                raise e

        return all_embeddings


@register_embedding_function
class LocalEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name: str, cache_folder: str | None = None):
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return self.model.encode(input).tolist()

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return LocalEmbeddingFunction(config["model"])


@lru_cache(maxsize=None)
def get_local_embedding_function(
    model_name: str, cache_folder: str | None = None
) -> LocalEmbeddingFunction:
    """
    Sử dụng theo singleton pattern để lấy đối tượng EmbeddingFunction dựa trên tên mô hình và đường dẫn mô hình.
    Lấy đối tượng EmbeddingFunction dựa trên tên mô hình và đường dẫn mô hình.
    :param model_name: Tên mô hình embedding (ví dụ: "all-MiniLM-L6-v2")
    :param cache_folder: Đường dẫn tới thư mục chứa mô hình (nếu có)
    :return: Đối tượng EmbeddingFunction
    """
    return LocalEmbeddingFunction(model_name=model_name, cache_folder=cache_folder)
