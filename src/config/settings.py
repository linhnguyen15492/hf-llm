from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:latest"
    openai_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"

    # Embedding
    embedding_provider: str = "sentence_transformer"
    local_embedding_model: str = "BAAI/bge-base-en-v1.5"
    local_embedding_model_cache_dir: str = "models/"

    # Reranker
    reranker_provider: str = "bge"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval
    top_k: int = 20
    rerank_top_k: int = 5

    # Storage
    data_dir: str = "data"

    chromadb_dir: str = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
