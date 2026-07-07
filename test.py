from src.retrieval.retriever import HybridRetriever
import json
from pathlib import Path
from vectordb.vector_store import ChromaVectorStore
from embeddings.embedding import get_local_embedding_function
from config.settings import settings

embedding_function = get_local_embedding_function(
    model_name=settings.local_embedding_model,
    cache_folder=settings.local_embedding_model_cache_dir,
)

vector_store = ChromaVectorStore(
    persistent_path=settings.chromadb_dir,
    collection_name="faq_collection",
    embedder=embedding_function,
)


corpus_path = "storage\\faq_corpus.json"


with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)


retriever = HybridRetriever(vectordb=vector_store, corpus_path=corpus_path)
documents = retriever.retrieve(["How can I get a certificate?"], top_k=5)
print(len(documents))
for doc in documents:
    print(doc)
    print("-" * 20)
