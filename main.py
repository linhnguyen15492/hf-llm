from llm.gemini_llm import GeminiLLM
from llm.openai_llm import OpenAILLM
from rag import PROMPT_TEMPLATE, FAQRag
from embeddings.embedding import LocalEmbeddingFunction
from retrieval.retriever import SimpleRetriever
from src.ingestion.loader import load_faq_data, ingest_data
from vectordb.vector_store import ChromaVectorStore
from src.config.settings import settings

MODEL_PATH = "models/"

INSTRUCTIONS = """
Your task is to answer questions from the course participants based on the provided context.

Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, just respond with "I don't know."
"""

PROMPT_TEMPLATE = """
CONTEXT: {context}
QUESTION: {question}
""".strip()


def main():
    # ingest_data()

    embedding_function = LocalEmbeddingFunction(
        model_name=settings.local_embedding_model,
        cache_folder=settings.local_embedding_model_cache_dir,
    )

    vector_store = ChromaVectorStore(
        persistent_path=settings.chromadb_dir,
        collection_name="faq_collection",
        embedder=embedding_function,
    )

    retriever = SimpleRetriever(vector_store)

    openai_llm = OpenAILLM(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
    )

    gemini_llm = GeminiLLM(
        model=settings.gemini_llm_model,
        api_key=settings.gemini_api_key,
    )

    questions = [
        "I just discovered the course. Can I join now?",
        "How do I get a certificate?",
    ]

    rag = FAQRag(
        llm_client=gemini_llm,
        retriever=retriever,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
    )

    for question in questions:
        answer = rag.ask(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()
