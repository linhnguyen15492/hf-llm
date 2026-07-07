from llm.gemini_llm import GeminiLLM
from llm.openai_llm import OpenAILLM
from rag import FAQRag
from embeddings.embedding import LocalEmbeddingFunction, get_local_embedding_function
from retrieval.retriever import HybridRetriever, SimpleRetriever
from ingestion.loader import ingest_data
from vectordb.vector_store import ChromaVectorStore
from prompts.prompt_builder import FAQPromptBuilder
from config.settings import settings
import gradio as gr


def main():

    ingest_data(settings.faq_corpus_dir)

    embedding_function = get_local_embedding_function(
        model_name=settings.local_embedding_model,
        cache_folder=settings.local_embedding_model_cache_dir,
    )

    vector_store = ChromaVectorStore(
        persistent_path=settings.chromadb_dir,
        collection_name="faq_collection",
        embedder=embedding_function,
    )

    retriever = HybridRetriever(
        vectordb=vector_store, corpus_path=settings.faq_corpus_dir
    )

    openai_llm = OpenAILLM(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
    )

    gemini_llm = GeminiLLM(
        model=settings.gemini_llm_model,
        api_key=settings.gemini_api_key,
    )

    prompt_builder = FAQPromptBuilder()

    rag = FAQRag(
        llm_client=gemini_llm,
        retriever=retriever,
        prompt_builder=prompt_builder,
    )

    # questions = [
    #     "I just discovered the course. Can I join now?",
    #     "How do I get a certificate?",
    # ]

    # for question in questions:
    #     answer = rag.ask(question)
    #     print(f"Question: {question}")
    #     print(f"Answer: {answer}")
    #     print("-" * 50)

    # Set up Gradio interface
    iface = gr.Interface(
        fn=rag.ask,
        inputs="text",
        outputs="text",
        title="RAG Application",
        description="Ask a question, and the app will retrieve relevant information and provide an answer.",
    )

    iface.launch()


if __name__ == "__main__":
    main()
