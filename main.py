import chromadb
import joblib
from llm.gemini_llm import GeminiLLM
from llm.openai_llm import OpenAILLM
import numpy as np
from chromadb.utils import embedding_functions
from qdrant_client import QdrantClient
from rag import PROMPT_TEMPLATE, FAQRag
from sentence_transformers import SentenceTransformer

from embeddings.embedding import LocalEmbeddingFunction
from models.document import FAQDocument
from retrieval.retriever import BM25Retriever, SemanticRetriever, SimpleRetriever
from src.prompts.prompt_templates import house_info_layout
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from src.llm.base_llm import generate_with_single_input
from src.utils.utils import (
    read_dataframe,
    pprint,
    cosine_similarity,
    reciprocal_rank_fusion,
)
from src.ingestion.loader import load_faq_data, ingest_data
from vectordb.vector_store import ChromaVectorStore
from src.config.settings import settings
from pathlib import Path

MODEL_PATH = "models/"

INSTRUCTIONS = """
Your task is to answer questions from the course participants based on the provided context.

Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, just respond with "I don't know."
"""

PROMPT_TEMPLATE = """
CONTEXT: {context}
QUESTION: {question}
""".strip()

# EMBEDDINGS = joblib.load("data/embeddings.joblib")
# NEWS_DATA = read_dataframe("data/news_data_dedup.csv")
# pprint(NEWS_DATA[5])
# CORPUS = [x['title'] + " " + x['description'] for x in NEWS_DATA]
# embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=MODEL_PATH)


def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    dataset (list or sequence): The dataset from which elements are to be retrieved. It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """

    # output = [NEWS_DATA[index] for index in indices]

    # return output

    pass


# def sample_rag():
#     print("Hello from hf-llm!")
#
#     # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
#     # model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", cache_dir=MODEL_PATH)
#     embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=MODEL_PATH)
#
#     # generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
#     # generator(
#     #     "In this course, we will teach you how to",
#     #     max_length=30,
#     #     num_return_sequences=2,
#     # )
#     # print(generator)
#
#     # You can add more functionality here as needed
#     house_data = [
#         {
#             "address": "123 Maple Street",
#             "city": "Springfield",
#             "state": "IL",
#             "zip": "62701",
#             "bedrooms": 3,
#             "bathrooms": 2,
#             "square_feet": 1500,
#             "price": 230000,
#             "year_built": 1998,
#         },
#         {
#             "address": "456 Elm Avenue",
#             "city": "Shelbyville",
#             "state": "TN",
#             "zip": "37160",
#             "bedrooms": 4,
#             "bathrooms": 3,
#             "square_feet": 2500,
#             "price": 320000,
#             "year_built": 2005,
#         },
#     ]
#
#     layout = house_info_layout(house_data)
#     print(layout)
#
#     # Load the 20 Newsgroups dataset
#     newsgroups_train: Any = fetch_20newsgroups(
#         subset="train", shuffle=True, random_state=42, data_home="./data"
#     )
#
#     print("Number of documents in the training set:", len(newsgroups_train))
#
#     # Convert the dataset to a DataFrame for easier handling
#     df = pd.DataFrame(
#         {"text": newsgroups_train.data, "category": newsgroups_train.target}
#     )
#
#     # Display some basic information about the dataset
#     print(df.head())
#     print("\nDataset Size:", df.shape)
#     print("\nNumber of Categories:", len(newsgroups_train.target_names))
#     print("\nCategories:", newsgroups_train.target_names)
#
#     generated_text = generate_with_single_input(
#         prompt="Write a short story about a robot learning to love.",
#         role="user",
#     )
#     print(generated_text)
#
#     bm25_retriever = BM25Retriever(CORPUS)
#     bm25_indices = bm25_retriever.retrieve("What are the recent news about GDP?")
#     print(bm25_indices)
#
#     # Example usage
#     query = "RAG is awesome"
#     # Using, but truncating the result to not pollute the output, don't truncate it in the exercise.
#     print(embedder.encode(query)[:40])
#
#     query1 = "What are the primary colors"
#     query2 = "Yellow, red and blue"
#     query3 = "Cats are friendly animals"
#
#     query1_embed = embedder.encode(query1)
#     query2_embed = embedder.encode(query2)
#     query3_embed = embedder.encode(query3)
#
#     print(f"Similarity between '{query1}' and '{query2}' = {cosine_similarity(query1_embed, query2_embed)[0]}")
#     print(f"Similarity between '{query1}' and '{query3}' = {cosine_similarity(query1_embed, query3_embed)[0]}")
#
#     query = "Taylor Swift"
#     query_embed = embedder.encode(query)
#     # The result is a matrix with one matrix per sample. Since there is only one sample (the query), it is a matrix with one matrix within.
#     # This is why you need to get the first element
#     similarity_scores = cosine_similarity(query_embed, EMBEDDINGS)
#     similarity_indices = np.argsort(
#         -similarity_scores)  # Sort on decreasing order (sort the negative on increasing order), but return the indices
#     # Top 2 indices
#     top_2_indices = similarity_indices[:2]
#     print(top_2_indices)
#
#     # Retrieving the data
#     print(query_news(top_2_indices))
#
#     semantic_retriever = SemanticRetriever(embedder, EMBEDDINGS)
#     semantic_indices = semantic_retriever.retrieve("What are the recent news about GDP?")
#     print(semantic_indices)
#
#     rrf_list = reciprocal_rank_fusion(semantic_indices, bm25_indices)
#     print(f"Semantic Search List: {semantic_indices}")
#     print(f"BM25 List: {bm25_indices}")
#     print(f"RRF List: {rrf_list}")
#


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
