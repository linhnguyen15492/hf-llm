import json
import os
from pathlib import Path
import requests
import chromadb
from tqdm import tqdm
from config.settings import settings
from embeddings.embedding import get_local_embedding_function
from models.document import FAQDocument
from dataclasses import asdict


def load_faq_data(save_path: str) -> list[FAQDocument]:
    path = Path(save_path)
    if path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [FAQDocument(**item) for item in data]

    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(FAQDocument(**item) for item in course_data)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump([asdict(doc) for doc in documents], f, ensure_ascii=False, indent=4)

    return documents


def ingest_data(save_path: str):
    faq_documents = load_faq_data(save_path=save_path)
    client = chromadb.PersistentClient(path=settings.chromadb_dir)

    try:
        collection = client.get_collection("faq_collection")
    except Exception as e:
        print(f"Could not connect to collection: {e}")
        collection = None

    if collection:
        print("Collection exists. Skipping ingestion.")
        return

    print("Collection does not exist. Ingesting data...")
    embedding_function = get_local_embedding_function(
        model_name=settings.local_embedding_model,
        cache_folder=settings.local_embedding_model_cache_dir,
    )

    collection = client.create_collection(
        name="faq_collection", embedding_function=embedding_function
    )

    ids = []
    texts = []
    metadatas = []
    count = 0
    for document in tqdm(faq_documents, desc="Processing documents"):
        ids.append(str(document.id))
        texts.append(document.question + " " + document.answer)
        metadatas.append(
            {
                "course": document.course,
                "section": document.section,
            }
        )
        count += 1
    print("Retrieving embeddings...")

    batch_size = 64
    for i in tqdm(range(0, len(ids), batch_size), desc="Retrieving embeddings"):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print(f"faq ingestion complete. Total documents ingested: {count}")
