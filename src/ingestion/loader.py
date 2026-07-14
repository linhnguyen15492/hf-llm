import json
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


def get_book_text_objects():
    # Source location
    text_objs = list()
    api_base_url = (
        "https://api.github.com/repos/progit/progit2/contents/book"  # Book base URL
    )
    chapter_urls = [
        "/01-introduction/sections",
        "/02-git-basics/sections",
    ]  # List of section URLs

    # Loop through book chapters
    for chapter_url in chapter_urls:
        response = requests.get(
            api_base_url + chapter_url
        )  # Get the JSON data for the section files in the chapter

        # Loop through inner files (sections)
        for file_info in response.json():
            if file_info["type"] == "file":  # Only process files (not directories)
                file_response = requests.get(file_info["download_url"])

                # Build objects including metadata
                chapter_title = file_info["download_url"].split("/")[-3]
                filename = file_info["download_url"].split("/")[-1]
                text_obj = {
                    "body": file_response.text,
                    "chapter_title": chapter_title,
                    "filename": filename,
                }
                text_objs.append(text_obj)
    return text_objs


def build_chunk_objs(book_text_obj, chunks):
    """
    Constructs a list of chunk objects from a given book text object
    and its associated chunks.

    Args:
        book_text_obj (dict): A dictionary containing metadata for the book text,
                              including 'chapter_title' and 'filename'.
        chunks (list): A list of chunks that represent parts of the book text.

    Returns:
        list: A list of dictionaries, each representing a chunk object
              with 'chapter_title', 'filename', 'chunk', and 'chunk_index'.
    """
    chunk_objs = list()  # Initialize an empty list to store chunk objects

    # Iterate over the chunks with an index
    for i, c in enumerate(chunks):
        # Create a dictionary for each chunk with its associated data
        chunk_obj = {
            "chapter_title": book_text_obj[
                "chapter_title"
            ],  # Chapter title from the book text object
            "filename": book_text_obj["filename"],  # Filename from the book text object
            "chunk": c,  # The actual chunk of text
            "chunk_index": i,  # The index of the chunk in the list
        }
        # Append the constructed chunk object to the list
        chunk_objs.append(chunk_obj)

    # Return the list of chunk objects
    return chunk_objs
