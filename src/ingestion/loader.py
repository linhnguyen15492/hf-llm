import requests
from chromadb import EmbeddingFunction
from models.document import FAQDocument
import chromadb


def load_faq_data() -> list[FAQDocument]:
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(FAQDocument(**item) for item in course_data)

    return documents


def ingest_faq(documents: list[FAQDocument], collection_name: str, persistent_path: str,
               embedding_function: EmbeddingFunction):
    client = chromadb.PersistentClient(path=persistent_path)
    collection = client.get_collection(collection_name)
    if collection:
        print("collection exists")
        return

    print("collection does not exist")
    print("collection created")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    ids = []
    texts = []
    metadatas = []
    for document in documents:
        ids.append(document.id)
        texts.append(document.question + " " + document.answer)
        metadatas.append({
            "course": document.course,
            "section": document.section,
        })

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    print("faq ingestion complete")
