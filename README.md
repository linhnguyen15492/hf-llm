# hf-llm

## RAG project folder structure

![alt text](./img/rag_folder.png)

## quickstart with Qdrant Locally

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant   
```