# hf-llm

## RAG architecture

```mermaid
flowchart TD
    U([User])

    APP[Application]

    DB[(DB)]
    DOCS[[D1 ... D5]]

    PROMPT[Build Prompt<br/>Question + Context]

    LLM[LLM]

    ANSWER([Answer])

    U -->|Question| APP

    APP -->|Query| DB
    DB -->|Retrieved Data| DOCS
    DOCS --> APP

    APP --> PROMPT
    PROMPT --> LLM

    LLM --> ANSWER
    ANSWER --> U
```

## RAG project folder structure

![alt text](./img/rag_folder.png)

## quickstart with Qdrant Locally

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

## Cai project thành một package (editable install)

```bash
uv pip install -e .
```

## Function Calling

The flow that broke:

```mermaid
flowchart TD
    U([User: How do I run Olama?])
    S[search - Olama - no useful results]
    A([LLM: I don't have information about Olama.])

    U --> S --> A
```

### The agent alternative

```mermaid
flowchart TD
    U([User: How do I run Olama?])
    L1[LLM: I'll search for 'Olama']
    S1[search - Olama - no useful results]
    L2[LLM: Hmm, no results. Maybe a typo for 'Ollama'?]
    S2[search - Ollama - found results!]
    A([LLM: Here's how to run Ollama locally...])

    U --> L1 --> S1 --> L2 --> S2 --> A
```
