from embeddings.embedding import APIEmbedder
import os
from llm.base_llm import BaseLLM
from retrieval.retriever import QdrantRetriever, Retriever
from src.prompts.prompt_builder import PromptBuilder
from vectordb.vector_store import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

INSTRUCTIONS = """
Your task is to answer questions from the course participants based on the provided context.

Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, respond with "I don't know."
"""

PROMPT_TEMPLATE = """
CONTEXT: {context}
QUESTION: {question}
""".strip()


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model, input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


def main():
    path = "data\\corpus_law_pub.json"

    embedder = APIEmbedder()

    qdrant_client = QdrantVectorStore(
        embedder=embedder,
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name="vietnam_law_corpus",
        vector_size=int(os.environ.get("EMBEDDING_DIM", "1536")),
    )

    retriever = QdrantRetriever(
        embedder=embedder,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name="vietnam_law_corpus",
    )

    documents = qdrant_client.process_and_chunk_law_json(path)
    qdrant_client.insert_chunks_to_vector_db(documents[:2])

    result = retriever.retrieve(
        "Nguyên đơn đòi tiền mua bán đất nhưng bị đơn không chịu trả quyền sở hữu",
        top_k=1,
    )
    print("Search Results:", result)


class FAQRag:
    def __init__(
        self,
        llm_client: BaseLLM,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.prompt_builder = prompt_builder

    def _build_context(self, search_results):
        return self.prompt_builder.build_context(search_results)

    def _build_prompt(self, query, context):
        return self.prompt_builder.build_prompt(query, context)

    def ask(self, query):
        search_results = self.retriever.retrieve(query, top_k=5)
        context = self._build_context(search_results)
        prompt = self._build_prompt(query, context)

        response = self.llm_client.generate_response(
            instructions=self.prompt_builder.INSTRUCTIONS,
            prompt=prompt,
        )

        return response
