from __future__ import annotations

import time
from textwrap import dedent
from ingestion.loader import load_faq_data
from tqdm.auto import tqdm
from rag import RAGBase
from config.settings import settings
from openai import OpenAI
from models.question import Questions
from pydantic import BaseModel
from typing import Any, TypeVar
import json
import pandas as pd

OutputType = TypeVar("OutputType", bound=BaseModel)


def calc_price(usage, input_price_per_million=0.75, output_price_per_million=4.50):
    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(
    usages, input_price_per_million=0.75, output_price_per_million=4.50
):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage, input_price_per_million, output_price_per_million)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type: type[OutputType],
    model,
) -> tuple[OutputType, Any]:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model, input=messages, text_format=output_type
    )

    if response is None:
        raise ValueError("Structured LLM response was None")

    if response.output_parsed is None:
        raise ValueError("Structured LLM response did not parse into an output")

    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type: type[OutputType],
    model,
    max_retries=3,
) -> tuple[OutputType, Any]:
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Structured LLM retry loop exited without a result")


def generate_ground_truth(
    doc,
    llm_client,
    data_gen_instructions,
    output_type: type[Questions],
    model,
):
    user_prompt = json.dumps(doc)

    try:
        out, usage = llm_structured_retry(
            llm_client, data_gen_instructions, user_prompt, output_type, model=model
        )
    except Exception as e:
        print(f"Error generating ground truth for document {doc['filename']}: {e}")
        return [], None

    if out is None or not hasattr(out, "questions") or out.questions is None:
        raise ValueError("LLM output is None or does not have 'questions' attribute")

    results = []

    for q in out.questions:
        results.append({"question": q, "document": doc["filename"]})

    return results, usage


def text_search(query):
    boost_dict = {"question": 3.0, "section": 0.5}

    return index.search(query, num_results=5, boost_dict=boost_dict)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model, input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results


def main():
    data_gen_instructions = dedent("""\
    You emulate a student who's taking our course.
    Formulate 5 questions this student might ask based on a FAQ record. The record
    should contain the answer to the questions, and the questions should be complete and not too short.
    If possible, use as fewer words as possible from the record.

    The output should resemble how people ask questions
    on the internet. Not too formal, not too short, not too long.
    """).strip()

    documents = load_faq_data(settings.faq_corpus_dir)
    documents_llm = []

    for doc in documents:
        if doc.course == "llm-zoomcamp":
            documents_llm.append(doc)

    documents = documents_llm

    openai_client = OpenAI(api_key=settings.openai_api_key)

    df_ground_truth = pd.read_csv("data/ground_truth-new.csv")
    ground_truth = df_ground_truth.to_dict(orient="records")


if __name__ == "__main__":
    main()
