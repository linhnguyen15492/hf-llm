from typing import List, Dict

import requests
import json
import numpy as np
import os
import pandas as pd
from dateutil import parser
from pprint import pprint as original_pprint


def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
        list1 (list[int]): A list of indices of the top-k documents that match the query.
        list2 (list[int]): Another list of indices of the top-k documents that match the query.
        top_k (int): The number of top documents to consider from each list for fusion. Defaults to 5.
        K (int): A constant used in the RRF formula. Defaults to 60.

    Returns:
        list[int]: A list of indices of the top-k documents sorted by their RRF scores.
    """

    # Create a dictionary to store the RRF scores for each document index
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        # Calculate the RRF score for each document index
        for rank, item in enumerate(
            lst, start=1
        ):  # Start = 1 set the first element as 1 and not 0.
            # This is a convention on how ranks work (the first element in ranking is denoted by 1 and not 0 as in lists)
            # If the item is not in the dictionary, initialize its score to 0
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Update the RRF score for each document index using the formula 1 / (rank + K)
            rrf_scores[item] += 1 / (rank + K)

    # Sort the document indices based on their RRF scores in descending order
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # Slice the list to get the top-k document indices
    top_k_indices = [int(x) for x in sorted_items[:top_k]]

    return top_k_indices


def cosine_similarity(v1, array_of_vectors):
    """
    Compute the cosine similarity between a vector and an array of vectors.

    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of cosine similarities between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store similarities
    similarities = []

    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]

    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Compute the dot product of v1 and v2
        dot_product = np.dot(v1, v2)
        # Compute the norms of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Compute the cosine similarity and append to the list
        similarity = dot_product / (norm_v1 * norm_v2)
        similarities.append(similarity)
    return np.array(similarities)


def euclidean_distance(v1, array_of_vectors):
    """
    Compute the Euclidean distance between a vector and an array of vectors.

    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of Euclidean distances between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store distances
    distances = []

    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]

    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Check if the input arrays have the same shape
        if v1.shape != v2.shape:
            raise ValueError(
                f"Shapes don't match: v1 shape: {v1.shape}, v2 shape: {v2.shape}"
            )
        # Calculate the Euclidean distance and append to the list
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        distances.append(dist)
    return distances


def format_date(date_string):
    # Parse the input string into a datetime object
    date_object = parser.parse(date_string)
    # Format the date to "YYYY-MM-DD"
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date


def pprint(*args, **kwargs):
    kwargs.setdefault("sort_dicts", False)
    original_pprint(*args, **kwargs)


def house_info_layout(houses):
    # Create an empty string
    layout = ""
    # Iterate over the houses
    for house in houses:
        # For each house, append the information to the string using f-strings
        # The following way using brackets is a good way to make the code readable as in each line you can start a new f-string that will appended to the previous one
        layout += (
            f"House located at {house['address']}, {house['city']}, {house['state']} {house['zip']} with "
            f"{house['bedrooms']} bedrooms, {house['bathrooms']} bathrooms, "
            f"{house['square_feet']} sq ft area, priced at ${house['price']}, "
            f"built in {house['year_built']}.\n"
        )  # Don't forget the new line character at the end!
    return layout


def read_dataframe(path):
    df = pd.read_csv(path)

    # Apply the custom date formatting function to the relevant columns
    df["published_at"] = df["published_at"].apply(format_date)
    df["updated_at"] = df["updated_at"].apply(format_date)

    # Convert the DataFrame to dictionary after formatting
    df = df.to_dict(orient="records")
    return df


def get_together_key():
    """
    Get the Together API key from environment variables.
    """
    return os.environ.get("TOGETHER_API_KEY", "")


# def generate_with_single_input(
#         prompt: str,
#         role: str = "user",
#         top_p: float = None,
#         temperature: float = None,
#         max_tokens: int = 500,
#         model: str = "Qwen/Qwen3.5-9B",
#         together_api_key=None,
#         **kwargs,
# ):
#     # Remove None parameters for Together API - don't set to string 'none'
#     if top_p is None:
#         payload_top_p = None
#     else:
#         payload_top_p = top_p
#     if temperature is None:
#         payload_temperature = None
#     else:
#         payload_temperature = temperature
#
#     payload = {
#         "model": model,
#         "messages": [{"role": role, "content": prompt}],
#         "max_tokens": max_tokens,
#         "reasoning": {"enabled": False},
#         **kwargs,
#     }
#     # Only add temperature and top_p if they're not None
#     if payload_temperature is not None:
#         payload["temperature"] = payload_temperature
#     if payload_top_p is not None:
#         payload["top_p"] = payload_top_p
#
#     if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
#         url = os.path.join(get_proxy_url(), "v1/chat/completions")
#         response = requests.post(url, json=payload, verify=False)
#         if not response.ok:
#             raise Exception(f"Error while calling LLM: {response.text}")
#         try:
#             json_dict = json.loads(response.text)
#         except Exception as e:
#             raise Exception(
#                 f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
#             )
#     else:
#         if together_api_key is None:
#             together_api_key = os.environ["TOGETHER_API_KEY"]
#         client = Together(api_key=together_api_key)
#         json_dict = client.chat.completions.create(**payload).model_dump()
#         json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
#             "message"
#         ]["role"].name.lower()
#     try:
#         output_dict = {
#             "role": json_dict["choices"][-1]["message"]["role"],
#             "content": json_dict["choices"][-1]["message"]["content"],
#         }
#     except Exception as e:
#         raise Exception(
#             f"Failed to get correct output dict. Please try again. Error: {e}"
#         )
#     return output_dict
#
#
# def generate_with_multiple_input(
#         messages: List[Dict],
#         top_p: float = None,
#         temperature: float = None,
#         max_tokens: int = 500,
#         model: str = "Qwen/Qwen3.5-9B",
#         together_api_key=None,
#         **kwargs,
# ):
#     # Remove None parameters for Together API
#     if top_p is None:
#         payload_top_p = None
#     else:
#         payload_top_p = top_p
#     if temperature is None:
#         payload_temperature = None
#     else:
#         payload_temperature = temperature
#
#     payload = {
#         "model": model,
#         "messages": messages,
#         "max_tokens": max_tokens,
#         "reasoning": {"enabled": False},
#         **kwargs,
#     }
#     # Only add temperature and top_p if they're not None
#     if payload_temperature is not None:
#         payload["temperature"] = payload_temperature
#     if payload_top_p is not None:
#         payload["top_p"] = payload_top_p
#
#     if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
#         url = os.path.join(get_proxy_url(), "v1/chat/completions")
#         response = requests.post(url, json=payload, verify=False)
#         if not response.ok:
#             raise Exception(f"Error while calling LLM: {response.text}")
#         try:
#             json_dict = json.loads(response.text)
#         except Exception as e:
#             raise Exception(
#                 f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
#             )
#     else:
#         if together_api_key is None:
#             together_api_key = os.environ["TOGETHER_API_KEY"]
#         client = Together(api_key=together_api_key)
#         json_dict = client.chat.completions.create(**payload).model_dump()
#         json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
#             "message"
#         ]["role"].name.lower()
#     try:
#         output_dict = {
#             "role": json_dict["choices"][-1]["message"]["role"],
#             "content": json_dict["choices"][-1]["message"]["content"],
#         }
#     except Exception as e:
#         raise Exception(
#             f"Failed to get correct output dict. Please try again. Error: {e}"
#         )
#     return output_dict


# def retrieve(model, query, top_k=5):
#     query_embedding = model.encode(query)
#
#     similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), EMBEDDINGS)[0]
#
#     similarity_indices = np.argsort(-similarity_scores)
#
#     top_k_indices = similarity_indices[:top_k]
#
#     return top_k_indices


# Define utility functions and classes
def generate_embedding(
    prompt: str,
):  # model: str = "BAAI/bge-base-en-v1.5", together_api_key = None, **kwargs):
    return model.encode(prompt).tolist()
    payload = {"model": model, "input": prompt, **kwargs}
    if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
        client = OpenAI(
            api_key="",  # Set any as dlai proxy does not use it. Set the together api key if using the together endpoint
            base_url="http://proxy.dlai.link/coursera_proxy/together/",  # If using together endpoint, add it here https://api.together.xyz/
            http_client=http_client,  # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint
        )
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict["data"][0]["embedding"]
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}"
            )
    else:
        if together_api_key is None:
            together_api_key = os.environ["TOGETHER_API_KEY"]
        client = Together(api_key=together_api_key)
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict["data"][0]["embedding"]
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}"
            )
