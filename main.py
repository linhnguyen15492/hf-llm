from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os
from src.prompts.prompt_templates import house_info_layout
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from src.llm.llm_client import generate_with_single_input

custom_path = "models/"

# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
# model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", cache_dir=custom_path)

model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=custom_path)

res = model.encode("RAG is awesome")
print(res.shape)


def main():
    print("Hello from hf-llm!")
    # generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    # generator(
    #     "In this course, we will teach you how to",
    #     max_length=30,
    #     num_return_sequences=2,
    # )
    # print(generator)

    # You can add more functionality here as needed
    house_data = [
        {
            "address": "123 Maple Street",
            "city": "Springfield",
            "state": "IL",
            "zip": "62701",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1500,
            "price": 230000,
            "year_built": 1998,
        },
        {
            "address": "456 Elm Avenue",
            "city": "Shelbyville",
            "state": "TN",
            "zip": "37160",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2500,
            "price": 320000,
            "year_built": 2005,
        },
    ]

    layout = house_info_layout(house_data)
    print(layout)

    # Load the 20 Newsgroups dataset
    newsgroups_train: Any = fetch_20newsgroups(
        subset="train", shuffle=True, random_state=42, data_home="./data"
    )

    print("Number of documents in the training set:", len(newsgroups_train))

    # Convert the dataset to a DataFrame for easier handling
    df = pd.DataFrame(
        {"text": newsgroups_train.data, "category": newsgroups_train.target}
    )

    # Display some basic information about the dataset
    print(df.head())
    print("\nDataset Size:", df.shape)
    print("\nNumber of Categories:", len(newsgroups_train.target_names))
    print("\nCategories:", newsgroups_train.target_names)

    generated_text = generate_with_single_input(
        prompt="Write a short story about a robot learning to love.",
        role="user",
    )
    print(generated_text)


if __name__ == "__main__":
    main()
