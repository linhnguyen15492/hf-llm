from transformers import pipeline


def main():
    print("Hello from hf-llm!")
    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    print(generator)


if __name__ == "__main__":
    main()
