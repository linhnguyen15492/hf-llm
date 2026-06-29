from src.embeddings.embedder import Embedder


def main():
    embed = Embedder()

    query = "How does approximate nearest neighbor search work?"

    v = embed.encode(query)

    print("Q1. Embedding a query", v[0])


if __name__ == "__main__":
    main()
