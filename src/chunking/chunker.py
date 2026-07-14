from typing import List


def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    """
    Splits a given text into chunks of a specified fixed size.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of words per chunk.

    Returns:
        List[str]: A list of text chunks, each containing up to 'chunk_size' words.
    """
    # Split the input text into individual words
    text_words = text.split()

    # Initialize a list to hold the chunks of words
    chunks = []

    # Iterate over the word indices in steps of 'chunk_size'
    for i in range(0, len(text_words), chunk_size):
        # Select a sublist of words from 'i' to 'i + chunk_size'
        chunk_words = text_words[i : i + chunk_size]

        # Join the selected words into a single string with spaces in between
        chunk = " ".join(chunk_words)

        # Add the chunk to the list of chunks
        chunks.append(chunk)

    # Return the list of word chunks
    return chunks


def get_chunks_fixed_size_with_overlap(
    text: str, chunk_size: int, overlap_fraction: float
) -> List[str]:
    """
    Splits a given text into chunks of a fixed size with a specified overlap fraction between consecutive chunks.

    Parameters:
    - text (str): The input text to be split into chunks.
    - chunk_size (int): The number of words each chunk should contain.
    - overlap_fraction (float): The fraction of the chunk size that should overlap with the adjacent chunk.
      For example, an overlap_fraction of 0.2 means 20% of the chunk size will be used as overlap.

    Returns:
    - List[str]: A list of chunks (each a string) where each chunk might overlap with its adjacent chunk.
    """

    # Split the text into individual words
    text_words = text.split()

    # Calculate the number of words to overlap between consecutive chunks
    overlap_int = int(chunk_size * overlap_fraction)

    # Initialize a list to store the resulting chunks
    chunks = []

    # Iterate over text in steps of chunk_size to create chunks
    for i in range(0, len(text_words), chunk_size):
        # Determine the start and end indices for the current chunk,
        # taking into account the overlap with the previous chunk
        chunk_words = text_words[max(i - overlap_int, 0) : i + chunk_size]

        # Join the selected words to form a chunk string
        chunk = " ".join(chunk_words)

        # Append the chunk to the list of chunks
        chunks.append(chunk)

    # Return the list of chunks
    return chunks


# Split the text into paragraphs
def get_chunks_by_paragraph(source_text: str) -> List[str]:
    return source_text.split("\n\n")


# Split the text by Asciidoc section markers
def get_chunks_by_asciidoc_sections(source_text: str) -> List[str]:
    return source_text.split("\n==")


def mixed_chunking(source_text):
    """
    Splits the given source_text into chunks using a mix of fixed-size and variable-size chunking.
    It first splits the text by Asciidoc markers and then processes the chunks to ensure they are
    of appropriate size. Smaller chunks are merged with the next chunk, and larger chunks can be
    further split at the middle or specific markers within the chunk.

    Args:
    - source_text (str): The text to be chunked.

    Returns:
    - list: A list of text chunks.
    """

    # Split the text by Asciidoc marker
    chunks = source_text.split("\n==")

    # Chunking logic
    new_chunks = []
    chunk_buffer = ""
    min_length = 25

    for chunk in chunks:
        new_buffer = chunk_buffer + chunk  # Create new buffer
        new_buffer_words = new_buffer.split(" ")  # Split into words
        if (
            len(new_buffer_words) < min_length
        ):  # Check whether buffer length is too small
            chunk_buffer = new_buffer  # Carry over to the next chunk
        else:
            new_chunks.append(new_buffer)  # Add to chunks
            chunk_buffer = ""

    if len(chunk_buffer) > 0:
        new_chunks.append(chunk_buffer)  # Add last chunk, if necessary

    return new_chunks
