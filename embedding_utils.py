"""OpenAI embedding helper functions."""
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to your local .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of non-empty strings and verify each vector's size."""
    if not texts:
        return []

    cleaned = [text if text.strip() else "[No text]" for text in texts]
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=cleaned,
        # Explicitly set dimensions so it matches the Qdrant collection.
        dimensions=EMBEDDING_DIMENSIONS,
    )
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    for i, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding {i} has {len(vector)} dimensions; expected "
                f"{EMBEDDING_DIMENSIONS}. Check config.py and the Qdrant collection."
            )
    return vectors
