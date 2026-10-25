"""Create the Qdrant collection used by Dash Dog. Run this before ingestion."""
from qdrant_client import QdrantClient, models
from config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_DIMENSIONS
)

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client.collection_exists(collection_name=COLLECTION_NAME):
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        print(f"Configured embedding size should be {EMBEDDING_DIMENSIONS}.")
        print("If its vector size differs, do not ingest until you resolve the mismatch.")
        print(info.config.params)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )
    print(
        f"Created collection '{COLLECTION_NAME}' with "
        f"{EMBEDDING_DIMENSIONS}-dimensional cosine vectors."
    )

if __name__ == "__main__":
    main()
