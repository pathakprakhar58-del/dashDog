import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

COLLECTION_NAME = "Finaal_Data_Collection"

def search_mits(query_text: str, top_k: int = 3):
    # 1. Convert question into an embedding vector
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    query_vector = response.data[0].embedding

    # 2. Query Qdrant for closest vector matches
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    # 3. Print the top matching chunks and source URLs
    print(f"\n🔍 Query: '{query_text}'\n" + "="*50)
    for i, point in enumerate(results, 1):
        payload = point.payload
        print(f"\nResult #{i} (Score: {point.score:.4f})")
        print(f"Source URL: {payload.get('source_url')}")
        print(f"Content Chunk:\n{payload.get('text')[:300]}...")

if __name__ == "__main__":
    query = input("enter the question : ")
    if query.strip():
        search_mits(query)