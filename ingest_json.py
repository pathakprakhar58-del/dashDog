"""Read scraped JSON, chunk its text, embed chunks, and upsert into Qdrant."""
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from config import (
    DATA_DIR, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE, UPSERT_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
)
from embedding_utils import embed_texts

CONTAINER_KEYS = ("records", "items", "results", "data", "pages", "documents")
URL_KEYS = {"url", "source_url", "page_url", "link", "href"}
TITLE_KEYS = {"title", "name", "heading", "page_title"}

def load_records(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ Skipping malformed JSON file '{path}': {e}")
        return []

def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    return ""

def record_to_text(record: Any) -> tuple[str, str | None, str | None]:
    """Turn a record's scalar fields into readable text and capture URL/title."""
    if not isinstance(record, dict):
        if isinstance(record, (str, int, float, bool)):
            return str(record).strip(), None, None
        return json.dumps(record, ensure_ascii=False, indent=2), None, None

    lines = []
    source_url = None
    title = None

    def walk(obj: Any, prefix: str = ""):
        nonlocal source_url, title
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key)
                key_lower = key_str.lower()
                path_name = f"{prefix}.{key_str}" if prefix else key_str

                if isinstance(value, (dict, list)):
                    walk(value, path_name)
                    continue

                value_str = scalar_text(value)
                if not value_str:
                    continue

                if key_lower in URL_KEYS and source_url is None:
                    source_url = value_str
                if key_lower in TITLE_KEYS and title is None:
                    title = value_str

                # Keep labels so values remain understandable after retrieval.
                lines.append(f"{path_name}: {value_str}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{prefix}[{i}]")
        else:
            value_str = scalar_text(obj)
            if value_str:
                lines.append(f"{prefix}: {value_str}" if prefix else value_str)

    walk(record)
    return "\n".join(lines).strip(), source_url, title

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks, preferring word boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("CHUNK_SIZE must be > 0 and CHUNK_OVERLAP must be smaller than it.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def stable_point_id(source_file: str, record_index: int, chunk_index: int, text: str) -> str:
    raw = f"{source_file}|{record_index}|{chunk_index}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")
    json_files = sorted(DATA_DIR.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {DATA_DIR}. Put your scraped JSON files in the data/ folder."
        )

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist. Run: python setup_qdrant.py"
        )

    info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    print(f"Using collection: {COLLECTION_NAME}")
    print(f"Configured embedding dimensions: {EMBEDDING_DIMENSIONS}")
    print("Collection configuration:", info.config.params)

    prepared = []
    total_records = 0
    for path in json_files:
        records = load_records(path)
        print(f"Reading {path.name}: {len(records)} record(s)")
        for record_index, record in enumerate(records):
            text, source_url, title = record_to_text(record)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_index, chunk in enumerate(chunks):
                prepared.append({
                    "text": chunk,
                    "source_file": str(path.relative_to(DATA_DIR)),
                    "source_url": source_url,
                    "title": title,
                    "record_index": record_index,
                    "chunk_index": chunk_index,
                })
            total_records += 1

    if not prepared:
        print("No usable text found in the JSON files.")
        return

    print(f"Prepared {total_records} record(s) into {len(prepared)} text chunk(s).")
    stored = 0

    for start in range(0, len(prepared), EMBEDDING_BATCH_SIZE):
        batch = prepared[start:start + EMBEDDING_BATCH_SIZE]
        vectors = embed_texts([item["text"] for item in batch])

        points = []
        for item, vector in zip(batch, vectors):
            point_id = stable_point_id(
                item["source_file"], item["record_index"], item["chunk_index"], item["text"]
            )
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": item["text"],
                    "source_file": item["source_file"],
                    "source_url": item["source_url"],
                    "title": item["title"],
                    "record_index": item["record_index"],
                    "chunk_index": item["chunk_index"],
                },
            ))

        for p_start in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[p_start:p_start + UPSERT_BATCH_SIZE],
                wait=True,
            )
            stored += len(points[p_start:p_start + UPSERT_BATCH_SIZE])
        print(f"Stored {stored}/{len(prepared)} chunks")

    print(f"Done. Stored {stored} chunks in Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
