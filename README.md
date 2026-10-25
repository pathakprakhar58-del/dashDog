# Dash Dog: scraped JSON → OpenAI embeddings → Qdrant

## 1. Requirements
- Python 3.10+
- A running Qdrant server at `http://localhost:6333`, or update `QDRANT_URL` for Qdrant Cloud.
- An OpenAI API key with access/credits for embeddings.

## 2. Set up the environment (Windows PowerShell)
Open a terminal in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can instead run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Configure the API key
Copy `.env.example` to `.env`, then replace the placeholder with your own key.
Do not paste your API key into chat, commit it to Git, or share `.env`.

## 4. Add scraped JSON files
Put your `.json` files in the `data/` folder. Subfolders are also scanned.

The script supports:
- a JSON list of records;
- a JSON object containing a list under a common key such as `records`, `items`, `results`, `data`, `pages`, or `documents`;
- a single JSON object.

It converts scalar fields into labeled text, chunks long text, and tries to capture `title`/`name` and `url`/`source_url`/`link` fields. Your exact scraped JSON schema may need a small adjustment for best results.

## 5. Start Qdrant
Start your Qdrant server/container before running the scripts. The Python client does not start the server itself.

## 6. Create the collection
```powershell
python setup_qdrant.py
```
This creates `mits_knowledge` with 1536-dimensional cosine vectors. If the collection already exists, inspect the printed configuration. Do not ingest if the existing collection has a different vector size.

## 7. Ingest the JSON
```powershell
python ingest_json.py
```
This reads the JSON files, prepares chunks, calls OpenAI embeddings, checks vector dimensions, and writes vectors plus metadata payloads to Qdrant.

## Important notes
- `text-embedding-3-small` defaults to 1536 dimensions. This project explicitly requests 1536 dimensions so the vectors match the collection.
- Qdrant payload fields do not need to have 1536 dimensions. Payload is metadata (text, title, source URL, file name, chunk index, etc.).
- This starter script uses character-based chunking. For production, inspect sample chunks and adapt extraction/chunking to the real JSON structure before embedding the entire dataset.
- Re-running ingestion uses stable point IDs for unchanged chunks, which helps update points rather than creating duplicates. Changed chunks can leave old points behind; clear/rebuild the collection carefully if you need a completely fresh index.
