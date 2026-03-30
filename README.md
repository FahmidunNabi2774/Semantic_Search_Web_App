# Semantic Search Web App

A full-stack semantic search engine for StackOverflow programming questions using FastAPI, sentence-transformers, and FAISS.

## Project Structure

- `app.py` - FastAPI app + HTML/CSS/JavaScript frontend.
- `search.py` - Query embedding and FAISS cosine similarity search.
- `data_loader.py` - JSON data loading and numpy vector preparation.
- `data.json` - Dataset of StackOverflow records (`question`, `answer`, `embedding`).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.
