"""FastAPI application for StackOverflow semantic search."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from data_loader import load_dataset
from search import SemanticSearcher

app = FastAPI(title="Semantic Search Web App")

searcher: SemanticSearcher | None = None


class SearchRequest(BaseModel):
    """Schema for semantic search requests."""

    query: str


class SearchResult(BaseModel):
    """Schema for an individual search result."""

    question: str
    answer: str
    answer_preview: str
    score: float


@app.on_event("startup")
def startup_event() -> None:
    """Preload dataset and FAISS index during API startup."""
    global searcher
    records = load_dataset("data.json")
    searcher = SemanticSearcher(records)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the single-page HTML frontend."""
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Semantic Search</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f6fb;
      color: #222;
    }
    .container {
      max-width: 900px;
      margin: 80px auto;
      padding: 0 20px;
    }
    h1 {
      text-align: center;
      margin-bottom: 30px;
    }
    .search-row {
      display: flex;
      gap: 10px;
      justify-content: center;
    }
    #query {
      width: 100%;
      max-width: 700px;
      font-size: 18px;
      padding: 16px;
      border: 1px solid #d0d5dd;
      border-radius: 10px;
      outline: none;
    }
    button {
      padding: 16px 22px;
      border: none;
      border-radius: 10px;
      background: #1f6feb;
      color: white;
      font-size: 16px;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.7;
      cursor: not-allowed;
    }
    #spinner {
      display: none;
      text-align: center;
      margin: 24px 0;
      font-size: 16px;
      color: #1f6feb;
    }
    .result {
      background: white;
      border: 1px solid #e4e7ec;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .question {
      font-weight: 700;
      margin-bottom: 8px;
    }
    .answer {
      color: #333;
      line-height: 1.45;
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Programming Semantic Search</h1>
    <div class=\"search-row\">
      <input id=\"query\" type=\"text\" placeholder=\"Ask a programming question...\" />
      <button id=\"searchBtn\">Search</button>
    </div>
    <div id=\"spinner\">Searching...</div>
    <div id=\"results\"></div>
  </div>

  <script>
    const queryInput = document.getElementById('query');
    const searchBtn = document.getElementById('searchBtn');
    const resultsContainer = document.getElementById('results');
    const spinner = document.getElementById('spinner');

    async function runSearch() {
      const query = queryInput.value.trim();
      if (!query) return;

      resultsContainer.innerHTML = '';
      spinner.style.display = 'block';
      searchBtn.disabled = true;

      try {
        const response = await fetch('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });

        const data = await response.json();

        if (!data.results || data.results.length === 0) {
          resultsContainer.innerHTML = '<p>No matching questions found.</p>';
          return;
        }

        for (const item of data.results) {
          const div = document.createElement('div');
          div.className = 'result';
          div.innerHTML = `
            <div class=\"question\">${item.question}</div>
            <div class=\"answer\">${item.answer_preview}</div>
          `;
          resultsContainer.appendChild(div);
        }
      } catch (error) {
        resultsContainer.innerHTML = '<p>Search failed. Please try again.</p>';
      } finally {
        spinner.style.display = 'none';
        searchBtn.disabled = false;
      }
    }

    searchBtn.addEventListener('click', runSearch);
    queryInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        runSearch();
      }
    });
  </script>
</body>
</html>
"""


@app.post("/search")
def search(request: SearchRequest) -> dict:
    """Return top semantic matches with answer previews."""
    if searcher is None:
        return {"results": []}

    matches = searcher.search(request.query, top_k=5)

    results: list[SearchResult] = []
    for match in matches:
        answer = match["answer"]
        preview = answer[:200] + ("..." if len(answer) > 200 else "")
        results.append(
            SearchResult(
                question=match["question"],
                answer=answer,
                answer_preview=preview,
                score=match["score"],
            )
        )

    return {"results": [result.model_dump() for result in results]}
