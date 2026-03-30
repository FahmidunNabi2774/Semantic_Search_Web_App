"""Semantic search utilities using sentence-transformers and FAISS."""

from __future__ import annotations

from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import Record, prepare_embedding_matrix


class SemanticSearcher:
    """Build and query a FAISS index for semantic similarity."""

    def __init__(self, records: List[Record], model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the model, normalize vectors, and preload the FAISS index."""
        self.records = records
        self.model = SentenceTransformer(model_name)

        matrix, dimension = prepare_embedding_matrix(records)
        self.index = faiss.IndexFlatIP(dimension)

        normalized_matrix = self._normalize_vectors(matrix)
        self.index.add(normalized_matrix)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding vector for a user query."""
        query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        query_vector = query_vector.astype(np.float32)
        return self._normalize_vectors(query_vector)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest neighbors and return the top matching records."""
        if not query.strip():
            return []

        query_vector = self.embed_query(query)
        scores, indices = self.index.search(query_vector, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue

            item = self.records[idx]
            results.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "score": float(score),
                }
            )

        return results
