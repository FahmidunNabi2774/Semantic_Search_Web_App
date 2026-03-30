"""Utilities for loading StackOverflow data and preparing vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


Record = Dict[str, Any]


def load_dataset(path: str | Path = "data.json") -> List[Record]:
    """Load the JSON dataset and validate required fields."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        records: List[Record] = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array of records")

    required_fields = {"question", "answer", "embedding"}
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record at index {idx} must be a JSON object")
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Record at index {idx} is missing fields: {sorted(missing)}")

    return records


def prepare_embedding_matrix(records: List[Record]) -> Tuple[np.ndarray, int]:
    """Convert record embeddings to a contiguous float32 numpy matrix."""
    if not records:
        raise ValueError("Dataset is empty")

    embeddings = [record["embedding"] for record in records]
    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Embeddings must form a 2D matrix")

    dim = int(matrix.shape[1])
    if dim == 0:
        raise ValueError("Embedding dimension cannot be zero")

    return np.ascontiguousarray(matrix), dim
