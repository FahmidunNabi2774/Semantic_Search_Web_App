"""Download the HuggingFace dataset and export first 3000 rows to data.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset, load_dataset


DATASET_NAME = "MartinElMolon/stackoverflow_preguntas_con_embeddings"
OUTPUT_FILE = Path("data.json")
MAX_ROWS = 3000


def _pick_first_available(record: Dict[str, Any], candidates: Iterable[str], default: Any = "") -> Any:
    """Return the first available value from a list of candidate keys."""
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _to_serializable_vector(value: Any) -> List[float]:
    """Convert an embedding value to a JSON-serializable list of floats."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a dataset row into the app's required JSON schema."""
    question = _pick_first_available(row, ["question", "pregunta", "title"], "")
    answer = _pick_first_available(row, ["answer", "respuesta", "body"], "")
    embedding = _pick_first_available(row, ["embedding", "embeddings", "vector"], [])

    return {
        "question": str(question),
        "answer": str(answer),
        "embedding": _to_serializable_vector(embedding),
    }


def export_first_rows(dataset: Dataset, output_path: Path, max_rows: int = MAX_ROWS) -> int:
    """Export the first rows from a HuggingFace dataset split to data.json."""
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if idx >= max_rows:
            break
        rows.append(normalize_record(row))

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def main() -> None:
    """Download dataset and write first 3000 normalized records to disk."""
    dataset_dict = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset_dict else next(iter(dataset_dict.keys()))
    dataset_split = dataset_dict[split_name]

    count = export_first_rows(dataset_split, OUTPUT_FILE, MAX_ROWS)
    print(f"Wrote {count} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
