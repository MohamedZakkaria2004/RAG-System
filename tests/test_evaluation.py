import json
from pathlib import Path

import pytest

from rag_system.evaluation import load_golden_dataset


def test_load_golden_dataset_validates_required_fields(tmp_path: Path):
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(json.dumps({"question": "Q"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_golden_dataset(dataset)


def test_load_golden_dataset_accepts_valid_rows(tmp_path: Path):
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Q",
                "ground_truth": "A",
                "reference_contexts": ["A context"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_golden_dataset(dataset)

    assert len(rows) == 1
