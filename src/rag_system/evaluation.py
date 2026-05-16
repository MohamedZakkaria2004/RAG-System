from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_system.schemas import EvaluationResult

REQUIRED_FIELDS = {"question", "ground_truth", "reference_contexts"}


def load_golden_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"Line {line_number} missing fields: {sorted(missing)}")
            if not row["question"] or not row["ground_truth"] or not row["reference_contexts"]:
                raise ValueError(f"Line {line_number} contains empty required fields")
            rows.append(row)
    if not rows:
        raise ValueError(f"Golden dataset is empty: {dataset_path}")
    return rows


def run_ragas_evaluation(
    dataset_path: str | Path,
    query_engine,
    threshold: float = 0.85,
) -> EvaluationResult:
    rows = load_golden_dataset(dataset_path)
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError("Install ragas and datasets to run evaluation.") from exc
    try:
        from ragas.metrics.collections import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for row in rows:
        response = query_engine.query(row["question"])
        questions.append(row["question"])
        answers.append(response.answer)
        contexts.append([chunk.text for chunk in response.retrieved_chunks])
        ground_truths.append(row["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    metrics = {key: float(value) for key, value in dict(result).items()}
    score = metrics.get("faithfulness", 0.0)
    return EvaluationResult(
        samples=len(rows),
        metrics=metrics,
        passed=score >= threshold,
        threshold=threshold,
    )
