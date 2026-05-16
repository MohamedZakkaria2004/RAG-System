from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_system.config import load_settings
from rag_system.evaluation import load_golden_dataset, run_ragas_evaluation
from rag_system.factory import build_services

app = typer.Typer(help="Production RAG system CLI.")


@app.command()
def ingest(
    path: str = typer.Option(..., "--path", "-p", help="File or directory to ingest."),
    recursive: bool = typer.Option(True, help="Recursively ingest supported files."),
    config: str = typer.Option("config/default.yaml", help="Path to YAML config."),
) -> None:
    services = build_services(load_settings(config))
    stats = services["ingestion"].ingest_path(path, recursive=recursive)
    typer.echo(stats.model_dump_json(indent=2))


@app.command("ingest-url")
def ingest_url(
    url: str = typer.Option(..., "--url", "-u", help="Web page URL to ingest."),
    config: str = typer.Option("config/default.yaml", help="Path to YAML config."),
) -> None:
    services = build_services(load_settings(config))
    stats = services["ingestion"].ingest_url(url)
    typer.echo(stats.model_dump_json(indent=2))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the indexed corpus."),
    config: str = typer.Option("config/default.yaml", help="Path to YAML config."),
) -> None:
    services = build_services(load_settings(config))
    response = services["query_engine"].query(question)
    typer.echo(response.model_dump_json(indent=2))


@app.command("eval")
def evaluate_command(
    dataset: str = typer.Option("data/eval/golden_qa.jsonl", help="Golden QA JSONL file."),
    threshold: float = typer.Option(0.85, help="Minimum allowed faithfulness score."),
    skip_if_missing: bool = typer.Option(False, help="Skip evaluation when dataset is absent."),
    validate_only: bool = typer.Option(False, help="Only validate the JSONL schema."),
    config: str = typer.Option("config/default.yaml", help="Path to YAML config."),
) -> None:
    dataset_path = Path(dataset)
    if skip_if_missing and not dataset_path.exists():
        typer.echo(f"Skipping evaluation because dataset is missing: {dataset_path}")
        return
    if validate_only:
        rows = load_golden_dataset(dataset_path)
        typer.echo(json.dumps({"valid": True, "samples": len(rows)}, indent=2))
        return

    services = build_services(load_settings(config))
    result = run_ragas_evaluation(
        dataset_path=dataset_path,
        query_engine=services["query_engine"],
        threshold=threshold,
    )
    typer.echo(result.model_dump_json(indent=2))
    if not result.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
