from pathlib import Path

from rag_system.loaders import load_path


def test_markdown_loader_preserves_source_metadata(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# Policy\n\nRefunds are available for 30 days.", encoding="utf-8")

    docs = load_path(source)

    assert len(docs) == 1
    assert docs[0].source == str(source)
    assert docs[0].title == "policy"
    assert docs[0].metadata["kind"] == "md"
    assert "Refunds" in docs[0].text
