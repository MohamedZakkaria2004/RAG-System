from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

from rag_system.schemas import RawDocument

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


def load_path(path: str | Path, recursive: bool = True) -> list[RawDocument]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if target.is_file():
        return _load_file(target)

    pattern = "**/*" if recursive else "*"
    documents: list[RawDocument] = []
    for file_path in sorted(target.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(_load_file(file_path))
    return documents


def load_url(url: str, timeout: int = 20) -> list[RawDocument]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
    text = _clean_text(soup.get_text("\n"))
    return [RawDocument(text=text, source=url, title=title, metadata={"kind": "web"})]


def _load_file(path: Path) -> list[RawDocument]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".html", ".htm"}:
        return [_load_html(path)]
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [
            RawDocument(
                text=_clean_text(text),
                source=str(path),
                title=path.stem,
                metadata={"kind": suffix.lstrip(".")},
            )
        ]
    return []


def _load_pdf(path: Path) -> list[RawDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to ingest PDFs.") from exc

    reader = PdfReader(str(path))
    documents: list[RawDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = _clean_text(text)
        if cleaned:
            documents.append(
                RawDocument(
                    text=cleaned,
                    source=str(path),
                    title=path.stem,
                    page=index,
                    metadata={"kind": "pdf"},
                )
            )
    return documents


def _load_html(path: Path) -> RawDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
    return RawDocument(
        text=_clean_text(soup.get_text("\n")),
        source=str(path),
        title=title,
        metadata={"kind": "html"},
    )


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
