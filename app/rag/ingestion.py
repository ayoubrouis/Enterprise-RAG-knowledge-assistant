"""Document loading and chunking.

Pipeline: raw files on disk -> parsed Documents (with metadata) -> chunks.

Metadata (source file, page number) is carried onto every chunk so that
retrieved results can be cited back to the exact document and page.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def _load_single_file(path: Path) -> list[Document]:
    """Load one file into LangChain Documents with metadata."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".docx":
        import docx2txt

        text = docx2txt.process(str(path))
        return [Document(page_content=text or "", metadata={"source": path.name})]
    return []


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """Load every supported document under ``docs_dir``."""
    docs_dir = docs_dir or settings.DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[Document] = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in settings.SUPPORTED_EXTENSIONS:
            continue
        loaded = _load_single_file(path)
        # Store just the file name (not the full path) so citations are
        # portable and clean regardless of where the project lives.
        for doc in loaded:
            doc.metadata["source"] = path.name
        documents.extend(loaded)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping, context-preserving chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    # Add a stable chunk id for traceability in citations.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_documents(docs_dir: Path | None = None) -> tuple[list[Document], list[Document]]:
    """Load + chunk documents. Returns (documents, chunks) for the caller."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError("No supported documents found in the docs directory.")
    chunks = chunk_documents(documents)
    return documents, chunks
