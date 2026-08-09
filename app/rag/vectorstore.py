"""FAISS vector store: build, persist, and load.

The index is just files in a folder, so "no cloud services" holds end to end.
Each tenant keeps its own index under ``data/tenants/<tenant_id>/vectorstore/``,
so retrieval is physically isolated between tenants.

Loading requires ``allow_dangerous_deserialization=True`` because FAISS stores
its document metadata as a pickle; we only load indexes we built ourselves.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import get_embeddings


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed all chunks and build a FAISS index."""
    return FAISS.from_documents(chunks, get_embeddings())


def save_vectorstore(store: FAISS, tenant_id: str | None = None) -> None:
    directory = settings.tenant_vectorstore_dir(tenant_id or settings.DEFAULT_TENANT)
    directory.mkdir(parents=True, exist_ok=True)
    store.save_local(str(directory))


def load_vectorstore(tenant_id: str | None = None) -> FAISS | None:
    """Load a tenant's persisted index, or None if it has not been built."""
    directory = settings.tenant_vectorstore_dir(tenant_id or settings.DEFAULT_TENANT)
    if not store_exists(tenant_id):
        return None
    return FAISS.load_local(
        str(directory),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def store_exists(tenant_id: str | None = None) -> bool:
    directory = settings.tenant_vectorstore_dir(tenant_id or settings.DEFAULT_TENANT)
    return (directory / "index.faiss").exists()
