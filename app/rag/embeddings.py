"""Embedding model wrapper.

Uses a free, open-source sentence-transformer model that runs locally.
The model is loaded once per process (lru_cache) and reused everywhere.
Normalizing embeddings makes FAISS's L2 index behave like cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
