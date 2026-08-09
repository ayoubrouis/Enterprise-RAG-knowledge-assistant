"""RAG orchestration: retrieve -> ground -> generate -> cite.

The pipeline owns the vector store and the LLM. It exposes:
  * retrieve()  - semantic search only (used by the evaluator)
  * answer()    - full RAG: search + grounded generation + citations
  * stats()     - quick index stats for dashboards
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS

from app.config import settings
from app.rag.llm import get_llm
from app.rag.vectorstore import load_vectorstore

PROMPT_TEMPLATE = """Answer the question using ONLY the information in the context below.
If the context does not contain the answer, reply with exactly: I don't know.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(
        self, tenant_id: str | None = None, store: FAISS | None = None
    ) -> None:
        self.tenant_id = tenant_id or settings.DEFAULT_TENANT
        self.store = store or load_vectorstore(self.tenant_id)
        if self.store is None:
            raise RuntimeError(
                f"No vector index for tenant '{self.tenant_id}'. "
                "Upload documents first (or run `python scripts/ingest.py`)."
            )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        """Return top-k chunks as {snippet, metadata, similarity} dicts."""
        k = k or settings.TOP_K
        hits = self.store.similarity_search_with_score(question, k=k)
        results = []
        for doc, distance in hits:
            results.append(
                {
                    "snippet": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "similarity": self._to_similarity(distance),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def answer(self, question: str, top_k: int | None = None) -> dict:
        """Full RAG query: returns {question, answer, sources}."""
        results = self.retrieve(question, k=top_k)

        context = "\n\n".join(
            f"[source: {r['source']}] {r['snippet']}" for r in results
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        llm = get_llm()
        answer = llm(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "similarity": round(r["similarity"], 4),
                    "snippet": r["snippet"][:400],
                }
                for r in results
            ],
        }

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Number of source documents and total chunks in the index."""
        sources: set[str] = set()
        for idx in self.store.index_to_docstore_id.values():
            doc = self.store.docstore.search(idx)
            sources.add(doc.metadata.get("source", "unknown"))
        return {
            "documents": len(sources),
            "chunks": int(self.store.index.ntotal),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_similarity(distance: float) -> float:
        """Convert FAISS L2 distance to a 0..1 cosine similarity.

        With normalized embeddings:  cos = 1 - d^2 / 2.
        """
        sim = 1.0 - (distance**2) / 2.0
        return max(0.0, min(1.0, sim))
