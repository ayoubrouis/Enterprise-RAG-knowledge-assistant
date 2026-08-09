"""Retrieval evaluation: precision@k, recall@k, and MRR.

A RAG system is only as good as its retriever - if the right chunks never reach
the LLM, the answer cannot be right. We therefore evaluate retrieval directly
using classic information-retrieval metrics against a small labeled test set.

The metric math is a pure function (no models involved) so it can be
unit-tested; ``evaluate_retrieval`` plugs it into the real pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.pipeline import RAGPipeline


@dataclass
class EvalCase:
    question: str
    relevant_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure metric math (unit-testable, no I/O or models)
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved sources that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for s in top_k if s in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant sources that appear in the top-k."""
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for s in retrieved[:k] if s in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/(rank of first relevant result); 0 if nothing relevant was found."""
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[EvalCase], k: int = 4) -> dict:
    """Run the pure metrics against a live pipeline and aggregate them."""
    per_case = []
    precision_total = recall_total = mrr_total = 0.0

    for case in cases:
        relevant = set(case.relevant_sources)
        # Document-level evaluation: multiple chunks from one file count once,
        # so dedupe while preserving rank order.
        retrieved = list(
            dict.fromkeys(r["source"] for r in pipeline.retrieve(case.question, k=k))
        )

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)

        precision_total += p
        recall_total += r
        mrr_total += rr
        per_case.append(
            {
                "question": case.question,
                "retrieved": retrieved,
                "relevant": case.relevant_sources,
                "precision@k": round(p, 3),
                "recall@k": round(r, 3),
                "mrr": round(rr, 3),
            }
        )

    n = len(cases) or 1
    return {
        "k": k,
        "num_cases": len(cases),
        "mean_precision@k": round(precision_total / n, 3),
        "mean_recall@k": round(recall_total / n, 3),
        "mean_mrr": round(mrr_total / n, 3),
        "per_case": per_case,
    }
