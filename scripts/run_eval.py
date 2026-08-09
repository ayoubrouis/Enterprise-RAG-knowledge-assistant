"""Run retrieval evaluation against the demo corpus.

Usage:
    python scripts/run_eval.py [k]

Requires a built index (run `python scripts/ingest.py` first).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.eval.evaluate import EvalCase, evaluate_retrieval
from app.rag.pipeline import RAGPipeline

# Labeled eval set: each question maps to the document that answers it.
EVAL_CASES = [
    EvalCase("What is the company 401(k) matching policy?", ["benefits_overview.pdf"]),
    EvalCase("How much does Nexora contribute to the HSA?", ["benefits_overview.pdf"]),
    EvalCase("How many days of paid time off do full-time employees get?", ["benefits_overview.pdf"]),
    EvalCase("What are the password requirements for company accounts?", ["security_policy.pdf"]),
    EvalCase("How quickly must a security incident be reported?", ["security_policy.pdf"]),
    EvalCase("What must you bring on your first day of onboarding?", ["onboarding_guide.txt"]),
    EvalCase("What are the minimum system requirements for the Atlas gateway?", ["atlas_gateway_manual.md"]),
    EvalCase("What were the revenues in Q3 2026?", ["q3_2026_report.txt"]),
    EvalCase("What was the customer retention rate last quarter?", ["q3_2026_report.txt"]),
]


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pipeline = RAGPipeline()
    report = evaluate_retrieval(pipeline, EVAL_CASES, k=k)

    print("\n=== Retrieval Evaluation Report ===")
    print(f"k={report['k']}  cases={report['num_cases']}")
    print(f"Mean Precision@{k}: {report['mean_precision@k']}")
    print(f"Mean Recall@{k}:    {report['mean_recall@k']}")
    print(f"Mean MRR:           {report['mean_mrr']}")

    frame = pd.DataFrame(report["per_case"])
    print("\n=== Per-case breakdown ===")
    print(frame[["question", "precision@k", "recall@k", "mrr"]])


if __name__ == "__main__":
    main()
