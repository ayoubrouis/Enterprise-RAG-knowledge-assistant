"""Build the FAISS vector index from a tenant's documents.

Usage:
    python scripts/ingest.py [--tenant <tenant_id>]

Outputs the persisted index to data/tenants/<tenant_id>/vectorstore/ for use by
the API and UI. Defaults to the configured default tenant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.ingestion import ingest_documents
from app.rag.vectorstore import build_vectorstore, save_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tenant",
        default=settings.DEFAULT_TENANT,
        help="Tenant whose docs folder is indexed (default: the configured default).",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    docs_dir = settings.tenant_docs_dir(args.tenant)

    documents, chunks = ingest_documents(docs_dir)
    print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")

    store = build_vectorstore(chunks)
    save_vectorstore(store, args.tenant)

    elapsed = time.perf_counter() - start
    print(f"Index built and saved in {elapsed:.1f}s -> {settings.tenant_vectorstore_dir(args.tenant)}")


if __name__ == "__main__":
    main()
