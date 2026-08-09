"""Move pre-multi-tenant data into the new tenant layout.

Before v2, documents lived in ``data/docs/`` and the index in
``data/vectorstore/``. The multi-tenant layout stores them per tenant under
``data/tenants/<tenant_id>/``. This script moves any legacy folders into the
default tenant so existing deployments keep working after upgrading.

Usage:
    python scripts/migrate_legacy_data.py [--tenant <tenant_id>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def _move_contents(src: Path, dst: Path) -> int:
    """Move every entry from src into dst. Returns number of entries moved."""
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for entry in src.iterdir():
        target = dst / entry.name
        if target.exists():
            continue  # keep the existing copy; never overwrite user data
        entry.rename(target)
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    args = parser.parse_args()

    legacy_docs = settings.ROOT_DIR / "data" / "docs"
    legacy_vs = settings.ROOT_DIR / "data" / "vectorstore"
    dst_docs = settings.tenant_docs_dir(args.tenant)
    dst_vs = settings.tenant_vectorstore_dir(args.tenant)

    for src, dst, label in (
        (legacy_docs, dst_docs, "documents"),
        (legacy_vs, dst_vs, "vector index"),
    ):
        if not src.exists():
            print(f"No legacy {label} folder at {src}.")
            continue
        moved = _move_contents(src, dst)
        print(f"Migrated {moved} {label} item(s): {src} -> {dst}")

    # Keep data/docs and data/vectorstore around; they are gitignored and
    # harmless. Operators can delete them once the migration output looks right.


if __name__ == "__main__":
    main()
