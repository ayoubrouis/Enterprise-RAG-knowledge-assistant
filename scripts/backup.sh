#!/usr/bin/env bash
# Snapshot the SQLite DB + all tenant data into a timestamped directory.
# Usage: bash scripts/backup.sh [output_dir]
#   output_dir defaults to ./backups/
set -euo pipefail

DATA_DIR="${RAG_DATA_DIR:-./data}"
OUT_DIR="${1:-./backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEST="${OUT_DIR}/${TIMESTAMP}"

mkdir -p "${DEST}"

# Checkpoint WAL before copying so the backup is consistent.
if command -v sqlite3 &>/dev/null && [ -f "${DATA_DIR}/system.db" ]; then
    sqlite3 "${DATA_DIR}/system.db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
fi

# Copy the database (and WAL/SHM if present).
for f in system.db system.db-wal system.db-shm; do
    [ -f "${DATA_DIR}/${f}" ] && cp "${DATA_DIR}/${f}" "${DEST}/"
done

# Copy all tenant data (indexes + documents).
if [ -d "${DATA_DIR}/tenants" ]; then
    cp -r "${DATA_DIR}/tenants" "${DEST}/tenants"
fi

echo "Backup saved to ${DEST}"
