#!/usr/bin/env bash
# Restore a backup snapshot into the data directory.
# Usage: bash scripts/restore.sh <backup_dir> [--force]
#   backup_dir: path containing system.db and/or tenants/
#   --force: overwrite existing data without prompting
set -euo pipefail

BACKUP_DIR="${1:?Usage: restore.sh <backup_dir> [--force]}"
FORCE="${2:-}"
DATA_DIR="${RAG_DATA_DIR:-./data}"

if [ ! -d "${BACKUP_DIR}" ]; then
    echo "Error: backup directory '${BACKUP_DIR}' not found." >&2
    exit 1
fi

# Safety check: refuse to overwrite live data unless --force is set.
if [ -d "${DATA_DIR}/tenants" ] && [ "${FORCE}" != "--force" ]; then
    echo "Error: ${DATA_DIR}/tenants already exists. Use --force to overwrite." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"

# Restore database files.
for f in system.db system.db-wal system.db-shm; do
    [ -f "${BACKUP_DIR}/${f}" ] && cp "${BACKUP_DIR}/${f}" "${DATA_DIR}/"
done

# Restore tenant data.
if [ -d "${BACKUP_DIR}/tenants" ]; then
    rm -rf "${DATA_DIR}/tenants"
    cp -r "${BACKUP_DIR}/tenants" "${DATA_DIR}/tenants"
fi

echo "Restore complete. Restart the API container to pick up changes."
