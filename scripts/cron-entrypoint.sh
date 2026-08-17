#!/usr/bin/env bash
# Cron entrypoint: runs the backup script on a schedule.
# Schedule is set via BACKUP_CRON env var (default: daily at 2 AM).
set -euo pipefail

CRON_SCHEDULE="${BACKUP_CRON:-0 2 * * *}"
BACKUP_DIR="${BACKUP_DIR:-/app/backups}"
DATA_DIR="${RAG_DATA_DIR:-/app/data}"

echo "${CRON_SCHEDULE} /app/scripts/backup.sh ${BACKUP_DIR}" > /etc/crontabs/root
echo "Backup cron: ${CRON_SCHEDULE} -> ${BACKUP_DIR}"
exec crond -f -l 8
