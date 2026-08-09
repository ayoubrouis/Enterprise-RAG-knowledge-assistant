#!/usr/bin/env bash
# Update the on-prem installation.
#
#   - Rebuilds/refreshes the images (or pulls from your registry if you point
#     docker-compose.yml at one).
#   - Restarts the stack with zero data loss (data/ + models/ are volumes).
#
# Usage:  ./scripts/update.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building images..."
docker compose build

echo "==> Starting stack..."
docker compose up -d

echo "==> Pruning old images..."
docker image prune -f

echo "Done. Check status with: docker compose ps"
