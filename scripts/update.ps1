# Update the on-prem installation from Windows.
#
#   - Rebuilds/refreshes the images (or pulls from your registry if you point
#     docker-compose.yml at one).
#   - Restarts the stack with zero data loss (data/ + models/ are volumes).
#
# Usage:  .\scripts\update.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Building images..."
docker compose build

Write-Host "==> Starting stack..."
docker compose up -d

Write-Host "==> Pruning old images..."
docker image prune -f

Write-Host "Done. Check status with: docker compose ps"
