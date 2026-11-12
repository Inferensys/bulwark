#!/usr/bin/env bash
# Print a user-delegation SAS for the container, for azcopy.
#   infra/azure/sas.sh [hours] [permissions]
# Defaults: valid 24 h, read/add/create/write/list. No delete: azcopy copy and sync (without
# --delete-destination) never delete blobs, so a leaked token cannot wipe the container.
set -euo pipefail
source "$(dirname "$0")/env.sh"
HOURS=${1:-24}
PERMISSIONS=${2:-racwl}
EXPIRY=$(date -u -v+"$HOURS"H +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u -d "+$HOURS hours" +%Y-%m-%dT%H:%MZ)
az storage container generate-sas --account-name "$BULWARK_STORAGE" --name "$BULWARK_CONTAINER" \
  --permissions "$PERMISSIONS" --expiry "$EXPIRY" --auth-mode login --as-user -o tsv
