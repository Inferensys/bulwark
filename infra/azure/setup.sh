#!/usr/bin/env bash
# One-time: resource group, storage account + container, your data-plane role on it, and a
# budget alert at the spending cap. The cap itself is enforced by spend.py (Azure budgets
# only alert; they cannot stop spending).
#   BULWARK_CAP_USD=<total-usd> BULWARK_ALERT_EMAIL=you@example.com infra/azure/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${BULWARK_CAP_USD:?Set BULWARK_CAP_USD to the total Azure spend allowed for Bulwark, in US dollars}"
: "${BULWARK_ALERT_EMAIL:?Set BULWARK_ALERT_EMAIL to the address that should get budget alerts}"
az group create -n "$BULWARK_RG" -l "$BULWARK_LOCATION" -o none
# Shared keys are off: every storage call uses Entra ID or a user-delegation SAS.
az storage account create -n "$BULWARK_STORAGE" -g "$BULWARK_RG" -l "$BULWARK_LOCATION" \
  --sku Standard_LRS --kind StorageV2 --access-tier Hot --min-tls-version TLS1_2 \
  --allow-blob-public-access false --allow-shared-key-access false -o none
# User-delegation SAS (sas.sh) and --auth-mode login need a data role; Owner alone has none.
az role assignment create --role "Storage Blob Data Contributor" \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --scope "$(az storage account show -n "$BULWARK_STORAGE" -g "$BULWARK_RG" --query id -o tsv)" -o none || true
sleep 30   # role assignments take a moment to apply
az storage container create -n "$BULWARK_CONTAINER" --account-name "$BULWARK_STORAGE" --auth-mode login -o none
# The cap counts spending from today; spend.py reads this date (git-ignored, per machine).
SPEND_START_FILE="$(dirname "$0")/.spend_start"
[ -s "$SPEND_START_FILE" ] || date -u +%F > "$SPEND_START_FILE"
# Alerts at 50%, 80% and 100% of the cap. Budgets reset every period, so use the longest
# (a year) rather than a month.
START=$(date -u +%Y-%m-01)
END="$(( $(date -u +%Y) + 2 ))-12-31"
az consumption budget create --budget-name bulwark-cap --amount "$BULWARK_CAP_USD" \
  --category cost --time-grain annually --start-date "$START" --end-date "$END" \
  --resource-group "$BULWARK_RG" \
  --notifications "{
    \"actual50\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":50,\"contactEmails\":[\"$BULWARK_ALERT_EMAIL\"]},
    \"actual80\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":80,\"contactEmails\":[\"$BULWARK_ALERT_EMAIL\"]},
    \"actual100\":{\"enabled\":true,\"operator\":\"GreaterThanOrEqualTo\",\"threshold\":100,\"contactEmails\":[\"$BULWARK_ALERT_EMAIL\"]}}" \
  -o none || echo "Budget alert not created (some subscription offers do not support budgets); spend.sh status still works."
echo "Storage: https://$BULWARK_STORAGE.blob.core.windows.net/$BULWARK_CONTAINER"
echo "Budget alerts at 50%, 80% and 100% of \$$BULWARK_CAP_USD. spend.sh enforces the cap on spending from"
echo "$(cat "$SPEND_START_FILE") as long as BULWARK_CAP_USD stays set (env.local.sh); check it with infra/azure/spend.sh status"
