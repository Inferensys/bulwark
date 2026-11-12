# Shared settings for the Azure scripts. Override any of these in the environment, or put
# per-machine values (alert email, spending cap) in infra/azure/env.local.sh (git-ignored;
# BULWARK_ENV_LOCAL names another file, as the tests do to keep yours out of them).
AZURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL="${BULWARK_ENV_LOCAL:-$AZURE_DIR/env.local.sh}"
if [ -f "$ENV_LOCAL" ]; then source "$ENV_LOCAL"; fi

export BULWARK_RG="${BULWARK_RG:-bulwark-train}"
export BULWARK_LOCATION="${BULWARK_LOCATION:-eastus2}"
export BULWARK_STORAGE="${BULWARK_STORAGE:-bulwarkdata$(az account show --query id -o tsv | cut -c1-6)}"
export BULWARK_CONTAINER="${BULWARK_CONTAINER:-bulwark}"
export BULWARK_VM_SIZE="${BULWARK_VM_SIZE:-Standard_NC24ads_A100_v4}"
# Data Science VM (Ubuntu 22.04): NVIDIA driver + CUDA preinstalled. Pinned so every job
# boots the same image. --sku is a prefix match (it also lists 2204-v100 and others), so
# filter on the exact sku, and confirm a version with `az vm image show --urn ...`:
#   az vm image list --publisher microsoft-dsvm --offer ubuntu-hpc --sku 2204 --all \
#     --query "[?sku=='2204'].version" -o tsv
export BULWARK_IMAGE="${BULWARK_IMAGE:-microsoft-dsvm:ubuntu-hpc:2204:22.04.2026100101}"
# Budget alerts go here (setup.sh requires it).
export BULWARK_ALERT_EMAIL="${BULWARK_ALERT_EMAIL:-}"

# Spending cap in US dollars (see infra/azure/spend.py): the total for all Bulwark Azure
# use, not per month. No default: set it in infra/azure/env.local.sh or the environment.
# spend.sh reserve, and so run_job.sh, refuse to start a job without it.
export BULWARK_CAP_USD="${BULWARK_CAP_USD:-}"
# Spot quota is often unavailable on pay-as-you-go subscriptions, so jobs run on-demand by
# default ($3.67/h list for an NC24ads_A100_v4 in eastus2). With Spot quota, set
# BULWARK_PRIORITY=Spot: the Spot list price is about $0.68/h, the job bids up to
# BULWARK_SPOT_MAX_PRICE ($0.85/h) and Azure evicts the VM rather than bill above that.
# spend.py budgets Spot hours at that max price.
export BULWARK_PRIORITY="${BULWARK_PRIORITY:-Regular}"
export BULWARK_ONDEMAND_PRICE="${BULWARK_ONDEMAND_PRICE:-3.67}"
export BULWARK_SPOT_MAX_PRICE="${BULWARK_SPOT_MAX_PRICE:-0.85}"
# Per VM-hour on top of compute: P10 OS disk (~$0.025/h) + Standard public IP ($0.005/h).
export BULWARK_HOURLY_EXTRAS="${BULWARK_HOURLY_EXTRAS:-0.03}"
# After its deadline a VM gets this long to sync, leave its exit marker and delete itself
# (run_job.sh); Azure auto-shutdown fires at the end of it. Reserved like training time.
export BULWARK_FINISH_MINUTES="${BULWARK_FINISH_MINUTES:-30}"
# Never handed to jobs: Blob storage + transactions (~$5) and a margin for billing lag (~$10).
export BULWARK_RESERVE_USD="${BULWARK_RESERVE_USD:-15}"
# Start of the period that counts against the cap; setup.sh records its run date here.
export BULWARK_SPEND_START="${BULWARK_SPEND_START:-$(cat "$AZURE_DIR/.spend_start" 2>/dev/null || true)}"
export BULWARK_LEDGER="${BULWARK_LEDGER:-$AZURE_DIR/ledger.jsonl}"
