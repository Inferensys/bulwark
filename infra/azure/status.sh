#!/usr/bin/env bash
# Show a job VM's state, the tail of its job log, and whether it reported done.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1
az vm get-instance-view -g "$BULWARK_RG" -n "$NAME" --query "instanceView.statuses[].displayStatus" -o tsv || echo "VM gone (finished and deleted, or evicted)"
# The job log never traces commands, but redact SAS signatures anyway before printing.
az vm run-command invoke -g "$BULWARK_RG" -n "$NAME" --command-id RunShellScript \
  --scripts "tail -n 20 /var/log/bulwark-job.log | sed -E 's/sig=[^&\" ]*/sig=REDACTED/g'" \
  --query "value[0].message" -o tsv 2>/dev/null || true
az storage blob exists --account-name "$BULWARK_STORAGE" -c "$BULWARK_CONTAINER" -n "jobs/$NAME.done" --auth-mode login --query exists -o tsv
