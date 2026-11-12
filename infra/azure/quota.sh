#!/usr/bin/env bash
# Show the GPU quota that Bulwark needs, and how to request it.
# Needs: 24 vCPUs of NCADS_A100_v4 (one A100 80GB VM); for Spot jobs, also >= 24 Spot vCPUs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
az vm list-usage --location "$BULWARK_LOCATION" \
  --query "[?contains(name.value,'NCADSA100v4') || contains(name.value,'lowPriority')].{quota:name.localizedValue, used:currentValue, limit:limit}" -o table
cat <<MSG

To raise a limit: Azure portal > Quotas > Compute > filter by $BULWARK_LOCATION > the
"Standard NCADS_A100_v4 Family vCPUs" row (or "Total Regional Spot vCPUs") > New quota
request. Some subscription types cannot raise GPU quota themselves; the portal then offers
to open a support request instead. Track open requests with:
  az support in-subscription tickets list -o table
MSG
