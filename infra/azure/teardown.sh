#!/usr/bin/env bash
# Delete a job VM and everything created with it (NIC, IP, NSG, OS disk). Blob data stays.
#   infra/azure/teardown.sh <vm-name>
#   infra/azure/teardown.sh --orphans     # public IPs, NICs and disks left by deleted VMs
# Exits non-zero if the VM still exists afterwards; it keeps billing until it is gone.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1
if [ "$NAME" = "--orphans" ]; then
  for id in $(az network public-ip list -g "$BULWARK_RG" --query "[?ipConfiguration==null].id" -o tsv); do
    az network public-ip delete --ids "$id" -o none && echo "Deleted $id"
  done
  for id in $(az network nic list -g "$BULWARK_RG" --query "[?virtualMachine==null].id" -o tsv); do
    az network nic delete --ids "$id" -o none && echo "Deleted $id"
  done
  for id in $(az disk list -g "$BULWARK_RG" --query "[?diskState=='Unattached'].id" -o tsv); do
    az disk delete --ids "$id" --yes -o none && echo "Deleted $id"
  done
  exit 0
fi

# 0 if the VM exists, 1 if it does not (az exits 3 for a missing resource); any other
# failure (not logged in, network) stops the script rather than pass for "deleted".
vm_exists() {
  local code=0
  az vm show -g "$BULWARK_RG" -n "$NAME" -o none 2>/dev/null || code=$?
  case $code in
    0) return 0 ;;
    3) return 1 ;;
    *) echo "az vm show $NAME failed (exit $code); is az logged in?" >&2; exit 1 ;;
  esac
}

# Whatever happens below, bring the ledger up to date.
trap '"$(dirname "$0")/spend.sh" reconcile || true' EXIT
if vm_exists; then
  # NIC and OS disk go with the VM (created with delete options), and so does the public IP
  # when run_job.sh set its delete option.
  az vm delete -g "$BULWARK_RG" -n "$NAME" --yes -o none || true
  if vm_exists; then
    echo "$NAME still exists and keeps billing; check the Azure portal or retry." >&2
    exit 1
  fi
fi
# Leftovers: a VM from an older launcher, or one whose delete options did not apply.
for kind in "network nic" "network public-ip" "network nsg"; do
  for id in $(az $kind list -g "$BULWARK_RG" --query "[?contains(name,'$NAME')].id" -o tsv || true); do
    az $kind delete --ids "$id" -o none || true
  done
done
for id in $(az disk list -g "$BULWARK_RG" --query "[?contains(name,'$NAME')].id" -o tsv || true); do
  az disk delete --ids "$id" --yes -o none || true
done
echo "Deleted $NAME"
