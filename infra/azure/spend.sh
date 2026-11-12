#!/usr/bin/env bash
# Spending cap: status, reserve, release, reconcile. See spend.py.
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec python3 "$(dirname "$0")/spend.py" "$@"
