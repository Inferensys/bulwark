#!/usr/bin/env python3
"""Hard spending cap for Bulwark's Azure use: BULWARK_CAP_USD, in US dollars, in total.

There is no default: without BULWARK_CAP_USD, `reserve` refuses every job (and so does
run_job.sh); `status`, `release` and `reconcile` still work.

Azure budgets only send alerts, so the cap is enforced here:

1. Every job VM reserves its worst case before it is created: (hours + the finish window)
   x (hourly price + disk and IP), the price being on-demand list or the Spot max price.
   The finish window covers the final syncs, the exit marker and the self-delete after the
   deadline. If reservations plus the storage reserve would pass the cap, the job does not
   start.
2. The VM deletes itself when the job ends or at its deadline, and an Azure auto-shutdown
   at the end of the finish window deallocates it if that fails. Its price per hour is
   fixed (Spot never bills above the max price), so compute stays within the reservation.
   A deallocated VM still bills its disk and IP until it is deleted; `status` flags it.
3. Running job VMs carry their reserved hours as tags, and any VM without an open ledger
   row is counted at that reservation. A lost ledger or a second clone therefore cannot
   hand out money a running VM may still spend. The ledger is locked while it changes.
4. Billed cost from Cost Management (it lags a few hours) is the backstop: whichever of
   it and the ledger is higher counts.
5. `reconcile` closes finished jobs at their measured runtime, which can be more or less
   than reserved, and clears leftover public IPs, NICs and disks once no job is running.

Run through spend.sh, which loads env.sh:
    infra/azure/spend.sh status
    infra/azure/spend.sh reserve <vm-name> <job> <hours>
    infra/azure/spend.sh release <vm-name>        # the VM was never created
    infra/azure/spend.sh reconcile
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

CAP_HELP = "Set BULWARK_CAP_USD to the total Azure spend allowed for Bulwark, in US dollars (e.g. in infra/azure/env.local.sh)."


def read_cap(value: str | None) -> float | None:
    """The cap from BULWARK_CAP_USD; None when it is unset or not a positive number."""
    try:
        cap = float(value or "nan")
    except ValueError:
        return None
    return cap if cap > 0 else None  # NaN compares False, so unset ends up here too


CAP = read_cap(os.environ.get("BULWARK_CAP_USD"))
RESERVE = float(os.environ.get("BULWARK_RESERVE_USD", "15"))
PRICE = (
    os.environ.get("BULWARK_SPOT_MAX_PRICE", "0.85")
    if os.environ.get("BULWARK_PRIORITY", "Regular") == "Spot"
    else os.environ.get("BULWARK_ONDEMAND_PRICE", "3.67")
)
RATE = float(PRICE) + float(os.environ.get("BULWARK_HOURLY_EXTRAS", "0.03"))
# After the deadline: final syncs, exit marker and self-delete (run_job.sh), then auto-shutdown.
FINISH_HOURS = float(os.environ.get("BULWARK_FINISH_MINUTES", "30")) / 60
LEDGER = Path(os.environ.get("BULWARK_LEDGER", Path(__file__).with_name("ledger.jsonl")))
RG = os.environ.get("BULWARK_RG", "bulwark-train")
STORAGE = os.environ.get("BULWARK_STORAGE", "")
CONTAINER = os.environ.get("BULWARK_CONTAINER", "bulwark")
START = os.environ.get("BULWARK_SPEND_START", "")  # setup.sh records it; see spend_start()
MAX_HOURS = 20  # auto-shutdown is a time of day, so deadline + finish window must stay < 24 h
TEARDOWN = Path(__file__).with_name("teardown.sh")


def az(*args: str) -> object | None:
    result = subprocess.run(["az", *args, "-o", "json"], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def ledger_lock() -> Iterator[None]:
    """Hold an exclusive lock on the ledger, so concurrent launches never lose a row."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER.with_suffix(".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def save(rows: list[dict]) -> None:
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(LEDGER)


def reservation(hours: float) -> float:
    return round((hours + FINISH_HOURS) * RATE, 2)


def charged(row: dict) -> float:
    return row["final_usd"] if row.get("final_usd") is not None else row["reserved_usd"]


def is_open(row: dict) -> bool:
    return row.get("final_usd") is None


def vms() -> dict[str, dict] | None:
    """Job VMs in the resource group: {name: {"power": ..., "tags": {...}}}, None if unknown."""
    listed = az("vm", "list", "-g", RG, "-d", "--query", "[].{name:name, power:powerState, tags:tags}")
    if not isinstance(listed, list):
        return None
    return {vm["name"]: {"power": vm.get("power") or "", "tags": vm.get("tags") or {}} for vm in listed}


def untracked_usd(rows: list[dict], running: dict[str, dict]) -> float:
    """Worst case of VMs this ledger has no open row for (another clone, or a lost ledger)."""
    tracked = {row["name"] for row in rows if is_open(row)}
    total = 0.0
    for name, vm in running.items():
        if name not in tracked:
            try:
                hours = float(vm["tags"].get("bulwark-hours", MAX_HOURS))
            except ValueError:
                hours = MAX_HOURS
            total += reservation(hours)
    return total


def spend_start(rows: list[dict]) -> str:
    """Start of the Cost Management window: the recorded setup date, else the first job,
    else a year back (counting older spending in the group only makes the cap stricter)."""
    if START:
        return START
    if rows:
        return min(row["started"] for row in rows)[:10]
    return (now() - timedelta(days=365)).strftime("%Y-%m-%d")


def actual_usd(start: str) -> float | None:
    """Billed cost of the resource group since start (lags a few hours behind)."""
    sub = az("account", "show", "--query", "id")
    if not sub:
        return None
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {"from": f"{start}T00:00:00Z", "to": now().strftime("%Y-%m-%dT%H:%M:%SZ")},
        "dataset": {"granularity": "None", "aggregation": {"cost": {"name": "Cost", "function": "Sum"}}},
    }
    url = (
        f"https://management.azure.com/subscriptions/{sub}/resourceGroups/{RG}"
        "/providers/Microsoft.CostManagement/query?api-version=2023-11-01"
    )
    result = az("rest", "--method", "post", "--url", url, "--body", json.dumps(body))
    if not isinstance(result, dict):
        return None  # e.g. the resource group does not exist yet
    rows = result.get("properties", {}).get("rows", [])
    return float(rows[0][0]) if rows else 0.0


def committed(rows: list[dict], running: dict[str, dict]) -> tuple[float, float | None]:
    ledger = sum(charged(row) for row in rows) + untracked_usd(rows, running)
    actual = actual_usd(spend_start(rows))
    return max(ledger, actual or 0.0), actual


def reserve(name: str, job: str, hours: float) -> int:
    if CAP is None:
        print(f"Refused: BULWARK_CAP_USD is not set to a positive amount. {CAP_HELP}", file=sys.stderr)
        return 2
    if not 0 < hours <= MAX_HOURS:
        print(f"hours must be in (0, {MAX_HOURS}]", file=sys.stderr)
        return 2
    with ledger_lock():
        rows = load()
        running = vms()
        if running is None:
            print(
                f"Refused: could not list the VMs in {RG}, so the cap cannot be checked. "
                "Is az logged in, and has infra/azure/setup.sh run?",
                file=sys.stderr,
            )
            return 1
        need = reservation(hours)
        spent, _ = committed(rows, running)
        left = CAP - RESERVE - spent
        if need > left:
            print(
                f"Refused: {job} needs up to ${need:.2f} ({hours:g} h + {FINISH_HOURS:g} h to finish, "
                f"x ${RATE:.2f}/h) but only ${left:.2f} of the ${CAP:.0f} cap is left (committed "
                f"${spent:.2f}, reserve ${RESERVE:.0f}). Run `spend.sh reconcile`, or shorten the job.",
                file=sys.stderr,
            )
            return 1
        rows.append(
            {
                "name": name,
                "job": job,
                "hours": hours,
                "rate": RATE,
                "reserved_usd": need,
                "started": now().isoformat(timespec="seconds"),
                "final_usd": None,
            }
        )
        save(rows)
    print(f"Reserved ${need:.2f} for {name}; ${left - need:.2f} of the cap left after it.")
    return 0


def release(name: str) -> int:
    with ledger_lock():
        rows = load()
        for row in rows:
            if row["name"] == name and is_open(row):
                row["final_usd"] = 0.0
                row["note"] = "released: VM never created"
        save(rows)
    return 0


def exit_marker_time(name: str) -> datetime | None:
    props = az(
        "storage", "blob", "show", "--account-name", STORAGE, "-c", CONTAINER,
        "-n", f"jobs/{name}.exit", "--auth-mode", "login", "--query", "properties.lastModified",
    )  # fmt: skip
    return datetime.fromisoformat(str(props).replace("Z", "+00:00")) if props else None


def clean_orphans() -> None:
    """Public IPs, NICs and disks that outlived their VMs bill until deleted."""
    subprocess.run([str(TEARDOWN), "--orphans"], check=False)


def reconcile() -> int:
    with ledger_lock():
        rows = load()
        running = vms()
        if running is None:
            print(f"Could not list the VMs in {RG}; nothing reconciled. Is az logged in?", file=sys.stderr)
            return 1
        for row in rows:
            if not is_open(row) or row["name"] in running:
                continue  # closed, or still running
            started = datetime.fromisoformat(row["started"])
            ended = exit_marker_time(row["name"])
            if ended is None:
                # Evicted or never booted: Azure deleted it at some unknown point. Keep the worst case.
                row["final_usd"] = row["reserved_usd"]
                row["note"] = "no exit marker; kept the full reservation"
                continue
            # Quarter-hour slack for the self-delete after the marker. Not capped at the
            # reservation: an overrun belongs in the ledger, not only in lagging billing data.
            hours = math.ceil(((ended - started).total_seconds() / 3600 + 0.25) * 4) / 4
            row["final_usd"] = round(hours * row["rate"], 2)
            row["note"] = f"ran ~{hours:g} h"
        save(rows)
        idle = not running and not any(is_open(row) for row in rows)
    if idle:
        clean_orphans()  # no job is running or launching, so nothing here is in use
    return status()


def status() -> int:
    rows = load()
    running = vms()
    spent, actual = committed(rows, running or {})
    cap = "not set" if CAP is None else f"${CAP:.0f}"
    print(f"Cap {cap}  |  reserve for storage + margin ${RESERVE:.0f}  |  rate ${RATE:.2f}/VM-hour")
    for row in rows:
        state = "open" if is_open(row) else (row.get("note") or "closed")
        print(f"  {row['name']:32s} {row['job']:8s} {row['hours']:>5g} h  ${charged(row):7.2f}  {state}")
    ledger = sum(charged(row) for row in rows)
    print(f"Jobs (worst case for open ones): ${ledger:.2f}")
    extra = untracked_usd(rows, running or {})
    if extra:
        print(f"VMs with no open ledger row, counted at their reservation: ${extra:.2f}")
    print(f"Billed since {spend_start(rows)} (Cost Management, lags hours): "
          f"{'unavailable' if actual is None else f'${actual:.2f}'}")  # fmt: skip
    if not START:
        print("  (BULWARK_SPEND_START is not set; infra/azure/setup.sh records it in infra/azure/.spend_start)")
    if CAP is None:
        print(f"New jobs are refused until the cap is set. {CAP_HELP}")
    else:
        left = CAP - RESERVE - spent
        print(f"Left for new jobs: ${left:.2f}  (~{max(0.0, left) / RATE:.0f} A100-hours)")
    if running is None:
        print(f"Could not list the VMs in {RG}.")
    for name, vm in sorted((running or {}).items()):
        note = ""
        if "deallocated" in vm["power"].lower():
            note = "  <- stopped but not deleted: disk and IP still bill; run teardown.sh " + name
        print(f"VM {name}: {vm['power'] or 'state unknown'}{note}")
    listed = az("network", "public-ip", "list", "-g", RG, "--query", "[?ipConfiguration==null].name")
    loose = [str(name) for name in listed] if isinstance(listed, list) else []
    if loose:
        print("Unattached public IPs ($0.12/day each): " + ", ".join(loose) + "  -> teardown.sh --orphans")
    return 0


def main(argv: list[str]) -> int:
    # Runs under the system python3 (3.9 on macOS), so no match statement.
    if argv == ["status"]:
        return status()
    if len(argv) == 4 and argv[0] == "reserve":
        return reserve(argv[1], argv[2], float(argv[3]))
    if len(argv) == 2 and argv[0] == "release":
        return release(argv[1])
    if argv == ["reconcile"]:
        return reconcile()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
