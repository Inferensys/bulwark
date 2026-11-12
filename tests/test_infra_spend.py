"""The Azure spending cap (infra/azure/spend.py) with `az` stubbed out: reservations up to
the cap, billed cost, the finish window, ledger locking, VMs the ledger does not know and
reconcile."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SPEND = Path(__file__).parents[1] / "infra" / "azure" / "spend.py"


@pytest.fixture
def spend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("spend", SPEND)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setattr(module, "CAP", 100.0)
    monkeypatch.setattr(module, "RESERVE", 15.0)
    monkeypatch.setattr(module, "RATE", 0.88)
    monkeypatch.setattr(module, "FINISH_HOURS", 0.5)
    monkeypatch.setattr(module, "LEDGER", tmp_path / "ledger.jsonl")
    module.billed = None  # what Cost Management reports
    module.alive: dict[str, dict] = {}  # VMs that still exist: name -> tags
    module.markers: dict[str, datetime] = {}  # jobs/<name>.exit timestamps
    module.orphan_sweeps = 0

    def fake_az(*args: str):  # type: ignore[no-untyped-def]
        if args[:2] == ("account", "show"):
            return "sub"
        if args[0] == "rest":
            return None if module.billed is None else {"properties": {"rows": [[module.billed, "USD"]]}}
        if args[:2] == ("vm", "list"):
            return [{"name": n, "power": "VM running", "tags": tags} for n, tags in module.alive.items()]
        if args[:3] == ("storage", "blob", "show"):
            name = args[args.index("-n") + 1].removeprefix("jobs/").removesuffix(".exit")
            marker = module.markers.get(name)
            return marker.isoformat().replace("+00:00", "Z") if marker else None
        return None

    def fake_clean_orphans() -> None:
        module.orphan_sweeps += 1

    monkeypatch.setattr(module, "az", fake_az)
    monkeypatch.setattr(module, "clean_orphans", fake_clean_orphans)
    return module


def test_reserve_until_the_cap(spend) -> None:  # type: ignore[no-untyped-def]
    for name in "abcd":
        assert spend.reserve(name, "lora", 20) == 0  # 4 x (20 + 0.5 h finish) x $0.88 = $72.16
    assert spend.reserve("e", "lora", 20) == 1  # $12.84 left under $100 - $15
    assert spend.reserve("e", "lora", 14) == 0  # 14.5 h x $0.88 = $12.76 fits
    assert sum(row["reserved_usd"] for row in spend.load()) <= 100 - 15
    assert spend.reserve("f", "lora", 25) == 2  # over the 20 h limit


def test_billed_cost_counts_when_higher(spend) -> None:  # type: ignore[no-untyped-def]
    spend.billed = 80.0  # something outside the ledger cost money
    assert spend.reserve("a", "lora", 6) == 1  # 80 + 15 + 6.5 h x 0.88 > 100
    assert spend.reserve("a", "lora", 5) == 0  # 80 + 15 + 5.5 h x 0.88 <= 100


def test_release_and_reconcile(spend) -> None:  # type: ignore[no-untyped-def]
    for name in ("done", "evicted", "running", "never"):
        spend.reserve(name, "lora", 10)
    spend.release("never")
    started = datetime.fromisoformat(spend.load()[0]["started"])
    spend.markers["done"] = started + timedelta(hours=1, minutes=10)
    spend.alive["running"] = {"bulwark-hours": "10"}
    spend.reconcile()
    rows = {row["name"]: row for row in spend.load()}
    assert rows["done"]["final_usd"] == pytest.approx(1.5 * 0.88)  # 1h10m + 15 min slack -> 1.5 h
    assert rows["evicted"]["final_usd"] == rows["evicted"]["reserved_usd"]  # worst case kept
    assert rows["running"]["final_usd"] is None
    assert rows["never"]["final_usd"] == 0.0
    assert json.loads(spend.LEDGER.read_text().splitlines()[0])["note"] == "ran ~1.5 h"


def test_reservation_covers_the_finish_window(spend) -> None:  # type: ignore[no-untyped-def]
    assert spend.reserve("a", "lora", 4) == 0
    assert spend.load()[0]["reserved_usd"] == pytest.approx(4.5 * 0.88)


def test_concurrent_reservations_keep_every_row(spend, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real_az = spend.az

    def slow_az(*args: str):  # type: ignore[no-untyped-def]
        if args[0] == "rest":
            time.sleep(0.2)  # widen the gap between reading and writing the ledger
        return real_az(*args)

    monkeypatch.setattr(spend, "az", slow_az)
    threads = [threading.Thread(target=spend.reserve, args=(name, "lora", 1)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(row["name"] for row in spend.load()) == ["a", "b", "c"]


def test_running_vms_without_a_ledger_row_count_against_the_cap(spend) -> None:  # type: ignore[no-untyped-def]
    # Launched from another clone (or the ledger was lost); untagged VMs count at the 20 h maximum.
    for index in range(3):
        spend.alive[f"bulwark-lora-{index}"] = {"bulwark-hours": "20"}
    spend.alive["bulwark-untagged"] = {}
    held = 4 * 20.5 * 0.88  # $72.16, leaving $12.84 under $100 - $15
    assert spend.reserve("a", "lora", 14.2) == 1  # 14.7 h x $0.88 = $12.94
    assert spend.reserve("a", "lora", 14) == 0  # 14.5 h x $0.88 = $12.76
    # Once the ledger has an open row for a VM, it is not counted twice.
    spend.alive["a"] = {"bulwark-hours": "14"}
    running = {name: {"power": "VM running", "tags": tags} for name, tags in spend.alive.items()}
    assert spend.untracked_usd(spend.load(), running) == pytest.approx(held)


def test_reserve_refuses_when_vms_cannot_be_listed(spend, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(spend, "vms", lambda: None)
    assert spend.reserve("a", "lora", 1) == 1
    assert spend.load() == []
    assert spend.reconcile() == 1


def test_reconcile_records_overruns_and_sweeps_orphans_only_when_idle(spend) -> None:  # type: ignore[no-untyped-def]
    spend.reserve("long", "lora", 2)
    spend.reserve("busy", "lora", 2)
    started = datetime.fromisoformat(spend.load()[0]["started"])
    spend.markers["long"] = started + timedelta(hours=2, minutes=40)  # ran past its 2.5 h
    spend.alive["busy"] = {"bulwark-hours": "2"}
    spend.reconcile()
    rows = {row["name"]: row for row in spend.load()}
    assert rows["long"]["final_usd"] == pytest.approx(3.0 * 0.88)  # 2h40m + 15 min -> 3 h, not capped
    assert rows["long"]["final_usd"] > rows["long"]["reserved_usd"]
    assert spend.orphan_sweeps == 0  # a job is still running

    del spend.alive["busy"]
    spend.markers["busy"] = started + timedelta(hours=1)
    spend.reconcile()
    assert spend.orphan_sweeps == 1


def test_status_flags_deallocated_vms(spend, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(spend, "vms", lambda: {"bulwark-x": {"power": "VM deallocated", "tags": {}}})
    spend.status()
    assert "still bill" in capsys.readouterr().out


def test_spend_start_falls_back_to_the_first_job(spend, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(spend, "START", "")
    rows = [{"started": "2026-10-30T10:00:00+00:00"}, {"started": "2026-10-29T09:00:00+00:00"}]
    assert spend.spend_start(rows) == "2026-10-29"
    assert spend.spend_start([]) < spend.now().strftime("%Y-%m-%d")
    monkeypatch.setattr(spend, "START", "2026-10-28")
    assert spend.spend_start(rows) == "2026-10-28"


def test_without_a_cap_no_job_is_reserved(spend, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(spend, "CAP", None)
    assert spend.reserve("a", "lora", 1) == 2
    assert spend.load() == []
    assert "BULWARK_CAP_USD" in capsys.readouterr().err
    assert spend.status() == 0  # status and reconcile still work
    assert "Cap not set" in capsys.readouterr().out


@pytest.mark.parametrize(("value", "cap"), [(None, None), ("", None), ("abc", None), ("0", None), ("-5", None), ("250", 250.0)])
def test_the_cap_has_no_default(spend, value, cap) -> None:  # type: ignore[no-untyped-def]
    assert spend.read_cap(value) == cap
