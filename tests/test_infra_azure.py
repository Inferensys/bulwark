"""Azure launcher and teardown, run against stub `az` and `azcopy` commands."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
AZURE = REPO / "infra" / "azure"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")

# Enough of the Azure CLI for run_job.sh, teardown.sh and spend.py. VMs live as files in
# $AZ_STATE/vms; AZ_FAIL lists steps that should fail. Like the real CLI, a missing
# resource exits 3 and an invalid argument value exits 2.
AZ_STUB = r'''#!/usr/bin/env python3
import json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

state = Path(os.environ["AZ_STATE"])
args = sys.argv[1:]
with open(state / "calls.jsonl", "a") as log:
    log.write(json.dumps(args) + "\n")
fail = set(os.environ.get("AZ_FAIL", "").split(","))
vms = state / "vms"
vms.mkdir(exist_ok=True)


def opt(name):
    return args[args.index(name) + 1] if name in args else None


def out(value=None):
    fmt = opt("-o") or "json"
    if fmt == "tsv" and value is not None:
        print("\n".join(map(str, value)) if isinstance(value, list) else value)
    elif fmt == "json":
        print(json.dumps(value))
    sys.exit(0)


words = [a for a in args if not a.startswith("-")][:3]
command = " ".join(words)
name = opt("-n")
if "--force-deletion" in args and opt("--force-deletion") not in {"true", "false", "yes", "no"}:
    sys.exit(2)
if command.startswith("account show"):
    out("sub-id")
if command.startswith("storage container generate-sas"):
    out("se=2026-10-30&sp=racwl&sig=SECRETSIG")
if command.startswith("vm create"):
    if "create" in fail:
        sys.exit(1)
    (vms / name).write_text("running")
    shutil.copy(opt("--custom-data"), state / "cloud-init.sh")
    out()
if command.startswith("vm show"):
    if "show" in fail:
        sys.exit(1)
    if not (vms / name).exists():
        sys.exit(3)
    rg = "/subscriptions/s/resourceGroups/rg/providers"
    out([f"{rg}/Microsoft.Compute/virtualMachines/{name}", "principal-1",
         f"{rg}/Microsoft.Network/networkInterfaces/{name}VMNic",
         f"{rg}/Microsoft.Compute/disks/{name}_OsDisk"])
if command.startswith("vm auto-shutdown"):
    sys.exit(1) if "auto-shutdown" in fail else out()
if command.startswith("network nic show"):
    out([f"ipconfig-test", "/subscriptions/s/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/ip-test"])
if command.startswith("network nic ip-config"):
    out()
if command.startswith("role assignment create"):
    sys.exit(1) if "role" in fail else out()
if command.startswith("vm delete"):
    if "delete" in fail:
        sys.exit(1)
    (vms / name).unlink(missing_ok=True)
    out()
if command.startswith("vm list"):
    out([{"name": vm.name, "power": "VM running", "tags": {}} for vm in vms.iterdir()])
if "list" in words[1:]:
    out([])
if command.startswith("storage blob show"):
    marker = state / "markers" / opt("-n").removeprefix("jobs/").removesuffix(".exit")
    if marker.exists():
        out(datetime.now(timezone.utc).isoformat())
    sys.exit(3)
if command.startswith("rest"):
    sys.exit(1)
print(f"az stub: unhandled {args}", file=sys.stderr)
sys.exit(2)
'''

AZCOPY_STUB = r"""#!/usr/bin/env bash
echo "$*" >> "$AZ_STATE/azcopy.log"
if [ "$1" = copy ] && [ "${2:-}" = --from-to ]; then
  cat > /dev/null
  case "$4" in
    */jobs/*.exit\?*) name=${4##*/jobs/}; mkdir -p "$AZ_STATE/markers"; touch "$AZ_STATE/markers/${name%%.exit\?*}" ;;
  esac
fi
"""


@pytest.fixture
def azure(tmp_path: Path) -> dict[str, str]:
    """Environment for running the Azure scripts against the stubs."""
    stubs, state, tmpdir = tmp_path / "bin", tmp_path / "state", tmp_path / "tmpdir"
    for path in (stubs, state, tmpdir):
        path.mkdir()
    for name, text in (("az", AZ_STUB), ("azcopy", AZCOPY_STUB)):
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "AZ_STATE": str(state),
        "TMPDIR": str(tmpdir),
        "BULWARK_RG": "rg",
        "BULWARK_STORAGE": "teststore",
        "BULWARK_LEDGER": str(tmp_path / "ledger.jsonl"),
        "BULWARK_SPEND_START": "2026-10-01",
        "BULWARK_SSH_KEY": str(tmp_path / "key" / "id_ed25519"),
        "ALLOW_DIRTY": "1",  # the worktree under test may hold uncommitted edits
        "BULWARK_CAP_USD": "100",
        # The tests set every value they need; a developer's own env.local.sh stays out.
        "BULWARK_ENV_LOCAL": str(tmp_path / "no-env.local.sh"),
        "PYTHON": sys.executable,  # runs infra/job_inputs.py (needs PyYAML)
    }


def calls(env: dict[str, str]) -> list[list[str]]:
    lines = (Path(env["AZ_STATE"]) / "calls.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def ledger(env: dict[str, str]) -> list[dict]:
    return [json.loads(line) for line in Path(env["BULWARK_LEDGER"]).read_text().splitlines()]


LORA = ("lora", "configs/train/lora.yaml", "--hours", "1", "--set", 'datasets=["drive_and_act"]')


def run_job(env: dict[str, str], *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(AZURE / "run_job.sh"), *(args or LORA)]
    return subprocess.run(command, cwd=REPO, env={**env, **extra}, capture_output=True, text=True, timeout=120)


def cloud_init_arrays(script: str) -> dict[str, list[str]]:
    """READS and WRITES as the VM's bash sees them."""
    lines = [line for line in script.splitlines() if line.startswith(("READS=(", "WRITES=("))]
    shown = 'printf "READS %s\\n" "${READS[@]}"; printf "WRITES %s\\n" "${WRITES[@]}"'
    out = subprocess.run(["bash", "-c", "\n".join([*lines, shown])], capture_output=True, text=True, check=True).stdout
    arrays: dict[str, list[str]] = {"READS": [], "WRITES": []}
    for line in out.splitlines():
        name, _, value = line.partition(" ")
        if value:
            arrays[name].append(value)
    return arrays


def test_run_job_launches_with_narrow_rights_and_a_safe_cloud_init(azure: dict[str, str]) -> None:
    result = run_job(azure)
    assert result.returncode == 0, result.stderr
    made = calls(azure)
    create = next(c for c in made if c[:2] == ["vm", "create"])
    assert "--role" not in create and "--scope" not in create
    assert "bulwark-hours=1" in create
    roles = [(c[c.index("--role") + 1], c[c.index("--scope") + 1]) for c in made if c[:3] == ["role", "assignment", "create"]]
    assert {scope.split("/")[-2] for _, scope in roles} == {
        "virtualMachines", "networkInterfaces", "disks", "publicIPAddresses"
    }  # fmt: skip
    first_role = made.index(next(c for c in made if c[:2] == ["role", "assignment"]))
    assert made.index(next(c for c in made if c[:2] == ["vm", "auto-shutdown"])) < first_role
    assert any("publicIPAddress.deleteOption=Delete" in c for c in made if c[:3] == ["network", "nic", "ip-config"])

    script = (Path(azure["AZ_STATE"]) / "cloud-init.sh").read_text()
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    assert not re.search(r"^\s*set -[a-z]*x", script, re.M)  # never trace: the log would hold the SAS
    assert script.count("SECRETSIG") == 1  # only in the SAS= assignment
    assert "export HOME=" in script and "uv sync --frozen" in script
    # Without CUDA the job would train on the CPU for its whole reservation: stop right after install.
    assert script.index("uv sync --frozen") < script.index("torch.cuda.is_available()") < script.index("fetch_pretrained")
    # No orphan cleanup at launch: it could delete the NIC or IP of another launch mid-create.
    assert not any(c[:2] in (["network", "public-ip"], ["network", "nic"]) and "list" in c for c in made)
    train = next(line for line in script.splitlines() if line.startswith(".venv/bin/bulwark train"))
    assert shlex.split(train) == [".venv/bin/bulwark", "train", "lora", "--config", "configs/train/lora.yaml",
                                  "--set", 'datasets=["drive_and_act"]']  # fmt: skip
    assert not list(Path(azure["TMPDIR"]).iterdir())  # the SAS-bearing temp files are gone
    (row,) = ledger(azure)
    assert row["final_usd"] is None and row["reserved_usd"] > 0
    # Only the init it reads and the out dir it writes, never the whole checkpoints and
    # features trees; what it writes is what gets synced back.
    assert cloud_init_arrays(script) == {"READS": ["checkpoints/heads-v0/best.pt"], "WRITES": ["checkpoints/b1-lora-v0"]}
    assert "$BASE/checkpoints?" not in script and "$BASE/data/features?" not in script
    assert "up_all 300" in script


def test_run_job_runs_an_evaluation_into_its_own_report_dir(azure: dict[str, str]) -> None:
    result = run_job(azure, "evaluate", "checkpoints/b1-lora-v0/best.pt", "--hours", "1", "--model", "b1", "--precision", "bf16")
    assert result.returncode == 0, result.stderr
    script = (Path(azure["AZ_STATE"]) / "cloud-init.sh").read_text()
    create = next(c for c in calls(azure) if c[:2] == ["vm", "create"])
    name = create[create.index("-n") + 1]
    command = next(line for line in script.splitlines() if line.startswith(".venv/bin/bulwark "))
    assert shlex.split(command) == [".venv/bin/bulwark", "evaluate", "--checkpoint", "checkpoints/b1-lora-v0/best.pt",
                                    "--model", "b1", "--precision", "bf16", "--out", f"runs/{name}"]  # fmt: skip
    assert cloud_init_arrays(script) == {"READS": ["checkpoints/b1-lora-v0/best.pt"], "WRITES": [f"runs/{name}"]}
    # The 5-minute sync leaves out files still being written (temporary names in
    # bulwark/eval/predict.py, bulwark/train/common.py and cache_features.py).
    (patterns,) = re.findall(r"--exclude-pattern '([^']*)'", script)
    skipped = {name for name in ("test.tmp.npz", "last.pt.tmp", "features-dmd-00000.npz.tmp", "test.npz", "best.pt")
               if any(fnmatch.fnmatch(name, pattern) for pattern in patterns.split(";"))}  # fmt: skip
    assert skipped == {"test.tmp.npz", "last.pt.tmp", "features-dmd-00000.npz.tmp"}


def test_run_job_reads_options_written_with_an_equals_sign(azure: dict[str, str]) -> None:
    # `--set=out=...` is what `bulwark train` writes to, so it is what the VM must sync back.
    result = run_job(azure, "lora", "configs/train/lora.yaml", "--hours=2", "--set=out=checkpoints/other")
    assert result.returncode == 0, result.stderr
    create = next(c for c in calls(azure) if c[:2] == ["vm", "create"])
    assert "bulwark-hours=2" in create
    script = (Path(azure["AZ_STATE"]) / "cloud-init.sh").read_text()
    assert cloud_init_arrays(script)["WRITES"] == ["checkpoints/other"]
    train = next(line for line in script.splitlines() if line.startswith(".venv/bin/bulwark train"))
    assert "--hours" not in train and shlex.split(train)[-1] == "--set=out=checkpoints/other"


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (LORA, {"BULWARK_CAP_USD": ""}, "BULWARK_CAP_USD"),
        (("lora", "configs/train/missing.yaml"), {}, "config not found"),
        (("evaluate", "runs/e1.onnx"), {}, "under checkpoints/"),
        (("lora", "configs/train/lora.yaml", "--set", "out=../out"), {}, "inside the repo"),
        (("evaluate", "checkpoints/e1-v0/best.pt", "--model", "e1", "--reference", "runs/b1"), {}, "--reference"),
        (("evaluate", "checkpoints/heads-v0/best.pt"), {}, "--model auto"),
        (("lora", "configs/train/lora.yaml", "--hours=two"), {}, "--hours must be a number"),
    ],
)
def test_run_job_refuses_before_reserving_or_creating_anything(azure: dict[str, str], args, env, message) -> None:  # type: ignore[no-untyped-def]
    result = run_job(azure, *args, **env)
    assert result.returncode != 0 and message in result.stderr
    made = calls(azure) if (Path(azure["AZ_STATE"]) / "calls.jsonl").exists() else []
    assert not any(c[:2] in (["vm", "create"], ["storage", "container"]) for c in made)
    assert not Path(azure["BULWARK_LEDGER"]).exists() or ledger(azure) == []


@pytest.mark.parametrize("step", ["auto-shutdown", "role"])
def test_a_launch_failing_after_create_deletes_the_vm(azure: dict[str, str], step: str) -> None:
    result = run_job(azure, AZ_FAIL=step)
    assert result.returncode != 0
    assert any(c[:2] == ["vm", "delete"] for c in calls(azure))
    assert not list((Path(azure["AZ_STATE"]) / "vms").iterdir())
    assert not list(Path(azure["TMPDIR"]).iterdir())
    (row,) = ledger(azure)
    assert row["final_usd"] is not None and row["note"].startswith("ran ~")  # charged for its minutes


def test_a_launch_failing_before_create_releases_the_reservation(azure: dict[str, str]) -> None:
    result = run_job(azure, AZ_FAIL="create")
    assert result.returncode != 0
    assert not any(c[:2] == ["vm", "delete"] for c in calls(azure))
    assert not list(Path(azure["TMPDIR"]).iterdir())
    (row,) = ledger(azure)
    assert row["final_usd"] == 0.0


def run_teardown(env: dict[str, str], *vms: str, **extra: str) -> subprocess.CompletedProcess[str]:
    for vm in vms:
        (Path(env["AZ_STATE"]) / "vms").mkdir(exist_ok=True)
        (Path(env["AZ_STATE"]) / "vms" / vm).write_text("running")
    command = ["bash", str(AZURE / "teardown.sh"), "bulwark-lora-1"]
    return subprocess.run(command, env={**env, **extra}, capture_output=True, text=True, timeout=60)


def test_teardown_deletes_the_vm_and_reconciles(azure: dict[str, str]) -> None:
    result = run_teardown(azure, "bulwark-lora-1")
    assert result.returncode == 0, result.stderr
    assert not (Path(azure["AZ_STATE"]) / "vms" / "bulwark-lora-1").exists()
    made = [c[:2] for c in calls(azure)]
    assert ["vm", "delete"] in made and "--force-deletion" not in calls(azure)[made.index(["vm", "delete"])]
    assert made.index(["vm", "list"]) > made.index(["vm", "delete"])  # spend.sh reconcile ran after


def test_teardown_fails_loudly_when_the_vm_survives(azure: dict[str, str]) -> None:
    result = run_teardown(azure, "bulwark-lora-1", AZ_FAIL="delete")
    assert result.returncode == 1 and "still exists" in result.stderr
    assert any(c[:2] == ["vm", "list"] for c in calls(azure))  # reconcile ran anyway


def test_teardown_does_not_mistake_an_az_error_for_a_deleted_vm(azure: dict[str, str]) -> None:
    result = run_teardown(azure, "bulwark-lora-1", AZ_FAIL="show")
    assert result.returncode == 1 and "logged in" in result.stderr
