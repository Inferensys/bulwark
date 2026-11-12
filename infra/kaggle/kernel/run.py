"""Kaggle entry point (templated by push_job.sh). Runs one `bulwark train <job>`, or
`bulwark evaluate` on a checkpoint (job `evaluate`).

The code travels inside this script (CODE_B64, a .tgz of the repo with PROVENANCE.json),
so a code change needs no re-upload of the data. Inputs come from /kaggle/input:

- the manifest only from the MANIFEST_DATASET dataset (bulwark-meta by default);
- shards from every attached dataset; one name supplied twice with different sizes is an
  error rather than a coin toss;
- features and checkpoints from attached datasets or previous sessions' outputs, limited
  to the directories the job reads or writes (infra/job_inputs.py). The directory it
  writes to is copied (so it can resume from it); the ones it only reads are linked,
  read-only, to keep the output small. One directory supplied by two inputs with
  different contents is an error, so a resume never picks a stale last.pt by path order.
  A path the job reads that no input supplies stops it before it starts.

Everything left in /kaggle/working becomes the notebook's output (20 GB at most), so the
links and the downloaded weights are removed at the end. run-inputs.json and
environment.txt record what the run used.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

JOB, CONFIG, EXTRA = "__JOB__", "__CONFIG__", "__EXTRA__"
MANIFEST_DATASET = "__MANIFEST_DATASET__"
CODE_B64 = "__CODE_B64__"
UV_VERSION = "0.8.17"  # the uv that wrote uv.lock
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
WORK = OUTPUT / "bulwark"
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
OUTPUT_LIMIT_GB = 18.0  # Kaggle drops output above 20 GB; stop with an error before that
PLACED = (("checkpoints", "checkpoints"), ("features", "data/features"))  # (record key, dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def found(pattern: str) -> list[Path]:
    # Recursive: Kaggle mounts datasets at /kaggle/input/<slug> and, at times, deeper.
    return sorted(Path(p) for p in glob.glob(str(INPUT / pattern), recursive=True))


def find_shard_files() -> list[Path]:
    """Every attached `.shard`. Kaggle unpacks an upload that is a single folder into the
    dataset root, so shards may sit at <input>/<slug>/x.shard rather than under shards/;
    shard names start with their dataset, so a recursive match cannot pick up strays."""
    return found("**/*.shard")


def pick_manifest(manifests: list[Path], dataset: str) -> Path:
    """The manifest in the `dataset` input; several copies of it must be identical."""
    chosen = [path for path in manifests if dataset in path.parts]
    if not chosen:
        raise SystemExit(
            f"No manifest.parquet in an attached '{dataset}' input (found: {[str(p) for p in manifests]}). "
            "Attach it (DATASETS in push_job.sh) or name the input that holds it (MANIFEST_DATASET)."
        )
    if len({sha256(path) for path in chosen}) > 1:
        raise SystemExit(f"'{dataset}' holds manifests that differ: {[str(p) for p in chosen]}")
    return chosen[0]


def link_shards(shards: list[Path], dest: Path) -> list[dict]:
    """Link every attached `<name>.shard` as dest/<name>.tar (Kaggle unpacks uploaded .tar)."""
    dest.mkdir(parents=True, exist_ok=True)
    linked: dict[str, Path] = {}
    for shard in sorted(shards):
        name = shard.name[: -len(".shard")] + ".tar"
        if name in linked:
            if linked[name].stat().st_size != shard.stat().st_size:
                raise SystemExit(f"{name} is attached twice with different sizes: {linked[name]} and {shard}")
            continue
        linked[name] = shard
        os.symlink(shard, dest / name)
    return [{"name": name, "source": str(path), "bytes": path.stat().st_size} for name, path in linked.items()]


def signature(path: Path, contents: bool = False) -> object:
    """Identity for comparing two copies: (path, size) of every file, or (path, sha256) with
    `contents`. Sizes are enough for read-only inputs; a checkpoint to resume from keeps
    its size from one session to the next, so it needs the hashes."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [(str(p.relative_to(path)), sha256(p) if contents else p.stat().st_size) for p in files]


def place(sources: list[Path], dest: Path, written: set[str] | None, wanted: set[str] | None = None) -> list[dict]:
    """Bring the `wanted` entries (None: all) of attached `features` or `checkpoints` dirs
    into dest: copy the ones this job writes (to resume from them), link the rest. An
    entry that two inputs supply with different contents is an error, never a silent
    choice by path order."""
    dest.mkdir(parents=True, exist_ok=True)
    placed: dict[str, Path] = {}
    records = []
    for source in sorted(sources):
        for entry in sorted(source.iterdir()):
            if entry.name == "pretrained":
                continue  # fetched fresh by scripts/fetch_pretrained.sh
            if wanted is not None and entry.name not in wanted:
                continue  # this job neither reads nor writes it
            target = dest / entry.name
            copy = written is None or entry.name in written
            if entry.name in placed:
                if signature(placed[entry.name], copy) != signature(entry, copy):
                    raise SystemExit(
                        f"{target} is attached twice with different contents: {placed[entry.name]} and {entry}. "
                        "Attach only the input to use (DATASETS / PREVIOUS_KERNELS in push_job.sh)."
                    )
                continue
            if copy:
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, target)
            else:
                os.symlink(entry, target)
            placed[entry.name] = entry
            records.append({"path": str(target), "source": str(entry), "mode": "copied" if copy else "linked"})
    return records


def require(reads: list[str]) -> None:
    """Stop before the job starts if a path it reads was not placed from any input."""
    missing = [path for path in reads if not Path(path).exists()]
    if missing:
        raise SystemExit(
            f"Not in any attached input: {', '.join(missing)}. Attach the dataset or the previous "
            "kernel whose output holds it (DATASETS / PREVIOUS_KERNELS in push_job.sh)."
        )


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str:
    return canonical(re.split(r"[\s=<>!~;\[(]", line.strip(), maxsplit=1)[0])


def torch_stack() -> set[str]:
    """torch, torchvision and everything they pull in, as installed in this image."""
    from importlib.metadata import PackageNotFoundError, requires

    seen: set[str] = set()
    todo = ["torch", "torchvision"]
    while todo:
        name = canonical(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            requirements = requires(name) or []
        except PackageNotFoundError:
            continue
        todo += [requirement_name(r) for r in requirements if "extra ==" not in r]
    return seen


def constraints(exported: str, skip: set[str]) -> str:
    """`uv export` output without the packages in skip."""
    kept = [
        line for line in exported.splitlines()
        if not line.strip() or line.lstrip().startswith("#") or requirement_name(line) not in skip
    ]  # fmt: skip
    return "\n".join(kept) + "\n"


def install() -> None:
    """Install the package with every dependency pinned to uv.lock, except the image's
    torch stack: reinstalling torch could mismatch Kaggle's CUDA driver."""
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, f"uv=={UV_VERSION}"], check=True)
    exported = subprocess.run(
        [sys.executable, "-m", "uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--extra", "export"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    path = OUTPUT / "constraints.txt"
    path.write_text(constraints(exported, torch_stack()))
    subprocess.run([*pip, "-c", str(path), "-e", ".[export]"], check=True)


def output_gb(root: Path) -> float:
    """Size of what Kaggle would keep: real files only, links not followed."""
    total = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                total += os.path.getsize(path)
    return total / 1e9


def remove_links(root: Path) -> None:
    for dirpath, dirnames, files in os.walk(root):
        for name in dirnames + files:
            path = Path(dirpath) / name
            if path.is_symlink():
                path.unlink()


def load_job_inputs(path: Path) -> types.ModuleType:
    """infra/job_inputs.py from the unpacked code (this script travels on its own)."""
    spec = importlib.util.spec_from_file_location("job_inputs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment() -> str:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
    return f"python {sys.version}\n\n{freeze}\n{gpu}"


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz") as archive:
        archive.extractall(WORK, filter="data")
    os.chdir(WORK)
    # The V-JEPA code and weights (1.7 GB) are fetched each run; keep them out of the output.
    for name in ("third_party", "checkpoints/pretrained"):
        (SCRATCH / name).mkdir(parents=True, exist_ok=True)
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        Path(name).symlink_to(SCRATCH / name)
    try:
        install()
        subprocess.run(["scripts/fetch_pretrained.sh"], check=True)
        jobs = load_job_inputs(WORK / "infra" / "job_inputs.py")
        paths = jobs.job_inputs(JOB, CONFIG, EXTRA)
        manifest = pick_manifest(found("**/manifest.parquet"), MANIFEST_DATASET)
        Path("data").mkdir(exist_ok=True)
        os.symlink(manifest, "data/manifest.parquet")
        inputs = {
            "job": JOB,
            "checkpoint" if JOB == "evaluate" else "config": CONFIG,
            "extra": EXTRA,
            "paths": paths,
            "code": json.loads(Path("PROVENANCE.json").read_text()),
            "manifest": {"source": str(manifest), "sha256": sha256(manifest)},
            "shards": link_shards(find_shard_files(), Path("data/shards")),
        }
        print(f"Manifest: {manifest} (sha256 {inputs['manifest']['sha256']})")
        for key, root in PLACED:
            wanted = jobs.entries_under(paths["read"] + paths["write"], root)
            if wanted != set():
                written = jobs.entries_under(paths["write"], root)
                inputs[key] = place(found(f"**/{root}"), Path(root), written, wanted)
        (OUTPUT / "run-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        require(paths["read"])
        (OUTPUT / "environment.txt").write_text(environment())
        if output_gb(OUTPUT) > OUTPUT_LIMIT_GB:
            raise SystemExit(f"Copied inputs already take {output_gb(OUTPUT):.1f} GB of the 20 GB output")
        command = ["evaluate", "--checkpoint", CONFIG] if JOB == "evaluate" else ["train", JOB, "--config", CONFIG]
        subprocess.run([sys.executable, "-m", "bulwark.cli", *command, *EXTRA], check=True)
    finally:
        remove_links(OUTPUT)
        size = output_gb(OUTPUT)
        print(f"Output: {size:.1f} GB in {OUTPUT}")
    if size > OUTPUT_LIMIT_GB:
        raise SystemExit(f"Output is {size:.1f} GB; Kaggle keeps at most 20 GB, so it may be lost")


if INPUT.is_dir():  # on Kaggle; tests import the helpers above without running anything
    main()
