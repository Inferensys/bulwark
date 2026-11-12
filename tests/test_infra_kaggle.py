"""Kaggle jobs: input selection in kernel/run.py, push_job.sh templating, push_data staging."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).parents[1]
KAGGLE = REPO / "infra" / "kaggle"


def load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def run():  # type: ignore[no-untyped-def]
    return load(KAGGLE / "kernel" / "run.py", "kaggle_run")


def write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_manifest_comes_only_from_the_named_input(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    old = write(tmp_path / "bulwark-data" / "manifest.parquet", b"old")
    meta = write(tmp_path / "bulwark-meta" / "manifest.parquet", b"new")
    same = write(tmp_path / "bulwark-meta" / "deeper" / "manifest.parquet", b"new")
    assert run.pick_manifest([old, meta, same], "bulwark-meta") == meta
    with pytest.raises(SystemExit, match="No manifest"):
        run.pick_manifest([old], "bulwark-meta")
    write(same, b"other")
    with pytest.raises(SystemExit, match="differ"):
        run.pick_manifest([old, meta, same], "bulwark-meta")


def test_shards_with_one_name_must_match(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a = write(tmp_path / "in1" / "shards" / "mrl_eye-00000.shard", b"abc")
    b = write(tmp_path / "in2" / "shards" / "mrl_eye-00000.shard", b"abc")
    c = write(tmp_path / "in2" / "shards" / "zenodo_driver-00000.shard", b"z")
    records = run.link_shards([a, b, c], tmp_path / "out")
    assert sorted(r["name"] for r in records) == ["mrl_eye-00000.tar", "zenodo_driver-00000.tar"]
    assert os.readlink(tmp_path / "out" / "mrl_eye-00000.tar") == str(a)
    write(b, b"abcd")
    with pytest.raises(SystemExit, match="different sizes"):
        run.link_shards([a, b], tmp_path / "out2")


def test_only_the_written_dir_is_copied(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    previous = tmp_path / "prev" / "checkpoints"
    meta = tmp_path / "meta" / "checkpoints"
    write(previous / "b1-lora-v0" / "last.pt", b"resume")
    write(previous / "pretrained" / "vjepa.pt", b"big")
    write(meta / "heads-v0" / "best.pt", b"heads")
    dest = tmp_path / "work" / "checkpoints"
    records = run.place([previous, meta], dest, {"b1-lora-v0"})
    assert {Path(r["path"]).name: r["mode"] for r in records} == {"b1-lora-v0": "copied", "heads-v0": "linked"}
    assert not (dest / "b1-lora-v0").is_symlink() and (dest / "heads-v0").is_symlink()
    assert not (dest / "pretrained").exists()

    run.remove_links(tmp_path / "work")
    assert not (dest / "heads-v0").exists() and (dest / "b1-lora-v0" / "last.pt").exists()
    assert run.output_gb(tmp_path / "work") == pytest.approx(len(b"resume") / 1e9)

    clash = tmp_path / "other" / "checkpoints"
    write(clash / "heads-v0" / "best.pt", b"different heads")
    with pytest.raises(SystemExit, match="different contents"):
        run.place([meta, clash], tmp_path / "work2" / "checkpoints", set())


def test_a_resumed_dir_from_two_inputs_must_be_identical(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # Two sessions' last.pt have the same size; only their contents tell them apart.
    old = write(tmp_path / "bulwark-lora-10" / "checkpoints" / "b1-lora-v0" / "last.pt", b"step-100")
    new = write(tmp_path / "bulwark-lora-9" / "checkpoints" / "b1-lora-v0" / "last.pt", b"step-900")
    sources = [old.parents[1], new.parents[1]]
    with pytest.raises(SystemExit, match="different contents"):
        run.place(sources, tmp_path / "work" / "checkpoints", {"b1-lora-v0"})

    write(new, b"step-100")  # the same checkpoint attached twice is fine
    records = run.place(sources, tmp_path / "work2" / "checkpoints", {"b1-lora-v0"})
    assert [(r["mode"], r["source"]) for r in records] == [("copied", str(old.parent))]
    assert (tmp_path / "work2" / "checkpoints" / "b1-lora-v0" / "last.pt").read_bytes() == b"step-100"


def test_only_the_dirs_the_job_uses_are_placed(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # Two inputs disagree about a features dir this job never reads: that must not stop it.
    first, second = tmp_path / "a" / "data" / "features", tmp_path / "b" / "data" / "features"
    write(first / "frozen" / "features-x.npz", b"frozen")
    write(first / "b1-lora-v0" / "features-x.npz", b"teacher")
    write(second / "b1-lora-v0" / "features-x.npz", b"another teacher")
    dest = tmp_path / "work" / "data" / "features"
    records = run.place([first, second], dest, set(), {"frozen"})
    assert [(Path(r["path"]).name, r["mode"]) for r in records] == [("frozen", "linked")]
    assert sorted(p.name for p in dest.iterdir()) == ["frozen"]


def test_a_read_that_no_input_supplies_stops_the_job(run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    attached = write(tmp_path / "input" / "meta" / "checkpoints" / "heads-v0" / "best.pt")
    run.place([attached.parents[1]], Path("checkpoints"), set())
    run.require(["checkpoints/heads-v0/best.pt"])  # linked from the input: found
    with pytest.raises(SystemExit, match="data/features/frozen.*PREVIOUS_KERNELS"):
        run.require(["checkpoints/heads-v0/best.pt", "data/features/frozen"])


def test_constraints_leave_the_image_torch_stack_alone(run) -> None:  # type: ignore[no-untyped-def]
    exported = "\n".join([
        "# This file was autogenerated by uv",
        "numpy==2.4.6 ; python_full_version == '3.11.*'",
        "    # via bulwark",
        "nvidia-cudnn-cu13==9.26.0.51 ; sys_platform == 'linux'",
        "sympy==1.14.0",
        "torch==2.15.0 ; python_full_version >= '3.11'",
        "Typing_Extensions==4.16.0",
    ])  # fmt: skip
    kept = run.constraints(exported, {"torch", "sympy", "nvidia-cudnn-cu13", "typing-extensions"})
    assert "numpy==2.4.6" in kept and "# via bulwark" in kept
    assert not any(name in kept for name in ("torch==", "sympy", "nvidia", "Typing"))


@pytest.fixture
def kaggle_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "kaggle"
    stub.write_text('#!/usr/bin/env bash\necho "$*" >> "$(dirname "$0")/kaggle.log"\n')
    stub.chmod(0o755)
    return stub


def push_job(tmp_path: Path, stub: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "KAGGLE_USER": "someone", "KAGGLE": str(stub), "STAGE": str(tmp_path / "stage"),
                   "ALLOW_DIRTY": "1", "PYTHON": sys.executable, **env}  # fmt: skip
    command = ["bash", str(KAGGLE / "push_job.sh"), *args]
    return subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True, timeout=120)


def test_push_job_passes_arguments_as_a_list(tmp_path: Path, kaggle_stub: Path) -> None:
    result = push_job(
        tmp_path, kaggle_stub, "lora", "configs/train/lora-kaggle.yaml", "--set", 'datasets=["drive_and_act"]',
        PREVIOUS_KERNELS="bulwark-lora bulwark-cache",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    job = load(tmp_path / "stage" / "run.py", "kaggle_run_templated")
    assert (job.JOB, job.CONFIG, job.MANIFEST_DATASET) == ("lora", "configs/train/lora-kaggle.yaml", "bulwark-meta")
    assert job.EXTRA == ["--set", 'datasets=["drive_and_act"]']
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(job.CODE_B64))) as archive:
        assert "PROVENANCE.json" in archive.getnames()
    metadata = json.loads((tmp_path / "stage" / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "someone/bulwark-drive-and-act", "someone/bulwark-dmd", "someone/bulwark-mrl", "someone/bulwark-auc",
        "someone/bulwark-meta",
    ]  # fmt: skip
    assert metadata["kernel_sources"] == ["someone/bulwark-lora", "someone/bulwark-cache"]
    assert "kernels push" in (tmp_path / "kaggle.log").read_text()


def test_push_job_runs_an_evaluation(tmp_path: Path, kaggle_stub: Path) -> None:
    result = push_job(
        tmp_path, kaggle_stub, "evaluate", "checkpoints/b1-lora-v0/best.pt", "--model", "b1", "--precision", "fp16",
        PREVIOUS_KERNELS="bulwark-lora",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    job = load(tmp_path / "stage" / "run.py", "kaggle_run_evaluate")
    assert (job.JOB, job.CONFIG) == ("evaluate", "checkpoints/b1-lora-v0/best.pt")
    assert job.EXTRA == ["--model", "b1", "--precision", "fp16"]
    assert json.loads((tmp_path / "stage" / "kernel-metadata.json").read_text())["kernel_sources"] == [
        "someone/bulwark-lora"
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("lora", "configs/train/missing.yaml"), {}, "config not found"),
        (("lora", "configs/train/lora-kaggle.yaml"), {"DATASETS": "bulwark-dmd"}, "must include bulwark-meta"),
        (("train", "configs/train/lora-kaggle.yaml"), {}, "unknown job"),
        (("lora", "configs/train/lora-kaggle.yaml", "--set", "out=/kaggle/working/x"), {}, "inside the repo"),
        (("evaluate", "runs/export/e1.onnx"), {}, "under checkpoints/"),
        (("evaluate", "checkpoints/b1-lora-v0/best.pt", "--model", "b1", "--reference", "runs/b1"), {}, "--reference"),
        (("evaluate", "checkpoints/heads-v0/best.pt"), {}, "--model auto"),
    ],
)
def test_push_job_fails_locally_before_any_upload(tmp_path: Path, kaggle_stub: Path, args, env, message) -> None:  # type: ignore[no-untyped-def]
    result = push_job(tmp_path, kaggle_stub, *args, **env)
    assert result.returncode == 2 and message in result.stderr
    assert not (tmp_path / "kaggle.log").exists()


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    module = load(KAGGLE / "stage_data.py", "stage_data")
    monkeypatch.setattr(module, "clearance", lambda storage: {"zenodo_driver": True, "dmd": False, "mrl_eye": None})
    root = tmp_path / "repo"
    for name in ("zenodo_driver-00000.tar", "dmd-00000.tar", "mrl_eye-00000.tar"):
        write(root / "data" / "shards" / name, name.encode())
    table = pa.table({"clip_id": ["z/1", "d/1", "m/1"], "dataset": ["zenodo_driver", "dmd", "mrl_eye"]})
    pq.write_table(table, write(root / "data" / "manifest.parquet"))
    return module, root


def test_stage_refuses_datasets_not_cleared_for_hosting(stage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    out = tmp_path / "staged"
    assert module.main(["--root", str(root), "--stage", str(out), "--shards", "*.tar"]) == 1
    assert not (out / "shards").exists()
    assert module.main(["--root", str(root), "--stage", str(out), "--shards", "zenodo_driver-*.tar"]) == 0
    assert [p.name for p in (out / "shards").iterdir()] == ["zenodo_driver-00000.shard"]
    assert module.main(["--root", str(root), "--stage", str(out), "--shards", "mrl_eye-*.tar", "--allow", "mrl_eye"]) == 0


def test_staged_manifest_never_loses_rows_silently(stage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    out = tmp_path / "meta"
    args = ["--root", str(root), "--stage", str(out), "--manifest"]
    assert module.main([*args, "--allow", "mrl_eye"]) == 1  # dmd rows would vanish
    assert not (out / "manifest.parquet").exists()
    assert module.main([*args, "--allow", "mrl_eye", "--allow", "dmd"]) == 0
    assert pq.read_table(out / "manifest.parquet").num_rows == 3

    assert module.main([*args, "--allow", "mrl_eye", "--drop-uncleared"]) == 0
    staged = pq.read_table(out / "manifest.parquet")
    assert sorted(staged.column("dataset").to_pylist()) == ["mrl_eye", "zenodo_driver"]
    assert not (out / "shards").exists()


def test_staged_manifest_is_never_empty(stage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    monkeypatch.setattr(module, "clearance", lambda storage: {"zenodo_driver": None, "dmd": False, "mrl_eye": None})
    out = tmp_path / "meta"
    assert module.main(["--root", str(root), "--stage", str(out), "--manifest", "--drop-uncleared"]) == 1
    assert not (out / "manifest.parquet").exists()
    assert "no dataset in the manifest is cleared" in capsys.readouterr().err


def test_stage_takes_several_globs_and_later_shard_runs(stage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    write(root / "data" / "shards" / "zenodo_driver-r001-00000.tar", b"added later")
    out = tmp_path / "staged"
    args = ["--root", str(root), "--stage", str(out), "--allow", "mrl_eye"]
    assert module.main([*args, "--shards", "zenodo_driver-*.tar mrl_eye-*.tar"]) == 0
    assert sorted(p.name for p in (out / "shards").iterdir()) == [
        "mrl_eye-00000.shard", "zenodo_driver-00000.shard", "zenodo_driver-r001-00000.shard"
    ]  # fmt: skip
    assert module.main([*args, "--shards", "zenodo_driver-*.tar typo-*.tar"]) == 1


@pytest.mark.parametrize("path", ["data/features/frozen", "data/manifest.parquet", "checkpoints/../data", "/etc"])
def test_stage_includes_only_checkpoints(stage, tmp_path: Path, path: str) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    write(root / "checkpoints" / "heads-v0" / "best.pt", b"heads")
    out = tmp_path / "staged"
    assert module.main(["--root", str(root), "--stage", str(out), "--include", path]) == 1
    assert module.main(["--root", str(root), "--stage", str(out), "--include", "checkpoints/heads-v0"]) == 0
    assert (out / "checkpoints" / "heads-v0" / "best.pt").read_bytes() == b"heads"


def test_stage_includes_finished_checkpoints_but_never_last_pt(stage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    run_dir = root / "checkpoints" / "heads-v0"
    for name in ("best.pt", "final.pt", "provenance.json", "last.pt", "log.jsonl", "stale-20261001T000000Z/best.pt"):
        write(run_dir / name, name.encode())
    out = tmp_path / "staged"
    assert module.main(["--root", str(root), "--stage", str(out), "--include", "checkpoints/heads-v0"]) == 0
    staged = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert staged == [f"checkpoints/heads-v0/{name}" for name in ("best.pt", "final.pt", "provenance.json")]

    from bulwark.train.common import RUN_FILES

    assert set(module.INCLUDE_FILES) == set(RUN_FILES) - {"last.pt", "log.jsonl"}

    args = ["--root", str(root), "--stage", str(out), "--include"]
    assert module.main([*args, "checkpoints/heads-v0/last.pt"]) == 1  # named explicitly: still refused
    assert module.main([*args, "checkpoints/missing"]) == 1
    assert module.main([*args, "checkpoints/heads-v0/final.pt"]) == 0
    assert [p.name for p in (out / "checkpoints" / "heads-v0").iterdir()] == ["final.pt"]


def test_stage_includes_exported_graphs(stage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # For `push_job.sh evaluate checkpoints/bulwark-e1.onnx --model onnx`.
    module, root = stage
    for name in ("bulwark-e1.onnx", "bulwark-e1.onnx.bench.json", "bulwark-e1.onnx.tmp"):
        write(root / "checkpoints" / name, name.encode())
    out = tmp_path / "staged"
    args = ["--root", str(root), "--stage", str(out), "--include"]
    assert module.main([*args, "checkpoints/bulwark-e1.onnx"]) == 0
    assert (out / "checkpoints" / "bulwark-e1.onnx").read_bytes() == b"bulwark-e1.onnx"
    assert module.main([*args, "checkpoints"]) == 0
    assert [p.name for p in (out / "checkpoints").iterdir()] == ["bulwark-e1.onnx"]
    assert module.main([*args, "checkpoints/bulwark-e1.onnx.tmp"]) == 1  # still being written


def test_shard_names_parse_the_same_everywhere() -> None:
    stage_data = load(KAGGLE / "stage_data.py", "stage_data_names")
    check = load(REPO / "scripts" / "check_shards.py", "check_shards_names")
    assert stage_data.SHARD_NAME.pattern == check.SHARD_NAME.pattern
    names = {"drive_and_act-00012.tar": "drive_and_act", "drive_and_act-r001-00000.tar": "drive_and_act",
             "mrl_eye-r012-00003.tar": "mrl_eye", "index.json": None}  # fmt: skip
    assert {name: stage_data.dataset_of(name) for name in names} == names


def test_clearance_needs_the_registry_field(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    from bulwark.data import registry

    module = load(KAGGLE / "stage_data.py", "stage_data_registry")
    monkeypatch.setattr(registry, "DATASETS", (types.SimpleNamespace(name="dmd"),))
    with pytest.raises(SystemExit, match="kaggle_storage_ok"):
        module.clearance("kaggle")
    spec = types.SimpleNamespace(name="dmd", kaggle_storage_ok=True, azure_storage_ok=None)
    monkeypatch.setattr(registry, "DATASETS", (spec,))
    assert module.clearance("kaggle") == {"dmd": True}
    assert module.clearance("azure") == {"dmd": None}
    with pytest.raises(SystemExit, match="Unknown storage"):
        module.clearance("s3")
    assert set(module.STORAGE_NAMES) == set(registry.STORAGE_FIELDS)


def test_each_storage_provider_is_cleared_on_its_own(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """Against the real registry: DMD is cleared for a private Kaggle copy, not yet for
    Azure Blob, so the same shards and manifest stage for one and are refused for the other."""
    module = load(KAGGLE / "stage_data.py", "stage_data_providers")
    root = tmp_path / "repo"
    write(root / "data" / "shards" / "dmd-00000.tar", b"dmd")
    pq.write_table(pa.table({"clip_id": ["d/1"], "dataset": ["dmd"]}), write(root / "data" / "manifest.parquet"))
    args = ["--root", str(root), "--stage", str(tmp_path / "staged"), "--shards", "dmd-*.tar", "--manifest"]
    assert module.main(args) == 0  # Kaggle is the default
    assert module.main([*args, "--storage", "kaggle"]) == 0
    assert module.main([*args, "--storage", "azure"]) == 1
    assert "not cleared for storage on Azure Blob (registry azure_storage_ok)" in capsys.readouterr().err
    assert module.main([*args, "--storage", "azure", "--allow", "dmd"]) == 0


def test_real_registry_never_stages_a_shrunken_manifest(tmp_path: Path) -> None:
    """Against bulwark/data/registry.py as it is: either the field is missing (a clear
    error), or the manifest is staged whole, or staging is refused. Never fewer rows."""
    from bulwark.data.registry import DATASETS

    module = load(KAGGLE / "stage_data.py", "stage_data_real")
    root = tmp_path / "repo"
    names = [spec.name for spec in DATASETS][:3]
    pq.write_table(pa.table({"clip_id": [f"{n}/1" for n in names], "dataset": names}),
                   write(root / "data" / "manifest.parquet"))  # fmt: skip
    out = tmp_path / "meta"
    try:
        code = module.main(["--root", str(root), "--stage", str(out), "--manifest"])
    except SystemExit as exc:
        assert "kaggle_storage_ok" in str(exc)
        return
    if code == 0:
        assert pq.read_table(out / "manifest.parquet").num_rows == len(names)
    else:
        assert not (out / "manifest.parquet").exists()


def test_push_data_needs_an_explicit_slug(tmp_path: Path, kaggle_stub: Path) -> None:
    environment = {**os.environ, "KAGGLE_USER": "someone", "KAGGLE": str(kaggle_stub), "SHARD_GLOB": ""}
    environment.pop("SLUG", None)
    result = subprocess.run(["bash", str(KAGGLE / "push_data.sh")], env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "SLUG" in result.stderr
    assert not (tmp_path / "kaggle.log").exists()


def test_stage_takes_only_finished_shards_and_copies_a_whole_manifest(stage, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    module, root = stage
    write(root / "data" / "shards" / "zenodo_driver-00001.tar.partial", b"killed mid-write")
    write(root / "data" / "shards" / "zenodo_driver.shards.json", b"{}")
    out = tmp_path / "staged"
    assert module.main(["--root", str(root), "--stage", str(out), "--shards", "zenodo_driver*"]) == 0
    assert [p.name for p in (out / "shards").iterdir()] == ["zenodo_driver-00000.shard"]
    assert "zenodo_driver-00001.tar.partial" in capsys.readouterr().err
    assert module.main(["--root", str(root), "--stage", str(out), "--shards", "zenodo_driver-*.partial"]) == 1

    # Nothing left out: the manifest is copied byte for byte, so its sha256 matches the local one.
    args = ["--root", str(root), "--stage", str(out), "--manifest", "--allow", "dmd", "--allow", "mrl_eye"]
    assert module.main(args) == 0
    assert (out / "manifest.parquet").read_bytes() == (root / "data" / "manifest.parquet").read_bytes()


def test_shards_are_found_whether_or_not_kaggle_kept_the_shards_folder(run, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A dataset whose upload was just the shards/ folder is unpacked into its root.
    monkeypatch.setattr(run, "INPUT", tmp_path)
    at_root = write(tmp_path / "datasets" / "me" / "bulwark-dmd" / "dmd-00000.shard", b"d")
    in_folder = write(tmp_path / "bulwark-mrl" / "shards" / "mrl_eye-00000.shard", b"m")
    assert run.find_shard_files() == sorted([at_root, in_folder])
