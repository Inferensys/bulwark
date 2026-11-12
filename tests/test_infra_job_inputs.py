"""infra/job_inputs.py: the checkpoint and feature paths each remote job reads and writes."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


@pytest.fixture
def jobs(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(REPO)  # configs are read relative to the repository root
    spec = importlib.util.spec_from_file_location("job_inputs", REPO / "infra" / "job_inputs.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.mark.parametrize(
    ("job", "config", "read", "write"),
    [
        ("cache", "configs/train/cache.yaml", [], ["data/features/frozen"]),
        ("cache", "configs/train/cache-b1.yaml", ["checkpoints/b1-lora-v0/best.pt"], ["data/features/b1-lora-v0"]),
        ("heads", "configs/train/heads.yaml", ["data/features/frozen"], ["checkpoints/heads-v0"]),
        ("lora", "configs/train/lora-kaggle.yaml", ["checkpoints/heads-v0/best.pt"], ["checkpoints/b1-lora-v0"]),
        ("distill", "configs/train/distill.yaml", ["data/features/b1-lora-v0"], ["checkpoints/e1-v0"]),
    ],
)
def test_each_job_reads_and_writes_what_its_config_names(jobs, job, config, read, write) -> None:  # type: ignore[no-untyped-def]
    assert jobs.job_inputs(job, config, []) == {"read": read, "write": write}


def test_overrides_win_and_a_read_that_is_also_written_counts_once(jobs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    extra = ["--set", "init=checkpoints/b1-lora-clean/final.pt", "--set", "features=data/features/b1-lora-clean"]
    paths = jobs.job_inputs("cache", "configs/train/cache.yaml", extra)
    assert paths == {"read": ["checkpoints/b1-lora-clean/final.pt"], "write": ["data/features/b1-lora-clean"]}
    paths = jobs.job_inputs("heads", "configs/train/heads.yaml", ["--set", "out=./checkpoints/heads-x/"])
    assert paths["write"] == ["checkpoints/heads-x"]


@pytest.mark.parametrize(
    ("extra", "write"),
    [
        (["--set=out=checkpoints/other"], ["checkpoints/other"]),  # click accepts the = form too
        (["--set", "out=checkpoints/x", "--set=out=checkpoints/y"], ["checkpoints/y"]),  # the last one wins
        (["--set", "out="], ["checkpoints/b1-lora-v0"]),  # null: load_config keeps the config's value
        (["--set", "out=checkpoints/x", "--set", "out=null"], ["checkpoints/b1-lora-v0"]),
    ],
)
def test_overrides_are_read_as_bulwark_train_reads_them(jobs, extra, write) -> None:  # type: ignore[no-untyped-def]
    assert jobs.job_inputs("lora", "configs/train/lora.yaml", extra)["write"] == write


def test_overrides_resolve_like_the_cli(jobs) -> None:  # type: ignore[no-untyped-def]
    from typer.main import get_command

    from bulwark.cli import _train_config, app

    lora = get_command(app).commands["train"].commands["lora"]  # type: ignore[attr-defined]
    for extra in (["--set=out=checkpoints/other"], ["--set", "out="], ["--set", "init=", "--set=init=checkpoints/a/best.pt"]):
        params = lora.make_context("lora", ["--config", "configs/train/lora.yaml", *extra]).params
        config = _train_config(params["config"], list(params["set_"]))
        assert jobs.job_inputs("lora", "configs/train/lora.yaml", extra) == {"read": [config.init], "write": [config.out]}


def test_evaluate_reads_its_checkpoint_and_features(jobs) -> None:  # type: ignore[no-untyped-def]
    assert jobs.job_inputs("evaluate", "checkpoints/b1-lora-v0/best.pt", ["--model", "b1", "--precision", "fp16"]) == {
        "read": ["checkpoints/b1-lora-v0/best.pt"], "write": ["runs/eval"]
    }  # fmt: skip
    extra = ["--features=data/features/frozen", "--out", "runs/heads-eval"]
    assert jobs.job_inputs("evaluate", "checkpoints/heads-v0/best.pt", extra) == {
        "read": ["checkpoints/heads-v0/best.pt", "data/features/frozen"], "write": ["runs/heads-eval"]
    }  # fmt: skip
    # heads and logits read the CLI's --features default, the frozen-encoder features that
    # configs/train/cache.yaml writes; b1 and e1 never read features.
    assert jobs.job_inputs("evaluate", "checkpoints/heads-v0/best.pt", ["--model=heads"])["read"] == [
        "checkpoints/heads-v0/best.pt", "data/features/frozen"
    ]  # fmt: skip
    extra = ["--model", "e1", "--features", "data/features/frozen", "--shards", "./data/shards/"]
    assert jobs.job_inputs("evaluate", "checkpoints/e1-v0/best.pt", extra)["read"] == ["checkpoints/e1-v0/best.pt"]
    assert jobs.job_inputs("evaluate", "checkpoints/e1-v0/e1.onnx", [])["read"] == ["checkpoints/e1-v0/e1.onnx"]


@pytest.mark.parametrize(
    ("job", "target", "extra", "message"),
    [
        ("train", "configs/train/heads.yaml", [], "unknown job"),
        ("heads", "configs/train/missing.yaml", [], "config not found"),
        ("heads", "../heads.yaml", [], "inside the repo"),
        ("heads", "configs/train/heads.yaml", ["--set", "out=/tmp/heads"], "inside the repo"),
        ("heads", "configs/train/heads.yaml", ["--set", "out=''"], "would write nowhere"),
        ("lora", "configs/train/lora.yaml", ["--set=init=runs/old/best.pt"], "only from checkpoints/ or data/features/"),
        ("evaluate", "runs/export/e1.onnx", [], "under checkpoints/"),
        ("evaluate", "checkpoints", [], "under checkpoints/"),
        ("evaluate", "checkpoints/e1-v0/final.pt", ["--model", "e1", "--out", "../elsewhere"], "inside the repo"),
        ("evaluate", "checkpoints/e1-v0/final.pt", ["--model", "e1", "--reference", "runs/b1"], "--reference is not available"),
        ("evaluate", "checkpoints/e1-v0/final.pt", ["--model", "e1", "--reference=runs/b1"], "--reference is not available"),
        ("evaluate", "checkpoints/e1-v0/final.pt", ["--model", "e1", "--shards", "data/other"], "drop --shards"),
        ("evaluate", "checkpoints/e1-v0/final.pt", ["--model", "e1", "--manifest=m.parquet"], "drop --manifest"),
        ("evaluate", "checkpoints/heads-v0/best.pt", [], "--model auto"),
        ("evaluate", "checkpoints/heads-v0/best.pt", ["--model", "frozen"], "--model must be one of"),
        ("evaluate", "checkpoints/heads-v0/best.pt", ["--model", "heads", "--features", "runs/f"], "only from checkpoints/"),
    ],
)
def test_jobs_that_would_fail_remotely_are_refused(jobs, job, target, extra, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(jobs.JobError, match=message):
        jobs.job_inputs(job, target, extra)


def test_entries_under(jobs) -> None:  # type: ignore[no-untyped-def]
    assert jobs.entries_under(["data/features/b1-lora-v0", "checkpoints/x"], "data/features") == {"b1-lora-v0"}
    assert jobs.entries_under(["data/features", "data/features/frozen"], "data/features") is None
    assert jobs.entries_under(["elsewhere/out"], "checkpoints") == set()


def test_defaults_match_the_training_config(jobs) -> None:  # type: ignore[no-untyped-def]
    from bulwark.train.common import TrainConfig

    defaults = {f.name: f.default for f in dataclasses.fields(TrainConfig)}
    assert {key: defaults[key] for key in jobs.DEFAULTS} == jobs.DEFAULTS
    assert set(jobs.TRAIN_JOBS) == {"cache", "heads", "lora", "distill"}


def test_evaluate_defaults_match_the_cli(jobs) -> None:  # type: ignore[no-untyped-def]
    from typer.main import get_command

    from bulwark.cli import app
    from bulwark.eval.predict import MODEL_KINDS

    evaluate = get_command(app).commands["evaluate"]  # type: ignore[attr-defined]
    defaults = {f"--{param.name}": param.default for param in evaluate.params}
    assert {flag: str(defaults[flag]) for flag in jobs.EVAL_DEFAULTS} == jobs.EVAL_DEFAULTS
    assert str(defaults["--out"]) == jobs.EVAL_OUT
    assert jobs.MODEL_KINDS == MODEL_KINDS and set(jobs.FEATURE_KINDS) < set(MODEL_KINDS)


def test_command_line_prints_one_line_per_path(jobs, capsys) -> None:  # type: ignore[no-untyped-def]
    assert jobs.main(["lora", "configs/train/lora.yaml"]) == 0
    assert capsys.readouterr().out.splitlines() == ["read checkpoints/heads-v0/best.pt", "write checkpoints/b1-lora-v0"]
    assert jobs.main(["lora", "configs/train/none.yaml"]) == 2
    assert "config not found" in capsys.readouterr().err


def test_kernel_and_cli_job_names_agree(jobs) -> None:  # type: ignore[no-untyped-def]
    from typer.main import get_command

    from bulwark.cli import app

    train = get_command(app).commands["train"]  # type: ignore[attr-defined]
    assert set(train.commands) == set(jobs.TRAIN_JOBS)
