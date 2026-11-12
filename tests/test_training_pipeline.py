"""The training chain end to end on CPU: cache -> heads -> lora -> cache B1 logits -> distill.

A tiny stand-in replaces the V-JEPA encoder (same interface: blocks with qkv/proj/fc1/fc2,
norms_block, a (B, T', H', W', D) token grid), so these run in seconds without weights.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.shards import ShardWriter
from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.backbone import BackboneConfig
from bulwark.model.bulwark import BulwarkEdge, load_payload
from bulwark.model.losses import MultiTaskLoss
from bulwark.train.common import ResumeError, load_config
from bulwark.train.provenance import ProvenanceError, features_identity, file_sha256, jsonable
from bulwark.train.stream import PHOTOMETRIC_POLICY

DIM = 16
# Taxonomy order. phone has video teacher targets of one class only, so it is masked.
SUPERVISED = [name for name in default_taxonomy().labels if name in {"eyes_closed", "activity", "face_visible"}]


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(DIM, DIM * 3)
        self.attn.proj = nn.Linear(DIM, DIM)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(DIM, DIM * 2)
        self.mlp.fc2 = nn.Linear(DIM * 2, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.attn.qkv(x).chunk(3, -1)
        x = x + self.attn.proj(torch.softmax(q @ k.transpose(-1, -2) / 4, -1) @ v)
        return x + self.mlp.fc2(torch.relu(self.mlp.fc1(x)))


class _TinyViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(3, DIM)
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.norms_block = nn.ModuleList([nn.LayerNorm(DIM) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norms_block[-1](x)


class StubEncoder(nn.Module):
    """VJepaEncoder's interface on a 4x4 patch grid."""

    def __init__(self, config: BackboneConfig | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        self.encoder = _TinyViT()
        self.embed_dim = DIM
        self.weights_path = None

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time = frames.shape[:2]
        steps = 1 if time == 1 else time // 2
        x = frames.float().permute(0, 4, 1, 2, 3) / 255.0
        x = F.adaptive_avg_pool3d(x, (steps, 4, 4)).permute(0, 2, 3, 4, 1).reshape(batch, steps * 16, 3)
        return self.encoder(x).reshape(batch, steps, 4, 4, DIM)


def _clips(dataset: str, count: int, frames: int, clean: bool, splits: tuple[int, int]) -> list[ClipRecord]:
    records = []
    for i in range(count):
        split = "train" if i < splits[0] else "val" if i < splits[1] else "test"
        labels: dict[str, Any] = {"eyes_closed": float(i % 2)} if frames == 1 else {"activity": ("texting", "eating")[i % 2]}
        teacher = {"face_visible": float(i % 3 > 0)}
        if frames > 1:
            teacher["phone"] = 1.0  # one class only, so it is masked rather than learned as a constant
        records.append(ClipRecord(
            clip_id=f"{dataset}/{i}", dataset=dataset, subject_id=f"{dataset}:{split}-{i % 6}", media_path=f"{i}",
            media_type="image" if frames == 1 else "video", licence="test", commercial_ok=clean, split=split,
            num_frames=frames, labels_gt=labels, labels_teacher=teacher,
        ))
    return records


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("bulwark.model.bulwark.VJepaEncoder", StubEncoder)
    monkeypatch.setattr("bulwark.train.distill.BulwarkEdge", functools.partial(BulwarkEdge, pretrained=False))
    images = _clips("img", 60, 1, clean=True, splits=(36, 48))   # commercially usable
    videos = _clips("vid", 24, 4, clean=False, splits=(16, 20))  # research only
    write_manifest(images + videos, tmp_path / "manifest.parquet")
    for name, group in (("img", images), ("vid", videos)):
        with ShardWriter(tmp_path / "shards", prefix=name, clips_per_shard=100) as writer:
            for i, r in enumerate(group):
                level = 200 if r.labels_gt.get("eyes_closed") or r.labels_gt.get("activity") == "texting" else 40
                writer.add(r.clip_id, np.full((r.num_frames, 32, 32, 3), level + i % 7, dtype=np.uint8))
    return tmp_path


def cfg(root: Path, **overrides: Any):  # type: ignore[no-untyped-def]
    base = {
        "manifest": str(root / "manifest.parquet"), "shards": str(root / "shards"), "device": "cpu",
        "precision": "fp32", "num_workers": 0, "batch_size": 4, "grad_accum": 2, "lr": 1e-3,
        "warmup_steps": 1, "log_every": 1, "lora_blocks": 1, "lora_rank": 2, "edge_img_size": 64,
        "edge_backbone": "resnet10t.c3_in1k",  # fast on CPU; E1 uses EfficientViT-B1
    }
    return load_config(None, **{**base, **overrides})


def run_cache(root: Path, **overrides: Any) -> Path:
    from bulwark.train.cache_features import run

    return run(cfg(root, features=str(root / "features" / "frozen"), batch_size=8, **overrides))


def run_heads(root: Path, out: str = "heads", **overrides: Any) -> Path:
    from bulwark.train.train_heads import run

    return run(cfg(root, features=str(root / "features" / "frozen"), out=str(root / out), epochs=3, batch_size=8, **overrides))


def run_lora(root: Path, out: str = "lora", **overrides: Any) -> Path:
    from bulwark.train.finetune_lora import run

    return run(cfg(root, out=str(root / out), **{"max_steps": 4, **overrides}))


def test_full_chain_records_provenance_and_refuses_reruns(data: Path) -> None:
    from bulwark.train.cache_features import run as cache
    from bulwark.train.distill import run as distill

    frozen = run_cache(data)
    meta = json.loads((frozen / "meta.json").read_text())
    assert meta["commercial_only"] is True and meta["release_only"] is True and meta["run"]["with_logits"] is False
    assert "logit_layout_sha256" not in meta["run"]  # frozen tokens do not depend on the taxonomy
    assert sorted(meta["shards"]) == ["img-00000.tar", "vid-00000.tar"]
    stamp = (frozen / "features-img-00000.npz").stat().st_mtime_ns
    run_cache(data)  # resumes: both shards match their fingerprints, nothing is redone
    assert (frozen / "features-img-00000.npz").stat().st_mtime_ns == stamp
    with pytest.raises(ProvenanceError, match="precision"):
        run_cache(data, precision="bf16")

    heads = run_heads(data)
    best = load_payload(heads / "best.pt")
    assert best["select"]["criterion"] == "gt_mean_headline" and "eyes_closed" in best["select"]["metrics"]
    assert best["select"]["support"]["eyes_closed"] == {"clips": 12, "subjects": 6}  # img val: clips 36-47
    assert "loss" in best and best["provenance"]["commercial_only"] is False
    assert set(best["provenance"]["datasets"]) == {"img", "vid"}
    assert best["supervised_labels"] == SUPERVISED and load_payload(heads / "last.pt")["supervised_labels"] == SUPERVISED
    assert "phone:video" in best["provenance"]["labels"]["masked"] and best["masked_labels"] == {"phone": ["video"]}
    assert best["provenance"]["release_only"] is False and best["provenance"]["taxonomy_version"] == default_taxonomy().version
    with pytest.raises(ResumeError, match="already complete"):
        run_heads(data)

    lora = run_lora(data, init=str(heads / "best.pt"), eval_every=2)
    final = load_payload(lora / "final.pt")
    assert final["step"] == 4 and final["lora_blocks"] == 1 and final["supervised_labels"] == SUPERVISED
    assert final["masked_labels"] == {"phone": ["video"]}
    assert set(final["provenance"]["datasets"]["img"]["splits"]) == {"train", "val"}  # val picks best.pt
    b1_best = load_payload(lora / "best.pt")
    assert b1_best["select"]["criterion"] == "gt_mean_headline" and b1_best["step"] in (2, 4)
    assert (b1_best["lora_blocks"], b1_best["lora_rank"]) == (1, 2) and set(b1_best["model"]) == set(final["model"])
    assert load_payload(lora / "last.pt")["best"] == b1_best["select"]["score"]
    assert final["provenance"]["inputs"]["init"]["sha256"]
    assert final["provenance"]["code"]["source"] in {"git", "PROVENANCE.json", "unknown"}
    assert not any("norms_block.0" in k for k in final["model"])  # only the final norm is trained
    log = [json.loads(line) for line in (lora / "log.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "start" and "provenance" in log[0]
    assert [e["step"] for e in log if e.get("event") == "val"] == [2, 4]  # eval_every, then the last step
    with pytest.raises(ResumeError, match="already complete"):
        run_lora(data, init=str(heads / "best.pt"))
    with pytest.raises(ResumeError, match=r"config\.lr"):
        run_lora(data, init=str(heads / "best.pt"), lr=5e-4)

    teacher = cache(cfg(data, init=str(lora / "final.pt"), features=str(data / "features" / "b1"), batch_size=8))
    teacher_meta = json.loads((teacher / "meta.json").read_text())
    assert "logit_layout_sha256" in teacher_meta["run"] and teacher_meta["supervised_labels"] == SUPERVISED
    assert teacher_meta["masked_labels"] == {"phone": ["video"]}
    assert teacher_meta["release_only"] is False  # the B1 behind it was not trained with release_only
    with pytest.raises(ProvenanceError, match="different run"):
        cache(cfg(data, init=str(lora / "last.pt"), features=str(teacher), batch_size=8))
    e1 = distill(cfg(data, teacher=str(teacher), out=str(data / "e1"), max_steps=2, lr=1e-3))
    payload = load_payload(e1 / "final.pt")
    assert payload["img_size"] == 64 and payload["supervised_labels"] == SUPERVISED
    assert payload["masked_labels"] == {"phone": ["video"]}
    assert set(payload["provenance"]["datasets"]["img"]["splits"]) == {"train", "val"}
    assert payload["provenance"]["distilled"] == SUPERVISED  # never the teacher's untrained outputs
    for name in ("last.pt", "best.pt"):  # inference reads the architecture from any of them
        other = load_payload(e1 / name)
        assert (other["backbone"], other["img_size"]) == ("resnet10t.c3_in1k", 64), name
    recorded = payload["provenance"]["inputs"]["teacher"]
    assert recorded["init_sha256"] == file_sha256(lora / "final.pt")
    # The E1 weights name the datasets its B1 teacher saw, and the encoder behind it.
    assert recorded["datasets"] == final["provenance"]["datasets"] and set(recorded["datasets"]) == {"img", "vid"}
    assert recorded["encoder"]["vjepa_commit"]
    assert payload["provenance"]["augment"] == jsonable(PHOTOMETRIC_POLICY)  # no crop against cached soft targets


def test_lora_resume_after_time_limit_matches_an_uninterrupted_run(data: Path) -> None:
    straight = load_payload(run_lora(data, out="straight", max_steps=8, eval_every=3) / "final.pt")
    sessions = 0
    while not (data / "sessions" / "final.pt").exists():
        run_lora(data, out="sessions", max_steps=8, eval_every=3, time_limit_minutes=1e-9)  # stops after each step
        sessions += 1
        assert sessions <= 10
    resumed = load_payload(data / "sessions" / "final.pt")
    assert sessions > 4  # crossed the epoch boundary (6 steps per epoch) mid-way
    for key, value in straight["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    logs = {run: [json.loads(line) for line in (data / run / "log.jsonl").read_text().splitlines()] for run in ("straight", "sessions")}
    assert [e.get("event") for e in logs["sessions"]].count("resume") == sessions - 1
    # Validation passes (step 3, the epoch end at 6, the last step) neither repeat nor perturb training.
    val = {run: [(e["step"], e["val_select"]) for e in log if e.get("event") == "val"] for run, log in logs.items()}
    assert [step for step, _ in val["sessions"]] == [3, 6, 8] and val["sessions"] == val["straight"]
    assert load_payload(data / "sessions" / "best.pt")["step"] == load_payload(data / "straight" / "best.pt")["step"]


def test_a_run_stopped_after_its_first_best_pt_resumes(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bulwark.train import finetune_lora

    step = finetune_lora.optimizer_step
    calls = 0

    def evicted_at_step_two(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("session evicted")
        return step(*args, **kwargs)

    monkeypatch.setattr(finetune_lora, "optimizer_step", evicted_at_step_two)
    with pytest.raises(RuntimeError, match="evicted"):
        run_lora(data, out="evicted", eval_every=1, checkpoint_minutes=1e6)  # no timed save in this run
    # best.pt from the step-1 val pass came with a last.pt, so the rerun is not refused.
    assert load_payload(data / "evicted" / "best.pt")["step"] == 1
    assert load_payload(data / "evicted" / "last.pt")["step"] == 1
    monkeypatch.setattr(finetune_lora, "optimizer_step", step)
    lora = run_lora(data, out="evicted", eval_every=1, checkpoint_minutes=1e6)
    assert load_payload(lora / "final.pt")["step"] == 4
    log = [json.loads(line) for line in (lora / "log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log if e.get("event") == "resume"] == [1]


def test_fresh_moves_the_old_run_aside_and_leftovers_are_refused(data: Path) -> None:
    run_cache(data)
    heads = run_heads(data)
    (heads / "last.pt").unlink()
    with pytest.raises(ResumeError, match="no last.pt"):
        run_heads(data)
    run_heads(data, fresh=True)
    stale = [p for p in heads.iterdir() if p.name.startswith("stale-")]
    assert len(stale) == 1 and (stale[0] / "final.pt").exists() and (heads / "final.pt").exists()


def test_commercial_only_refuses_research_init_teacher_and_resume(data: Path) -> None:
    from bulwark.train.cache_features import run as cache
    from bulwark.train.distill import run as distill

    run_cache(data)
    research = run_heads(data)
    clean = run_heads(data, out="heads-clean", commercial_only=True)
    clean_best = load_payload(clean / "best.pt")
    assert clean_best["provenance"]["commercial_only"] is True
    assert set(clean_best["provenance"]["datasets"]) == {"img"}

    with pytest.raises(ProvenanceError, match="commercial_only=false"):
        run_lora(data, out="lora-clean", init=str(research / "best.pt"), commercial_only=True)
    lora_clean = run_lora(data, out="lora-clean", init=str(clean / "best.pt"), commercial_only=True)
    assert load_payload(lora_clean / "final.pt")["provenance"]["commercial_only"] is True

    lora = run_lora(data, init=str(research / "best.pt"), time_limit_minutes=1e-9)  # research run, unfinished
    with pytest.raises(ResumeError, match="cannot be overridden"):
        run_lora(data, init=str(clean / "best.pt"), commercial_only=True, allow_mismatch=True)

    (data / "legacy").mkdir()
    torch.save({"model": {}, "step": 1}, data / "legacy" / "last.pt")  # no provenance: refused even with allow_mismatch
    with pytest.raises(ProvenanceError, match="records no provenance"):
        run_lora(data, out="legacy", init=str(clean / "best.pt"), commercial_only=True, allow_mismatch=True)

    research_teacher = cache(cfg(data, init=str(lora / "last.pt"), features=str(data / "features" / "b1"), batch_size=8))
    with pytest.raises(ProvenanceError, match="commercial_only=false"):
        distill(cfg(data, teacher=str(research_teacher), out=str(data / "e1-clean"), commercial_only=True, max_steps=1))
    clean_teacher = cache(cfg(data, init=str(lora_clean / "final.pt"), features=str(data / "features" / "b1-clean"),
                              batch_size=8, commercial_only=True))
    assert json.loads((clean_teacher / "meta.json").read_text())["commercial_only"] is True
    b1_datasets = load_payload(lora_clean / "final.pt")["provenance"]["datasets"]
    assert features_identity(clean_teacher)["datasets"] == b1_datasets and set(b1_datasets) == {"img"}
    e1 = distill(cfg(data, teacher=str(clean_teacher), out=str(data / "e1-clean"), commercial_only=True, max_steps=1))
    e1_provenance = load_payload(e1 / "final.pt")["provenance"]
    assert set(e1_provenance["datasets"]) == {"img"} and e1_provenance["inputs"]["teacher"]["datasets"] == b1_datasets


def test_cache_leaves_out_and_lists_clips_with_non_finite_outputs(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bulwark.train import cache_features

    forward = cache_features._Encode.forward

    def overflow(self, frames):  # type: ignore[no-untyped-def]
        summary, logits = forward(self, frames)
        summary[0, 0, 0] = 1e6  # past fp16's range: inf once stored as fp16
        return summary, logits

    monkeypatch.setattr(cache_features._Encode, "forward", overflow)
    frozen = run_cache(data)
    meta = json.loads((frozen / "meta.json").read_text())
    dropped = [c for ids in meta["nonfinite"].values() for c in ids]
    assert dropped and sorted(meta["nonfinite"]) == ["img-00000.tar", "vid-00000.tar"]
    with np.load(frozen / "features-img-00000.npz") as npz:
        kept = set(npz["clip_ids"].tolist())
        assert np.isfinite(npz["tokens"]).all()
    assert len(kept) == 60 - len(meta["nonfinite"]["img-00000.tar"]) and not kept & set(dropped)


def test_cache_overwrite_clears_only_another_runs_dir(data: Path) -> None:
    frozen = run_cache(data)
    stamp = (frozen / "features-img-00000.npz").stat().st_mtime_ns
    # A later session of the same cache job, pushed with the same --set overwrite=true.
    run_cache(data, overwrite=True)
    assert (frozen / "features-img-00000.npz").stat().st_mtime_ns == stamp
    (frozen / "meta.json").unlink()
    with pytest.raises(ProvenanceError, match="without meta.json"):
        run_cache(data)
    run_cache(data, overwrite=True)
    assert (frozen / "features-img-00000.npz").stat().st_mtime_ns != stamp
    assert sorted(json.loads((frozen / "meta.json").read_text())["shards"]) == ["img-00000.tar", "vid-00000.tar"]


class _FlakyLoss(MultiTaskLoss):
    """Non-finite on the listed calls (1-based), like an fp16 forward that overflowed."""

    bad: set[int] = set()
    calls = 0

    def forward(self, logits, targets):  # type: ignore[no-untyped-def]
        loss, parts = super().forward(logits, targets)
        type(self).calls += 1
        return (loss * float("nan"), parts) if type(self).calls in self.bad else (loss, parts)


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch) -> type[_FlakyLoss]:
    monkeypatch.setattr("bulwark.train.finetune_lora.MultiTaskLoss", _FlakyLoss)
    # fp16 loss scaling as on a T4, on CPU.
    monkeypatch.setattr("bulwark.train.finetune_lora.grad_scaler", lambda device, dtype: torch.amp.GradScaler("cpu"))
    _FlakyLoss.calls = 0
    return _FlakyLoss


def test_windows_of_non_finite_losses_are_skipped_under_loss_scaling(data: Path, flaky: type[_FlakyLoss]) -> None:
    flaky.bad = {1, 2, 5, 6}  # every micro-batch of the first and third windows (grad_accum=2)
    lora = run_lora(data, out="flaky", max_steps=4)
    assert load_payload(lora / "final.pt")["step"] == 4
    log = [json.loads(line) for line in (lora / "log.jsonl").read_text().splitlines()]
    assert sum(e.get("event") == "nonfinite_loss" for e in log) == 4
    assert [e["skipped_steps"] for e in log if "skipped_steps" in e][-1] >= 2


def test_a_run_of_non_finite_losses_stops_without_overwriting_last_pt(data: Path, flaky: type[_FlakyLoss]) -> None:
    flaky.bad = set(range(3, 100))  # finite for the first step, then never again
    with pytest.raises(FloatingPointError, match="10 non-finite losses"):
        run_lora(data, out="diverged", max_steps=20, checkpoint_minutes=0)  # saves after every healthy step
    assert load_payload(data / "diverged" / "last.pt")["step"] == 1


def test_missing_or_filtered_out_data_fails_before_training(data: Path) -> None:
    (data / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No \\*.tar shards"):
        run_lora(data, shards=str(data / "empty"))
    with pytest.raises(ValueError, match="No train clips"):
        run_lora(data, datasets=["nope"])
    with pytest.raises(ValueError, match="fewer than grad_accum"):
        run_lora(data, batch_size=64)
