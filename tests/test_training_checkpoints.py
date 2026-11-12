from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import load_payload
from bulwark.model.heads import BulwarkHeads
from bulwark.model.losses import MultiTaskLoss
from bulwark.train.common import (
    Checkpointer,
    ResumeError,
    autocast_dtype,
    data_parallel,
    load_config,
    load_feature_dir,
    load_teacher_logits,
    loss_param_group,
    param_groups,
)
from bulwark.train.provenance import ProvenanceError, require_clean


def provenance(**config: object) -> dict:
    base = {"lr": 1e-3, "commercial_only": False, "out": "a", "device": "cpu", "time_limit_minutes": 0}
    return {"job": "lora", "config": {**base, **config}, "manifest_sha256": "m", "inputs": {"shards": {"a.tar": "1"}}}


def test_autocast_auto_picks_fp16_on_a_t4_and_bf16_on_ampere(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    assert autocast_dtype(cuda, "auto") == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    assert autocast_dtype(cuda, "auto") == torch.bfloat16
    assert autocast_dtype(torch.device("cpu"), "auto") is None


def test_param_groups_do_not_decay_norms_biases_tokens_or_loss_weights() -> None:
    taxonomy = default_taxonomy()
    heads = BulwarkHeads(8, taxonomy, width=16)
    decay, keep = param_groups(heads.named_parameters(), 0.05)
    names = {id(p): n for n, p in heads.named_parameters()}
    kept = {names[id(p)] for p in keep["params"]}
    assert {"pos", "cls", "norm.weight", "norm.bias", "proj.0.weight", "out.bias"} <= kept
    assert all(p.ndim >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.05 and keep["weight_decay"] == 0.0
    (loss_group,) = loss_param_group(MultiTaskLoss(taxonomy), 1e-2)
    assert loss_group["weight_decay"] == 0.0 and loss_group["lr"] == 1e-2 and loss_group["scheduled"] is False
    assert loss_param_group(MultiTaskLoss(taxonomy, learn_weights=False), 1e-2) == []


def test_load_payload_refuses_pickled_code(tmp_path: Path) -> None:
    torch.save({"model": {"w": torch.ones(1)}, "step": 3, "best": None}, tmp_path / "ok.pt")
    assert load_payload(tmp_path / "ok.pt")["step"] == 3
    torch.save({"model": {}, "hook": Path("x")}, tmp_path / "bad.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_payload(tmp_path / "bad.pt")


def test_resume_refuses_a_different_run_and_names_what_changed(tmp_path: Path) -> None:
    Checkpointer(tmp_path, 15, provenance()).save("last.pt", {"model": {}, "step": 1})
    with pytest.raises(ResumeError, match=r"config\.lr.*allow_mismatch"):
        Checkpointer(tmp_path, 15, provenance(lr=2e-3)).resume()
    # Where it runs and how often it saves may change between sessions.
    assert Checkpointer(tmp_path, 15, provenance(out="b", device="cuda", time_limit_minutes=660)).resume()["step"] == 1
    assert Checkpointer(tmp_path, 15, provenance(lr=2e-3)).resume(allow_mismatch=True)["step"] == 1
    assert json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-2])["event"] == "allow_mismatch"
    with pytest.raises(ResumeError, match="cannot be overridden"):
        Checkpointer(tmp_path, 15, provenance(commercial_only=True)).resume(allow_mismatch=True)


def test_resume_refuses_a_last_pt_without_a_fingerprint(tmp_path: Path) -> None:
    torch.save({"model": {}, "step": 5}, tmp_path / "last.pt")  # written before provenance existed
    with pytest.raises(ResumeError, match="no run fingerprint"):
        Checkpointer(tmp_path, 15, provenance()).resume()


def test_require_clean_needs_recorded_commercial_only() -> None:
    require_clean(None, "init")
    require_clean({"path": "x", "has_provenance": True, "commercial_only": True, "datasets": {"a": {"commercial_ok": True}}}, "init")
    with pytest.raises(ProvenanceError, match="no provenance"):
        require_clean({"path": "x", "has_provenance": False}, "init")
    with pytest.raises(ProvenanceError, match="commercial_only=false"):
        require_clean({"path": "x", "has_provenance": True, "commercial_only": False}, "teacher")
    with pytest.raises(ProvenanceError, match="not commercially usable"):
        require_clean({"path": "x", "has_provenance": True, "commercial_only": True, "datasets": {"a": {"commercial_ok": False}}}, "init")


def _npz(path: Path, ids: list[str], **logits: np.ndarray) -> None:
    np.savez(path, clip_ids=np.array(ids), tokens=np.zeros((len(ids), 2, 3), dtype=np.float16), **{f"logit__{k}": v for k, v in logits.items()})


def test_load_feature_dir_rejects_mixed_logits_short_logits_and_duplicates(tmp_path: Path) -> None:
    (tmp_path / "mixed").mkdir()
    _npz(tmp_path / "mixed" / "features-a.npz", ["a"])
    _npz(tmp_path / "mixed" / "features-b.npz", ["b"], yawn=np.zeros(1))
    with pytest.raises(ValueError, match="mixes runs"):
        load_feature_dir(tmp_path / "mixed")
    (tmp_path / "short").mkdir()
    _npz(tmp_path / "short" / "features-a.npz", ["a", "b"], yawn=np.zeros(1))
    with pytest.raises(ValueError, match="one row per clip"):
        load_feature_dir(tmp_path / "short")
    (tmp_path / "dup").mkdir()
    _npz(tmp_path / "dup" / "features-a.npz", ["a"])
    _npz(tmp_path / "dup" / "features-b.npz", ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        load_feature_dir(tmp_path / "dup")
    (tmp_path / "dup" / "features-c.npz.tmp").write_bytes(b"partial")  # never read
    (tmp_path / "dup" / "features-b.npz").unlink()
    assert load_feature_dir(tmp_path / "dup")[0] == ["a"]


def test_load_teacher_logits_matches_the_full_load_without_reading_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Distillation reads only the logits: the v0 tokens alone are 6.5 GB, and holding them
    # got the first Kaggle distillation run killed for host memory.
    _npz(tmp_path / "features-a.npz", ["a", "b"], yawn=np.array([0.1, 0.2]))
    _npz(tmp_path / "features-b.npz", ["c"], yawn=np.array([0.3]))
    ids, _, logits = load_feature_dir(tmp_path)
    read: list[str] = []
    real_getitem = np.lib.npyio.NpzFile.__getitem__
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", lambda self, key: read.append(key) or real_getitem(self, key))
    teacher_ids, teacher_logits = load_teacher_logits(tmp_path)
    assert teacher_ids == ids == ["a", "b", "c"]
    assert teacher_logits.keys() == logits.keys() and np.array_equal(teacher_logits["yawn"], logits["yawn"])
    assert read and "tokens" not in read
    (tmp_path / "short").mkdir()
    _npz(tmp_path / "short" / "features-a.npz", ["a", "b"], yawn=np.zeros(1))
    with pytest.raises(ValueError, match="one row per clip"):
        load_teacher_logits(tmp_path / "short")


def test_config_types_and_single_device_passthrough() -> None:
    config = load_config(None, lr="1e-3", fresh=True, checkpoint_minutes=5)
    assert config.lr == 1e-3 and isinstance(config.checkpoint_minutes, float) and config.fresh is True
    with pytest.raises(ValueError, match="true or false"):
        load_config(None, commercial_only="yes")
    model = torch.nn.Linear(2, 2)
    assert data_parallel(model, torch.device("cpu")) is model
