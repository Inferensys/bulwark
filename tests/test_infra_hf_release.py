"""scripts/hf_release.py: staging each model for its own Hugging Face model repository, the
cards rendered from release/hf/, and the Hub calls of upload, open and collection, against a
fake HfApi and a fake validate-yaml endpoint (no test touches the network)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from safetensors.numpy import save_file
from test_docs_project_files import _link_targets, _pyproject
from typer.testing import CliRunner

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper  # noqa: E402

from bulwark import __version__  # noqa: E402
from bulwark.cli import app  # noqa: E402
from bulwark.data.registry import BY_NAME, MAINTAINERS_RELEASE_DECISION  # noqa: E402
from bulwark.export.onnx import export_edge, write_benchmark  # noqa: E402
from bulwark.export.release import MODEL_RELEASE, MODEL_REPOS, RESEARCH_LICENCE, model_card_url  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COMMIT = "0123456789ab" * 3 + "0123"
# A dataset whose weights release the registry has not determined (DMD, which the fixtures
# use otherwise, is cleared by the maintainers' decision of 2026-10-30).
UNDECIDED = "yawdd"
MANIFEST = "ab" * 32
EDGE = {"backbone": "test_efficientnet", "img_size": 64}
B1, E1 = MODEL_REPOS["b1"], MODEL_REPOS["e1"]
SPACE = "Inferensys/bulwark-demo"  # the demo Space, the collection's first item once it exists
# collection.yaml with the demo Space's item, which the real file holds commented out until
# the Space exists.
SPACE_ITEM = (
    "  - type: space\n    id: Inferensys/bulwark-demo\n    at_launch: true\n    note: >-\n"
    "      Bulwark demo: runs Bulwark-B1 v0 or Bulwark-E1 v0 on AI-generated example clips of a driver.\n"
    "      Research preview, not certified. Weights: CC BY-NC 4.0, non-commercial research only.\n"
)


def with_space() -> dict[str, str]:
    real = (REPO / "release" / "hf" / "collection.yaml").read_text()
    assert real.count("\nitems:\n") == 1
    return {"release/hf/collection.yaml": real.replace("\nitems:\n", "\nitems:\n" + SPACE_ITEM)}
runner = CliRunner()


def load_hf_release():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("hf_release", REPO / "scripts" / "hf_release.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["hf_release"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


hf = load_hf_release()


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real Hub client: offline mode (read once at import, so set on the constant), an
    HfApi that cannot be built and no HTTP request (validate-yaml). Credentials in the
    environment would change the scan."""
    import huggingface_hub
    import huggingface_hub.constants

    class NoHub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("a test built a real HfApi")

    def no_request(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test sent an HTTP request")

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub, "HfApi", NoHub)
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    for name in hf.SECRET_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def released_b1(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixtures' B1 checkpoint ("b1" * 32) stands for the released one."""
    monkeypatch.setattr(hf, "B1_CHECKPOINT_SHA256", "b1" * 32)


def clear(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Mark datasets as cleared for a weights release in the registry."""
    for name in names:
        monkeypatch.setitem(BY_NAME, name, dataclasses.replace(BY_NAME[name], weights_release_ok=True))


def git(root: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout


CARD = """# Model card

## Model details

Weights: the [B1 repository](https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring). The
[data card](docs/data-card.md#data) describes the data; licences: [CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt),
code [LICENSE](LICENSE), [NOTICE](NOTICE), [examples](examples). See [details](#model-details).
"""
# Templates of a test checkout: every check the stage makes, without *at release* items.
TEMPLATES = {
    "b1.md": "# Bulwark-B1: test card\n\n{{ labels_table }}\n\n## Results\n\n{{ results_table }}\n\n"
             "## Licence and dataset terms\n\n{{> licence }}\n\n## Commercial models\n\n{{> product }}\n\n"
             "Read the [data card]({{ code_url }}/docs/data-card.md#data), the [report, which needs access](evaluation/b1/report.html) "
             "and the [LICENSE](LICENSE).\n{{#released e1}}\nSmaller: [E1]({{ e1_url }}).\n{{/released}}\n"
             "{{#unreleased e1}}\nSmaller: E1, in development.\n{{/unreleased}}\n",
    "e1.md": "# Bulwark-E1: test card\n\n{{ labels_table }}\n\n## Results and speed\n\n{{ results_table }}\n\n{{ speed_table }}\n\n"
             "## Licence and dataset terms\n\n{{> licence }}\n\n## Commercial models\n\n{{> product }}\n\n"
             "Read the [report, which needs access](evaluation/e1/report.html). Larger: [B1]({{ b1_url }}).\n",
    "shared/licence.md": "The weights are under [CC BY-NC 4.0]({{ legal_code_url }}).\n",
    "shared/product.md": "[Contact Inferensys]({{ contact_url }}) or write to {{ contact_email }}.\n",
}  # fmt: skip


# The launch of a test checkout: B1 alone, whatever the real release/hf/launch.yaml lists;
# and one that releases E1 too, as the real one does.
B1_ONLY = {"release/hf/launch.yaml": "released: [b1]\n"}
E1_RELEASED = {"release/hf/launch.yaml": "released: [b1, e1]\n"}
# The real collection.yaml before the collection exists: its address is not known yet.
COLLECTION_PENDING = {
    "release/hf/collection.yaml": re.sub(
        r"^url: .*$", "url: null", (REPO / "release" / "hf" / "collection.yaml").read_text(), flags=re.MULTILINE
    )
}


def checkout(tmp_path: Path, files: dict[str, str] | None = None, *, card: str = CARD) -> Path:
    """A committed checkout with the files a stage copies, test templates, the real
    names.yaml, collection.yaml and organisation card, and a launch of B1 alone
    (B1_ONLY); no *at release* items."""
    root = tmp_path / "repo"
    contents = {
        "MODEL_CARD.md": card,
        "docs/data-card.md": "# Data card\n\n## Data\n\nThe [model card](../MODEL_CARD.md#model-details).\n",
        "LICENSE": "Apache License\n",
        "LICENSES/CC-BY-NC-4.0.txt": "Attribution-NonCommercial 4.0 International\n",
        "NOTICE": "Bulwark\n",
        "examples/run.py": "print()\n",
        **{f"release/hf/{name}": text for name, text in TEMPLATES.items()},
        **{f"release/hf/{name}": (REPO / "release" / "hf" / name).read_text() for name in ("names.yaml", "collection.yaml", "org.yaml", "org-card.md")},
        **B1_ONLY,
        **(files or {}),
    }  # fmt: skip
    for name, text in contents.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def source(job: str, checkpoint: str, datasets: tuple[str, ...] = ("dmd",)) -> dict[str, Any]:
    return {
        "checkpoint": {"file": "best.pt", "sha256": checkpoint}, "step": 10, "job": job, "commercial_only": False,
        "datasets": {name: {"licence": "t", "commercial_ok": False, "splits": {"train": 4}} for name in datasets},
        "manifest_sha256": MANIFEST, "code_commit": COMMIT, "code_dirty": False,
        "upstream_datasets": {}, "unrecorded_inputs": [],
    }  # fmt: skip


def release_dir(
    path: Path, *, kind: str = "b1", checkpoint: str = "b1" * 32, cleared: bool = False, model_card: str | None = None,
    licence: str = RESEARCH_LICENCE, header: dict[str, str] | None = None, dataset: str = "dmd",
) -> Path:  # fmt: skip
    """A release directory in the format `bulwark release` writes, with a one-tensor model."""
    path.mkdir(parents=True)
    save_file({"w": np.zeros((2, 2), np.float32)}, str(path / "model.safetensors"), metadata={"format": "bulwark-release", **(header or {})})
    if kind.startswith("b1"):
        (path / "LICENSE-vjepa2").write_text("MIT License\n\nCopyright (c) Meta Platforms, Inc. and affiliates.\n")
    config = {
        "format": "bulwark-release", "format_version": 1, "kind": kind, "labels": ["yawn", "phone"],
        "masked_labels": {"phone": ["image"]}, "weights_licence": licence, "commercial_only": False,
        "source": source("lora" if kind.startswith("b1") else "distill", checkpoint, (dataset,)),
        "model_card": model_card or model_card_url(kind), "model_release": MODEL_RELEASE,
        "release_check": {"cleared": cleared, "datasets": {}},
    }  # fmt: skip
    (path / "config.json").write_text(json.dumps(config, indent=2))
    return path


def graph(
    path: Path, *, checkpoint: str = "e1" * 32, cleared: bool = False, frames: int = 16, model_card: str | None = None,
    props: dict[str, str] | None = None, doc_string: str = "", dataset: str = "dmd", teacher: str = "b1" * 32,
) -> Path:  # fmt: skip
    """A tiny ONNX graph with the metadata `bulwark export` writes."""
    node = helper.make_node("Identity", ["frames"], ["yawn"], doc_string=doc_string)
    tensor = helper.make_tensor_value_info
    model = helper.make_model(helper.make_graph([node], "e1", [tensor("frames", TensorProto.FLOAT, [1])], [tensor("yawn", TensorProto.FLOAT, [1])]))
    provenance = {
        "job": "distill", "commercial_only": False, "manifest_sha256": MANIFEST, "code": {"commit": COMMIT, "dirty": False},
        "datasets": {dataset: {"licence": "t", "commercial_ok": False, "splits": {"train": 4}}},
    }  # fmt: skip
    contract = {
        "input": {"name": "frames", "shape": ["batch", frames, 256, 256, 3]},
        "checkpoint": {"path": "best.pt", "sha256": checkpoint, "provenance": provenance},
        "training": {"teacher_init_sha256": teacher}, "release_check": {"cleared": cleared},
        "supervised_labels": ["yawn", "phone"], "masked_labels": {"phone": ["image"]},
        "label_status": {"ground_truth": ["yawn"], "teacher_only": ["phone"]},
    }  # fmt: skip
    helper.set_model_props(model, {
        "bulwark_contract": json.dumps(contract), "frames": str(frames), "license": RESEARCH_LICENCE,
        "model_card": model_card or model_card_url("e1"), "model_release": MODEL_RELEASE, "release_cleared": json.dumps(cleared),
        **(props or {}),
    })  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def metrics(checkpoint: str, *, limit: int = 0) -> dict[str, Any]:
    """metrics.json as `bulwark evaluate` writes it, cut to what the cards read: DMD scored
    per clip, Drive&Act per event (one moment from several cameras), and Drive&Act's
    inner-mirror camera held out."""
    def score(value: float, n: int, subjects: int, datasets: list[str]) -> dict[str, Any]:
        return {"metric": "auroc", "value": value, "n": n, "n_subjects": subjects, "datasets": datasets, "baseline": {"auroc": 0.5}}

    policy = {"subject_group_datasets": ["auc_v2"], "holdout_views": [["drive_and_act", "inner_mirror"]], "holdout_datasets": ["uta_rldd"]}
    return {
        "meta": {
            "limit_per_split": limit, "checkpoint": {"path": "best.pt", "sha256": checkpoint},
            "predicted_with": {"created_utc": "2026-10-30T11:31:42+00:00", "manifest": {"sha256": MANIFEST}, "split_policy": policy},
        },
        "training_support": {"yawn": {"status": "gt"}, "phone": {"status": "gt"}},
        "coverage": {
            "test": {d: {"manifest": n, "predicted": n} for d, n in (("dmd", 10), ("auc_v2", 5), ("drive_and_act", 12))},
            "holdout": {"drive_and_act": {"manifest": 2, "predicted": 2}},
        },
        "holdout_groups": {
            "drive_and_act": {"kind": "new camera (inner_mirror) and new subjects", "clips": 2, "subjects": 1},
            "uta_rldd": {"kind": "dataset never trained on", "clips": 0, "subjects": 0, "status": "not available (not ingested)"},
        },
        "splits": {
            "test": {
                "counts": {"dmd": {"clips": 10, "subjects": 3, "events": 10}, "auc_v2": {"clips": 5, "subjects": 1, "events": 5},
                           "drive_and_act": {"clips": 12, "subjects": 3, "events": 4}},
                "units": {"dmd": "clip", "auc_v2": "clip", "drive_and_act": "event"},
                "summary": {"yawn": score(0.9041, 724, 3, ["dmd"]), "phone": score(0.98856, 40, 4, ["dmd", "auc_v2"])},
                "datasets": {"dmd": {"yawn": {"n": 724, "n_subjects": 3, "auroc": 0.9041, "baseline": {"auroc": 0.5}}},
                             "drive_and_act": {"phone": {"n": 4, "n_subjects": 3, "auroc": 0.9963, "baseline": {"auroc": 0.5}}}},
                "per_clip": {"drive_and_act": {"phone": {"n": 12, "auroc": 0.9421}}},
            },
            "holdout": {
                "units": {"drive_and_act": "clip"},
                "datasets": {"drive_and_act": {"phone": {"n": 2, "n_subjects": 1, "auroc": 0.8077, "baseline": {"auroc": 0.5}}}},
            },
        },
    }  # fmt: skip


def evaluation(path: Path, checkpoint: str, *, limit: int = 0, report: str = "<html><body>scores</body></html>") -> Path:
    """`bulwark evaluate` output: the report, metrics, and files that are never staged."""
    (path / "predictions").mkdir(parents=True)
    (path / "predictions" / "test.npz").write_bytes(b"\0")
    (path / "calibration.json").write_text("{}")
    (path / "report.html").write_text(report)
    (path / "metrics.json").write_text(json.dumps(metrics(checkpoint, limit=limit)))
    return path


def provenance_file(path: Path, job: str, **extra: Any) -> Path:
    data = {"job": job, "manifest_sha256": MANIFEST, "code": {"commit": COMMIT, "dirty": False}, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def bundle(tmp_path: Path, model: str = "b1", *, cleared: bool = False, dataset: str = "dmd") -> dict[str, Any]:
    """`stage` keyword arguments of a complete release of `model`, trained on `dataset`: B1's
    release directory, or E1's 16-frame and 1-frame graphs with their B1 teacher's evaluation;
    an evaluation of the weights and the checkpoint's provenance."""
    inputs = tmp_path / "inputs"
    if model == "b1":
        return {
            "release": release_dir(inputs / "bulwark-b1-v0", cleared=cleared, dataset=dataset),
            "evaluation": evaluation(inputs / "eval-b1", "b1" * 32),
            "provenance": provenance_file(inputs / "b1" / "provenance.json", "lora"),
        }
    graphs = [graph(inputs / f"e1-{frames}.onnx", frames=frames, cleared=cleared, dataset=dataset) for frames in (16, 1)]
    for path, frames in zip(graphs, (16, 1), strict=True):
        write_benchmark(path, [{"ms_p50": 30.0, "ms_p95": 40.0, "threads": 1, "onnxruntime": "1.31.1", "cpu": "Test CPU",
                                "input": f"1x{frames}x256x256x3 uint8", "scope": "model only"}])  # fmt: skip
    return {
        "graphs": graphs, "evaluation": evaluation(inputs / "eval-e1", "e1" * 32),
        "teacher_evaluation": evaluation(inputs / "eval-b1", "b1" * 32),
        "provenance": provenance_file(inputs / "e1" / "provenance.json", "distill"),
    }  # fmt: skip


def staged(
    tmp_path: Path, root: Path, model: str = "b1", *, cleared: bool = False, public: bool = False,
    out: Path | None = None, dataset: str = "dmd",
) -> Path:  # fmt: skip
    out = out or tmp_path / f"stage-{model}"
    assert hf.stage(model, out, **bundle(tmp_path, model, cleared=cleared, dataset=dataset), public=public, root=root) == []
    return out


# ---- a fake Hub ------------------------------------------------------------------------------


def http_error(cls: type, status: int) -> Exception:
    """A huggingface_hub HTTP error built offline (works with the httpx and httpx2 releases)."""
    return cls(f"{status}", response=SimpleNamespace(headers={}, request=None, status_code=status))


class FakeHub:
    """The HfApi calls the tool makes, recorded as (method, args, kwargs), for one model
    repository at a time and one collection. Any other method (delete_repo,
    delete_collection, super_squash_history, ...) raises AttributeError; the one removal it
    has, delete_collection_item, is what `collection` uses for deferred items only."""

    endpoint = "https://huggingface.co"

    def __init__(self, *, exists: bool = True, private: bool = True, gated: Any = False) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.exists, self.private, self.gated = exists, private, gated
        self.commits: dict[str, dict[str, SimpleNamespace]] = {"init": {}}  # commit -> path -> sibling
        self.history: dict[str, list[str]] = {"init": ["init"]}  # commit -> itself and its ancestors
        self.main = "init"
        self.tags: dict[str, str] = {}
        self.pulls: dict[str, str] = {}
        self.takes_gating = True  # False: settings report gated False whatever is sent
        self.collection: SimpleNamespace | None = None
        # Other models' repositories, by id: private (True), public (False) or missing (None).
        # Any id not listed here is the one repository above.
        self.others: dict[str, bool | None] = {}

    def _log(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((method, args, kwargs))

    def methods(self) -> list[str]:
        return [method for method, _, _ in self.calls]

    def add_commit(self, files: dict[str, SimpleNamespace], parent: str | None = None) -> str:
        commit = f"{len(self.commits):040x}"
        self.commits[commit] = files
        self.history[commit] = [commit, *self.history[parent or self.main]]
        return commit

    def repo_info(self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None, files_metadata: bool = False) -> SimpleNamespace:
        from huggingface_hub.errors import RepositoryNotFoundError

        self._log("repo_info", repo_id, repo_type=repo_type, revision=revision, files_metadata=files_metadata)
        if repo_id in self.others:
            if self.others[repo_id] is None:
                raise http_error(RepositoryNotFoundError, 404)
            return SimpleNamespace(private=self.others[repo_id], gated=False if self.others[repo_id] else "auto", siblings=[])
        if not self.exists:
            raise http_error(RepositoryNotFoundError, 404)
        files = self.commits[revision or self.main]
        siblings = [SimpleNamespace(rfilename=".gitattributes", blob_id="0" * 40, lfs=None), *files.values()]
        if not files_metadata:  # as the Hub: blob ids and LFS hashes only with files_metadata
            siblings = [SimpleNamespace(rfilename=s.rfilename, blob_id=None, lfs=None) for s in siblings]
        return SimpleNamespace(private=self.private, gated=self.gated, siblings=siblings)

    def create_repo(self, repo_id: str, *, repo_type: str | None = None, private: bool | None = None, exist_ok: bool = False) -> None:
        self._log("create_repo", repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok)
        self.exists, self.private = True, bool(private)

    def list_repo_refs(self, repo_id: str, *, repo_type: str | None = None, include_pull_requests: bool = False) -> SimpleNamespace:
        self._log("list_repo_refs", repo_id, repo_type=repo_type, include_pull_requests=include_pull_requests)

        def refs(entries: dict[str, str]) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name, target_commit=commit) for name, commit in entries.items()]

        return SimpleNamespace(
            branches=refs({"main": self.main}), tags=refs(self.tags), converts=[],
            pull_requests=refs(self.pulls) if include_pull_requests else None,
        )  # fmt: skip

    def list_repo_commits(self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None) -> list[SimpleNamespace]:
        self._log("list_repo_commits", repo_id, repo_type=repo_type, revision=revision)
        return [SimpleNamespace(commit_id=commit) for commit in self.history[revision or self.main]]

    def upload_folder(self, **kwargs: Any) -> SimpleNamespace:
        self._log("upload_folder", **kwargs)
        folder = Path(kwargs["folder_path"])
        self.main = self.add_commit({rel: sibling(folder / rel, rel) for rel in hf._files(folder)})
        return SimpleNamespace(oid=self.main, commit_url=f"{self.endpoint}/{kwargs['repo_id']}/commit/{self.main}")

    def create_tag(self, repo_id: str, *, repo_type: str | None = None, tag: str, revision: str | None = None, tag_message: str | None = None) -> None:
        self._log("create_tag", repo_id, repo_type=repo_type, tag=tag, revision=revision, tag_message=tag_message)
        self.tags[tag] = revision or self.main

    def update_repo_settings(self, repo_id: str, *, repo_type: str | None = None, gated: Any = None, private: bool | None = None) -> None:
        self._log("update_repo_settings", repo_id, repo_type=repo_type, gated=gated, private=private)
        if gated is not None and self.takes_gating:
            self.gated = gated
        if private is not None:
            self.private = private

    # collections (huggingface_hub 2.3.0 signatures)

    def create_collection(self, title: str, *, namespace: str | None = None, description: str | None = None, private: bool = False, exists_ok: bool = False) -> SimpleNamespace:
        self._log("create_collection", title, namespace=namespace, description=description, private=private, exists_ok=exists_ok)
        if self.collection is None:
            self.collection = SimpleNamespace(slug=f"{namespace}/bulwark-0123", title=title, description=description, private=private, items=[])
        return self.collection

    def get_collection(self, slug: str) -> SimpleNamespace:
        self._log("get_collection", slug)
        assert self.collection is not None and slug == self.collection.slug
        return self.collection

    def update_collection_metadata(self, slug: str, *, title: str | None = None, description: str | None = None, private: bool | None = None) -> None:
        self._log("update_collection_metadata", slug, title=title, description=description, private=private)
        assert self.collection is not None
        for key, value in {"title": title, "description": description, "private": private}.items():
            if value is not None:
                setattr(self.collection, key, value)

    def add_collection_item(self, slug: str, item_id: str, item_type: str, *, note: str | None = None, exists_ok: bool = False) -> None:
        self._log("add_collection_item", slug, item_id, item_type, note=note, exists_ok=exists_ok)
        assert self.collection is not None
        items = self.collection.items
        items.append(SimpleNamespace(item_object_id=f"obj{len(items)}", item_id=item_id, item_type=item_type, note=note, position=len(items)))

    def delete_collection_item(self, slug: str, item_object_id: str, *, missing_ok: bool = False) -> None:
        self._log("delete_collection_item", slug, item_object_id, missing_ok=missing_ok)
        assert self.collection is not None
        self.collection.items = [i for i in self.collection.items if i.item_object_id != item_object_id]
        for index, entry in enumerate(self.collection.items):
            entry.position = index

    def update_collection_item(self, slug: str, item_object_id: str, *, note: str | None = None, position: int | None = None) -> None:
        self._log("update_collection_item", slug, item_object_id, note=note, position=position)
        assert self.collection is not None
        item = next(i for i in self.collection.items if i.item_object_id == item_object_id)
        if note is not None:
            item.note = note
        if position is not None:
            self.collection.items.remove(item)
            self.collection.items.insert(position, item)
            for index, entry in enumerate(self.collection.items):
                entry.position = index


class Hubs:
    """One Hub client over both model repositories, each a FakeHub of its own: a call goes
    to the FakeHub of the repository it names, and is recorded here too, in order. Only
    FakeHub's methods exist (no destructive ones)."""

    endpoint = "https://huggingface.co"

    def __init__(self, b1: FakeHub | None = None, e1: FakeHub | None = None) -> None:
        self.repos = {B1: b1 or FakeHub(exists=False), E1: e1 or FakeHub(exists=False)}
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not callable(getattr(FakeHub, name, None)) or name in ("methods", "add_commit"):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            repo = kwargs["repo_id"] if "repo_id" in kwargs else args[0]
            self.calls.append((name, args, kwargs))
            return getattr(self.repos[repo], name)(*args, **kwargs)

        return call

    def methods(self) -> list[str]:
        return [method for method, _, _ in self.calls]


def sibling(path: Path, rel: str) -> SimpleNamespace:
    """How the Hub reports a file: weights in LFS (sha256), other files by git blob id."""
    if rel.endswith(hf.WEIGHT_SUFFIXES):
        return SimpleNamespace(rfilename=rel, blob_id="f" * 40, lfs=SimpleNamespace(sha256=hf.sha256_file(path)))
    return SimpleNamespace(rfilename=rel, blob_id=hf.git_blob_sha1(path), lfs=None)


def foreign(rel: str) -> SimpleNamespace:
    return SimpleNamespace(rfilename=rel, blob_id="e" * 40, lfs=SimpleNamespace(sha256="dd" * 32))


# ---- repositories, front matter and validate-yaml ----------------------------------------------


def test_each_model_has_its_own_repository_and_the_cards_link_the_code_at_its_release_tag() -> None:
    assert B1 == "Inferensys/bulwark-b1-driver-monitoring" and E1 == "Inferensys/bulwark-e1-driver-monitoring-onnx"
    assert f"{hf.ENDPOINT}/{B1}" == model_card_url("b1") and f"{hf.ENDPOINT}/{E1}" == model_card_url("e1")
    assert _pyproject()["project"]["urls"]["Repository"] == hf.SOURCE_URL
    assert hf.CODE_TAG == f"v{__version__}" == "v0.1.0"


def test_git_blob_sha1_is_the_git_object_id(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("Bulwark\n")
    expected = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    assert hf.git_blob_sha1(path) == expected == hashlib.sha1(b"blob 8\0Bulwark\n").hexdigest()


@pytest.mark.parametrize("model", ["b1", "e1"])
def test_front_matter_is_the_hub_metadata_and_access_form_of_each_model(model: str) -> None:
    from huggingface_hub import ModelCard

    datasets = ["auc_v2", "dmd", "drive_and_act", "mrl_eye"]
    text = hf.front_matter(model, datasets)
    metadata = yaml.safe_load(text.split("---\n")[1])
    card = ModelCard(text + "# Card\n").data  # parses offline, as the Hub parses the README
    assert card.license == metadata["license"] == "cc-by-nc-4.0" and metadata["pipeline_tag"] == "video-classification"
    assert metadata["library_name"] == {"b1": "pytorch", "e1": "onnx"}[model] and metadata["metrics"] == ["roc_auc", "f1", "mae"]
    assert len(metadata["tags"]) == len(set(metadata["tags"])) == {"b1": 16, "e1": 18}[model]
    if model == "b1":
        assert "base_model" not in metadata and {"v-jepa", "lora"} <= set(metadata["tags"])
    else:
        assert metadata["base_model"] == "timm/efficientvit_b1.r224_in1k" and metadata["base_model_relation"] == "finetune"
        assert {"efficientvit", "knowledge-distillation", "onnxruntime", "cpu"} <= set(metadata["tags"]) and "v-jepa" not in metadata["tags"]
    never = {"occupant-monitoring", "oms", "gaze-estimation", "eye-tracking", "face-recognition", "dms", "adas", "fleet", "real-time", "edge-ai", "timm"}
    assert not never & set(metadata["tags"])
    # The owner's form: name, affiliation, intended use, the licence checkbox and an opt-in
    # question, Yes or No (strings, not YAML 1.1 booleans), with no "." or "$" in any key.
    assert metadata["extra_gated_fields"] == hf.GATE_FIELDS == {
        "Name": "text",
        "Company or affiliation": "text",
        "Intended use": "text",
        "I accept the CC BY-NC licence and the training datasets' terms, and will use these weights only for non-commercial "
        "research": "checkbox",
        "May Inferensys contact you about commercially licensed models?": {"type": "select", "options": ["Yes", "No"]},
    }
    assert not any(re.search(r"[.$]", key) for key in metadata["extra_gated_fields"])
    assert "\n? " not in text  # every key is a plain key, readable in review
    # The heading never suggests that these weights can be licensed: the commercial offer is
    # separate, commercially licensed models.
    assert metadata["extra_gated_heading"] == "Free for non-commercial research. These weights are not licensed for commercial use."
    assert len(metadata["extra_gated_heading"].split()) <= 12
    prompt = metadata["extra_gated_prompt"]
    assert len(re.sub(r"\]\([^)]*\)", "]", prompt).split()) <= 80 and not re.search(r"^#", prompt, re.MULTILINE)
    assert "the four training datasets: Drive&Act, DMD, MRL Eye and AUC Distracted Driver v2." in prompt
    assert f"{model_card_url(model)}#licence-and-dataset-terms" in prompt and "https://inferensys.com/contact" in prompt
    assert "[inferensys.com/contact](https://inferensys.com/contact) or hello@inferensys.com" in prompt  # the commercial route
    assert "Access is granted as soon as you submit this form" in prompt
    description = metadata["extra_gated_description"]
    assert "username, email address and answers" in description and "only if you answer Yes" in description
    assert "to contact you about commercially licensed models" in description and "licensing" not in description
    assert "https://inferensys.com/privacy" in description and metadata["extra_gated_button_content"] == "Request access"


class Answer(io.BytesIO):
    def __enter__(self) -> Answer:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_validate_sends_the_front_matter_only_and_reports_errors_and_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []
    answers = [b'{"errors":[],"warnings":[]}', b'{"errors":[],"warnings":[{"message":"tag unknown"}]}']

    def urlopen(request: urllib.request.Request, timeout: float) -> Answer:
        sent.append(request)
        return Answer(answers.pop(0))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    readme = hf.front_matter("b1", ["dmd"]) + "# Card\n\nBody text that is not sent.\n"
    assert hf.validate_metadata(readme, "model") == []
    assert sent[0].full_url == "https://huggingface.co/api/validate-yaml" and sent[0].get_method() == "POST"
    body = json.loads(sent[0].data)
    assert body["repoType"] == "model" and body["content"] == readme.split("# Card")[0].rstrip("\n") + "\n"
    assert hf.validate_metadata(readme, "model") == ["validate-yaml warning: tag unknown"]

    def refused(request: urllib.request.Request, timeout: float) -> Answer:
        error = b'{"errors":[{"message":"You can\'t have a key with a \\".\\" or \\"$\\" character","type":"error"}],"warnings":[]}'
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(error))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    assert hf.validate_metadata(readme, "model") == ['validate-yaml error: You can\'t have a key with a "." or "$" character']


def test_stage_with_validate_refuses_metadata_the_hub_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "validate_metadata", lambda readme, kind: ["validate-yaml error: bad"] if kind == "model" else [])
    out = tmp_path / "stage"
    assert hf.stage("b1", out, **bundle(tmp_path), validate=True, root=checkout(tmp_path)) == ["validate-yaml error: bad"]
    assert not out.exists()


# ---- stage: real artifacts with the repository's own templates and docs ------------------------


@pytest.fixture(scope="module")
def realistic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """E1's 16-frame and 1-frame graphs from `bulwark export`'s writer (with their
    bench.json) of one tiny E1 checkpoint trained on DMD, which the maintainers' decision of
    2026-10-30 clears for a weights release, with its evaluation and provenance; a B1 release
    directory in `bulwark release`'s format with its evaluation and provenance."""
    root = tmp_path_factory.mktemp("realistic")
    torch.manual_seed(0)
    provenance = {
        "job": "distill", "commercial_only": False, "manifest_sha256": MANIFEST, "code": {"commit": COMMIT, "dirty": False},
        "datasets": {"dmd": {"licence": "CC BY-NC-ND 4.0", "commercial_ok": False, "splits": {"train": 4}}},
        "config": {"out": "/Users/someone/checkpoints/e1-v0"},  # a whole path: scrubbed to its last part
    }  # fmt: skip
    checkpoint = save_checkpoint(
        root / "best.pt", BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, "supervised_labels": ["yawn", "phone"], "provenance": provenance}
    )
    graphs = [export_edge(checkpoint, root / f"e1-{f}.onnx", frames=f, input_size=64, embeddings=False) for f in (16, 1)]
    for path in graphs:
        write_benchmark(path, [{"ms_p50": 12.34, "ms_p95": 20.5, "threads": 1, "onnxruntime": "1.31.1", "cpu": "Test CPU",
                                "input": "1x16x64x64x3 uint8", "scope": "model only"}])  # fmt: skip
    (root / "e1-v0").mkdir()
    (root / "e1-v0" / "provenance.json").write_text(json.dumps(provenance))
    return {
        "e1": {"graphs": graphs, "evaluation": evaluation(root / "eval-e1", hf.sha256_file(checkpoint)), "provenance": root / "e1-v0" / "provenance.json"},
        "b1": {"release": release_dir(root / "bulwark-b1-v0", cleared=True), "evaluation": evaluation(root / "eval-b1", "b1" * 32),
               "provenance": provenance_file(root / "b1" / "provenance.json", "lora", config={"init": "/home/u/heads/best.pt"})},
    }  # fmt: skip


@pytest.fixture(scope="module")
def private_stages(realistic: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Both models staged with this repository's templates, docs and registry."""
    outs = {}
    for model in ("b1", "e1"):
        outs[model] = tmp_path_factory.mktemp("staging") / model
        assert hf.stage(model, outs[model], **realistic[model]) == []
    return outs


def test_b1_stage_holds_its_repository_layout_and_nothing_else(private_stages: dict[str, Path], realistic: dict[str, Any]) -> None:
    out = private_stages["b1"]
    assert set(hf._files(out)) == {
        "README.md", "MODEL_CARD.md", "config.json", "model.safetensors", "LICENSE", "LICENSE-vjepa2", "NOTICE", "SHA256SUMS",
        "provenance.json", "release-check.json", "evaluation/b1/report.html", "evaluation/b1/metrics.json",
    }  # fmt: skip  # no predictions/, calibration.json or .partial
    assert not out.with_name("b1.partial").exists()
    release = realistic["b1"]["release"]
    for name in ("config.json", "model.safetensors", "LICENSE-vjepa2"):
        assert (out / name).read_bytes() == (release / name).read_bytes(), name
    assert (out / "LICENSE").read_bytes() == (REPO / "LICENSES" / "CC-BY-NC-4.0.txt").read_bytes()  # what visitors of a gated repository can read
    assert (out / "NOTICE").read_bytes() == (REPO / "NOTICE").read_bytes()
    assert json.loads((out / "provenance.json").read_text())["config"]["init"] == "best.pt"
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest, rel
    assert sorted(line.split("  ", 1)[1] for line in (out / "SHA256SUMS").read_text().splitlines()) == sorted(set(hf._files(out)) - {"SHA256SUMS"})


def test_e1_stage_holds_both_graphs_their_contracts_and_the_timm_licence(private_stages: dict[str, Path], realistic: dict[str, Any]) -> None:
    out = private_stages["e1"]
    assert set(hf._files(out)) == {
        "README.md", "MODEL_CARD.md", "config.json", "LICENSE", "LICENSE-timm", "NOTICE", "SHA256SUMS", "provenance.json",
        "release-check.json", "bulwark-e1-video.onnx", "bulwark-e1-image.onnx", "bulwark-e1-video.onnx.bench.json",
        "bulwark-e1-image.onnx.bench.json", "evaluation/e1/report.html", "evaluation/e1/metrics.json",
    }  # fmt: skip
    video, image = realistic["e1"]["graphs"]
    assert (out / "bulwark-e1-video.onnx").read_bytes() == video.read_bytes()
    assert (out / "bulwark-e1-image.onnx").read_bytes() == image.read_bytes()
    assert (out / "LICENSE-timm").read_bytes() == (REPO / "LICENSE").read_bytes()  # Apache-2.0, section 4(a)
    config = json.loads((out / "config.json").read_text())
    assert config["model"] == "e1" and config["model_card"] == model_card_url("e1") and config["model_release"] == MODEL_RELEASE
    assert {name: entry["frames"] for name, entry in config["graphs"].items()} == {"bulwark-e1-video.onnx": 16, "bulwark-e1-image.onnx": 1}
    contract = config["graphs"]["bulwark-e1-video.onnx"]["bulwark_contract"]
    assert contract["outputs"] and contract["activation"] and contract["supervised_labels"] == ["yawn", "phone"]
    assert config["graphs"]["bulwark-e1-image.onnx"]["sha256"] == hf.sha256_file(image)


@pytest.mark.parametrize("model", ["b1", "e1"])
def test_readme_is_the_front_matter_over_the_card_rendered_from_the_templates(private_stages: dict[str, Path], model: str) -> None:
    from huggingface_hub import ModelCard

    out = private_stages[model]
    readme = (out / "README.md").read_text(encoding="utf-8")
    records = json.loads((out / "release-check.json").read_text())
    card, problems = hf.render_card(model, out, records)
    assert problems == [] and readme == hf.front_matter(model, ["dmd"]) + card
    assert ModelCard(readme).data.license == "cc-by-nc-4.0"
    assert "{{" not in readme and "}}" not in readme
    lines = card.splitlines()
    assert sum(line.startswith("# ") for line in lines) == 1 and "## Licence and dataset terms" in lines
    assert "| Yawn | `yawn` |" in card  # the labels table, from the weights' own record
    assert "| Yawn | AUROC | 0.90 (0.50) |" in card  # results to 2 decimals, with the baseline
    assert ("| Test CPU | 1 | onnxruntime 1.31.1 |" in card) is (model == "e1")  # E1's speed, from bench.json
    assert "MODEL_CARD.md" in card and "Inferensys/bulwark/blob/v0.1.0/MODEL_CARD.md" in card
    # Only the datasets of these weights' release check are cited (DMD here), from names.yaml.
    assert "- [DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset" in card and "arxiv.org/abs/1901.09097" not in card
    assert "| DMD | CC BY-NC-ND 4.0, academic use | [DMD website](https://dmd.vicomtech.org/) |" in card
    if model == "e1":  # what the export checked, from each graph's own record; config.json's size
        assert re.search(r"`bulwark export` checked each graph against the PyTorch model on [^:]+: the logits differ by at most ", card)
        assert re.search(r"\| `config\.json` \| both graphs' contracts: [^|]+ \| \d+\.\d (kB|MB) \|", card)


def test_the_e1_card_says_first_that_e1_v0_scores_well_below_b1(private_stages: dict[str, Path]) -> None:
    """E1 v0 is released as a research preview that scores well below B1 (RELEASING.md,
    release set). Its card says so first, under its opening sentence and before the
    licence: what it does not detect, where it works, what it is for, and that B1 is the
    model for research. The scores themselves are only in the tables the stage renders."""
    readme = (private_stages["e1"] / "README.md").read_text(encoding="utf-8")
    card = readme.split("\n# Bulwark-E1", 1)[1]
    top = card.split("\n- **Licence:**", 1)[0]
    assert f"> **E1 v0 scores well below Bulwark-B1. For research, use [Bulwark-B1]({model_card_url('b1')}).**" in top
    for claim in (
        "it does not detect closed eyes in driver video. On DMD's test videos it scores at chance.",
        "on a camera view it was not trained on, its phone, eating and drinking detection falls to chance or below.",
        "its activity results are weak.",
        "on close-up still images of eyes (MRL Eye), and on phone use seen from the camera views it was trained on.",
        "Use it to try CPU and ONNX deployment and Bulwark's output format, not for its accuracy. A retrain is planned.",
    ):
        assert claim in top, claim
    assert f"| Larger model | [Bulwark-B1]({model_card_url('b1')}), PyTorch; use it for research |" in card
    assert "| Results | Indicative only, and well below Bulwark-B1's: [Results and speed](#results-and-speed) |" in card


def test_both_cards_hold_every_shared_section_word_for_word(private_stages: dict[str, Path]) -> None:
    readmes = {model: (out / "README.md").read_text(encoding="utf-8") for model, out in private_stages.items()}
    values = hf.shared_values()
    for path in sorted((REPO / "release" / "hf" / "shared").glob("*.md")):
        problems: list[str] = []
        text = hf.fill(path.read_text(encoding="utf-8"), values, path.name, problems)
        assert problems == [] and all(text in readme for readme in readmes.values()), path.name
    product = (REPO / "release" / "hf" / "shared" / "product.md").read_text(encoding="utf-8")
    assert product.startswith(  # the owner's wording
        "The weights on this page can't be used commercially, including pilots, proofs of concept (POCs) and customer "
        "evaluations. Inferensys builds commercially licensed models for your cameras and use case, trained on data "
        "cleared for commercial use. Tell us your use case, cameras (RGB or infrared), hardware and where you'll deploy."
    )
    assert "[Contact Inferensys]({{ contact_url }})" in product and "{{ contact_email }}" in product


def test_release_check_records_carry_the_maintainers_decision_with_the_real_registry(private_stages: dict[str, Path]) -> None:
    for model, names in (("b1", ["model.safetensors"]), ("e1", ["bulwark-e1-video.onnx", "bulwark-e1-image.onnx"])):
        records = json.loads((private_stages[model] / "release-check.json").read_text())
        assert list(records) == names
        for name in names:
            record = records[name]
            assert record["artifact"] == name and record["publishable"] is True and record["problems"] == []
            assert record["model_card"] == model_card_url(model) and record["model_release"] == MODEL_RELEASE
            dmd = record["release_check"]["datasets"]["dmd"]
            assert record["release_check"]["cleared"] is True and dmd["weights_release_ok"] is True
            assert dmd["weights_release_basis"] == MAINTAINERS_RELEASE_DECISION  # what the clearance rests on
            assert set(record["checked_with"]) == {"bulwark", "commit", "dirty"}


@pytest.mark.parametrize("model", ["b1", "e1"])
def test_every_link_of_a_stage_reaches_a_staged_file_or_this_repository_at_the_release_tag(private_stages: dict[str, Path], model: str) -> None:
    out = private_stages[model]
    tracked = hf.tracked_files(REPO)
    for rel in ("README.md", "MODEL_CARD.md"):
        for target in _link_targets((out / rel).read_text(encoding="utf-8")):
            file_part, _, anchor = target.partition("#")
            if target.startswith(hf.SOURCE_URL + "/blob/") or target.startswith(hf.SOURCE_URL + "/tree/"):
                kind, ref, path = re.match(re.escape(hf.SOURCE_URL) + r"/(blob|tree)/([^/]+)/([^#]*)", target).groups()  # type: ignore[union-attr]
                assert ref == hf.CODE_TAG, target
                assert (path in tracked) if kind == "blob" else any(t.startswith(path.rstrip("/") + "/") for t in tracked), target
                if anchor:
                    assert anchor in hf.heading_anchors((REPO / path).read_text(encoding="utf-8")), target
            elif not re.match(r"[a-z]+:", target):
                if file_part:
                    assert rel == "README.md" and (out / file_part).is_file(), f"{rel}: {target}"  # MODEL_CARD.md links go to GitHub
                else:
                    assert anchor in hf.heading_anchors((out / rel).read_text(encoding="utf-8")), f"{rel}: {target}"


def test_every_placeholder_is_documented_and_every_template_placeholder_is_known() -> None:
    folder = REPO / "release" / "hf"
    rows = [line.split("|")[1] for line in (folder / "README.md").read_text().splitlines() if line.startswith("| `")]
    names = set(re.findall(r"`([a-z][a-z0-9_]*)`", "\n".join(rows)))  # the Values table's first column
    assert names == set(hf.shared_values()) | set(hf.card_value_names())
    for path in [*folder.glob("*.md"), *folder.glob("shared/*.md")]:
        if path.name == "README.md":
            continue
        used = set(hf.PLACEHOLDER.findall(path.read_text()))
        allowed = set(hf.shared_values()) if path.parent.name == "shared" or path.name == "org-card.md" else names
        assert used <= allowed, (path.name, used - allowed)
    names_yaml = yaml.safe_load((folder / "names.yaml").read_text())
    from bulwark.data.taxonomy import default_taxonomy

    assert set(names_yaml["outputs"]) == set(default_taxonomy().labels)


# ---- the card: placeholders, includes, tables and structure ------------------------------------


def test_unknown_placeholders_braces_missing_includes_and_model_values_in_shared_sections_refuse_the_stage(tmp_path: Path) -> None:
    files = {
        "release/hf/b1.md": TEMPLATES["b1.md"] + "\n{{ no_such_value }} and {{> missing }} and {{ broken\n",
        "release/hf/shared/licence.md": "Results: {{ results_table }}. {{> product }}\n",
    }
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=checkout(tmp_path, files))
    assert any("{{ no_such_value }} is not a placeholder" in p for p in problems), problems
    assert any("includes release/hf/shared/missing.md, which does not exist" in p for p in problems)
    assert any("release/hf/b1.md: '{{' or '}}' outside a placeholder or include" in p for p in problems)
    assert any("shared/licence.md: {{ results_table }} differs between the models" in p for p in problems)
    assert any("shared/licence.md: a shared section cannot include another" in p for p in problems)
    assert not (tmp_path / "stage").exists()


def test_values_without_a_source_render_as_at_release_items_that_keep_the_stage_private(tmp_path: Path) -> None:
    root = checkout(
        tmp_path, {"release/hf/b1.md": TEMPLATES["b1.md"] + "\nCollection: {{ collection_link }}.\n", **COLLECTION_PENDING}
    )
    inputs = bundle(tmp_path)
    del inputs["evaluation"]
    out = tmp_path / "stage"
    assert hf.stage("b1", out, **inputs, root=root) == []  # a private stage may be incomplete
    readme = (out / "README.md").read_text()
    assert "*At release: no evaluation of these weights is staged (evaluation/b1/).*" in readme
    assert "*At release: the collection's address (release/hf/collection.yaml `url`).*" in readme
    pending = hf.public_problems(out, "b1", json.loads((out / "release-check.json").read_text()), root)
    assert "No evaluation of these weights is staged (--eval; evaluation/b1/)." in pending
    assert "README.md: link evaluation/b1/report.html reaches no staged file." in pending
    assert any(p.startswith("README.md: 2 items marked *at release*") for p in pending)


def test_results_tables_give_two_decimals_the_baseline_and_the_test_people() -> None:
    from bulwark.data.taxonomy import default_taxonomy

    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    b1, e1 = metrics("b1" * 32), metrics("e1" * 32)
    e1["splits"]["test"]["summary"]["yawn"]["value"] = 0.8
    table = hf._results_table("b1", default_taxonomy(), names, b1, None).splitlines()
    assert table[:2] == ["| What it detects | Metric | B1 (baseline) | Test people |", "|---|---|---|---|"]
    assert table[2:] == ["| Yawn | AUROC | 0.90 (0.50) | 3 |", "| Phone | AUROC | 0.99 (0.50) | 3 + 1 group |"]  # taxonomy order; AUC is one group
    table = hf._results_table("e1", default_taxonomy(), names, e1, b1).splitlines()
    assert table[0] == "| What it detects | Metric | E1 (baseline) | B1 | Test people |"
    assert table[2] == "| Yawn | AUROC | 0.80 (0.50) | 0.90 | 3 |"
    # B1's summary is a mean over datasets: over others than E1's, it is not set beside E1's.
    assert hf._results_table("e1", default_taxonomy(), names, e1, b1).endswith(
        "\n\nThe B1 column is Bulwark-B1's score on the same test clips, from its own evaluation."
    )  # no "not comparable" cell, so no sentence about one
    b1["splits"]["test"]["summary"]["phone"]["datasets"] = ["dmd", "drive_and_act"]
    table = hf._results_table("e1", default_taxonomy(), names, e1, b1)
    assert table.splitlines()[3] == "| Phone | AUROC | 0.99 (0.50) | not comparable | 3 + 1 group |"
    assert table.endswith('It reads "not comparable" where the two models were scored on different datasets for that output.')
    holdout = hf._holdout("b1", default_taxonomy(), names, b1)  # UTA-RLDD, with no clips, is left out
    assert holdout.splitlines()[0] == "**Drive&Act, inner-mirror camera** (a camera that training never used, filming the 1 test person; 2 clips)"
    assert "| Phone | AUROC | 0.81 (0.50) | 2 |" in holdout and "UTA" not in holdout and "inner_mirror" not in holdout
    # The holdout is scored per clip, one camera; Drive&Act's test side per event, over several
    # cameras: the sentence sets the holdout beside the test cameras' per-clip scores.
    assert hf._holdout_summary(default_taxonomy(), names, b1) == (
        "**A camera that training never used.** On Drive&Act's inner-mirror camera, filming the 1 test person, every score is "
        "worse than on the clips of Drive&Act's test cameras, each scored on its own as here: phone AUROC 0.81 against 0.94."
    )
    per_event_only = copy.deepcopy(b1)
    del per_event_only["splits"]["test"]["per_clip"]
    assert hf._holdout_summary(default_taxonomy(), names, per_event_only) == ""  # never per event against per clip
    per_clip = copy.deepcopy(b1)
    per_clip["splits"]["test"]["units"]["drive_and_act"] = "clip"
    assert hf._holdout_summary(default_taxonomy(), names, per_clip).endswith("than on Drive&Act's test cameras: phone AUROC 0.81 against 1.00.")
    assert hf._test_split(names, b1).splitlines()[2:] == [  # in the order of names.yaml
        "| Drive&Act | 3 | 12 |", "| DMD | 3 | 10 |", "| AUC Distracted Driver v2 | one group, not identified per image | 5 |",
    ]  # fmt: skip
    b1["splits"]["test"]["counts"]["mrl_eye"] = {"clips": 4, "subjects": 3}
    assert hf._test_people_sentences(names, b1) == (  # AUC spelled out at its first mention
        "Drive&Act, DMD and MRL Eye each have 3 test people. AUC Distracted Driver v2 (from the American University in Cairo, "
        "AUC) does not identify the person in each image, so its test people count as one group.",
        "3 test people per dataset",
    )
    b1["splits"]["test"]["counts"]["mrl_eye"]["subjects"] = 2
    assert hf._test_people_sentences(names, b1)[1] == "2 to 3 test people per dataset"


def test_a_summary_row_whose_datasets_disagree_is_given_per_dataset() -> None:
    """A mean of a score at chance and a near-perfect one reads as moderate: such an output
    gets one row per dataset, with B1's score on that dataset beside E1's."""
    from bulwark.data.taxonomy import default_taxonomy

    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    b1, e1 = metrics("b1" * 32), metrics("e1" * 32)
    for data, (dmd, mrl) in ((b1, (0.902, 0.998)), (e1, (0.4968, 0.9984))):
        test = data["splits"]["test"]
        test["counts"]["mrl_eye"] = {"clips": 8, "subjects": 3, "events": 8}
        test["summary"]["eyes_closed"] = {"metric": "auroc", "value": (dmd + mrl) / 2, "n": 30, "n_subjects": 6,
                                          "datasets": ["dmd", "mrl_eye"], "baseline": {"auroc": 0.5}}  # fmt: skip
        test["datasets"]["dmd"]["eyes_closed"] = {"n": 20, "auroc": dmd, "baseline": {"auroc": 0.5}}
        test["datasets"]["mrl_eye"] = {"eyes_closed": {"n": 8, "auroc": mrl, "baseline": {"auroc": 0.5}}}
    table = hf._results_table("e1", default_taxonomy(), names, e1, b1)
    assert table.splitlines()[2:4] == ["| Eyes closed: DMD | AUROC | 0.50 (0.50) | 0.90 | 3 |", "| Eyes closed: MRL Eye | AUROC | 1.00 (0.50) | 1.00 | 3 |"]
    assert "| Eyes closed | AUROC |" not in table
    assert table.endswith("Eyes closed is given per dataset, because its scores differ by more than 0.25 between datasets and a mean would hide that.")
    assert "| Eyes closed | AUROC | 0.95 (0.50) | 6 |" in hf._results_table("b1", default_taxonomy(), names, b1, None)  # 0.90 and 1.00 agree


def test_a_dataset_scored_per_event_counts_events_not_clips() -> None:
    from bulwark.data.taxonomy import default_taxonomy

    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    text = hf._by_dataset("b1", default_taxonomy(), names, metrics("b1" * 32), "test")
    drive_and_act, dmd = text.split("\n\n**DMD**")
    assert drive_and_act.startswith("**Drive&Act** (3 test people, 12 clips of 4 events)\n\nDrive&Act films each moment from several cameras.")
    assert "| What it detects | Metric | B1 (baseline) | Events |\n|---|---|---|---|\n| Phone | AUROC | 1.00 (0.50) | 4 |" in drive_and_act
    assert dmd.startswith(" (3 test people, 10 clips)\n\n| What it detects | Metric | B1 (baseline) | Clips |")


def test_sentences_about_empty_lists_are_left_out() -> None:
    from bulwark.data.taxonomy import default_taxonomy

    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    assert hf._masked_on(names, ["yawn", "phone"], {"phone": {"image"}}, "image") == "On still images, this output was not trained and is left out: phone."
    assert hf._masked_on(names, ["yawn", "phone"], {"phone": {"image"}}, "video") == ""
    support = {"training_support": {"activity": {"classes_gt": {"smoking": 0, "texting": 3}}}}
    assert hf._untrained_classes(names, default_taxonomy(), ["activity"], support) == "This class was never trained: smoking (activity)."
    assert hf._untrained_classes(names, default_taxonomy(), ["yawn"], support) == ""


def test_an_official_page_that_does_not_answer_is_linked_with_its_archived_copy() -> None:
    """MRL Eye's page did not answer when the B1 card was staged: both tables that link it,
    the dataset terms (names.yaml `archive`) and B1's training data, add the Internet
    Archive's copy, which the registry's terms were read from."""
    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    archive = "https://web.archive.org/web/20260516/https://mrl.cs.vsb.cz/eyedataset.html"
    table = hf._datasets_table(names, ["dmd", "mrl_eye"])
    assert f"| [MRL Eye website](https://mrl.cs.vsb.cz/eyedataset.html) ([archived copy]({archive})) |" in table
    assert "| DMD | CC BY-NC-ND 4.0, academic use | [DMD website](https://dmd.vicomtech.org/) |" in table
    assert f"| [MRL Eye](https://mrl.cs.vsb.cz/eyedataset.html) ([archived copy]({archive})) |" in (REPO / "release" / "hf" / "b1.md").read_text()


def test_e1_says_whether_it_trained_the_outputs_of_its_b1_teacher() -> None:
    teacher = {"checkpoint_labels": {"supervised_labels": ["yawn", "phone"]}}
    assert hf._outputs_as_b1(["phone", "yawn"], teacher) == "the same"
    assert hf._outputs_as_b1(["yawn"], teacher) == "most of the same"
    assert hf._outputs_as_b1(["yawn"], None).startswith("*At release: the evaluation of E1's B1 teacher is not staged")


def test_the_labels_table_names_each_output_its_range_and_whether_ground_truth_checked_it() -> None:
    from bulwark.data.taxonomy import default_taxonomy

    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    status = {"yawn": "gt", "head_yaw": "teacher_only", "modality": "gt"}
    trained = ["yawn", "head_yaw", "modality"]
    b1 = hf._labels_table("b1", default_taxonomy(), names, trained, {}, status).splitlines()
    assert b1[0] == "| Group | What it detects | Output key | Value | Checked against ground truth |"
    assert "| Mouth | Yawn | `yawn` | Yes/no probability | Yes |" in b1
    assert "| Head | Head yaw | `head_yaw` | -1 to 1 (deg / 90) | No: learned from another model's labels |" in b1  # as returned
    assert "| Camera conditions | Camera type | `modality` | One of 3 classes | Sanity check only |" in b1
    trained += ["blink", "eyeglasses"]
    status |= {"blink": "gt", "eyeglasses": "gt"}
    e1 = hf._labels_table("e1", default_taxonomy(), names, trained, {"modality": {"image"}, "yawn": {"video"}}, status).splitlines()
    assert e1[0] == "| Group | What it detects | Output key | Checked against ground truth | Trained for |"  # five columns, as B1
    # Trained for: where the training clips had labels (names.yaml `labelled`), less the masked media types.
    assert "| Camera conditions | Camera type | `modality` | Sanity check only | Video only |" in e1  # masked on images
    assert "| Head | Head yaw | `head_yaw` | No: learned from another model's labels | Video and still images |" in e1
    assert "| Eyes | Blink | `blink` | Yes | Video (still images: copies B1 only) |" in e1  # no image has a blink label
    assert "| Hands and objects | Eyeglasses | `eyeglasses` | Yes | Still images (video: copies B1 only) |" in e1
    assert "| Mouth | Yawn | `yawn` | Yes | Copies B1 only |" in e1  # masked on video, and no image has a yawn label
    assert all(row.count("|") == 6 for row in [*b1, *e1])  # the group is a column: no row of empty cells
    # An output a reader must not use (names.yaml `note`): v0's head_nod labels came from a
    # rule with the pitch sign reversed. The note replaces the reason in the check, in both tables.
    nod, nod_status = ["head_yaw", "head_nod"], {"head_yaw": "teacher_only", "head_nod": "teacher_only"}
    rows = hf._labels_table("e1", default_taxonomy(), names, nod, {}, nod_status).splitlines()
    assert "| Head | Head nod | `head_nod` | No: wrong in v0, do not use | Video (still images: copies B1 only) |" in rows
    rows = hf._labels_table("b1", default_taxonomy(), names, nod, {}, nod_status).splitlines()
    assert "| Head | Head nod | `head_nod` | Yes/no probability | No: wrong in v0, do not use |" in rows
    assert "| Head | Head yaw | `head_yaw` | -1 to 1 (deg / 90) | No: learned from another model's labels |" in rows
    noted = {**names, "outputs": {**names["outputs"], "yawn": {**names["outputs"]["yawn"], "note": "see the notes"}}}
    rows = hf._labels_table("b1", default_taxonomy(), noted, ["yawn", "modality"], {}, status | {"yawn": "gt"}).splitlines()
    assert "| Mouth | Yawn | `yawn` | Yes/no probability | Yes: see the notes |" in rows  # any check, any output
    assert "| Camera conditions | Camera type | `modality` | One of 3 classes | Sanity check only |" in rows
    assert hf._copied_on_images("e1", names, trained, {"yawn": {"video"}}) == (
        "On a still image, ignore yawn and blink: no training image had a label for them, so on still images E1 only learned "
        "to copy Bulwark-B1's outputs, which B1 learned from video."
    )
    assert hf._copied_on_images("b1", names, trained, {}) == "" and hf._copied_on_images("e1", names, ["head_yaw"], {}) == ""
    unknown = {**names, "outputs": {**names["outputs"], "blink": {"name": "Blink", "group": "Eyes"}}}
    assert hf._trained_for(unknown, "blink", {}).startswith("*At release: release/hf/names.yaml records no `labelled` media types")


def test_names_yaml_labelled_media_types_are_those_of_the_data_card_coverage_table() -> None:
    """names.yaml `labelled` (the E1 card's Trained for column) lists, for each output, the
    media types whose train clips have a label for it in docs/data-card.md's coverage table."""
    names = yaml.safe_load((REPO / "release" / "hf" / "names.yaml").read_text())
    found: dict[str, set[str]] = {}
    for line in (REPO / "docs" / "data-card.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("`") and cells[2] in ("video", "image") and cells[4].replace(",", "").isdigit():
            found.setdefault(cells[0].strip("`"), set()).add(cells[2])
    assert len(found) > 15  # the table was read
    labelled = {label: set(entry["labelled"]) for label, entry in names["outputs"].items() if "labelled" in entry}
    assert labelled == found


def test_card_structure_rules() -> None:
    good = "# Model: test (research preview)\n\n## Results\n\n### Per dataset\n\n```bash\nls\n```\n"
    assert hf.structure_problems(good) == []
    assert hf.structure_problems("## Results\n") == ["README.md: the card needs exactly one H1 heading"]
    assert hf.structure_problems("# A\n\n### Deep\n") == ["README.md: heading 'Deep' skips a level"]
    assert hf.structure_problems("# A\n\n## Using it in a product?\n")[0].startswith("README.md: heading 'Using it in a product?'")
    assert hf.structure_problems("# A\n\n```\nx\n```\n") == ["README.md: a code fence without a language (bash, python, bibtex, text)"]
    assert hf.structure_problems("# A\n\n```python\n# not a heading\n```\n") == []


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("[notes](docs/notes.md)", "README.md: link docs/notes.md reaches no file of this stage"),
        ("[main](https://github.com/Inferensys/bulwark/blob/main/README.md)", "go to the release tag v0.1.0, not main"),
        ("[gone]({{ code_url }}/docs/gone.md)", "docs/gone.md is not a file git tracks"),
        ("[folder]({{ code_url }}/examples)", "examples is not a file git tracks"),
        ("![image]({{ code_raw_url }}/docs/media/gone.webp)", "docs/media/gone.webp is not a file git tracks"),
        ("![image](https://raw.githubusercontent.com/Inferensys/bulwark/main/README.md)", "go to the release tag v0.1.0, not main"),
        ("[anchor]({{ code_url }}/docs/data-card.md#nowhere)", "README.md: no heading for #nowhere in docs/data-card.md"),
        ("[here](#nowhere)", "README.md: no heading for #nowhere"),
        ("[ref]\n\n[ref]: docs/data-card.md", "README.md: reference-style link docs/data-card.md"),
        ('<a href="LICENSE">licence</a>', "README.md: HTML link LICENSE"),
        ("[spaced](docs/a b.md)", "README.md: link '](docs/a b.md)"),
    ],
)
def test_a_card_link_that_reaches_neither_the_stage_nor_the_code_at_its_tag_refuses_the_stage(tmp_path: Path, link: str, expected: str) -> None:
    root = checkout(tmp_path, {"release/hf/b1.md": TEMPLATES["b1.md"] + f"\n{link}\n"})
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=root)
    assert any(expected in p for p in problems), problems
    assert not (tmp_path / "stage").exists()


def test_the_full_card_is_staged_with_its_links_pointing_at_the_code_on_github(tmp_path: Path) -> None:
    out = staged(tmp_path, checkout(tmp_path))
    card = (out / "MODEL_CARD.md").read_text()
    tag = f"{hf.SOURCE_URL}/blob/{hf.CODE_TAG}"
    assert f"[data card]({tag}/docs/data-card.md#data)" in card and f"[CC BY-NC 4.0]({tag}/LICENSES/CC-BY-NC-4.0.txt)" in card
    assert f"code [LICENSE]({tag}/LICENSE)" in card  # the repository's own LICENSE, not the staged legal code
    assert f"[examples]({hf.SOURCE_URL}/tree/{hf.CODE_TAG}/examples)" in card and "[details](#model-details)" in card


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("[cli](bulwark/cli.py)", "MODEL_CARD.md: link bulwark/cli.py: bulwark/cli.py is not a file or folder git tracks"),
        ("[new](docs/new.md)", "MODEL_CARD.md: link docs/new.md: docs/new.md is not a file or folder git tracks"),
        ("[anchor](docs/data-card.md#nowhere)", "MODEL_CARD.md: no heading for #nowhere in docs/data-card.md"),
        ("[up](../outside.md)", "MODEL_CARD.md: link ../outside.md: ../outside.md is not a file or folder"),
        ("[self](#nowhere)", "MODEL_CARD.md: no heading for #nowhere"),
        ('[titled](docs/gone.md "Gone")', "MODEL_CARD.md: link docs/gone.md: docs/gone.md is not a file or folder"),
    ],
)
def test_a_full_card_link_to_an_untracked_path_or_missing_heading_refuses_the_stage(tmp_path: Path, link: str, expected: str) -> None:
    root = checkout(tmp_path, card=CARD + f"\n{link}\n")
    (root / "docs" / "new.md").write_text("# New\n")  # not committed
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=root)
    assert any(p.startswith(expected) for p in problems), problems


def test_fixed_files_must_be_tracked(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    git(root, "rm", "-q", "--cached", "NOTICE")
    git(root, "commit", "-qm", "untrack NOTICE")
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=root)
    assert "NOTICE: not tracked by git, so not a published repository file" in problems


# ---- stage: text, paths, dataset material and inputs -------------------------------------------


@pytest.mark.parametrize(
    "text",
    ['"/Users/someone/x"', "file:///Users/me/x", "/home/u/run", "/kaggle/working/e1.pt", "C:\\data\\x", "D:/runs",
     "see ~/runs/x", "/Volumes/disk/x", "/private/var/folders/ab/x", "/content/drive/x", "/opt/ml/model"],
)  # fmt: skip
def test_local_paths_are_found(text: str) -> None:
    assert any(p.startswith("a local path") for p in hf._text_problems([text], {})), text


@pytest.mark.parametrize(
    "text",
    ["infra/kaggle/push_data.sh", "cp data/manifest.parquet /tmp/remerged.parquet", "https://www.kaggle.com/c/state-farm",
     "runs/eval/b1/report.html", model_card_url("b1"), "https://github.com/Inferensys/bulwark/tree/v0.1.0/examples",
     "Write to hello@inferensys.com."],
)  # fmt: skip
def test_relative_paths_urls_and_the_public_contact_address_pass(text: str) -> None:
    assert hf._text_problems([text], {}) == []


def test_secrets_emails_credential_values_and_subject_ids_are_found() -> None:
    def found(text: str, **kwargs: Any) -> list[str]:
        return hf._text_problems([text], kwargs.get("values", {}))

    for secret in ("hf_" + "a1" * 17, "https://x.blob.core.windows.net/c/x?sv=1&sig=abc", "AccountKey=abc;",
                   "-----BEGIN OPENSSH PRIVATE KEY-----", '{"username": "u", "key": "' + "0f" * 16 + '"}'):  # fmt: skip
        assert found(secret) == ["what looks like a token, key or signed URL"], secret
    assert found("contact someone@example.org") == ["an email address"]
    assert found("hello@inferensys.com or someone@inferensys.com") == ["an email address"]
    assert found("someuser/bulwark-dmd", values={"KAGGLE_USER": "someuser"}) == ["the value of $KAGGLE_USER"]
    assert found('"subject_ids": ["drive_and_act:vp5"]')[0].startswith("dataset subject ids (drive_and_act:vp5)")
    # the data card names AUC's two driver groups, which hold no person's id
    assert found("is `auc_v2:train_drivers` or auc_v2:test_drivers.") == []
    assert found("dmd:gA_1 and `auc_v2:train_drivers`")[0].startswith("dataset subject ids (dmd:gA_1)")


def test_subject_ids_in_a_template_or_the_full_card_refuse_the_stage(tmp_path: Path) -> None:
    row = "| Holdout drivers | 3 (drive_and_act:vp11, drive_and_act:vp13, drive_and_act:vp5) |\n"
    root = checkout(tmp_path, {"release/hf/b1.md": TEMPLATES["b1.md"] + row}, card=CARD + row)
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=root)
    for rel in ("MODEL_CARD.md", "README.md"):  # README.md is the public card of the gated repository
        assert any(p.startswith(f"{rel}: dataset subject ids (drive_and_act:vp11") for p in problems), problems


def test_the_value_of_a_credential_variable_refuses_the_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = bundle(tmp_path)
    provenance_file(inputs["provenance"], "lora", data="someuser/bulwark-dmd")  # a Kaggle slug
    monkeypatch.setenv("KAGGLE_USER", "someuser")
    problems = hf.stage("b1", tmp_path / "stage", **inputs, root=checkout(tmp_path))
    assert "provenance.json: the value of $KAGGLE_USER" in problems


def test_onnx_text_reads_metadata_doc_strings_and_attributes_but_not_names(tmp_path: Path) -> None:
    path = graph(tmp_path / "g.onnx", doc_string="traced in /home/u/train.py", props={"note": "/kaggle/working/e1.pt"})
    texts = hf.onnx_text(path)
    assert "traced in /home/u/train.py" in texts and "/kaggle/working/e1.pt" in texts
    assert "yawn" not in texts and "Identity" not in texts  # names and op types are identifiers, not free text


def _edit_json(path: Path, **values: Any) -> None:
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data))


def _config_note(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["release"] / "config.json", note="/Users/someone/x")


def regraph(path: Path, **kwargs: Any) -> Path:
    """A graph written again in place, measured again (its bench.json names its sha256)."""
    graph(path, **kwargs)
    write_benchmark(path, [{"ms_p50": 30.0, "threads": 1}])
    return path


def _onnx_note(inputs: dict[str, Any]) -> None:
    regraph(inputs["graphs"][0], props={"note": "/kaggle/working/e1.pt"})


def _metrics_path(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["evaluation"] / "metrics.json", files={"test": "C:\\data\\test.npz"})


def _header_path(inputs: dict[str, Any]) -> None:
    save_file({"w": np.zeros((2, 2), np.float32)}, str(inputs["release"] / "model.safetensors"), metadata={"source": "/home/u/b1.pt"})


def _provenance_text(inputs: dict[str, Any]) -> None:
    provenance_file(inputs["provenance"], "lora", notes="logs in ~/runs/b1 on the old machine")


def _binary_report(inputs: dict[str, Any]) -> None:
    (inputs["evaluation"] / "report.html").write_bytes(b"\xff\xfe\x00binary")


def _subject_ids(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["teacher_evaluation"] / "metrics.json", holdout_groups={"drive_and_act": {"subject_ids": ["drive_and_act:vp5"]}})


@pytest.mark.parametrize(
    ("model", "change", "expected"),
    [
        ("b1", _config_note, "config.json: a local path"),
        ("e1", _onnx_note, "bulwark-e1-video.onnx: a local path ('/kaggle/working/e1.pt'"),
        ("b1", _metrics_path, "evaluation/b1/metrics.json: a local path"),
        ("b1", _header_path, "model.safetensors: a local path"),
        ("b1", _provenance_text, "provenance.json: a local path"),
        ("b1", _binary_report, "evaluation/b1/report.html: unexpected binary or unreadable file"),
        ("e1", _subject_ids, "evaluation/b1/metrics.json: dataset subject ids (drive_and_act:vp5)"),
    ],
)
def test_staged_text_with_local_paths_subject_ids_or_binary_data_is_refused(tmp_path: Path, model: str, change: Any, expected: str) -> None:
    root = checkout(tmp_path)
    inputs = bundle(tmp_path, model)
    change(inputs)
    out = tmp_path / "stage"
    problems = hf.stage(model, out, **inputs, root=root)
    assert any(p.startswith(expected) for p in problems), problems
    assert not out.exists() and not out.with_name("stage.partial").exists()


def test_a_whole_local_path_in_provenance_is_scrubbed_not_refused(tmp_path: Path) -> None:
    inputs = bundle(tmp_path)
    provenance_file(inputs["provenance"], "lora", config={"features": "/Users/someone/data/features/b1"})
    out = tmp_path / "stage"
    assert hf.stage("b1", out, **inputs, root=checkout(tmp_path)) == []
    assert json.loads((out / "provenance.json").read_text())["config"]["features"] == "b1"


def _frames(inputs: dict[str, Any]) -> None:
    (inputs["release"] / "frames").mkdir()


def _manifest(inputs: dict[str, Any]) -> None:
    (inputs["release"] / "manifest.parquet").write_bytes(b"PAR1")


def _limit(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["evaluation"] / "metrics.json", meta={"limit_per_split": 96, "checkpoint": {"sha256": "b1" * 32}})


def _other_weights(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["evaluation"] / "metrics.json", meta={"limit_per_split": 0, "checkpoint": {"sha256": "cc" * 32}})


def _part_of_test(inputs: dict[str, Any]) -> None:
    coverage = metrics("b1" * 32)["coverage"]
    coverage["test"]["dmd"]["predicted"] = 4  # shards or cached features missing
    coverage["test"]["auc_v2"]["predicted"] = 0
    _edit_json(inputs["evaluation"] / "metrics.json", coverage=coverage)


def _other_split(inputs: dict[str, Any]) -> None:
    coverage = metrics("b1" * 32)["coverage"]
    coverage["holdout"]["drive_and_act"]["predicted_in_other_splits"] = 3
    _edit_json(inputs["evaluation"] / "metrics.json", coverage=coverage)


def _no_coverage(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["evaluation"] / "metrics.json", coverage={})


def _teacher_on_other_clips(inputs: dict[str, Any]) -> None:
    data = json.loads((inputs["teacher_evaluation"] / "metrics.json").read_text())
    data["meta"]["predicted_with"]["manifest"]["sha256"] = "0" * 64
    (inputs["teacher_evaluation"] / "metrics.json").write_text(json.dumps(data))


def _teacher_under_other_split(inputs: dict[str, Any]) -> None:
    data = json.loads((inputs["teacher_evaluation"] / "metrics.json").read_text())
    data["meta"]["predicted_with"]["split_policy"]["holdout_views"] = []
    (inputs["teacher_evaluation"] / "metrics.json").write_text(json.dumps(data))


def _other_teacher(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["teacher_evaluation"] / "metrics.json", meta={"limit_per_split": 0, "checkpoint": {"sha256": "cc" * 32}})


def _bench(inputs: dict[str, Any]) -> None:
    path = inputs["graphs"][0]
    path.with_name(path.name + ".bench.json").write_text(json.dumps({"onnx": {"sha256": "00" * 32}}))


def _provenance(inputs: dict[str, Any]) -> None:
    provenance_file(inputs["provenance"], "lora", code={"commit": "f" * 40, "dirty": False})


def _no_vjepa_notice(inputs: dict[str, Any]) -> None:
    (inputs["release"] / "LICENSE-vjepa2").unlink()


def _apache(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["release"] / "config.json", weights_licence="Apache-2.0")


def _e1_release(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["release"] / "config.json", kind="e1")


def _symlink(inputs: dict[str, Any]) -> None:
    target = inputs["release"].with_name("real-release")
    inputs["release"].rename(target)
    inputs["release"].symlink_to(target)


def _format_version(inputs: dict[str, Any]) -> None:
    _edit_json(inputs["release"] / "config.json", format_version=99)


def _two_video_graphs(inputs: dict[str, Any]) -> None:
    inputs["graphs"][1] = regraph(inputs["graphs"][1], frames=16)


def _eight_frames(inputs: dict[str, Any]) -> None:
    inputs["graphs"][1] = regraph(inputs["graphs"][1], frames=8)


def _two_checkpoints(inputs: dict[str, Any]) -> None:
    inputs["graphs"][1] = regraph(inputs["graphs"][1], frames=1, checkpoint="cc" * 32)


def _graph_apache(inputs: dict[str, Any]) -> None:
    regraph(inputs["graphs"][0], props={"license": "Apache-2.0"})


@pytest.mark.parametrize(
    ("model", "change", "expected"),
    [
        ("b1", _frames, "holds frames; a release directory holds only"),
        ("b1", _manifest, "holds manifest.parquet; a release directory holds only"),
        ("b1", _limit, "a --limit run (96 clips per split)"),
        ("b1", _other_weights, "an evaluation of other weights (sha256 cccccccccccc) than the staged artifacts"),
        ("b1", _part_of_test, "predicted only part of a split (test auc_v2: 0 of 5; test dmd: 4 of 10 clips)"),
        ("b1", _other_split, "predicted clips its manifest puts in another split (holdout drive_and_act: 3)"),
        ("b1", _no_coverage, "records no coverage of the test split"),
        ("e1", _part_of_test, "predicted only part of a split"),
        ("e1", _other_teacher, "an evaluation of other weights (sha256 cccccccccccc) than the B1 teacher of these graphs"),
        ("e1", _teacher_on_other_clips, "scored other clips than"),
        ("e1", _teacher_under_other_split, "scored other clips than"),
        ("e1", _bench, "measured on another graph (onnx.sha256 differs)"),
        ("b1", _provenance, "not the provenance of the staged weights' checkpoint"),
        ("b1", _no_vjepa_notice, "a B1 release needs LICENSE-vjepa2"),
        ("b1", _apache, "licence 'Apache-2.0' is not research-only"),
        ("b1", _e1_release, "a release of kind 'e1'; the B1 repository holds a B1 release directory"),
        ("b1", _symlink, "a symbolic link; pass the file or directory itself"),
        ("b1", _format_version, "not a bulwark-release v1 directory"),
        ("e1", _two_video_graphs, "a second 16-frame graph"),
        ("e1", _eight_frames, "a graph for 8 frames; the E1 repository holds a 16-frame and a 1-frame graph"),
        ("e1", _two_checkpoints, "The graphs come from different checkpoints"),
        ("e1", _graph_apache, "licence 'Apache-2.0' is not research-only"),
    ],
)
def test_inputs_that_are_not_a_release_are_refused(tmp_path: Path, model: str, change: Any, expected: str) -> None:
    root = checkout(tmp_path)
    inputs = bundle(tmp_path, model)
    change(inputs)
    out = tmp_path / "stage"
    problems = hf.stage(model, out, **inputs, root=root)
    assert any(expected in p for p in problems), problems
    assert not out.exists() and not out.with_name("stage.partial").exists()


def test_a_stage_takes_only_its_own_model_s_inputs(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    b1, e1 = bundle(tmp_path / "b", "b1"), bundle(tmp_path / "e", "e1")
    assert hf.stage("b1", tmp_path / "s1", release=b1["release"], graphs=e1["graphs"], root=root) == ["A B1 stage takes no --onnx and no --teacher-eval."]
    assert hf.stage("e1", tmp_path / "s2", release=b1["release"], graphs=e1["graphs"], root=root) == [
        "An E1 stage takes no --release: its repository holds the ONNX graphs."
    ]
    assert hf.stage("e1", tmp_path / "s3", root=root)[0].startswith("An E1 stage needs --onnx")
    assert hf.stage("b1", tmp_path / "s4", root=root)[0].startswith("A B1 stage needs --release")


@pytest.mark.parametrize("folder", ["data", "ops", "docs/local"])  # as .gitignore
def test_inputs_under_private_folders_and_a_used_output_folder_are_refused(tmp_path: Path, folder: str) -> None:
    root = checkout(tmp_path)
    release = release_dir(root / folder / "bulwark-b1-v0")
    out = tmp_path / "stage"
    out.mkdir()
    (out / "old.txt").write_text("x")
    problems = hf.stage("b1", out, release=release, root=root)
    assert f"{release}: under {folder}/, which holds dataset material or private notes" in problems
    assert f"{out} exists and is not an empty folder; stage into a new one" in problems


def test_an_output_folder_inside_an_input_is_refused(tmp_path: Path) -> None:
    inputs = bundle(tmp_path)
    out = inputs["release"] / "stage"
    problems = hf.stage("b1", out, **inputs, root=checkout(tmp_path))
    assert f"{inputs['release']}: overlaps the output folder {out}" in problems


def test_the_record_refuses_a_licence_that_is_not_research_only_and_another_model_s_card(tmp_path: Path) -> None:
    release = release_dir(tmp_path / "r", cleared=True, licence="MIT")
    record = hf.artifact_record(release, "b1", checkout(tmp_path))
    assert record["publishable"] is False and "Its licence 'MIT' is not research-only." in record["problems"]
    assert record["release_check"]["checked_for"] == "commercial licence"
    other = hf.artifact_record(graph(tmp_path / "g.onnx", model_card=model_card_url("b1")), "e1", checkout(tmp_path / "x"))
    assert f"It records model card {model_card_url('b1')}, not {model_card_url('e1')}: export it again with `bulwark export`." in other["problems"]


def test_public_stage_with_the_real_registry_lists_the_uncleared_datasets_and_writes_nothing(tmp_path: Path) -> None:
    root = checkout(tmp_path, E1_RELEASED)
    out = tmp_path / "stage"
    problems = hf.stage("e1", out, **bundle(tmp_path, "e1", cleared=True, dataset=UNDECIDED), public=True, root=root)
    assert any(p.startswith("bulwark-e1-video.onnx: Publishing weights trained on these datasets is not cleared") and f"{UNDECIDED} (not determined)" in p for p in problems), problems
    assert not out.exists()


@pytest.mark.parametrize("model", ["b1", "e1"])
def test_public_stage_of_a_complete_cleared_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    clear(monkeypatch, "dmd")
    out = staged(tmp_path, checkout(tmp_path, E1_RELEASED), model, cleared=True, public=True)
    records = json.loads((out / "release-check.json").read_text())
    assert records and all(record["publishable"] for record in records.values())


def _old_card_link(inputs: dict[str, Any], root: Path) -> None:
    _edit_json(inputs["release"] / "config.json", model_card="https://huggingface.co/Inferensys/bulwark")


def _no_eval(inputs: dict[str, Any], root: Path) -> None:
    del inputs["evaluation"]


def _no_provenance(inputs: dict[str, Any], root: Path) -> None:
    del inputs["provenance"]


def _template_item(inputs: dict[str, Any], root: Path) -> None:
    (root / "release" / "hf" / "b1.md").write_text(TEMPLATES["b1.md"] + "\n*At release:* the quick start.\n")
    git(root, "commit", "-qam", "template item")


def _card_item(inputs: dict[str, Any], root: Path) -> None:
    (root / "MODEL_CARD.md").write_text(CARD + "\n*At release:* the evaluation.\n")
    git(root, "commit", "-qam", "card item")


def _data_card_item(inputs: dict[str, Any], root: Path) -> None:
    (root / "docs" / "data-card.md").write_text("# Data card\n\n## Data\n\n*At release:* counts.\n")
    git(root, "commit", "-qam", "data card item")


def _built_uncleared(inputs: dict[str, Any], root: Path) -> None:
    _edit_json(inputs["release"] / "config.json", release_check={"cleared": False})


def _dirty(inputs: dict[str, Any], root: Path) -> None:
    (root / "NOTICE").write_text("changed\n")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_old_card_link, "model.safetensors: It records model card https://huggingface.co/Inferensys/bulwark, not"),
        (_no_eval, "No evaluation of these weights is staged (--eval; evaluation/b1/)."),
        (_no_provenance, "No provenance of the checkpoint is staged (--provenance; provenance.json)."),
        (_template_item, "README.md: 1 items marked *at release*"),
        (_card_item, "MODEL_CARD.md: 1 items marked *at release*"),
        (_data_card_item, "docs/data-card.md: 1 items marked *at release*"),
        (_built_uncleared, "model.safetensors: Its own record says it is not cleared for release"),
        (_dirty, "The checkout has uncommitted changes (NOTICE)"),
    ],
)
def test_public_stage_refuses_what_a_public_repository_may_not_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: Any, expected: str
) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    inputs = bundle(tmp_path, cleared=True)
    change(inputs, root)
    out = tmp_path / "stage"
    problems = hf.stage("b1", out, **inputs, public=True, root=root)
    assert any(p.startswith(expected) for p in problems), problems
    assert not out.exists()


def test_an_e1_stage_whose_teacher_evaluation_scored_other_clips_is_refused_on_upload(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root, "e1")
    _teacher_on_other_clips({"teacher_evaluation": out / "evaluation" / "b1"})
    hf.write_sums(out)
    hub = FakeHub()
    problems = hf.upload(out, "e1", hub, root=root)
    assert any("so the card cannot set their scores side by side" in p for p in problems), problems
    assert hub.calls == []


def test_public_stage_needs_weights_of_the_released_b1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both cards name one B1: B1's weights must be built from it, E1 distilled from it."""
    clear(monkeypatch, "dmd")
    monkeypatch.setattr(hf, "B1_CHECKPOINT_SHA256", "cc" * 32)
    root = checkout(tmp_path, E1_RELEASED)
    problems = hf.stage("b1", tmp_path / "b1", **bundle(tmp_path / "b", cleared=True), public=True, root=root)
    assert any(p.startswith("model.safetensors: built from checkpoint b1b1b1b1b1b1, not the released Bulwark-B1 checkpoint (cccccccccccc") for p in problems), problems
    problems = hf.stage("e1", tmp_path / "e1", **bundle(tmp_path / "e", "e1", cleared=True), public=True, root=root)
    assert any(p.startswith("The graphs were distilled from B1 checkpoint b1b1b1b1b1b1, not the released Bulwark-B1 (cccccccccccc") for p in problems), problems
    assert not (tmp_path / "b1").exists() and not (tmp_path / "e1").exists()
    # A private stage takes them, and lists what keeps them private.
    out = tmp_path / "private-e1"
    assert hf.stage("e1", out, **bundle(tmp_path / "p", "e1", cleared=True), root=root) == []
    records = json.loads((out / "release-check.json").read_text())
    assert any(p.startswith("The graphs were distilled from B1 checkpoint") for p in hf.public_problems(out, "e1", records, root))


def test_public_e1_stage_needs_both_graphs_and_the_teacher_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    inputs = bundle(tmp_path, "e1", cleared=True)
    inputs["graphs"] = inputs["graphs"][:1]
    del inputs["teacher_evaluation"]
    problems = hf.stage("e1", tmp_path / "stage", **inputs, public=True, root=checkout(tmp_path, E1_RELEASED))
    assert "The release needs a 1-frame E1 graph (`bulwark export` --frames 1); none is staged." in problems
    assert "No evaluation of E1's B1 teacher is staged (--teacher-eval; evaluation/b1/)." in problems


# ---- the release set: the models a launch releases -----------------------------------------------


def test_the_release_set_is_read_from_launch_yaml(tmp_path: Path) -> None:
    released, problems = hf.release_set()  # the real one: B1 at least, from the first launch on
    assert "b1" in released and problems == []
    assert hf.release_set(checkout(tmp_path)) == (frozenset({"b1"}), [])
    assert hf.release_set(checkout(tmp_path / "with-e1", E1_RELEASED)) == (frozenset({"b1", "e1"}), [])
    wrong = ["released: [b1, x1]\n", "released: []\n", "released: b1\n", "released: [b1, b1]\n", "other: [b1]\n"]
    for number, text in enumerate(wrong):
        root = tmp_path / f"wrong-{number}"
        (root / "release" / "hf").mkdir(parents=True)
        (root / "release" / "hf" / "launch.yaml").write_text(text)
        assert hf.release_set(root) == (frozenset(), ["release/hf/launch.yaml: `released` must list one or more of b1, e1, each once"]), text
    assert hf.release_set(tmp_path / "nowhere")[0] == frozenset()  # without the file, nothing is released


def test_release_blocks_keep_or_drop_their_lines_by_the_release_set() -> None:
    text = "a\n{{#released e1}}\nwith E1\n{{/released}}\n{{#unreleased e1}}\nwithout E1\n\nstill\n{{/unreleased}}\nz\n"
    problems: list[str] = []
    assert hf.release_blocks(text, {"b1"}, "t.md", problems) == "a\nwithout E1\n\nstill\nz\n"
    assert hf.release_blocks(text, {"b1", "e1"}, "t.md", problems) == "a\nwith E1\nz\n"
    assert problems == []
    assert hf.release_blocks("{{#released x1}}\nx\n{{/released}}\n", {"b1"}, "t.md", problems) == ""
    assert problems == ["t.md: {{#released x1}} names no model (b1 or e1)"]


def test_a_release_block_that_holds_another_block_tag_is_refused_whatever_the_release_set() -> None:
    """A block runs to the next closing tag of its own kind. A wrong closing tag or a nested
    block would make a dropped block swallow the text up to it, unseen, so either is a
    problem under every release set, not only once the block is kept."""
    mismatched = (
        "{{#released e1}}\nLINK E1\n{{/unreleased}}\n{{#unreleased e1}}\nE1 in development\n{{/unreleased}}\n"
        "middle text that must stay\n{{#released e1}}\nLINK E1 again\n{{/released}}\ntail\n"
    )
    nested = "a\n{{#released e1}}\nouter\n{{#unreleased e1}}\ninner\n{{/unreleased}}\nafter\n{{/released}}\nz\n"
    for text, opening in ((mismatched, "{{#released e1}}"), (nested, "{{#released e1}}")):
        for released in ({"b1"}, {"b1", "e1"}):
            problems: list[str] = []
            hf.release_blocks(text, released, "t.md", problems)
            assert problems == [
                f"t.md: a {opening} block holds another block tag; blocks do not nest, and each closes with "
                "{{/released}} on a line of its own"
            ], (text, released)


def test_the_b1_card_says_e1_is_in_development_until_launch_yaml_releases_it(private_stages: dict[str, Path], tmp_path: Path) -> None:
    """The real b1.md: while launch.yaml releases B1 alone, E1 appears as in development in
    the first screen, the At a glance table and the related models, and its private
    repository is never linked. Listing e1 in launch.yaml, as the real one does, brings the
    three links back, each saying that E1 v0 scores well below B1."""
    out = private_stages["b1"]
    records = json.loads((out / "release-check.json").read_text())
    cards = {}
    for name, launch in (("b1-only", B1_ONLY), ("with-e1", E1_RELEASED)):
        shutil.copytree(REPO / "release", tmp_path / name / "release")
        for rel, text in launch.items():
            (tmp_path / name / rel).write_text(text)
        cards[name], problems = hf.render_card("b1", out, records, tmp_path / name)
        assert problems == [] and "{{" not in cards[name], name
    card = cards["b1-only"]
    assert E1 not in card
    assert "\n- **Smaller model:** a smaller ONNX version for CPU, Bulwark-E1, is in development. ONNX is the Open Neural Network Exchange format.\n" in card
    assert "\n| Smaller model | Bulwark-E1, an ONNX version for CPU, is in development |\n" in card
    assert "\n- **Smaller model:** Bulwark-E1, a smaller ONNX version for CPU, is in development and not released yet.\n" in card
    card = cards["with-e1"]
    assert card.count(f"[Bulwark-E1]({model_card_url('e1')})") == 3 and "in development" not in card
    lines = [line for line in card.splitlines() if f"[Bulwark-E1]({model_card_url('e1')})" in line]
    assert all("E1 v0" in line and "scores well below B1" in line for line in lines), lines
    first_screen = card.split("\n## ", 1)[0]
    assert "Its first version, E1 v0, scores well below B1: it is for trying CPU deployment, not for its accuracy." in first_screen
    # The stage of the real checkout follows the real launch.yaml.
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert (E1 in readme) is ("e1" in hf.release_set()[0])


def test_a_card_that_names_the_repository_of_a_model_that_is_not_released_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In the model's card (README.md) at any stage, since the templates' release blocks
    decide it; in the full card (MODEL_CARD.md), a GitHub document, for a public stage."""
    named = f"names {E1}, the repository of Bulwark-E1, which is not released (release/hf/launch.yaml); link it only once e1 is listed there"
    files = {"release/hf/b1.md": TEMPLATES["b1.md"] + "\nAlso [E1]({{ e1_url }}).\n{{#unreleased e1}}\nan unclosed block\n"}
    problems = hf.stage("b1", tmp_path / "stage", **bundle(tmp_path), root=checkout(tmp_path, files))
    assert f"README.md: {named}" in problems
    assert "release/hf/b1.md: '{{' or '}}' outside a placeholder or include" in problems  # the unclosed block
    assert not (tmp_path / "stage").exists()
    clear(monkeypatch, "dmd")
    card = CARD + "\nE1 is at https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx.\n"
    root = checkout(tmp_path / "card", card=card)
    problems = hf.stage("b1", tmp_path / "public", **bundle(tmp_path / "card", cleared=True), public=True, root=root)
    assert problems == [f"MODEL_CARD.md: {named}"] and not (tmp_path / "public").exists()
    # Once launch.yaml releases E1, both may link it.
    files["release/hf/b1.md"] = TEMPLATES["b1.md"] + "\nAlso [E1]({{ e1_url }}).\n"
    root = checkout(tmp_path / "with-e1", {**files, **E1_RELEASED}, card=card)
    assert hf.stage("b1", tmp_path / "public-2", **bundle(tmp_path / "with-e1", cleared=True), public=True, root=root) == []


def test_public_steps_refuse_a_model_that_is_not_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """E1 is not in the release set: a public stage, `upload --public` and `open` refuse it
    with one message and before any Hub call; a private stage and upload still work, as an
    untagged private preview that lists why."""
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    message = hf.not_released("e1")
    assert message.startswith("Bulwark-E1 is not released yet: release/hf/launch.yaml does not list e1 under `released`")
    assert hf.stage("e1", tmp_path / "public", **bundle(tmp_path / "p", "e1", cleared=True), public=True, root=root) == [message]
    assert not (tmp_path / "public").exists()
    out = staged(tmp_path, root, "e1", cleared=True)
    records = json.loads((out / "release-check.json").read_text())
    assert all(record["publishable"] for record in records.values())  # the weights are cleared; the model is not released
    assert hf.public_problems(out, "e1", records, root) == [message]
    hub = FakeHub()
    assert hf.upload(out, "e1", hub, public=True, root=root) == [message] and hub.calls == []
    assert hf.open_repo(out, "e1", hub, yes=True, root=root) == [message] and hub.calls == []
    assert hf.upload(out, "e1", hub, root=root) == []  # a private preview
    assert "create_tag" not in hub.methods() and hub.private is True
    upload = next(kwargs for method, _, kwargs in hub.calls if method == "upload_folder")
    assert upload["commit_message"].endswith("(private preview)") and message in upload["commit_description"]
    assert message in capsys.readouterr().out
    # B1 is released: the same steps take it.
    b1 = staged(tmp_path / "b", root, "b1", cleared=True, public=True)
    hub = FakeHub(exists=False)
    assert hf.upload(b1, "b1", hub, root=root) == [] and "create_tag" in hub.methods()
    assert hf.open_repo(b1, "b1", hub, yes=True, root=root) == [] and hub.private is False
    # B1's card names no other repository, so only B1's is asked about.
    assert {args[0] for method, args, _ in hub.calls if method == "repo_info"} == {B1}


TOGETHER = "Repositories whose cards link each other open together, in one run: `open --model b1 --model e1`."


def test_a_card_goes_public_only_once_every_repository_it_links_is_public(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Being listed in launch.yaml is not being public. Once e1 is listed, B1's card links
    E1's repository, which stays private until it is opened. Until the Hub reports it
    public, B1's card may go to B1's private repository, but `upload` to the public
    repository and an `open` of B1 alone refuse it, before anything is written; `open`
    says to open both in one run (RELEASING.md, launch order)."""
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path, E1_RELEASED)
    out = staged(tmp_path, root, "b1", cleared=True, public=True)
    assert E1 in (out / "README.md").read_text(encoding="utf-8")
    closed = [f"{E1} is not public yet; open it before publishing a card that links it (RELEASING.md, launch order)."]
    for state in (True, None):  # private, or not visible to this token
        hub = FakeHub(private=False, gated="auto")  # B1's public repository, whose main holds these weights
        hub.main = hub.add_commit({"model.safetensors": sibling(out / "model.safetensors", "model.safetensors")})
        hub.others[E1] = state
        assert hf.upload(out, "b1", hub, tag=None, public=True, root=root) == closed
        assert "upload_folder" not in hub.methods() and [args[0] for m, args, _ in hub.calls if m == "repo_info"] == [B1, E1]
    hub.others[E1] = False
    assert hf.upload(out, "b1", hub, tag=None, public=True, root=root) == [] and "upload_folder" in hub.methods()
    # The private repository takes the card, and `open` waits for E1's repository.
    hub = FakeHub(exists=False)
    hub.others[E1] = True
    assert hf.upload(out, "b1", hub, root=root) == [] and "create_tag" in hub.methods()
    assert E1 not in {args[0] for method, args, _ in hub.calls if method == "repo_info"}
    hub.calls.clear()
    assert hf.open_repo(out, "b1", hub, yes=True, root=root) == [*closed, TOGETHER]
    assert "update_repo_settings" not in hub.methods() and hub.private is True
    hub.others[E1] = False
    assert hf.open_repo(out, "b1", hub, yes=True, root=root) == [] and hub.private is False


# ---- open: both repositories in one run -----------------------------------------------------------


def upload_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hubs: Hubs | None = None, *, private: dict[str, str] | None = None
) -> tuple[Path, dict[str, Path], Hubs]:
    """Cleared public stages of both models from a checkout that releases both, each card
    linking the other model's repository, uploaded to their private repositories. A model
    in `private` gets a stage that may not go public: "item", an *at release* item in its
    card; "dataset", weights trained on a dataset whose release is undecided."""
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path, E1_RELEASED)
    private = private or {}
    for model in (m for m, kind in private.items() if kind == "item"):
        template = root / "release" / "hf" / f"{model}.md"
        template.write_text(template.read_text() + "\n*At release:* the quick start.\n")
        git(root, "commit", "-qam", "template item")
    stages = {
        model: staged(tmp_path / model, root, model, cleared=True, public=model not in private,
                      dataset=UNDECIDED if private.get(model) == "dataset" else "dmd")
        for model in ("b1", "e1")
    }  # fmt: skip
    assert E1 in (stages["b1"] / "README.md").read_text() and B1 in (stages["e1"] / "README.md").read_text()
    hubs = hubs or Hubs()
    for model, out in stages.items():
        assert hf.upload(out, model, hubs, root=root) == []
    assert all(hub.private is True for hub in hubs.repos.values())
    hubs.calls.clear()
    return root, stages, hubs


def test_one_open_run_opens_both_repositories_whose_cards_link_each_other(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, stages, hubs = upload_both(tmp_path, monkeypatch)
    assert hf.open_repos(stages, hubs, yes=True, root=root) == []
    assert all(hub.private is False and hub.gated == "auto" for hub in hubs.repos.values())
    changes = [(args[0], kwargs) for method, args, kwargs in hubs.calls if method == "update_repo_settings"]
    assert changes == [(repo, {"repo_type": "model", "gated": "auto", "private": False}) for repo in (B1, E1)]
    # Every check of both repositories ran before the first change: main, every revision.
    checks = hubs.calls[: hubs.methods().index("update_repo_settings")]
    assert {args[0] for method, args, _ in checks if method == "list_repo_commits"} == {B1, E1}
    # The cards' links to each other were not asked of the Hub: this run opens both.
    assert all(kwargs.get("files_metadata") for method, _, kwargs in checks if method == "repo_info")


def test_one_open_run_refuses_both_before_any_hub_call_when_a_stage_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, stages, hubs = upload_both(tmp_path, monkeypatch)
    _edit_readme(stages["e1"])
    problems = hf.open_repos(stages, hubs, yes=True, root=root)
    assert "Bulwark-E1: README.md: changed since staging" in problems and hubs.calls == []
    assert all(hub.private is True for hub in hubs.repos.values())


def _changed_main(hub: FakeHub) -> None:
    hub.main = hub.add_commit({**hub.commits[hub.main], "README.md": foreign("README.md")})  # an older card, say


def _public(hub: FakeHub) -> None:
    hub.private, hub.gated = False, "auto"


@pytest.mark.parametrize("failing", ["b1", "e1"])
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("history", "commit "),  # other weights in a revision: history_problems
        ("item", "README.md: 1 items marked *at release*"),  # the stage may not go public: check_stage's pending list
        ("dataset", "Publishing weights trained on these datasets is not cleared"),
        ("main", "Main does not mirror this stage"),  # remote_differences
        ("public", "is already public"),
    ],
)
def test_one_open_run_changes_nothing_when_a_check_of_either_model_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str, kind: str, expected: str
) -> None:
    """Every check runs for each model of the run, the second as well as the first: one
    failure of either model leaves both repositories as they were."""
    private = {failing: kind} if kind in ("item", "dataset") else None
    root, stages, hubs = upload_both(tmp_path, monkeypatch, private=private)
    hub = hubs.repos[MODEL_REPOS[failing]]
    {"history": _tagged_preview, "main": _changed_main, "public": _public}.get(kind, lambda hub: None)(hub)
    problems = hf.open_repos(stages, hubs, yes=True, root=root)
    name = "Bulwark-B1" if failing == "b1" else "Bulwark-E1"
    assert any(p.startswith(f"{name}: ") and expected in p for p in problems), problems
    assert "update_repo_settings" not in hubs.methods()
    assert all(h.private is (h is not hub or kind != "public") for h in hubs.repos.values())


def test_opening_either_model_alone_refuses_while_the_other_model_s_repository_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stages, hubs = upload_both(tmp_path, monkeypatch)
    for model, other in (("b1", E1), ("e1", B1)):
        closed = f"{other} is not public yet; open it before publishing a card that links it (RELEASING.md, launch order)."
        assert hf.open_repo(stages[model], model, hubs, yes=True, root=root) == [closed, TOGETHER]
    assert "update_repo_settings" not in hubs.methods()
    assert all(hub.private is True for hub in hubs.repos.values())


def test_open_asks_for_each_repository_id_before_changing_any(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, stages, hubs = upload_both(tmp_path, monkeypatch)
    answers = iter([B1, B1])  # the second answer is not E1's id
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert hf.open_repos(stages, hubs, root=root) == ["Not confirmed; nothing changed."]
    assert "update_repo_settings" not in hubs.methods()
    answers = iter([B1, E1])
    assert hf.open_repos(stages, hubs, root=root) == []
    assert all(hub.private is False for hub in hubs.repos.values())


class StaysPublic(FakeHub):
    """A Hub that opens the repository, then fails to make it private again."""

    def update_repo_settings(self, repo_id: str, **kwargs: Any) -> None:
        from huggingface_hub.errors import HfHubHTTPError

        if kwargs.get("private") is True and self.private is False:
            raise http_error(HfHubHTTPError, 503)
        super().update_repo_settings(repo_id, **kwargs)


def test_when_a_change_fails_midway_the_run_says_which_repositories_are_public(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All or none: E1's change fails after B1's took, so B1 is made private again too, and
    every repository of the run is accounted for. When B1 cannot be made private again,
    the run says it is public and how to recover."""
    failed = f"The settings change failed (HfHubHTTPError: 400), so {E1} was made private again; check its settings on the Hub."
    root, stages, hubs = upload_both(tmp_path / "back", monkeypatch, Hubs(e1=FailingSettings("fail")))
    assert hf.open_repos(stages, hubs, yes=True, root=root) == [
        failed,
        f"{B1} was opened in this run and was made private again: the run opens all of its repositories or none.",
        f"Public now: none. Private: {B1}, {E1}. Nothing is public. Fix the cause, then run `open --model b1 --model e1` again.",
    ]
    assert all(hub.private is True for hub in hubs.repos.values())

    root, stages, hubs = upload_both(tmp_path / "stuck", monkeypatch, Hubs(b1=StaysPublic(), e1=FailingSettings("fail")))
    assert hf.open_repos(stages, hubs, yes=True, root=root) == [
        failed,
        f"{B1} was opened in this run, and making it private again failed (HfHubHTTPError: 503): it is public, gated with "
        "automatic approval.",
        f"Public now: {B1}. Private: {E1}. A public card may now link a repository that is not public. Make each public "
        "repository above private in its settings on the Hub, or, once the Hub takes the change, open the others: "
        "`open --model e1`.",
    ]
    assert hubs.repos[B1].private is False and hubs.repos[E1].private is True

    # The first repository's change fails: the second is never touched.
    root, stages, hubs = upload_both(tmp_path / "first", monkeypatch, Hubs(b1=FailingSettings("fail")))
    assert hf.open_repos(stages, hubs, yes=True, root=root) == [
        f"The settings change failed (HfHubHTTPError: 400), so {B1} was made private again; check its settings on the Hub.",
        f"{E1} was not changed and stays private.",
        f"Public now: none. Private: {B1}, {E1}. Nothing is public. Fix the cause, then run `open --model b1 --model e1` again.",
    ]
    assert [args[0] for method, args, _ in hubs.calls if method == "update_repo_settings"] == [B1, B1]


class Interrupted(FakeHub):
    """A Hub where the change to public takes, and then the run is interrupted (Ctrl-C)."""

    def update_repo_settings(self, repo_id: str, **kwargs: Any) -> None:
        super().update_repo_settings(repo_id, **kwargs)
        if kwargs.get("private") is False:
            raise KeyboardInterrupt


def test_an_interrupt_midway_makes_every_repository_of_the_run_private_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """All or none, even when the run is stopped: E1's change is interrupted after B1's took,
    so both are made private again, the run says so, and the interrupt goes on."""
    root, stages, hubs = upload_both(tmp_path, monkeypatch, Hubs(e1=Interrupted()))
    capsys.readouterr()
    with pytest.raises(KeyboardInterrupt):
        hf.open_repos(stages, hubs, yes=True, root=root)
    assert all(hub.private is True for hub in hubs.repos.values())
    err = capsys.readouterr().err
    assert f"The settings change was interrupted (KeyboardInterrupt), so {E1} was made private again" in err
    assert f"{B1} was opened in this run and was made private again" in err
    assert f"Public now: none. Private: {B1}, {E1}." in err


def test_the_open_command_takes_each_model_once_with_its_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hf, "release_set", lambda root=hf.REPO: (frozenset({"b1", "e1"}), []))  # a launch of both
    for argv in (["--model", "b1", "--model", "b1"], ["--model", "b1", "--model", "e1", "--stage", str(tmp_path)]):
        with pytest.raises(SystemExit) as usage:
            hf.main(["open", *argv])
        assert usage.value.code == 2
    capsys.readouterr()
    missing = [str(tmp_path / "missing-b1"), str(tmp_path / "missing-e1")]
    argv = ["open", "--model", "b1", "--model", "e1", "--stage", missing[0], "--stage", missing[1], "--yes"]
    assert hf.main(argv, api=FakeHub()) == 1
    err = capsys.readouterr().err
    assert "Bulwark-B1: " in err and "run `stage --model b1` first" in err and "run `stage --model e1` first" in err


# ---- upload --------------------------------------------------------------------------------------


def test_upload_of_an_uncleared_stage_creates_a_private_repository_and_does_not_tag(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root, dataset=UNDECIDED)
    hub = FakeHub(exists=False)
    assert hf.upload(out, "b1", hub, root=root) == []
    assert hub.calls[1] == ("create_repo", (B1,), {"repo_type": "model", "private": True, "exist_ok": True})
    upload = next(kwargs for method, _, kwargs in hub.calls if method == "upload_folder")
    assert upload["repo_id"] == B1 and upload["repo_type"] == "model" and Path(upload["folder_path"]) == out
    assert upload["delete_patterns"] == "*" and upload["commit_message"].endswith("(private preview)")
    assert "create_tag" not in hub.methods() and hub.private is True


def test_each_stage_goes_to_its_own_model_s_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path, E1_RELEASED)
    out = staged(tmp_path, root, "e1", cleared=True)
    hub = FakeHub(exists=False)
    assert hf.upload(out, "e1", hub, root=root) == []
    assert {args[0] for _, args, _ in hub.calls if args} == {E1}
    assert next(kwargs for method, _, kwargs in hub.calls if method == "upload_folder")["repo_id"] == E1
    other = FakeHub()
    problems = hf.upload(out, "b1", other, root=root)  # an E1 stage is not a B1 stage
    assert problems and other.calls == []


def test_upload_of_a_cleared_but_unfinished_stage_says_what_keeps_it_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    inputs = bundle(tmp_path, cleared=True)
    del inputs["evaluation"]
    out = tmp_path / "stage"
    assert hf.stage("b1", out, **inputs, root=root) == []
    hub = FakeHub()
    assert hf.upload(out, "b1", hub, root=root) == []
    assert "create_tag" not in hub.methods()
    assert "No evaluation of these weights is staged" in capsys.readouterr().out


def test_upload_refuses_a_public_repository_unless_told_and_then_an_uncleared_stage(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root, dataset=UNDECIDED)
    hub = FakeHub(private=False, gated="auto")
    assert hf.upload(out, "b1", hub, root=root) == [f"{B1} is public; pass --public to upload to a public repository (only a stage `stage --public` would take)."]
    problems = hf.upload(out, "b1", hub, public=True, root=root)
    assert problems[-1] == f"{B1} is public, so only a stage that `stage --public` would take may go there."
    assert "upload_folder" not in hub.methods()


@pytest.mark.parametrize("gated", [False, "manual"])
def test_upload_refuses_a_public_repository_that_is_not_gated_with_automatic_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gated: Any) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(private=False, gated=gated)
    problems = hf.upload(out, "b1", hub, public=True, root=root)
    assert problems == [f"{B1} is public but not gated with automatic approval (gated: {gated!r}); fix its settings on the Hub first."]


def _edit_readme(out: Path) -> None:
    (out / "README.md").write_text("edited\n")


def _add_file(out: Path) -> None:
    (out / "extra.txt").write_text("x")


def _with_new_sums(change: Any) -> Any:
    def edit(out: Path) -> None:
        change(out)
        hf.write_sums(out)

    edit.__name__ = change.__name__
    return edit


def _manifest(out: Path) -> None:
    (out / "manifest.parquet").write_bytes(b"PAR1")


def _local_path(out: Path) -> None:
    (out / "config.json").write_text((out / "config.json").read_text().replace('"lora"', '"/home/u/lora"'))


def _labels_table(out: Path) -> None:
    (out / "labels.csv").write_text("clip,yawn\n")


def _dotfile(out: Path) -> None:
    (out / ".env").write_text("x=1\n")


def _pickle(out: Path) -> None:
    (out / "best.pt").write_bytes(b"\x80\x04")


def _orphan_bench(out: Path) -> None:
    (out / "bulwark-e1-image.onnx.bench.json").write_text("{}")


def _no_vjepa(out: Path) -> None:
    (out / "LICENSE-vjepa2").unlink()


def _limit_eval(out: Path) -> None:
    _edit_json(out / "evaluation" / "b1" / "metrics.json", meta={"limit_per_split": 96, "checkpoint": {"sha256": "b1" * 32}})


def _other_eval(out: Path) -> None:
    _edit_json(out / "evaluation" / "b1" / "metrics.json", meta={"limit_per_split": 0, "checkpoint": {"sha256": "cc" * 32}})


def _partial_eval(out: Path) -> None:
    coverage = metrics("b1" * 32)["coverage"]
    coverage["holdout"]["drive_and_act"]["predicted"] = 1
    _edit_json(out / "evaluation" / "b1" / "metrics.json", coverage=coverage)


def _other_provenance(out: Path) -> None:
    provenance_file(out / "provenance.json", "lora", code={"commit": "f" * 40, "dirty": False})


def _readme_body(out: Path) -> None:
    (out / "README.md").write_text((out / "README.md").read_text().replace("Bulwark-B1: test card", "Bulwark-B1: edited card"))


def _licence_text(out: Path) -> None:
    (out / "LICENSE").write_text("MIT License\n")


def _no_licence(out: Path) -> None:
    (out / "LICENSE").unlink()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_edit_readme, "README.md: changed since staging"),
        (_add_file, "extra.txt: not in SHA256SUMS (added after staging)"),
        (_with_new_sums(_manifest), "manifest.parquet: dataset material or a pickle"),
        (_with_new_sums(_local_path), "config.json: a local path"),
        (_with_new_sums(_labels_table), "labels.csv: not part of a Bulwark-B1 stage"),
        (_with_new_sums(_dotfile), ".env: a dotfile"),
        (_with_new_sums(_pickle), "best.pt: dataset material or a pickle"),
        (_with_new_sums(_orphan_bench), "bulwark-e1-image.onnx.bench.json: not part of a Bulwark-B1 stage"),
        (_with_new_sums(_no_vjepa), "a B1 release needs LICENSE-vjepa2"),
        (_with_new_sums(_limit_eval), "a --limit run (96 clips per split)"),
        (_with_new_sums(_other_eval), "an evaluation of other weights (sha256 cccccccccccc)"),
        (_with_new_sums(_partial_eval), "predicted only part of a split (holdout drive_and_act: 1 of 2 clips)"),
        (_with_new_sums(_other_provenance), "not the provenance of the staged weights' checkpoint"),
        (_with_new_sums(_readme_body), "README.md: not what `stage` writes for these artifacts"),
        (_with_new_sums(_licence_text), "LICENSE: not what `stage` writes for these artifacts"),
        (_with_new_sums(_no_licence), "LICENSE: missing; a Bulwark-B1 stage holds it"),
    ],
)
def test_upload_refuses_a_stage_changed_after_staging_before_any_hub_call(tmp_path: Path, change: Any, expected: str) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root)
    change(out)
    hub = FakeHub()
    problems = hf.upload(out, "b1", hub, root=root)
    assert any(expected in p for p in problems), problems
    assert hub.calls == []


def test_upload_refuses_a_stage_whose_templates_changed_since_staging(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root)
    (root / "release" / "hf" / "shared" / "product.md").write_text("Edited.\n")
    hub = FakeHub()
    assert hf.upload(out, "b1", hub, root=root) == ["README.md: not what `stage` writes for these artifacts from this checkout's files; stage again"]
    assert hub.calls == []


def test_an_edited_e1_config_is_refused(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root, "e1")
    _edit_json(out / "config.json", model_release="v9")
    hf.write_sums(out)
    problems = hf.upload(out, "e1", FakeHub(), root=root)
    assert "config.json: not what `stage` writes for these artifacts from this checkout's files; stage again" in problems


def test_scan_reads_credential_values_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = checkout(tmp_path)
    out = staged(tmp_path, root)
    (out / "NOTICE").write_text("uploaded by someuser\n")
    hf.write_sums(out)
    monkeypatch.setenv("KAGGLE_USERNAME", "someuser")
    assert "NOTICE: the value of $KAGGLE_USERNAME" in hf.upload(out, "b1", FakeHub(), root=root)


@pytest.mark.parametrize("change", ["cleared", "withdrawn"])
def test_upload_refuses_when_the_registry_changed_since_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    root = checkout(tmp_path)
    dataset = UNDECIDED if change == "cleared" else "dmd"
    out = staged(tmp_path, root, dataset=dataset)
    monkeypatch.setitem(BY_NAME, dataset, dataclasses.replace(BY_NAME[dataset], weights_release_ok=change == "cleared"))
    hub = FakeHub()
    problems = hf.upload(out, "b1", hub, root=root)
    assert any("its release check now differs from the one staged with it" in p for p in problems), problems
    assert hub.calls == []


def test_upload_of_a_cleared_stage_tags_the_uploaded_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(exists=False)
    assert hf.upload(out, "b1", hub, root=root) == []
    tag = next(call for call in hub.calls if call[0] == "create_tag")
    assert tag == ("create_tag", (B1,), {"repo_type": "model", "tag": "v0", "revision": hub.main, "tag_message": "Bulwark-B1 v0"})
    upload = next(kwargs for method, _, kwargs in hub.calls if method == "upload_folder")
    assert upload["commit_message"] == "Bulwark-B1 v0: model.safetensors" and "model.safetensors: cleared" in upload["commit_description"]


def test_upload_of_a_cleared_stage_to_the_public_gated_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(private=False, gated="auto")
    assert hf.upload(out, "b1", hub, public=True, root=root) == []
    assert "upload_folder" in hub.methods() and "update_repo_settings" not in hub.methods()


def test_upload_never_moves_a_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub()
    hub.tags["v0"] = "init"
    problems = hf.upload(out, "b1", hub, root=root)
    assert problems == [
        f"Tag v0 already exists in {B1}, and tags are never moved: upload with another --tag, or with --no-tag "
        "if main already holds these weights."
    ]
    assert hf.upload(out, "b1", hub, tag=None, root=root) == [] and hub.tags == {"v0": "init"}


def test_new_weights_in_the_public_repository_need_a_new_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(private=False, gated="auto")
    problems = hf.upload(out, "b1", hub, tag=None, public=True, root=root)
    assert problems[0].startswith("The upload would add or change weights in the public repository without a tag: model.safetensors")
    assert hf.upload(out, "b1", hub, tag="v0.1", public=True, root=root) == []
    assert hf.upload(out, "b1", hub, tag=None, public=True, root=root) == []  # main already holds these weights: a card fix


def test_upload_refuses_to_remove_weights_from_the_public_repository_unless_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(private=False, gated="auto")
    hub.main = hub.add_commit({"old.onnx": foreign("old.onnx")})
    problems = hf.upload(out, "b1", hub, public=True, root=root)
    assert problems == ["The upload would remove weights from the public repository: old.onnx. Pass --allow-weight-deletion if that is intended."]
    assert hf.upload(out, "b1", hub, public=True, allow_weight_deletion=True, root=root) == []


def test_upload_to_the_private_repository_warns_of_weights_that_open_will_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub()
    hub.main = hub.add_commit({"preview.safetensors": foreign("preview.safetensors")})
    assert hf.upload(out, "b1", hub, root=root) == []
    assert "preview.safetensors (weights that are not this release's cleared weights)" in capsys.readouterr().out


def test_upload_refuses_another_endpoint_and_a_dirty_checkout_for_a_cleared_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub()
    hub.endpoint = "https://mirror.example.org"
    assert hf.upload(out, "b1", hub, root=root) == [f"The Hub client talks to https://mirror.example.org, not {hf.ENDPOINT}; nothing was sent."]
    hub.endpoint = hf.ENDPOINT
    (root / "examples" / "run.py").write_text("edited\n")  # the registry that cleared the stage may be edited too
    problems = hf.upload(out, "b1", hub, root=root)
    assert any(p.startswith("The checkout has uncommitted changes (examples/run.py)") for p in problems) and hub.calls == []


def test_command_line_reports_hub_errors_with_the_login_steps(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, OfflineModeIsEnabled

    def failing(error: Exception) -> FakeHub:
        class Failing(FakeHub):
            def create_collection(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
                raise error

        return Failing()

    for error, expected in (
        (http_error(HfHubHTTPError, 401), "HfHubHTTPError: 401"),
        (OfflineModeIsEnabled("offline"), "HF_HUB_OFFLINE is set"),
        (http_error(GatedRepoError, 403), "GatedRepoError: 403"),
    ):
        assert hf.main(["collection"], api=failing(error)) == 1
        err = capsys.readouterr().err
        assert expected in err and ("`hf auth login`" in err) is (expected != "HF_HUB_OFFLINE is set")
    monkeypatch.setattr(hf, "release_set", lambda root=hf.REPO: (frozenset({"b1"}), []))  # the first launch, B1 alone
    assert hf.main(["open", "--model", "e1", "--stage", str(tmp_path / "missing")], api=FakeHub()) == 1
    assert "Bulwark-E1 is not released yet: release/hf/launch.yaml does not list e1" in capsys.readouterr().err
    assert hf.main(["open", "--model", "b1", "--stage", str(tmp_path / "missing")], api=FakeHub()) == 1
    assert "run `stage --model b1` first" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:
        hf.main(["stage", "--out", str(tmp_path / "x")])  # --model is required
    assert usage.value.code == 2


# ---- open --------------------------------------------------------------------------------------


@pytest.fixture
def uploaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, FakeHub]:
    """A cleared B1 stage from a committed checkout, uploaded to a new private repository."""
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FakeHub(exists=False)
    assert hf.upload(out, "b1", hub, root=root) == []
    hub.calls.clear()
    return root, out, hub


def test_open_turns_on_automatic_gating_and_makes_the_repository_public(uploaded: tuple[Path, Path, FakeHub]) -> None:
    root, out, hub = uploaded
    assert hf.open_repo(out, "b1", hub, yes=True, root=root) == []
    settings = [call for call in hub.calls if call[0] == "update_repo_settings"]
    assert hf.GATED == "auto"  # access requests are approved as soon as they are sent
    assert settings == [("update_repo_settings", (B1,), {"repo_type": "model", "gated": "auto", "private": False})]
    assert hub.methods()[-2:] == ["update_repo_settings", "repo_info"]
    scanned = {call[2]["revision"] for call in hub.calls if call[0] == "list_repo_commits"}
    assert scanned == {hub.main}  # main and the v0 tag point at the uploaded commit
    assert hub.private is False and hub.gated == "auto"


def test_open_calls_update_repo_settings_as_huggingface_hub_defines_it() -> None:
    """The settings call `open` makes binds to huggingface_hub's own signature (2.3.0:
    `gated: Literal["auto", "manual", False] | None`, `private`, `repo_type`), and so do the
    collection calls."""
    import inspect

    from huggingface_hub.hf_api import HfApi

    # The class itself: the `offline` fixture replaces only the huggingface_hub.HfApi name.
    signature = inspect.signature(HfApi.update_repo_settings)
    signature.bind(object(), B1, repo_type="model", gated=hf.GATED, private=False)
    assert "'auto'" in str(signature.parameters["gated"].annotation)
    inspect.signature(HfApi.create_collection).bind(object(), "t", namespace="n", description="d", private=True, exists_ok=True)
    inspect.signature(HfApi.add_collection_item).bind(object(), "s", "i", "model", note="n", exists_ok=True)
    inspect.signature(HfApi.update_collection_item).bind(object(), "s", "o", note="n")
    inspect.signature(HfApi.update_collection_item).bind(object(), "s", "o", position=1)
    inspect.signature(HfApi.update_collection_metadata).bind(object(), "s", title="t", description="d")
    inspect.signature(HfApi.update_collection_metadata).bind(object(), "s", private=False)
    inspect.signature(HfApi.delete_collection_item).bind(object(), "s", "o")


def test_the_access_form_is_reviewed_in_the_staged_readme_since_only_open_gates(uploaded: tuple[Path, Path, FakeHub]) -> None:
    """Upload creates the repository private and never gated, so the private page shows no
    access-request form before `open`; RELEASING has it reviewed in the staged README.md."""
    _, out, hub = uploaded
    assert hub.private is True and hub.gated is False
    header = (out / "README.md").read_text(encoding="utf-8").split("---\n", 2)[1]
    assert yaml.safe_load(header)["extra_gated_fields"] == hf.GATE_FIELDS
    releasing = " ".join((REPO / "RELEASING.md").read_text(encoding="utf-8").split())
    assert "the private page does not show the form" in releasing and "front matter of the staged README.md" in releasing


def test_open_asks_for_the_repository_id(uploaded: tuple[Path, Path, FakeHub], monkeypatch: pytest.MonkeyPatch) -> None:
    root, out, hub = uploaded
    monkeypatch.setattr("builtins.input", lambda prompt: E1)  # the other model's repository
    assert hf.open_repo(out, "b1", hub, root=root) == ["Not confirmed; nothing changed."]
    assert "update_repo_settings" not in hub.methods() and hub.private is True
    monkeypatch.setattr("builtins.input", lambda prompt: B1)
    assert hf.open_repo(out, "b1", hub, root=root) == [] and hub.private is False


def test_open_refuses_an_uncleared_stage_without_calling_the_hub(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    hub = FakeHub()
    problems = hf.open_repo(staged(tmp_path, root, dataset=UNDECIDED), "b1", hub, yes=True, root=root)
    assert any(f"{UNDECIDED} (not determined)" in p for p in problems) and hub.calls == []


def test_open_refuses_another_endpoint_and_a_dirty_checkout(uploaded: tuple[Path, Path, FakeHub]) -> None:
    root, out, hub = uploaded
    hub.endpoint = "https://mirror.example.org"
    assert hf.open_repo(out, "b1", hub, yes=True, root=root) == [f"The Hub client talks to https://mirror.example.org, not {hf.ENDPOINT}; nothing was sent."]
    hub.endpoint = hf.ENDPOINT
    (root / "examples" / "run.py").write_text("edited\n")  # the registry that cleared the stage may be edited too
    problems = hf.open_repo(out, "b1", hub, yes=True, root=root)
    assert any(p.startswith("The checkout has uncommitted changes (examples/run.py)") for p in problems)
    assert hub.calls == [] and hub.private is True


def test_open_refuses_a_repository_that_is_already_public(uploaded: tuple[Path, Path, FakeHub]) -> None:
    root, out, hub = uploaded
    hub.private, hub.gated = False, "auto"
    assert hf.open_repo(out, "b1", hub, yes=True, root=root) == [f"{B1} is already public (gated: 'auto'); open changes nothing."]


def _missing_file(hub: FakeHub) -> None:
    hub.main = hub.add_commit({k: v for k, v in hub.commits[hub.main].items() if k != "NOTICE"})


def _other_weights_in_head(hub: FakeHub) -> None:
    hub.main = hub.add_commit({**hub.commits[hub.main], "model.safetensors": foreign("model.safetensors")})


def _older_graph(hub: FakeHub) -> None:
    old = hub.add_commit({"bulwark-e1-video.onnx": foreign("bulwark-e1-video.onnx")}, parent="init")
    hub.history[hub.main] = [hub.main, old, "init"]


def _tagged_preview(hub: FakeHub) -> None:
    hub.tags["preview"] = hub.add_commit({"preview.safetensors": foreign("preview.safetensors")}, parent="init")


def _pull_request(hub: FakeHub) -> None:
    hub.pulls["refs/pr/1"] = hub.add_commit({"labels.csv": SimpleNamespace(rfilename="labels.csv", blob_id="1" * 40, lfs=None)}, parent="init")


def _dotfile_commit(hub: FakeHub) -> None:
    hub.tags["x"] = hub.add_commit({".env": SimpleNamespace(rfilename=".env", blob_id="1" * 40, lfs=None)}, parent="init")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_missing_file, "NOTICE: staged, not in the repository"),
        (_other_weights_in_head, "model.safetensors: differs from the staged file"),
        (_older_graph, "bulwark-e1-video.onnx (weights that are not this release's cleared weights)"),
        (_tagged_preview, "preview.safetensors (weights that are not this release's cleared weights)"),
        (_pull_request, "labels.csv (not a file of a Bulwark-B1 stage)"),
        (_dotfile_commit, ".env (a dotfile)"),
    ],
)
def test_open_refuses_a_repository_that_differs_from_the_stage_or_holds_other_files_in_any_revision(
    uploaded: tuple[Path, Path, FakeHub], change: Any, expected: str
) -> None:
    root, out, hub = uploaded
    change(hub)
    problems = hf.open_repo(out, "b1", hub, yes=True, root=root)
    assert any(expected in p for p in problems), problems
    assert "update_repo_settings" not in hub.methods() and hub.private is True


def test_open_makes_the_repository_private_again_when_gating_did_not_take(uploaded: tuple[Path, Path, FakeHub]) -> None:
    root, out, hub = uploaded
    hub.takes_gating = False
    problems = hf.open_repo(out, "b1", hub, yes=True, root=root)
    assert problems == [f"The settings change reported private=False, gated=False, so {B1} was made private again; check its settings on the Hub."]
    settings = [call[2] for call in hub.calls if call[0] == "update_repo_settings"]
    assert settings == [{"repo_type": "model", "gated": "auto", "private": False}, {"repo_type": "model", "gated": None, "private": True}]
    assert hub.private is True


class FailingSettings(FakeHub):
    """A Hub that applies a settings change and then fails: the change made public without
    gating (`fail`), the check after it (`fail_check`), or also the change back (`fail_back`)."""

    def __init__(self, fail: str) -> None:
        super().__init__()
        self.fail = fail

    def update_repo_settings(self, repo_id: str, **kwargs: Any) -> None:
        from huggingface_hub.errors import HfHubHTTPError

        if self.fail == "fail_back" and kwargs.get("private") is True:
            raise http_error(HfHubHTTPError, 503)
        super().update_repo_settings(repo_id, **{**kwargs, "gated": None if self.fail != "fail_check" else kwargs.get("gated")})
        if self.fail in ("fail", "fail_back") and kwargs.get("private") is False:
            raise http_error(HfHubHTTPError, 400)

    def repo_info(self, repo_id: str, **kwargs: Any) -> SimpleNamespace:
        if self.fail == "fail_check" and any(method == "update_repo_settings" for method in self.methods()):
            raise TimeoutError("read timed out")
        return super().repo_info(repo_id, **kwargs)


@pytest.mark.parametrize("fail", ["fail", "fail_check"])
def test_open_makes_the_repository_private_again_when_the_change_or_its_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FailingSettings(fail)
    assert hf.upload(out, "b1", hub, root=root) == []
    problems = hf.open_repo(out, "b1", hub, yes=True, root=root)
    expected = "HfHubHTTPError: 400" if fail == "fail" else "TimeoutError: read timed out"
    assert problems == [f"The settings change failed ({expected}), so {B1} was made private again; check its settings on the Hub."]
    assert hub.private is True and hub.calls[-1][2]["private"] is True


def test_open_says_so_when_the_repository_cannot_be_made_private_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch, "dmd")
    root = checkout(tmp_path)
    out = staged(tmp_path, root, cleared=True)
    hub = FailingSettings("fail_back")
    assert hf.upload(out, "b1", hub, root=root) == []
    problems = hf.open_repo(out, "b1", hub, yes=True, root=root)
    assert len(problems) == 1 and "It may be public without gating: check its settings on the Hub now." in problems[0]
    assert hub.private is False and hub.gated is False  # what the message warns of


def test_the_fake_hub_has_no_destructive_methods() -> None:
    """Apart from removing a collection item, which `collection` does only for the items of
    a model that is not released (test_collection_defers_the_items_of_a_model_that_is_not_released)."""
    for name in ("delete_repo", "delete_tag", "delete_branch", "super_squash_history", "delete_file", "delete_folder",
                 "delete_collection"):  # fmt: skip
        assert not hasattr(FakeHub(), name), name


# ---- collection and organisation card -----------------------------------------------------------


def test_collection_is_created_private_with_the_items_and_notes_in_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = checkout(tmp_path)
    spec, problems = hf.collection_spec(root)
    assert problems == [] and spec["title"] == "Bulwark: driver drowsiness and distraction detection models"
    assert len(spec["title"]) <= 60 and len(spec["description"]) <= 150 and all(len(i["note"]) <= 500 for i in spec["items"])
    hub = FakeHub()
    assert hf.collection(hub, root=root) == []
    assert hub.calls[0] == ("create_collection", (spec["title"],), {
        "namespace": "Inferensys", "description": spec["description"], "private": True, "exists_ok": True,
    })  # fmt: skip
    collection = hub.collection
    assert collection is not None and collection.private is True
    assert [(i.item_type, i.item_id) for i in collection.items] == [("model", B1), ("paper", "2008.12085"), ("paper", "2603.14482")]
    # E1 and its backbone's paper wait for E1's release; the demo Space and the AUC paper
    # wait for launch: the paper's page appears once a public card cites it.
    assert all(i.note for i in collection.items)
    assert all(kwargs.get("private") in (None, True) for method, _, kwargs in hub.calls if method == "update_collection_metadata")
    printed = capsys.readouterr().out
    assert "Record its address in release/hf/collection.yaml (url: https://huggingface.co/collections/Inferensys/bulwark-0123)" in printed
    assert f"Deferred until their model is released (release/hf/launch.yaml): model {E1}, paper 2205.14756" in printed
    assert "delete_collection_item" not in hub.methods()  # nothing deferred was there
    hub.calls.clear()
    assert hf.collection(hub, root=root) == []  # again: nothing to add, nothing moved
    assert "add_collection_item" not in hub.methods() and "update_collection_item" not in hub.methods()


def test_collection_refuses_a_public_collection_and_goes_public_only_after_the_repositories(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    hub = FakeHub(private=True)
    problems = hf.collection(hub, public=True, root=root)
    # Only the released models' repositories: B1 launches alone.
    assert problems == [f"{B1} is not public yet; open it before the collection (RELEASING.md, launch order)."]
    assert [args[0] for method, args, _ in hub.calls if method == "repo_info"] == [B1]
    assert hf.collection(hub, public=True, root=checkout(tmp_path / "with-e1", E1_RELEASED)) == [
        f"{B1} is not public yet; open it before the collection (RELEASING.md, launch order).",
        f"{E1} is not public yet; open it before the collection (RELEASING.md, launch order).",
    ]
    assert "create_collection" not in hub.methods()
    hub.private = False
    assert hf.collection(hub, public=True, root=root) == []
    collection = hub.collection
    assert collection is not None and collection.private is False and collection.items[-1].item_id == "1901.09097"
    assert hub.methods()[-1] == "update_collection_metadata" and hub.calls[-1][2] == {"title": None, "description": None, "private": False}
    assert hf.collection(hub, root=root) == ["The collection Inferensys/bulwark-0123 is public; pass --public to update a public collection."]


def test_collection_public_run_removes_deferred_items_before_going_public(tmp_path: Path) -> None:
    """The launch itself: the private collection holds E1 and its backbone's paper from
    before B1 launched alone. `collection --public` removes both before it makes the
    collection public, so they are never public, and adds the AUC paper."""
    hub = FakeHub(private=False)  # B1's repository is public
    hub.collection = SimpleNamespace(slug="Inferensys/bulwark-0123", private=True, items=[
        SimpleNamespace(item_object_id=f"o{n}", item_id=item_id, item_type=kind, note=None, position=n)
        for n, (kind, item_id) in enumerate([("model", B1), ("model", E1), ("paper", "2008.12085"), ("paper", "2603.14482"),
                                             ("paper", "2205.14756")])
    ])  # fmt: skip
    assert hf.collection(hub, public=True, root=checkout(tmp_path, with_space())) == []
    deletions = [n for n, (method, _, _) in enumerate(hub.calls) if method == "delete_collection_item"]
    public = [n for n, (method, _, kwargs) in enumerate(hub.calls) if method == "update_collection_metadata" and kwargs["private"] is False]
    assert len(deletions) == 2 and len(public) == 1 and max(deletions) < public[0]
    assert [(i.item_type, i.item_id) for i in hub.collection.items] == [
        ("space", SPACE), ("model", B1), ("paper", "2008.12085"), ("paper", "2603.14482"), ("paper", "1901.09097"),
    ]
    assert hub.collection.private is False


def test_collection_adds_the_demo_space_first_at_launch(tmp_path: Path) -> None:
    """The demo Space is the first item of collection.yaml: a space item, by its Hub id, marked
    at_launch. Private runs leave it out; `collection --public` adds it as a space
    (huggingface_hub `add_collection_item` item_type "space") in first place, so the
    organisation page previews the Space, B1 and E1."""
    root = checkout(tmp_path, {**E1_RELEASED, **with_space()})
    spec, problems = hf.collection_spec(root)
    space = spec["items"][0]
    assert problems == [] and space == {"type": "space", "id": SPACE, "note": space["note"], "at_launch": True, "for": None}
    assert "AI-generated example clips" in space["note"] and "Research preview" in space["note"]
    assert "Bulwark-B1 v0 or Bulwark-E1 v0" in space["note"] and "non-commercial research only" in space["note"]
    hub = FakeHub()
    assert hf.collection(hub, root=root) == []
    assert SPACE not in [i.item_id for i in hub.collection.items] and hub.collection.private is True
    hub.private = False  # B1's and E1's repositories are public
    hub.calls.clear()
    assert hf.collection(hub, public=True, root=root) == []
    assert [args[1:] for method, args, _ in hub.calls if method == "add_collection_item"] == [(SPACE, "space"), ("1901.09097", "paper")]
    items = hub.collection.items
    assert [(i.item_type, i.item_id) for i in items][:3] == [("space", SPACE), ("model", B1), ("model", E1)]
    assert items[0].note == space["note"] and hub.collection.private is False
    hub.calls.clear()
    assert hf.collection(hub, public=True, root=root) == []  # again: nothing to add, nothing moved
    assert "add_collection_item" not in hub.methods() and "update_collection_item" not in hub.methods()


def test_collection_public_waits_for_the_demo_space(tmp_path: Path) -> None:
    """At launch the Space must exist and be public before `collection --public` changes
    anything: the org needs a paid plan for a Gradio Space, so it may not exist yet."""
    root = checkout(tmp_path, {**E1_RELEASED, **with_space()})
    for state in (None, True):  # missing, then private
        hub = FakeHub(private=False)  # B1's and E1's repositories are public
        hub.others[SPACE] = state
        problems = hf.collection(hub, public=True, root=root)
        assert problems == [
            f"The Space {SPACE} does not exist or is not public; create it (RELEASING.md, launch order), or change "
            f"or remove its item in {hf.TEMPLATES}/collection.yaml, before collection --public."
        ]
        assert "create_collection" not in hub.methods() and hub.collection is None
        assert ("repo_info", (SPACE,), {"repo_type": "space"}) in [(m, a[:1], {"repo_type": k.get("repo_type")}) for m, a, k in hub.calls]


def test_collection_leaves_other_items_in_place_and_fixes_notes_and_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = checkout(tmp_path, E1_RELEASED)
    hub = FakeHub()
    hub.collection = SimpleNamespace(slug="Inferensys/bulwark-0123", private=True, items=[
        SimpleNamespace(item_object_id="old", item_id="someone/other", item_type="model", note=None, position=0),
        SimpleNamespace(item_object_id="e1", item_id=E1, item_type="model", note="old note", position=1),
    ])  # fmt: skip
    assert hf.collection(hub, root=root) == []
    items = hub.collection.items
    assert [i.item_id for i in items][:2] == [B1, E1] and items[1].note.startswith("Bulwark-E1, the CPU model")
    assert any(i.item_id == "someone/other" for i in items)
    assert "Left in place (not in collection.yaml): model someone/other" in capsys.readouterr().out


def test_collection_defers_the_items_of_a_model_that_is_not_released(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A collection made before E1 left the launch holds E1 and its backbone's paper: a run
    removes those two, prints each removal, and leaves every other item in place. Once
    launch.yaml releases E1, they come back in their place."""
    hub = FakeHub()
    hub.collection = SimpleNamespace(slug="Inferensys/bulwark-0123", private=True, items=[
        SimpleNamespace(item_object_id=f"o{n}", item_id=item_id, item_type=kind, note=None, position=n)
        for n, (kind, item_id) in enumerate([("model", B1), ("model", E1), ("paper", "2008.12085"), ("paper", "2603.14482"),
                                             ("paper", "2205.14756"), ("model", "someone/other")])
    ])  # fmt: skip
    assert hf.collection(hub, root=checkout(tmp_path)) == []
    removed = [args[1] for method, args, _ in hub.calls if method == "delete_collection_item"]
    assert removed == ["o1", "o4"]  # E1 and the EfficientViT paper, nothing else
    assert [(i.item_type, i.item_id) for i in hub.collection.items] == [
        ("model", B1), ("paper", "2008.12085"), ("paper", "2603.14482"), ("model", "someone/other"),
    ]
    printed = capsys.readouterr().out
    assert f"Removed model {E1}: it waits for Bulwark-E1, which is not released (release/hf/launch.yaml)." in printed
    assert "Removed paper 2205.14756: it waits for Bulwark-E1" in printed
    assert "Left in place (not in collection.yaml): model someone/other" in printed
    hub.calls.clear()
    assert hf.collection(hub, root=checkout(tmp_path / "with-e1", E1_RELEASED)) == []
    assert "delete_collection_item" not in hub.methods()
    assert [(i.item_type, i.item_id) for i in hub.collection.items][:5] == [
        ("model", B1), ("model", E1), ("paper", "2008.12085"), ("paper", "2603.14482"), ("paper", "2205.14756"),
    ]


def test_collection_spec_enforces_the_hub_s_limits(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    path = root / "release" / "hf" / "collection.yaml"
    data = yaml.safe_load(path.read_text())
    data["description"] = "x" * 151
    data["items"][0]["note"] = "y" * 501
    n = len(data["items"])  # the real file's items; the bad ones follow
    dmd = 1 + next(i for i, item in enumerate(data["items"]) if item.get("id") == "2008.12085")
    data["items"].append({"type": "dataset", "id": "x"})
    data["items"].append({"type": "paper", "id": "1234.5678", "for": "x1"})
    # The DMD paper again, deferred with E1: a run would delete it and then miss it.
    data["items"].append({"type": "paper", "id": "2008.12085", "for": "e1"})
    data["items"].append({"type": "space"})  # a space needs its id
    path.write_text(yaml.safe_dump(data))
    _, problems = hf.collection_spec(root)
    assert "collection.yaml: the description is longer than 150 characters" in problems
    assert "collection.yaml: the note of item 1 is longer than 500 characters" in problems
    assert f"collection.yaml: item {n + 1} needs type model (with model b1 or e1), space or paper (with id)" in problems
    assert f"collection.yaml: item {n + 2} is `for` x1, which is not a model (b1 or e1)" in problems
    assert f"collection.yaml: item {n + 3} repeats item {dmd} (paper 2008.12085)" in problems
    assert f"collection.yaml: item {n + 4} needs type model (with model b1 or e1), space or paper (with id)" in problems
    hub = FakeHub()
    assert f"collection.yaml: item {n + 3} repeats item {dmd} (paper 2008.12085)" in hf.collection(hub, root=root) and hub.calls == []


def test_org_card_writes_the_space_readme_and_prints_the_settings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = checkout(tmp_path, COLLECTION_PENDING)
    out = tmp_path / "org"
    assert hf.org_card(out, root=root) == []
    assert hf._files(out) == ["README.md"]
    text = (out / "README.md").read_text()
    metadata = yaml.safe_load(text.split("---\n")[1])
    assert metadata == {"title": "README", "sdk": "static", "pinned": False}
    body = text.split("---\n", 2)[2]
    assert body.lstrip().startswith("# Inferensys\n") and model_card_url("b1") in body
    # With B1 alone in launch.yaml, the card names E1 as in development and never links its private repository.
    assert E1 not in body and "- Bulwark-E1: a smaller version for CPU, in the ONNX (Open Neural Network Exchange) format, in development." in body and "Both" not in body
    assert "{{" not in body and "hello@inferensys.com" in body
    printed = capsys.readouterr().out
    assert "details: Inference Systems (Inferensys) builds production AI" in printed and "website: https://inferensys.com" in printed
    assert "Not ready to publish: 1 *at release* items" in printed  # the collection's address is not known yet
    # Offline, it cannot ask the Hub whether the repositories it links are public: it lists them.
    assert f"It links {B1}: make the Space public, or update it, only once each is public" in printed
    assert hf.org_card(out, root=root) == [f"{out} exists and is not an empty folder; stage into a new one"]
    # Once collection.yaml records the address, the card links to it and nothing is pending.
    recorded = yaml.safe_load((REPO / "release" / "hf" / "collection.yaml").read_text())["url"]
    assert recorded and recorded.startswith("https://huggingface.co/collections/Inferensys/")
    capsys.readouterr()
    assert hf.org_card(tmp_path / "org-2", root=checkout(tmp_path / "recorded")) == []
    assert recorded in (tmp_path / "org-2" / "README.md").read_text()
    assert "Not ready to publish" not in capsys.readouterr().out
    # Once launch.yaml releases E1, the card links it and calls both research previews.
    assert hf.org_card(tmp_path / "org-3", root=checkout(tmp_path / "with-e1", E1_RELEASED)) == []
    body = (tmp_path / "org-3" / "README.md").read_text()
    assert f"- [Bulwark-E1]({model_card_url('e1')}): a smaller version" in body and "in development" not in body
    assert "Its first version, E1 v0, scores well below Bulwark-B1: use it to try CPU deployment, and Bulwark-B1 for research." in body
    # A link to E1's repository outside a release block is refused while B1 launches alone.
    template = (REPO / "release" / "hf" / "org-card.md").read_text() + "\nAlso [E1]({{ e1_url }}).\n"
    root = checkout(tmp_path / "linked", {"release/hf/org-card.md": template})
    assert hf.org_card(tmp_path / "org-4", root=root) == [
        f"README.md: names {E1}, the repository of Bulwark-E1, which is not released (release/hf/launch.yaml); "
        "link it only once e1 is listed there"
    ]
    assert not (tmp_path / "org-4").exists()


def test_the_release_cli_writes_the_card_of_the_model_s_own_repository(tmp_path: Path) -> None:
    torch.manual_seed(0)
    provenance = {"job": "distill", "commercial_only": False, "code": {"commit": COMMIT, "dirty": False},
                  "datasets": {"dmd": {"licence": "t", "commercial_ok": False, "splits": {"train": 4}}}}  # fmt: skip
    checkpoint = save_checkpoint(tmp_path / "e1.pt", BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, "supervised_labels": ["yawn"], "provenance": provenance})
    result = runner.invoke(app, ["release", str(checkpoint), str(tmp_path / "release")])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "release" / "config.json").read_text())["model_card"] == model_card_url("e1")
    shutil.rmtree(tmp_path / "release")
