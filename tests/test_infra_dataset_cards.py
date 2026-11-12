"""infra/kaggle/dataset_cards.py: Kaggle metadata for the private training datasets."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from PIL import Image

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.registry import DATASETS
from bulwark.data.sources.base import clip_stride

REPO = Path(__file__).parents[1]


@pytest.fixture
def cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "dataset_cards", REPO / "infra" / "kaggle" / "dataset_cards.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_cards_fit_kaggle_limits_and_point_to_the_contact_page(cards) -> None:  # type: ignore[no-untyped-def]
    for slug, name in cards.SLUGS.items():
        card = cards.card(slug)
        assert 6 <= len(card["title"]) <= 50 and 20 <= len(card["subtitle"]) <= 80
        assert (
            card["isPrivate"] is True
            and card["licenses"] == [{"name": "other"}]
            and card["keywords"]
        )
        assert "https://inferensys.com/contact" in card["description"]
        assert "commercial" in card["description"].split("## Contact")[1]
        assert "0 clips" not in card["description"]  # no manifest here: no count line
        if name:
            spec = next(d for d in DATASETS if d.name == name)
            assert (
                spec.citation.strip() in card["description"] and spec.licence in card["description"]
            )


def test_every_card_has_sources_and_an_accepted_update_frequency(cards) -> None:  # type: ignore[no-untyped-def]
    accepted = {
        "not specified",
        "never",
        "annually",
        "quarterly",
        "monthly",
        "weekly",
        "daily",
        "hourly",
    }
    for slug, name in cards.SLUGS.items():
        card = cards.card(slug)
        assert card["expectedUpdateFrequency"] in accepted
        sources = card["userSpecifiedSources"]
        names = [name] if name else [n for n in cards.SLUGS.values() if n]
        for spec in (d for d in DATASETS if d.name in names):
            assert spec.title in sources and spec.url in sources
            if name:  # the shard cards give the paper and the licence
                assert spec.licence in sources and "Paper: " in sources
    assert cards.card("bulwark-meta")["expectedUpdateFrequency"] == "monthly"
    assert cards.card("bulwark-dmd")["expectedUpdateFrequency"] == "never"
    meta = cards.card("bulwark-meta")["userSpecifiedSources"]
    assert "MediaPipe Face Landmarker" in meta and "subject" in meta and "split_policy" in meta
    mrl = cards.card("bulwark-mrl")["userSpecifiedSources"]
    assert "doi.org/10.1007/978-3-030-03801-4_38" in mrl and "identical" in mrl
    # The check was against an Internet Archive capture, and the sources say so.
    assert "mrlEyes_2018_01.zip as captured by the Internet Archive (2026-04-23" in mrl
    # Checkpoints build on third-party pretrained weights, which the sources name.
    assert "V-JEPA 2.1" in meta and "EfficientViT-B1" in meta


def test_paper_lines_come_from_bibtex_and_plain_citations(cards) -> None:  # type: ignore[no-untyped-def]
    line = cards.paper("@article{x,\n  author = {Doe, J. and Roe, R.},\n  title = {{A Title}},\n"
                       "  journal = {J. Things},\n  year = {2020},\n  doi = {10.1/abc},\n}")  # fmt: skip
    assert line == "Doe, J.; Roe, R. (2020). A Title. J. Things. https://doi.org/10.1/abc"
    assert cards.paper("Plain  citation,\n 2019.") == "Plain citation, 2019."


def test_cards_count_clips_from_the_manifest(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = [
        ClipRecord(clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i % 3}", media_path=f"{i}.mp4",
                   media_type="video", licence="t", commercial_ok=False, split=("train", "val", "test")[i % 3])
        for i in range(9)
    ]  # fmt: skip
    write_manifest(records, tmp_path / "data" / "manifest.parquet")
    assert "9 clips: 3 train, 3 val, 3 test" in cards.card("bulwark-dmd")["description"]
    assert "| DMD | 9 |" in cards.card("bulwark-meta")["description"]
    assert "Splits are subject-disjoint" in cards.card("bulwark-dmd")["description"]


def test_meta_card_counts_the_staged_manifest(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dmd_manifest(tmp_path, n=9)
    stage = tmp_path / "stage"
    dmd_manifest(stage / "x", n=6)  # a cut copy, as DROP_UNCLEARED stages it
    stage.mkdir(exist_ok=True)
    (stage / "x" / "data" / "manifest.parquet").rename(stage / "manifest.parquet")
    card = cards.card("bulwark-meta", ["manifest.parquet"], stage)
    assert "| DMD | 6 |" in card["description"] and "6 clips" in card["description"]
    assert "(6 clips)" in card["resources"][0]["description"]
    assert "| DMD | 9 |" in cards.card("bulwark-meta")["description"]  # unstaged: data/


def test_holdout_and_excluded_are_not_called_subject_disjoint(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    splits = ("train", "val", "test", "holdout", "excluded")
    records = [
        ClipRecord(clip_id=f"drive_and_act/{i}", dataset="drive_and_act", subject_id=f"drive_and_act:{i}",
                   media_path=f"{i}.mp4", media_type="video", licence="t", commercial_ok=False, split=split)
        for i, split in enumerate(splits)
    ]  # fmt: skip
    write_manifest(records, tmp_path / "data" / "manifest.parquet")
    text = " ".join(cards.card("bulwark-drive-and-act")["description"].split())
    assert "Splits are subject-disjoint" not in text
    assert "Train, val and test are subject-disjoint" in text
    assert "holdout is the inner-mirror view of the test drivers" in text


def test_auc_card_describes_still_images_and_the_authors_driver_split(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    records = [
        ClipRecord(clip_id=f"auc_v2/Camera 1/{split}/c0/{i}.jpg", dataset="auc_v2",
                   subject_id=f"auc_v2:{split}_drivers", media_path=f"Camera 1/{split}/c0/{i}.jpg",
                   media_type="image", licence="t", commercial_ok=False, split=split)
        for i, split in enumerate(("train", "train", "test"))
    ]  # fmt: skip
    write_manifest(records, tmp_path / "data" / "manifest.parquet")
    shard(tmp_path / "data" / "shards" / "auc_v2-00000.tar", [r.clip_id for r in records])
    card = cards.card("bulwark-auc", cards.local_files("bulwark-auc"))
    text = " ".join(card["description"].split())
    assert card["title"] == "Bulwark training shards: AUC Distracted Driver v2"
    # Kaggle's 80-character limit is met by shorter wording, not by cutting a word.
    assert card["subtitle"].endswith("for training Bulwark") and len(card["subtitle"]) <= 80
    assert "still images from AUC Distracted Driver v2's two cabin cameras, one frame each" in text
    assert "one 256 px JPEG frame per image" in text and "16 frames" not in text
    assert "3 clips: 2 train, 1 test. Train and test follow the authors' split by driver" in text
    assert "Splits are subject-disjoint" not in text
    (resource,) = card["resources"]
    assert resource["path"] == "auc_v2-00000.shard" and "one 256 px JPEG frame per clip" in resource["description"]
    assert "frames letterboxed to 256 px" in card["userSpecifiedSources"]
    assert "clips cut" not in card["userSpecifiedSources"]
    meta = cards.card("bulwark-meta")
    assert "`bulwark-auc`" in meta["description"] and "| AUC Distracted Driver v2 | 3 |" in meta["description"]
    sources = meta["userSpecifiedSources"]
    assert "AUC Distracted Driver v2 keeps its authors' train and test split by driver, with one correction" in sources
    assert "from 1 source dataset: (1) AUC Distracted Driver Dataset v2" in sources


def test_meta_card_names_only_the_datasets_in_its_manifest(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dmd_manifest(tmp_path, n=3)  # a manifest without AUC, as the first full run's
    meta = cards.card("bulwark-meta")
    sources = meta["userSpecifiedSources"]
    assert "from 1 source dataset: (1) " in sources and "Driver Monitoring Dataset" in sources
    assert "AUC" not in sources and "Drive&Act" not in sources and "MRL" not in sources
    # Checkpoints are not said to be trained on whatever the manifest holds.
    assert "on these data" not in sources and "its provenance.json lists" in sources
    assert "(`bulwark-dmd`)" in meta["description"] and "bulwark-auc" not in meta["description"]
    # Without a manifest, every shard dataset is named.
    (tmp_path / "data" / "manifest.parquet").unlink()
    sources = cards.card("bulwark-meta")["userSpecifiedSources"]
    assert "from 4 source datasets" in sources and "AUC Distracted Driver Dataset v2" in sources


def test_clip_id_column_gives_both_suffix_forms(cards) -> None:  # type: ignore[no-untyped-def]
    assert "`@<start frame>`" in cards.COLUMNS["clip_id"]
    assert "`@<start frame>s<stride>:<chunk id>`" in cards.COLUMNS["clip_id"]


def test_cards_give_the_clip_length_the_stride_rule_produces(cards) -> None:  # type: ignore[no-untyped-def]
    # 16 frames at stride max(1, round(2 s x fps) // 16) cover about 1.5 s at 30 fps and
    # about 1.07 s at 15 fps, never the full 2 s window.
    assert clip_stride(30, 2.0, 16) == 3 and clip_stride(15, 2.0, 16) == 1
    for slug in cards.SLUGS:
        description = cards.card(slug)["description"]
        assert not re.search(r"\b2[- ]?(s|sec|second|seconds)\b", description), slug
    assert "about 1.07 s" in cards.card("bulwark-drive-and-act")["description"]
    assert "about 1.5 s" in cards.card("bulwark-dmd")["description"]


def shard(path: Path, clip_ids: list[str], extra: int = 0) -> Path:
    """A shard as `bulwark shard` writes it: frame members, then index.json last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {f"k{i}": clip_id for i, clip_id in enumerate(clip_ids)}
    with tarfile.open(path, "w") as tar:
        for key in index:
            for frame in range(2 + extra):
                data = b"j" * 700
                info = tarfile.TarInfo(f"{key}/{frame:02d}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        data = json.dumps(index).encode()
        info = tarfile.TarInfo("index.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


def dmd_manifest(root: Path, n: int = 9) -> None:
    records = [
        ClipRecord(clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i % 3}", media_path=f"{i}.mp4",
                   media_type="video", licence="t", commercial_ok=False, split=("train", "val", "test")[i % 3])
        for i in range(n)
    ]  # fmt: skip
    write_manifest(records, root / "data" / "manifest.parquet")


def test_shard_index_is_read_from_the_end_of_the_tar(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ids = [f"dmd/{i}" for i in range(50)]
    assert cards.shard_clip_ids(shard(tmp_path / "a.tar", ids, extra=40)) == ids
    no_index = tmp_path / "b.tar"
    with tarfile.open(no_index, "w") as tar:
        tar.addfile(tarfile.TarInfo("x/00.jpg"), io.BytesIO(b""))
    assert cards.shard_clip_ids(no_index) is None


def test_resources_describe_exactly_the_staged_files(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dmd_manifest(tmp_path)
    shard(tmp_path / "data" / "shards" / "dmd-00000.tar", ["dmd/0", "dmd/1", "dmd/2", "dmd/9"])
    stage = tmp_path / "stage"
    shard(stage / "shards" / "dmd-00001.shard", ["dmd/3", "dmd/4"])  # only staged, not under data/
    (stage / "shards" / "dmd-00000.shard").write_bytes(
        (tmp_path / "data" / "shards" / "dmd-00000.tar").read_bytes()
    )
    for name in ("dataset-metadata.json", "dataset-cover-image.png"):
        (stage / name).write_text("{}")
    (stage / ".git").mkdir()
    (stage / ".git" / "HEAD").write_text("x")
    files = cards.staged_files(stage)
    assert files == ["shards/dmd-00000.shard", "shards/dmd-00001.shard"]
    # The kaggle CLI uploads dotfiles, so a stray one is listed and the card refuses it.
    (stage / "shards" / ".DS_Store").write_text("x")
    assert ".DS_Store" in cards.staged_files(stage)[0]
    with pytest.raises(SystemExit, match="No description for shards/.DS_Store"):
        cards.card("bulwark-dmd", cards.staged_files(stage), stage)
    (stage / "shards" / ".DS_Store").unlink()
    card = cards.card("bulwark-dmd", files, stage)
    assert [r["path"] for r in card["resources"]] == files
    first, second = (r["description"] for r in card["resources"])
    assert "shard 00000 of 2: 4 clips (1 train, 1 val, 1 test, 1 not in the manifest)" in first
    assert "shard 00001 of 2: 2 clips (1 train, 1 val)" in second
    with pytest.raises(SystemExit, match="No description for notes.txt"):
        cards.resources(["notes.txt"])


def test_manifest_resource_describes_every_column(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dmd_manifest(tmp_path)
    stage = tmp_path / "stage"
    (stage / "checkpoints" / "heads-v0").mkdir(parents=True)
    (stage / "manifest.parquet").write_bytes((tmp_path / "data" / "manifest.parquet").read_bytes())
    for name in ("best.pt", "final.pt", "provenance.json"):
        (stage / "checkpoints" / "heads-v0" / name).write_text("x")
    (stage / "checkpoints" / "bulwark-e1.onnx").write_text("x")
    files = cards.staged_files(stage)
    card = cards.card("bulwark-meta", files, stage)
    by_path = {r["path"]: r for r in card["resources"]}
    assert sorted(by_path) == sorted(files) and len(files) == 5
    fields = by_path["manifest.parquet"]["schema"]["fields"]
    columns = pq.read_schema(stage / "manifest.parquet").names
    assert [f["name"] for f in fields] == columns == list(ClipRecord.__dataclass_fields__)
    assert all(
        f["description"] and f["type"] in {"string", "integer", "number", "boolean"} for f in fields
    )
    types = {f["name"]: f["type"] for f in fields}
    assert (
        types["commercial_ok"] == "boolean"
        and types["fps"] == "number"
        and types["num_frames"] == "integer"
    )
    assert "9 clips" in by_path["manifest.parquet"]["description"]
    assert "checkpoints/heads-v0" in by_path["checkpoints/heads-v0/best.pt"]["description"]
    assert "ONNX" in by_path["checkpoints/bulwark-e1.onnx"]["description"]
    # Without a local manifest the columns come from ClipRecord, with the same types.
    (tmp_path / "data" / "manifest.parquet").unlink()
    bare = cards.manifest_resource("manifest.parquet", None)["schema"]["fields"]
    assert [(f["name"], f["type"]) for f in bare] == [(f["name"], f["type"]) for f in fields]


def test_kaggle_file_list_follows_pages(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    stub = tmp_path / "kaggle"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" --page-token p2 "* ]]; then printf "name,size,creationDate\\nb.shard,1,x\\n";\n'
        'else printf "Next Page Token = p2\\nname,size,creationDate\\na.shard,1,x\\n"; fi\n'
    )
    stub.chmod(0o755)
    assert cards.kaggle_files("me/bulwark-dmd", str(stub)) == ["a.shard", "b.shard"]


def test_apply_waits_until_kaggle_lists_the_staged_files(
    cards, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    stage = tmp_path / "stage"
    for name in ("dmd-00000.shard", "dmd-00001.shard"):
        shard(stage / "shards" / name, ["dmd/0"])
    lists = iter([["dmd-00000.shard"], ["dmd-00000.shard"], ["dmd-00000.shard", "dmd-00001.shard"]])
    monkeypatch.setattr(cards, "kaggle_files", lambda ref, kaggle: next(lists))
    calls: list[list[str]] = []
    monkeypatch.setattr(cards.subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setattr(cards, "cover", lambda slug, target: target)
    cards.apply("bulwark-dmd", "me", "kaggle", stage, tries=3, wait=0)
    assert len(calls) == 1 and "--update" in calls[0]
    # A list that never catches up stops before the update, naming the missing file.
    monkeypatch.setattr(cards, "kaggle_files", lambda ref, kaggle: ["dmd-00000.shard"])
    with pytest.raises(SystemExit, match="not on Kaggle: dmd-00001.shard"):
        cards.apply("bulwark-dmd", "me", "kaggle", stage, tries=2, wait=0)
    assert len(calls) == 1


def test_apply_waits_while_kaggle_lists_the_previous_checkpoint_directory(
    cards, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    # A new bulwark-meta version whose checkpoint files have the same names as the old
    # version's, in another directory: the old list must not count as the new one.
    stage = tmp_path / "stage"
    (stage / "checkpoints" / "heads-v1").mkdir(parents=True)
    for name in ("best.pt", "final.pt"):
        (stage / "checkpoints" / "heads-v1" / name).write_bytes(b"\0")
    (stage / "manifest.parquet").write_bytes(b"\0")
    old = ["manifest.parquet", "checkpoints/heads-v0/best.pt", "checkpoints/heads-v0/final.pt"]
    new = ["manifest.parquet", "checkpoints/heads-v1/best.pt", "checkpoints/heads-v1/final.pt"]
    lists = iter([old, old, new])
    monkeypatch.setattr(cards, "kaggle_files", lambda ref, kaggle: next(lists))
    seen: list[list[str]] = []
    monkeypatch.setattr(cards, "card", lambda slug, files, stage: seen.append(files) or {})
    monkeypatch.setattr(cards.subprocess, "run", lambda command, check: None)
    monkeypatch.setattr(cards, "cover", lambda slug, target: target)
    cards.apply("bulwark-meta", "me", "kaggle", stage, tries=3, wait=0)
    assert seen == [new]


def test_apply_waits_while_a_new_dataset_is_still_being_created(
    cards, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    stage = tmp_path / "stage"
    shard(stage / "shards" / "auc_v2-00000.shard", ["auc_v2/0"])
    answers = iter([None, None, ["auc_v2-00000.shard"]])

    def listed(ref: str, kaggle: str) -> list[str]:
        files = next(answers)
        if files is None:  # Kaggle answers 403 until the first version is ready
            raise cards.subprocess.CalledProcessError(1, ["kaggle", "datasets", "files", ref])
        return files

    monkeypatch.setattr(cards, "kaggle_files", listed)
    calls: list[list[str]] = []
    monkeypatch.setattr(cards.subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setattr(cards, "cover", lambda slug, target: target)
    cards.apply("bulwark-auc", "me", "kaggle", stage, tries=3, wait=0)
    assert len(calls) == 1 and "--update" in calls[0]


def test_json_mode_writes_resources_and_the_cover(cards, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "elsewhere"
    dmd_manifest(root)
    shard(root / "data" / "shards" / "dmd-00000.tar", ["dmd/0"])
    out, covers = tmp_path / "card.json", tmp_path / "cover"
    covers.mkdir()
    cards.main(
        ["--root", str(root), "--slug", "bulwark-dmd", "--json", str(out), "--cover", str(covers)]
    )
    card = json.loads(out.read_text())
    assert [r["path"] for r in card["resources"]] == ["dmd-00000.shard"]
    assert "shard 00000 of 1: 1 clip (1 train)" in card["resources"][0]["description"]
    with Image.open(covers / "dataset-cover-image.png") as image:
        assert image.size == (560, 280)


def test_every_slug_has_a_committed_560x280_cover(cards) -> None:  # type: ignore[no-untyped-def]
    covers = load_covers()
    assert set(covers.COVERS) == set(cards.SLUGS)
    for slug in cards.SLUGS:
        with Image.open(cards.COVERS / f"{slug}.png") as image:
            assert image.size == (560, 280)
        assert covers.render(slug).size == (560, 280)


def test_the_kaggle_cli_never_uploads_the_cover_or_the_card_as_files() -> None:
    KaggleApi = pytest.importorskip("kaggle.api.kaggle_api_extended").KaggleApi  # the ops extra

    skipped = [KaggleApi.DATASET_METADATA_FILE, *KaggleApi.DATASET_COVER_IMAGE_FILES]
    assert "dataset-cover-image.png" in skipped and "dataset-metadata.json" in skipped


def test_staged_files_follow_the_kaggle_cli_skip_rules(cards) -> None:  # type: ignore[no-untyped-def]
    api = pytest.importorskip("kaggle.api.kaggle_api_extended")  # the ops extra
    KaggleApi = api.KaggleApi
    names = {
        KaggleApi.DATASET_METADATA_FILE,
        KaggleApi.OLD_DATASET_METADATA_FILE,
        KaggleApi.KERNEL_METADATA_FILE,
        KaggleApi.MODEL_METADATA_FILE,
        KaggleApi.MODEL_INSTANCE_METADATA_FILE,
    }
    assert names == cards.CLI_SKIPPED_NAMES
    assert all(n.startswith("dataset-cover-image") for n in KaggleApi.DATASET_COVER_IMAGE_FILES)
    dirs = {p.strip("*/").rstrip("/") for p in api.DEFAULT_IGNORE_PATTERNS}
    assert dirs == cards.CLI_SKIPPED_DIRS


def load_covers():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "kaggle_covers", REPO / "infra" / "kaggle" / "covers.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
