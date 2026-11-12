"""Kaggle metadata (the whole dataset card) for Bulwark's private datasets.

    python infra/kaggle/dataset_cards.py --slug bulwark-dmd            # print one card
    python infra/kaggle/dataset_cards.py --apply                       # update every slug
    python infra/kaggle/dataset_cards.py --apply --slug bulwark-meta   # update one
    python infra/kaggle/dataset_cards.py --root ../bulwark --slug bulwark-mrl --json card.json

A card is everything Kaggle scores a dataset's usability on, except a public notebook (the
datasets are private): title, subtitle, description, keywords, licence, sources
(`userSpecifiedSources`), expected update frequency, a cover image
(infra/kaggle/covers/<slug>.png, rendered by covers.py) and a description of every file,
with column descriptions for the manifest (`resources`).

Names, links, licences and citations come from bulwark/data/registry.py, clip counts
from data/manifest.parquet (or, with --stage, the manifest.parquet staged there, which
DROP_UNCLEARED may have cut) and each shard's index.json, so the cards stay true as the
data changes. --root names the checkout whose data/ (and checkpoints/) the counts come
from (default: this repository). Which files `resources` describes: with --stage, the
files staged there (push_data.sh); with --apply or --from-kaggle, the dataset's current
file list on Kaggle (`kaggle datasets files`, read only; --apply --stage first waits
until that list matches the staged files, since right after an upload Kaggle may still
list the previous version's); otherwise the files
push_data.sh would stage by default (every finished shard of the dataset, or the
manifest). A file the card cannot describe is an error, never a guess.

push_data.sh writes the card into each upload, but Kaggle's create and version calls
take only the title, subtitle, description, keywords and per-file descriptions of
top-level files from it; --apply (which push_data.sh runs after each upload) sets the
rest through a metadata update without uploading files again. Needs the kaggle CLI
(`uv pip install -e ".[ops]"`) for --apply and --from-kaggle.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bulwark.data.manifest import ClipRecord, read_manifest  # noqa: E402
from bulwark.data.registry import DATASETS, DatasetSpec  # noqa: E402
from bulwark.data.splits import SplitPolicy  # noqa: E402

# Where data/ and checkpoints/ live (--root); the code always comes from REPO.
ROOT = REPO
# The manifest a card describes when one is staged (card(..., stage)): the copy that goes
# up, which DROP_UNCLEARED may have cut, rather than data/manifest.parquet under ROOT.
STAGED_MANIFEST: Path | None = None
COVERS = Path(__file__).resolve().parent / "covers"
COVER_NAME = "dataset-cover-image.png"  # the kaggle CLI uploads it as the cover, never as a file
# What the kaggle CLI (KaggleApi.upload_files) leaves out of an upload: these top-level
# names (the card, the cover, other metadata files) and DEFAULT_IGNORE_PATTERNS'
# directories. Everything else, dotfiles included, goes up as a data file.
CLI_SKIPPED_NAMES = {
    "dataset-metadata.json",
    "datapackage.json",
    "kernel-metadata.json",
    "model-metadata.json",
    "model-instance-metadata.json",
}
CLI_SKIPPED_DIRS = {".git", ".cache", ".huggingface"}

CONTACT = "https://inferensys.com/contact"
WEBSITE = "https://inferensys.com"
GITHUB = "https://github.com/Inferensys"
KEYWORDS = ["computer vision", "video", "deep learning", "automobiles and vehicles", "image"]

# Kaggle slug -> registry dataset (None: the manifest and checkpoints dataset).
SLUGS: dict[str, str | None] = {
    "bulwark-drive-and-act": "drive_and_act",
    "bulwark-dmd": "dmd",
    "bulwark-mrl": "mrl_eye",
    "bulwark-auc": "auc_v2",
    "bulwark-meta": None,
}
# Kaggle's accepted values: not specified, never, annually, quarterly, monthly, weekly,
# daily, hourly. A shard dataset changes only with a new data release (new or re-cut
# clips, a removal request), which is irregular and rare, so no schedule would be true;
# "never" says the published version is not refreshed on a schedule, and any change
# comes as a new version. The manifest and starting checkpoints change as training and
# labelling proceed, about monthly.
FREQUENCY = {
    "bulwark-drive-and-act": "never",
    "bulwark-dmd": "never",
    "bulwark-mrl": "never",
    "bulwark-auc": "never",
    "bulwark-meta": "monthly",
}
SHORT = {
    "drive_and_act": "Drive&Act",
    "dmd": "DMD",
    "mrl_eye": "MRL Eye",
    "auc_v2": "AUC Distracted Driver v2",
}
# Clip lengths follow bulwark.data.sources.base.clip_stride: 16 frames at a stride of
# max(1, round(2 s x fps) // 16), so a clip covers less than its 2 s window.
WHAT = {
    "drive_and_act": (
        "16-frame clips (about 1.5 s of video at 30 fps; about 1.07 s for the 15 fps Kinect "
        "colour view) from Drive&Act's synchronised cabin cameras"
    ),
    "dmd": "16-frame clips (about 1.5 s of video at 30 fps) from DMD's face camera",
    "mrl_eye": "infrared eye crops from MRL Eye, one frame each",
    "auc_v2": "still images from AUC Distracted Driver v2's two cabin cameras, one frame each",
}
# Datasets whose clips are still images (one frame each), not cut from video.
STILLS = {"mrl_eye", "auc_v2"}
# A shard's file name on Kaggle (.shard) or on disk (.tar); same rule as stage_data.SHARD_NAME.
SHARD_FILE = re.compile(r"^(?P<dataset>.+?)(?:-r\d{3})?-(?P<number>\d{5})\.(?:shard|tar)$")
# Checkpoint files push_data.sh may stage (stage_data.INCLUDE_FILES and *.onnx graphs).
CHECKPOINT_FILES = {
    "best.pt": (
        "PyTorch checkpoint of the training run in {run}: the weights selected on "
        "ground-truth validation labels (by k-fold cross-validation for head runs that "
        "set it), with the run's provenance. A starting point for later Bulwark jobs; "
        "research use only."
    ),
    "final.pt": (
        "PyTorch checkpoint of the training run in {run}: the weights at the end of its "
        "step budget, with the run's provenance. A starting point for later Bulwark jobs; "
        "research use only."
    ),
    "provenance.json": (
        "Provenance of the training run in {run}: resolved config, datasets and their "
        "licences, manifest hash, split salt, inputs, labels, code commit and library "
        "versions (bulwark.train.provenance)."
    ),
}
ONNX_FILE = (
    "Bulwark-E1 exported to ONNX by `bulwark export`: input `frames`, uint8 RGB of shape "
    "(batch, T, S, S, 3); one raw-logit output per taxonomy label. The contract and the "
    "checkpoint's provenance are stored in the graph's metadata. Research use only."
)
# Every manifest column (bulwark.data.manifest.ClipRecord), as Kaggle column descriptions.
COLUMNS: dict[str, str] = {
    "clip_id": (
        "Unique clip id: the dataset, the source file's path and, for video, the clip's "
        "position: `@<start frame>` (DMD and other sources), or "
        "`@<start frame>s<stride>:<chunk id>` for Drive&Act (the annotation chunk the clip "
        "is cut from). Images (MRL Eye, AUC Distracted Driver v2) have none. Each shard's "
        "index.json maps its entries to these ids."
    ),
    "dataset": "Source dataset (Bulwark registry name): drive_and_act, dmd, mrl_eye or auc_v2.",
    "subject_id": (
        "Person id, prefixed with the dataset (`dmd:gX_0`). Splits are subject-disjoint: a "
        "person is in only one of train, val and test. AUC Distracted Driver v2 publishes no "
        "driver ids, so its two ids name the train and test driver groups of its authors' "
        "split, corrected for one driver found on both sides (`auc_v2:train_drivers`, "
        "`auc_v2:test_drivers`; meta authors_split marks the moved images)."
    ),
    "media_path": "Source video or image, relative to the source dataset's own download (not included here).",
    "media_type": "`video` (a 16-frame clip) or `image` (one frame).",
    "licence": "Licence of the source dataset, as recorded in Bulwark's dataset registry.",
    "commercial_ok": (
        "Whether the source dataset's published terms appear to permit commercial use "
        "(the maintainers' reading, not legal advice)."
    ),
    "start_frame": "First source frame of the clip (0 for images).",
    "num_frames": "Frames in the clip: 16 for video, 1 for images.",
    "fps": "Frame rate of the source video (0 for images).",
    "frame_stride": "The clip takes every k-th source frame: k = max(1, round(2 s x fps) // 16).",
    "view": (
        "Camera or sensor the clip comes from: a Drive&Act cabin view, DMD's face camera, an "
        "MRL Eye sensor or one of AUC Distracted Driver v2's two cabin cameras (cam1, cam2)."
    ),
    "crop": "JSON [x, y, width, height] in source pixels applied before resizing, or null for the whole frame.",
    "split": "train, val, test, holdout (scored only, never trained on), excluded (neither) or unassigned.",
    "labels_gt": "JSON object: the dataset's own annotations, mapped to Bulwark's label taxonomy.",
    "labels_teacher": (
        "JSON object: automatic labels, never ground truth. Eye, mouth and head-pose signals "
        "from MediaPipe Face Landmarker on video clips, and labels derived from a dataset's "
        "annotations by a Bulwark rule (DMD microsleep). labels_gt wins where both exist."
    ),
    "meta": "JSON object: other per-clip fields from the source (session, native activity, official split, ...).",
}
# Frictionless types Kaggle maps (string, integer, number, boolean), from the parquet type.
ARROW_TYPES = {"string": "string", "large_string": "string", "bool": "boolean", "double": "number",
               "float": "number", "int64": "integer", "int32": "integer"}  # fmt: skip
PYTHON_TYPES = {"str": "string", "bool": "boolean", "float": "number", "int": "integer"}

ABOUT_BULWARK = f"""## About Bulwark

Bulwark is a research model for the state of a person on camera: fatigue, distraction,
attention and risky actions. A V-JEPA 2.1 video encoder with multi-task heads (Bulwark-B1)
is distilled into a small ONNX model for edge devices (Bulwark-E1). It is meant as a base
for products such as driver monitoring, industrial-operator safety and workplace safety,
each built as a small adapter on top. Weights trained on research datasets like this one
are for research use only; a model trained on commercially licensed data is the
route to commercial use.

## Contact

To reach Inferensys with questions about this data, removal requests, or commercial
licensing and custom work, use the contact page: {CONTACT}.
Website: {WEBSITE}. GitHub: {GITHUB}."""


def _manifest_path() -> Path:
    return STAGED_MANIFEST or ROOT / "data" / "manifest.parquet"


def _manifest_datasets() -> list[str]:
    """The source datasets of the manifest the card describes, shard slugs' order first;
    without a manifest, every dataset that has a shard slug."""
    with_slug = [name for name in SLUGS.values() if name is not None]
    path = _manifest_path()
    if not path.exists():
        return with_slug
    import pandas as pd

    present = set(pd.read_parquet(path, columns=["dataset"])["dataset"])
    return [d for d in with_slug if d in present] + sorted(present - set(with_slug))


def _counts(dataset: str | None) -> Counter[str]:
    path = _manifest_path()
    if not path.exists():
        return Counter()
    return Counter(r.split for r in read_manifest(path) if dataset is None or r.dataset == dataset)


def _splits(counts: Counter[str]) -> str:
    order = ["train", "val", "test", "holdout", "excluded", "unassigned"]
    return ", ".join(f"{counts[s]:,} {s}" for s in order if counts[s])


def _spec(name: str) -> DatasetSpec:
    return next(d for d in DATASETS if d.name == name)


def paper(citation: str) -> str:
    """One line for the paper a registry citation names: authors (year). Title. Venue. DOI."""
    if not citation.lstrip().startswith("@"):
        return " ".join(citation.split())
    fields = {
        key.lower(): value.replace("{", "").replace("}", "").strip()
        for key, value in re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", citation, flags=re.M)
    }
    authors = "; ".join(a.strip() for a in fields.get("author", "").split(" and "))
    venue = fields.get("booktitle") or fields.get("journal", "")
    parts = [
        f"{authors} ({fields['year']})." if "year" in fields else f"{authors}.",
        f"{fields.get('title', '')}.",
    ]
    if venue:
        parts.append(f"{venue}" + (f", vol. {fields['volume']}" if "volume" in fields else "")
                     + (f", pp. {fields['pages']}" if "pages" in fields else "") + ".")  # fmt: skip
    if "doi" in fields:
        parts.append(f"https://doi.org/{fields['doi']}")
    elif "url" in fields:
        parts.append(fields["url"])
    return " ".join(part for part in parts if part.strip(" ."))


def _copy_note(spec: DatasetSpec) -> str:
    """A short note on where the ingested copy came from, when it is not the official one."""
    if not spec.source_copy:
        return ""
    verified = spec.source_copy_fingerprint and spec.source_copy_fingerprint in spec.source_copy
    if verified and "identical" in spec.source_copy:
        archive = re.search(
            r"official archive (\S+) as captured by the Internet Archive on (\d{4}-\d{2}-\d{2})",
            spec.source_copy,
        )
        against = (
            f"the official archive {archive[1]} as captured by the Internet Archive "
            f"({archive[2]}; the publisher's server was unreachable)"
            if archive
            else "the official archive"
        )
        return (
            f" The copy ingested was a mirror, compared with {against} and found identical "
            "(the same images, with the same image fingerprint)."
        )
    return " The copy ingested was not the official download; see the description."


def source_line(spec: DatasetSpec) -> str:
    line = f"{spec.title}, {spec.url}."
    if spec.citation:
        line += f" Paper: {paper(spec.citation)}"
    return f"{line} Licence: {spec.licence}.{_copy_note(spec)}"


def sources(slug: str) -> str:
    """Kaggle's Provenance "Sources" (userSpecifiedSources): one free-text string."""
    name = SLUGS[slug]
    if name is not None:
        made = "frames letterboxed" if name in STILLS else "clips cut, frames letterboxed"
        return (
            f"{source_line(_spec(name))} Prepared by Inferensys with Bulwark's data pipeline "
            f"({made} to 256 px and stored as JPEG in tar shards); all rights in the source "
            "data remain with its owners."
        )
    names = _manifest_datasets()
    datasets = [_spec(d) for d in names]
    listed = "; ".join(f"({i}) {spec.title}, {spec.url}" for i, spec in enumerate(datasets, 1))
    count = f"{len(datasets)} source dataset{'s' if len(datasets) != 1 else ''}"
    splits = []
    if "drive_and_act" in names:
        splits.append(
            "Drive&Act keeps its official split_0 subject split (its inner-mirror view of the "
            "test drivers is a cross-view holdout)"
        )
    if "auc_v2" in names:
        splits.append(
            "AUC Distracted Driver v2 keeps its authors' train and test split by driver, with "
            "one correction (83 camera-1 test images of a driver who is also in train are moved "
            "to train), as one driver group per side and no val (it publishes no driver ids)"
        )
    splits.append("other datasets are split by a stable hash of the subject id")
    return (
        f"Clip manifest built by Inferensys from {count}: {listed} "
        "(papers, licences and terms are in each shard dataset's card). "
        "Labels: labels_gt maps each dataset's own annotations to Bulwark's taxonomy; "
        "labels_teacher holds automatic labels, from MediaPipe Face Landmarker (Apache-2.0) "
        "run on the video clips (not the eye crops), and labels derived from a dataset's "
        "annotations by a Bulwark rule (DMD microsleep). Train, val and test are "
        f"subject-disjoint: {'; '.join(splits)}. The policy is stored in the parquet "
        "metadata (split_policy). Checkpoints, when present, were trained by Inferensys, "
        "each on the datasets its provenance.json lists (not necessarily all of these), "
        "starting from third-party pretrained weights: Bulwark-B1 checkpoints build on "
        "Meta's V-JEPA 2.1 ViT-B encoder (its licence applies to them), and Bulwark-E1 is "
        "initialised from EfficientViT-B1."
    )


# --- resources: a description for every file ------------------------------------------------


_SPLIT_OF: dict[Path, dict[str, str]] = {}


def _split_of() -> dict[str, str]:
    """{clip id: split} from the manifest the card describes (empty without one); read once."""
    path = _manifest_path()
    if path not in _SPLIT_OF:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["clip_id", "split"]) if path.exists() else None
        _SPLIT_OF[path] = (
            {} if frame is None else dict(zip(frame["clip_id"], frame["split"], strict=True))
        )
    return _SPLIT_OF[path]


def shard_clip_ids(path: Path) -> list[str] | None:
    """Clip ids in a shard, from its index.json, or None if the shard has none.

    `bulwark shard` writes index.json as the tar's last member, so it is read from the end
    of the file (one read, instead of walking every member's header); a shard laid out
    otherwise falls back to tarfile.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        want = 1 << 20
        while True:
            start = max(0, size - want) // 512 * 512
            handle.seek(start)
            data = handle.read(size - start)
            for offset in range(len(data) - 512, -1, -512):
                block = data[offset : offset + 512]
                if block[:11] == b"index.json\0" and block[257:262] == b"ustar":
                    length = int(block[124:136].split(b"\0")[0].strip() or b"0", 8)
                    body = data[offset + 512 : offset + 512 + length]
                    if len(body) == length:
                        return list(json.loads(body).values())
            if start == 0:
                break
            want *= 8
    import tarfile

    with tarfile.open(path) as tar:
        try:
            member = tar.extractfile("index.json")
        except KeyError:
            return None
        return None if member is None else list(json.loads(member.read()).values())


def _shard_description(name: str, total: int, staged: Path | None) -> str:
    match = SHARD_FILE.match(Path(name).name)
    assert match is not None
    dataset, number = match["dataset"], match["number"]
    label = SHORT.get(dataset, dataset)
    frames = (
        "one 256 px JPEG frame per clip"
        if dataset in STILLS
        else "16 frames per clip, 256 px JPEG"
    )
    local = ROOT / "data" / "shards" / f"{Path(name).name.rsplit('.', 1)[0]}.tar"
    source = (
        local if local.is_file() else staged if staged is not None and staged.is_file() else None
    )
    clip_ids = shard_clip_ids(source) if source is not None else None
    counted = ""
    if clip_ids is not None:
        split_of = _split_of()
        counts = Counter(split_of[c] for c in clip_ids if c in split_of)
        missing = len(clip_ids) - sum(counts.values())
        detail = ", ".join(
            x for x in (_splits(counts), f"{missing:,} not in the manifest" if missing else "") if x
        )
        counted = f": {len(clip_ids):,} clip{'' if len(clip_ids) == 1 else 's'}" + (
            f" ({detail})" if detail and split_of else ""
        )
    return (
        f"{label} shard {number} of {total}{counted}. A tar archive (renamed .shard because "
        f"Kaggle unpacks .tar uploads) of {frames}, with index.json mapping each entry to its "
        "clip id. Labels and splits are in manifest.parquet (bulwark-meta)."
    )


def _column_type(name: str, parquet: Path | None) -> str:
    if parquet is not None:
        import pyarrow.parquet as pq

        arrow = str(pq.read_schema(parquet).field(name).type)
        return ARROW_TYPES.get(arrow, "string")
    annotation = str(ClipRecord.__dataclass_fields__[name].type).split("[")[0].split(" ")[0]
    return PYTHON_TYPES.get(annotation, "string")


def manifest_resource(path: str, staged: Path | None) -> dict[str, Any]:
    """manifest.parquet with a description of every column it has."""
    parquet = staged if staged is not None and staged.is_file() else _manifest_path()
    if parquet.is_file():
        import pyarrow.parquet as pq

        columns = list(pq.read_schema(parquet).names)
    else:
        parquet, columns = None, list(ClipRecord.__dataclass_fields__)  # type: ignore[assignment]
    missing = [c for c in columns if c not in COLUMNS]
    if missing:
        raise SystemExit(
            f"manifest.parquet: no description for column(s) {', '.join(missing)}; add them to COLUMNS"
        )
    total = sum(_counts(None).values())
    rows = f" ({total:,} clips)" if total else ""
    return {
        "path": path,
        "description": (
            f"Bulwark's clip manifest{rows}: one row per training clip, with its source video "
            "and frames, subject, split, licence, the dataset's own labels and automatic "
            "(teacher) labels. Label columns are JSON text. The parquet metadata records the "
            "split policy."
        ),
        "schema": {
            "fields": [
                {"name": c, "description": COLUMNS[c], "type": _column_type(c, parquet)}
                for c in columns
            ]
        },
    }


def resources(files: list[str], stage: Path | None = None) -> list[dict[str, Any]]:
    """Kaggle `resources`: a description for each of `files` (paths as Kaggle or the staging
    dir names them). A file the card cannot describe stops it: no file goes undescribed."""
    shards = Counter(m["dataset"] for f in files if (m := SHARD_FILE.match(Path(f).name)))
    out: list[dict[str, Any]] = []
    unknown = []
    for name in sorted(files):
        staged = stage / name if stage is not None else None
        base = Path(name).name
        match = SHARD_FILE.match(base)
        if match:
            out.append(
                {
                    "path": name,
                    "description": _shard_description(name, shards[match["dataset"]], staged),
                }
            )
        elif base == "manifest.parquet":
            out.append(manifest_resource(name, staged))
        elif base in CHECKPOINT_FILES or base.endswith(".onnx"):
            run = Path(name).parent.as_posix()
            run = run if run not in ("", ".") else "checkpoints/"
            text = CHECKPOINT_FILES[base].format(run=run) if base in CHECKPOINT_FILES else ONNX_FILE
            out.append({"path": name, "description": text})
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(
            f"No description for {', '.join(unknown)}; add one to infra/kaggle/dataset_cards.py"
        )
    return out


def staged_files(stage: Path) -> list[str]:
    """Files under the staging dir that the kaggle CLI uploads, by the CLI's own rules
    (CLI_SKIPPED_NAMES, the cover, CLI_SKIPPED_DIRS). A stray dotfile such as .DS_Store is
    uploaded, so it is listed, and the card then refuses it for want of a description."""
    return sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
        and not any(part in CLI_SKIPPED_DIRS for part in path.relative_to(stage).parts[:-1])
        and not (
            path.parent == stage
            and (path.name in CLI_SKIPPED_NAMES or path.name.startswith("dataset-cover-image"))
        )
    )


def local_files(slug: str) -> list[str]:
    """What push_data.sh stages for `slug` by default: every finished shard of its dataset
    (as .shard), or the manifest."""
    name = SLUGS[slug]
    if name is None:
        return ["manifest.parquet"] if _manifest_path().exists() else []
    found = (ROOT / "data" / "shards").glob(f"{name}-*.tar")
    return sorted(
        f"{p.stem}.shard" for p in found if (m := SHARD_FILE.match(p.name)) and m["dataset"] == name
    )


def kaggle_files(ref: str, kaggle: str) -> list[str]:
    """The dataset's current file names on Kaggle (`kaggle datasets files`, every page)."""
    names: list[str] = []
    token = None
    while True:
        command = [kaggle, "datasets", "files", ref, "--csv", "--page-size", "200"]
        if token:
            command += ["--page-token", token]
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        token = None
        lines = []
        for line in output.splitlines():
            if line.startswith("Next Page Token = "):
                token = line.split("=", 1)[1].strip()
            elif line.strip() and line.strip() != "No files found":
                lines.append(line)
        names += [row["name"] for row in csv.DictReader(io.StringIO("\n".join(lines)))]
        if not token:
            return names


def card(slug: str, files: list[str] | None = None, stage: Path | None = None) -> dict[str, object]:
    """The card for `slug`; with `files`, also `resources` describing each of them. With a
    `stage` that holds manifest.parquet, counts come from that staged copy."""
    global STAGED_MANIFEST
    staged = stage / "manifest.parquet" if stage is not None else None
    STAGED_MANIFEST = staged if staged is not None and staged.is_file() else None
    base = _meta_card() if SLUGS[slug] is None else _shard_card(slug)
    base["userSpecifiedSources"] = sources(slug)
    base["expectedUpdateFrequency"] = FREQUENCY[slug]
    if files is not None:
        base["resources"] = resources(files, stage)
    return base


def _shard_card(slug: str) -> dict[str, object]:
    name = SLUGS[slug]
    assert name is not None
    spec = _spec(name)
    counts = _counts(name)
    total = sum(counts.values())
    frames = (
        "one 256 px JPEG frame per image"
        if name in STILLS
        else "16 frames per clip, 256 px JPEG"
    )
    source_copy = f"\n\nThe copy used here: {spec.source_copy}" if spec.source_copy else ""
    if name in SplitPolicy().subject_group_datasets:
        # AUC v2: one subject group per side of the authors' split (no driver ids).
        disjoint = (
            "Train and test follow the authors' split by driver, except 83 camera-1 test\n"
            "  images of a driver who is also in train, moved to train; no per-image driver\n"
            "  id is published, so there is no val split, and one driver's images cannot be\n"
            "  removed on their own (see Bulwark's data card)."
        )
    elif counts["holdout"] or counts["excluded"]:
        # Drive&Act: the same drivers appear in more than one view, so holdout and excluded
        # share subjects with test and train/val; only train, val and test are disjoint.
        disjoint = (
            "Train, val and test are subject-disjoint (no person appears in more than one\n"
            "  of them); holdout is the inner-mirror view of the test drivers (a cross-view\n"
            "  check, never trained on) and excluded is the inner-mirror view of the train\n"
            "  and val drivers (unused)."
        )
    else:
        disjoint = "Splits are subject-disjoint (no person appears in\n  more than one split)."
    clips_line = f"- {total:,} clips: {_splits(counts)}. {disjoint}" if total else f"- {disjoint}"
    description = f"""This private dataset holds training shards for **Bulwark**, the Inferensys research
model for driver and operator state (see below). It contains {WHAT[name]}, prepared for
Bulwark's Kaggle training jobs. It is not a public release of the source data.

## Contents

- `{name}-NNNNN.shard`: tar archives, renamed from `.tar` because Kaggle unpacks tar
  uploads. Each holds {frames}, and an `index.json` that maps each entry to its clip id.
{clips_line}
- Labels are not in the shards. They live in Bulwark's clip manifest (the
  `bulwark-meta` dataset), which records each clip's source video, frames, split,
  dataset labels and automatic (teacher) labels.

## Source and terms

Derived from **{spec.title}** ({spec.url}). Licence: {spec.licence}. All rights in the
source data remain with its owners. This copy exists only so Bulwark can train on Kaggle's
GPUs; it must not be shared or redistributed. To use the data, obtain it from its
owners under their terms.{source_copy}

Please cite the original dataset:

```
{spec.citation.strip()}
```

{ABOUT_BULWARK}
"""
    subtitle = f"{SHORT[name]} prepared as JPEG clip shards for training the Bulwark model"
    if len(subtitle) > 80:  # Kaggle's limit: shorten the wording rather than cut a word
        subtitle = f"{SHORT[name]} prepared as JPEG clip shards for training Bulwark"
    return {
        "title": f"Bulwark training shards: {SHORT[name]}",
        "subtitle": subtitle,
        "description": description,
        "keywords": KEYWORDS,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }


def _meta_card() -> dict[str, object]:
    counts = _counts(None)
    per_dataset = (
        Counter(r.dataset for r in read_manifest(_manifest_path())) if counts else Counter()
    )
    rows = "\n".join(f"| {SHORT.get(d, d)} | {n:,} |" for d, n in sorted(per_dataset.items()))
    present = set(_manifest_datasets())
    shard_slugs = ", ".join(f"`{slug}`" for slug, name in SLUGS.items() if name in present)
    uses = f" ({shard_slugs})" if shard_slugs else ""
    description = f"""This private dataset holds the clip manifest (and, when a job needs them,
starting checkpoints) for **Bulwark**, the Inferensys research model for driver and operator
state. Every Bulwark training job on Kaggle attaches it together with the shard datasets
it uses{uses}; a job trains on and scores only the clips whose shards it attaches.

## Contents

- `manifest.parquet`: one row per clip{f" ({sum(counts.values()):,} clips: {_splits(counts)})" if counts else ""}:
  dataset, subject, source video and frame range, split, the dataset's own labels,
  automatic (teacher) labels, licence and a commercial-use flag. Its metadata records the
  split policy it was made with.
- `checkpoints/` (when present): starting checkpoints that a later job starts from (heads,
  LoRA adapters, Bulwark-E1 weights or ONNX graphs), each with its run's provenance.
  Bulwark-B1 checkpoints build on Meta's V-JEPA 2.1 ViT-B encoder, whose licence applies
  to them; Bulwark-E1 is initialised from EfficientViT-B1.

{("| Dataset | Clips |" + chr(10) + "|---|---|" + chr(10) + rows + chr(10)) if rows else ""}
Each source dataset keeps its own licence and terms; see the shard datasets for citations.
None of this may be redistributed.

{ABOUT_BULWARK}
"""
    return {
        "title": "Bulwark training manifest and checkpoints",
        "subtitle": "Clip manifest, labels and starting checkpoints for Bulwark training jobs",
        "description": description,
        "keywords": KEYWORDS,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }


def cover(slug: str, target_dir: Path) -> Path:
    """Copy the slug's cover next to dataset-metadata.json, where the kaggle CLI finds it."""
    target = target_dir / COVER_NAME
    shutil.copyfile(COVERS / f"{slug}.png", target)
    return target


def _names(files: list[str]) -> Counter[str]:
    return Counter(files)


def _as_listed(staged: list[str]) -> list[str]:
    """The paths Kaggle lists for these staged files: an upload that is a single folder is
    unpacked into the dataset root (shards/x.shard is listed as x.shard); otherwise folders
    keep their names (checkpoints/heads-v0/best.pt). Full paths, so a new version whose
    files share names with the old one's (another checkpoint directory) is not mistaken for
    it while Kaggle still lists the old version."""
    tops = {f.split("/", 1)[0] for f in staged}
    if len(tops) == 1 and all("/" in f for f in staged):
        return [f.split("/", 1)[1] for f in staged]
    return staged


def _listed_while_processing(ref: str, kaggle: str) -> list[str]:
    """kaggle_files, or [] while Kaggle refuses to list a dataset that is still being
    created (it answers 403 until the first version is ready)."""
    try:
        return kaggle_files(ref, kaggle)
    except subprocess.CalledProcessError:
        return []


def apply(
    slug: str, user: str, kaggle: str, stage: Path | None = None, tries: int = 10, wait: float = 30
) -> None:
    """Update the dataset's metadata on Kaggle, describing the files Kaggle lists. With
    `stage` (the upload just made), wait until Kaggle lists exactly the staged files: right
    after `datasets version` it may still list the previous version's files."""
    ref = f"{user}/{slug}"
    if stage is None:
        files = kaggle_files(ref, kaggle)
    else:
        expected = _names(_as_listed(staged_files(stage)))
        files = _listed_while_processing(ref, kaggle)
        for _ in range(tries - 1):
            if _names(files) == expected:
                break
            time.sleep(wait)
            files = _listed_while_processing(ref, kaggle)
        if _names(files) != expected:
            extra = sorted((_names(files) - expected).elements())
            missing = sorted((expected - _names(files)).elements())
            raise SystemExit(
                f"{ref}: Kaggle's file list does not match the staged files yet (not on Kaggle: "
                f"{', '.join(missing) or 'none'}; not staged: {', '.join(extra) or 'none'})."
            )
    with tempfile.TemporaryDirectory() as tmp:
        info = {"datasetSlug": slug, "ownerUser": user, **card(slug, files, stage)}
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps({"info": info}, indent=2))
        cover(slug, Path(tmp))
        subprocess.run(
            [kaggle, "datasets", "metadata", f"{user}/{slug}", "--update", "-p", tmp], check=True
        )


def _kaggle() -> str:
    return (
        os.environ.get("KAGGLE") or shutil.which("kaggle") or str(REPO / ".venv" / "bin" / "kaggle")
    )


def _user() -> str:
    return os.environ.get("KAGGLE_USER") or sys.exit("set KAGGLE_USER")


def main(argv: list[str] | None = None) -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", choices=sorted(SLUGS))
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="checkout whose data/ and checkpoints/ to describe"
    )
    parser.add_argument("--apply", action="store_true", help="update the metadata on Kaggle")
    parser.add_argument(
        "--json", type=Path, help="write the card as dataset-metadata.json fields to this file"
    )
    parser.add_argument(
        "--stage", type=Path, help="describe the files staged in this dir (push_data.sh)"
    )
    parser.add_argument(
        "--from-kaggle", action="store_true", help="describe the files the dataset has on Kaggle"
    )
    parser.add_argument(
        "--cover", type=Path, help=f"also copy the slug's cover into this dir as {COVER_NAME}"
    )
    parser.add_argument(
        "--list-slugs", action="store_true", help="print the slugs that have a card"
    )
    args = parser.parse_args(argv)
    if args.list_slugs:
        print("\n".join(sorted(SLUGS)))
        return
    ROOT = args.root.resolve()
    if args.stage is not None and not args.slug:
        parser.error("--stage describes one upload: give its --slug")
    slugs = [args.slug] if args.slug else sorted(SLUGS)

    def files(slug: str) -> list[str]:
        if args.stage is not None:
            return staged_files(args.stage)
        if args.from_kaggle:
            return kaggle_files(f"{_user()}/{slug}", _kaggle())
        return local_files(slug)

    if args.json:
        args.json.write_text(
            json.dumps(card(slugs[0], files(slugs[0]), args.stage), indent=2) + "\n"
        )
        if args.cover:
            cover(slugs[0], args.cover)
        return
    if not args.apply:
        for slug in slugs:
            c = card(slug, files(slug), args.stage)
            print(f"# {c['title']}\n_{c['subtitle']}_\n\n{c['description']}")
            print(
                f"Sources: {c['userSpecifiedSources']}\nUpdate frequency: {c['expectedUpdateFrequency']}"
            )
            print(f"Cover: {COVERS / (slug + '.png')}\nFiles ({len(c['resources'])}):")  # type: ignore[arg-type]
            for r in c["resources"]:  # type: ignore[attr-defined]
                print(f"- {r['path']}: {r['description']}")
        return
    user, kaggle = _user(), _kaggle()
    for slug in slugs:
        apply(slug, user, kaggle, args.stage)
        print(f"updated {user}/{slug}")


if __name__ == "__main__":
    main()
