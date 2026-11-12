from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from bulwark.data.manifest import (
    SPLITS,
    ClipRecord,
    merge_manifests,
    read_manifest,
    read_manifest_metadata,
    write_manifest,
)
from bulwark.data.registry import DATASETS, datasets_markdown
from bulwark.data.removals import drop_removed_subjects
from bulwark.data.splits import assign_splits, check_disjoint, policy_metadata
from bulwark.export.release import RESEARCH_LICENCE
from bulwark.label.objects_teacher import MODEL_SIZES as OBJECT_MODEL_SIZES

if TYPE_CHECKING:
    from bulwark.inference import BulwarkModel
    from bulwark.train.common import TrainConfig

TAGLINE = "Research video model for driver monitoring systems: eye closure, PERCLOS, blinks, yawns, phone use, eating, drinking and activity from a 2-second in-cabin clip."
app = typer.Typer(no_args_is_help=True, help=TAGLINE)
train_app = typer.Typer(no_args_is_help=True, help="Training jobs (CPU, Apple MPS or CUDA).")
label_app = typer.Typer(no_args_is_help=True, help="Teacher pseudo-labelling, and teacher agreement with ground truth.")
app.add_typer(train_app, name="train")
app.add_typer(label_app, name="label")
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[1]
# Dataset roots on this machine. Relative roots are relative to the directory that holds
# configs/ (the repository root), so dataset roots resolve the same from any working
# directory; the data/, runs/ and checkpoints/ defaults of the options are relative to the
# working directory.
DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"

ManifestOption = Annotated[Path, typer.Option("--manifest", help="Merged, split clip manifest (parquet)")]
ShardsOption = Annotated[Path, typer.Option("--shards", help="Directory of tar shards written by `bulwark shard`")]
LabelsOption = Annotated[Path, typer.Option("--labels", help="Directory of teacher label files (<teacher>.jsonl)")]
LimitOption = Annotated[int, typer.Option("--limit", help="Stop after N clips (0 = all)")]
PatternOption = Annotated[str, typer.Option("--pattern", help="Shard glob, e.g. 'dmd-*.tar'")]
# What loading a model or checkpoint can raise for a bad file: reported without a traceback.
# torch.load(weights_only=True) raises UnpicklingError for files that are not plain checkpoints.
LOAD_ERRORS = (ValueError, KeyError, OSError, ImportError, pickle.UnpicklingError)


def _fail(exc: BaseException) -> typer.Exit:
    """Print an error without a traceback; returns the Exit to raise."""
    message = str(exc)
    if isinstance(exc, pickle.UnpicklingError):
        detail = next((line.strip() for line in message.splitlines() if "Unsupported global" in line), "")
        message = (
            "Not loaded: torch.load(weights_only=True) refuses the file. It is not a checkpoint, or it holds Python "
            f"objects besides tensors and plain values, which Bulwark checkpoints never do. {detail.split(' was not')[0]}"
        ).strip()
    console.print(f"[red]{message}[/red]")
    return typer.Exit(1)


def _roots(config: Path | None = None, *, required: bool = True) -> dict[str, Path]:
    """Dataset roots from configs/data.yaml. Raises when the file is missing, unless not `required`."""
    config = config or DATA_CONFIG
    if not config.exists():
        if required:
            console.print(f"[red]{config} is missing; it maps each dataset to its local root.[/red]")
            raise typer.Exit(1)
        return {}
    base = config.resolve().parent.parent
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return {name: (base / Path(path).expanduser()).resolve() for name, path in (data.get("roots") or {}).items()}


@app.command()
def datasets(
    markdown: Annotated[Path | None, typer.Option("--markdown", help="Write the dataset table as Markdown to this file instead of printing it")] = None,
) -> None:
    """List known datasets, or write docs/datasets.md."""
    if markdown:
        markdown.write_text(datasets_markdown(), encoding="utf-8")
        console.print(f"Wrote {markdown}")
        return
    roots = _roots(required=False)
    table = Table("dataset", "licence", "commercial", "GB", "adapter", "local")
    for spec in DATASETS:
        local = roots.get(spec.name)
        table.add_row(
            spec.name, spec.licence[:40], "yes" if spec.commercial_ok else "no", f"{spec.size_gb:g}",
            "yes" if spec.adapter else "-", "yes" if local and local.exists() else "-",
        )
    console.print(table)


@app.command()
def ingest(
    dataset: Annotated[str, typer.Argument(help="Dataset name, as listed by `bulwark datasets`")],
    root: Annotated[Path | None, typer.Option("--root", help="Dataset root directory; defaults to roots.<dataset> in configs/data.yaml")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Directory for <dataset>.parquet")] = Path("data/manifests"),
    limit: Annotated[int, typer.Option("--limit", help="Stop after N records (0 = all)")] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Parse and print a sample record without writing the manifest")] = False,
) -> None:
    """Convert one dataset's native annotations into data/manifests/<dataset>.parquet."""
    from bulwark.data.sources import get_adapter

    root = root or _roots().get(dataset)
    if root is None or not root.exists():
        raise typer.BadParameter(f"No local root for {dataset}; pass --root or set it in {DATA_CONFIG}")
    records = []
    for record in get_adapter(dataset, root).records():
        records.append(record.validate())
        if limit and len(records) >= limit:
            break
    records, removed = drop_removed_subjects(records)
    if removed:
        console.print(f"{dataset}: dropped {removed} clips of removed subjects (bulwark/data/removed_subjects.yaml)")
    console.print(f"{dataset}: {len(records)} clips, {len({r.subject_id for r in records})} subjects")
    if records:
        sample = records[len(records) // 2]
        console.print(json.dumps({"clip_id": sample.clip_id, "labels_gt": sample.labels_gt, "meta": sample.meta}, indent=2))
    if not dry_run and records:
        path = write_manifest(records, out_dir / f"{dataset}.parquet")
        console.print(f"Wrote {path}")


@app.command()
def split(
    manifests: Annotated[Path, typer.Option("--manifests", help="Directory of per-dataset manifests to merge")] = Path("data/manifests"),
    out: Annotated[Path, typer.Option("--out", help="Merged manifest to write")] = Path("data/manifest.parquet"),
    dedup: Annotated[bool, typer.Option("--dedup/--no-dedup", help="Drop cross-dataset duplicate images before splitting")] = True,
) -> None:
    """Merge per-dataset manifests, drop removed subjects and cross-dataset duplicate images, assign splits.

    `excluded` holds clips neither trained on nor scored by `bulwark evaluate`: every clip of a dataset the split policy excludes (zenodo_driver, by a maintainer decision), and the held-out camera view's clips of subjects outside the test split (the Drive&Act mid-level probe, `bulwark probe drive_and_act_midlevel`, still uses those). The merged manifest records the split policy in its file metadata, which `bulwark evaluate` and `report` read.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.data.splits import SplitPolicy

    policy = SplitPolicy()
    paths = sorted(manifests.glob("*.parquet"))
    if not paths:
        console.print(f"[red]No per-dataset manifests in {manifests}; run `bulwark ingest <dataset>` first.[/red]")
        raise typer.Exit(1)
    records, removed = drop_removed_subjects(merge_manifests(paths))
    if removed:
        console.print(f"Dropped {removed} clips of removed subjects (bulwark/data/removed_subjects.yaml)")
    if dedup and len({r.dataset for r in records if r.media_type == "image"}) > 1:
        from bulwark.data.dedup import drop_cross_dataset_duplicates

        # Datasets with person ids first, then those whose ids name groups of people (AUC v2's
        # two driver groups), so a copy is kept where it can be scored per person.
        present = {r.dataset for r in records} - set(policy.train_only_datasets)
        groups = set(policy.subject_group_datasets)
        evaluable = sorted(present - groups) + sorted(present & groups)
        commercial_before = {r.clip_id for r in records if r.commercial_ok}
        try:
            records, dropped = drop_cross_dataset_duplicates(records, _roots(), prefer_evaluable=evaluable)
        except ValueError as exc:
            console.print(f"[red]De-duplication failed: {exc} Fix the roots in {DATA_CONFIG}, or pass --no-dedup.[/red]")
            raise typer.Exit(1) from exc
        kept = [r for r in records if r.meta.get("near_duplicate_of")]
        lost = sum(1 for r in kept if r.clip_id in commercial_before and not r.commercial_ok)
        console.print(
            f"Dropped {dropped} cross-dataset duplicate images; {len(kept)} kept images have copies in other datasets "
            f"(meta near_duplicate_of), {lost} of them lost commercial_ok because a copy is not commercially usable"
        )
    records = assign_splits(records, policy)
    check_disjoint(records, policy)
    write_manifest(records, out, metadata=policy_metadata(policy))
    counts = Counter((r.dataset, r.split) for r in records)
    columns = ("train", "val", "test", "holdout", "excluded")
    table = Table("dataset", *columns)
    for name in sorted({d for d, _ in counts}):
        table.add_row(name, *(str(counts[(name, s)]) for s in columns))
    console.print(table)
    console.print(f"Wrote {out} ({len(records)} clips)")


def _sharded(out: Path, name: str) -> tuple[set[str], int, list[Path]]:
    """Clip ids already in a dataset's `<name>-NNNNN.tar` shards, the next free shard number,
    and shards that cannot be read. The writer renames a shard to `.tar` only once its
    index.json is written (an interrupted one leaves `.tar.partial`, which is not read), so
    an unreadable `.tar` comes from an older version of `bulwark shard` or a damaged copy.

    Clips in `<name>-rNNN-NNNNN.tar` files, a name an earlier version of this command wrote
    for added clips, count as sharded too (scripts/check_shards.py reads both names)."""
    import re
    import tarfile

    from bulwark.data.shards import shard_index

    pattern = re.compile(rf"{re.escape(name)}(-r\d{{3}})?-(\d{{5}})\.tar")
    ids: set[str] = set()
    numbers = [-1]
    broken: list[Path] = []
    for path in sorted(out.glob(f"{name}-*.tar")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        if not match[1]:
            numbers.append(int(match[2]))
        try:
            ids.update(shard_index(path).values())
        except (ValueError, tarfile.TarError):
            broken.append(path)
    return ids, max(numbers) + 1, broken


def shard_record_path(out: Path, name: str) -> Path:
    """`<dataset>.shards.json` next to a dataset's shards: frame size, clips that failed to
    decode and the fingerprint of how the sharded clips are cut (see `shard_fingerprint`)."""
    return out / f"{name}.shards.json"


def shard_fingerprint(records: Iterable[ClipRecord]) -> dict[str, str]:
    """sha256 digests of the sharded clips' ids and of how each is cut from its source
    (media path, start frame, frame count, stride, crop). Clip ids name a source and a start
    frame only, so an adapter change in sampling or cropping keeps them; the second digest
    catches it. scripts/check_shards.py computes the same digests from the manifest."""
    ids, cuts = hashlib.sha256(), hashlib.sha256()
    for r in sorted(records, key=lambda r: r.clip_id):
        crop = None if r.crop is None else list(r.crop)
        ids.update(f"{r.clip_id}\n".encode())
        cuts.update((json.dumps([r.clip_id, r.media_path, int(r.start_frame), int(r.num_frames), int(r.frame_stride), crop]) + "\n").encode())
    return {"clip_ids": ids.hexdigest(), "cuts": cuts.hexdigest()}


def _read_shard_record(out: Path, name: str) -> dict[str, Any]:
    path = shard_record_path(out, name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.command()
def shard(
    manifest: ManifestOption = Path("data/manifest.parquet"),
    out: Annotated[Path | None, typer.Option("--out", help="Shard directory (default data/shards, or data/shards-sample with --sample)")] = None,
    size: Annotated[int, typer.Option("--size", help="Frame side length in pixels (frames are letterboxed)")] = 256,
    clips_per_shard: Annotated[int, typer.Option("--clips-per-shard", help="Clips per tar shard")] = 1000,
    dataset: Annotated[str | None, typer.Option("--dataset", help="Only shard this dataset")] = None,
    sample: Annotated[int, typer.Option("--sample", help="Random clips per split, for smoke runs (0 = all)")] = 0,
) -> None:
    """Decode clips to letterboxed JPEG frame stacks (256 px by default) in tar shards, one shard set per dataset.

    Re-running adds only the manifest clips that are not in a shard yet, in new shard files numbered after the existing ones; existing shards are never rewritten. Clips in shards but no longer in the manifest are reported as stale (training ignores them). <dataset>.shards.json records the clips that fail to decode and a fingerprint of how the sharded clips are cut; a re-run refuses when the manifest now cuts them differently under the same clip ids, and scripts/check_shards.py fails. Every manifest clip is sharded, whatever its split: the Drive&Act mid-level probe (`bulwark probe`) uses the `excluded` camera-view clips too. Clips of an excluded dataset (zenodo_driver) are sharded as well, but nothing trains on or scores them.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    import random
    from datetime import datetime, timezone

    import cv2

    from bulwark.data.clips import load_clip
    from bulwark.data.shards import ShardWriter

    out = out or Path("data/shards-sample" if sample else "data/shards")
    roots = _roots()
    manifest_records = [r for r in read_manifest(manifest) if dataset is None or r.dataset == dataset]
    records = manifest_records
    if sample:
        rng = random.Random(0)
        picked = []
        for split_name in sorted({r.split for r in records}):
            group = [r for r in records if r.split == split_name]
            picked += rng.sample(group, min(sample, len(group)))
        records = sorted(picked, key=lambda r: (r.media_path, r.start_frame))  # sequential reads
    by_dataset: dict[str, list[ClipRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
    for name, group in by_dataset.items():
        existing, next_shard, broken = _sharded(out, name)
        if broken:
            console.print(
                f"[red]{name}: {len(broken)} shard(s) have no index or cannot be read (written by an older, interrupted "
                f"`bulwark shard`, or damaged), e.g. {broken[0]}. Delete them and run again.[/red]"
            )
            raise typer.Exit(1)
        in_manifest = {r.clip_id: r for r in manifest_records if r.dataset == name}
        stale = len(existing - set(in_manifest))
        previous = _read_shard_record(out, name)
        recorded = previous.get("fingerprint") or {}
        now = shard_fingerprint(in_manifest[c] for c in existing if c in in_manifest)
        if recorded.get("clip_ids") == now["clip_ids"] and recorded.get("cuts") != now["cuts"]:
            console.print(
                f"[red]{name}: the manifest cuts already-sharded clips differently than when they were sharded (media path, "
                f"start frame, frame count, stride or crop changed under the same clip ids). Delete {out}/{name}-*.tar and rebuild.[/red]"
            )
            raise typer.Exit(1)
        missing = [r for r in group if r.clip_id not in existing]
        console.print(f"{name}: {len(group) - len(missing)} of {len(group)} clips already sharded, {len(missing)} to add, {stale} stale")
        if stale:
            console.print(f"[yellow]{name}: {stale} sharded clips are not in the manifest; delete {out}/{name}-*.tar and rebuild to drop them.[/yellow]")
        if not missing:
            continue
        if name not in roots:
            console.print(f"[red]{name}: no root in {DATA_CONFIG}[/red]")
            raise typer.Exit(1)
        if existing and previous.get("frame_size", size) != size:
            console.print(f"[red]{name}: the existing shards hold {previous['frame_size']} px frames; pass --size {previous['frame_size']}, or delete {out}/{name}-*.tar and rebuild.[/red]")
            raise typer.Exit(1)
        errors: dict[str, str] = {}
        # Numbering continues after the existing shards, so files from earlier runs are never overwritten.
        with ShardWriter(out, prefix=name, clips_per_shard=clips_per_shard, first_shard=next_shard) as writer:
            for record in missing:
                try:
                    writer.add(record.clip_id, load_clip(record, roots[name], size))
                except (FileNotFoundError, ValueError, cv2.error) as exc:
                    message = str(exc).strip()
                    errors[record.clip_id] = message.splitlines()[0][:200] if message else type(exc).__name__
                    if len(errors) <= 5:
                        console.print(f"[yellow]{record.clip_id}: {exc}[/yellow]")
        failed = len(errors)
        attempted = {r.clip_id for r in missing}
        # Failures of clips this run did not try again (another --sample) stay listed.
        earlier = {c: e for c, e in (previous.get("failed") or {}).items() if c not in attempted and c not in existing}
        sharded = (existing | (attempted - set(errors))) & set(in_manifest)
        shard_record_path(out, name).write_text(json.dumps({
            "dataset": name,
            "frame_size": size,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": shard_fingerprint(in_manifest[c] for c in sharded),
            "failed": dict(sorted({**earlier, **errors}.items())),
        }, indent=1) + "\n", encoding="utf-8")
        console.print(f"{name}: {len(missing) - failed} clips -> {len(writer.written)} shards ({failed} failed)")
        if failed and failed == len(missing):
            if not existing:
                console.print(f"[red]{name}: every clip failed; is its root in {DATA_CONFIG} right ({roots[name]})?[/red]")
                raise typer.Exit(1)
            # A re-run retries clips that failed before; when only those remain, that is not an error.
            console.print(f"[yellow]{name}: {failed} clips still fail to decode and stay out of the shards.[/yellow]")


@label_app.command("face")
def label_face(
    shards: ShardsOption = Path("data/shards"),
    labels: LabelsOption = Path("data/labels"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    model: Annotated[Path, typer.Option("--model", help="MediaPipe face landmarker model file")] = Path("checkpoints/pretrained/face_landmarker.task"),
    limit: LimitOption = 0,
    yaw_sign: Annotated[float, typer.Option("--yaw-sign", help="Flip the sign of MediaPipe yaw (+1 or -1)")] = 1.0,
    pattern: PatternOption = "*.tar",
) -> None:
    """MediaPipe eyes / mouth / head-pose teacher (skips eye-crop images, which hold no face)."""
    from bulwark.label.face_teacher import FaceTeacher
    from bulwark.label.run import run_teacher

    only, eye_crops = _teacher_clips("face", _read_manifest(manifest))
    console.print(f"Face teacher on {len(only)} clips; {eye_crops} eye crops skipped")
    teacher = FaceTeacher(model, yaw_sign=yaw_sign)
    console.print(f"Wrote {run_teacher('face', teacher.clip, sorted(shards.glob(pattern)), labels, only=only, limit=limit)}")


@label_app.command("objects")
def label_objects(
    shards: ShardsOption = Path("data/shards"),
    labels: LabelsOption = Path("data/labels"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    size: Annotated[str, typer.Option("--size", help=f"RF-DETR model size: {', '.join(OBJECT_MODEL_SIZES)}")] = "medium",
    limit: LimitOption = 0,
    pattern: PatternOption = "*.tar",
) -> None:
    """RF-DETR phone / drink / food negatives (a detection never makes a positive); skips eye-crop images."""
    from bulwark.label.objects_teacher import ObjectsTeacher
    from bulwark.label.run import run_teacher

    if size not in OBJECT_MODEL_SIZES:
        raise typer.BadParameter(f"choose from {', '.join(OBJECT_MODEL_SIZES)}, got {size!r}", param_hint="--size")
    only, eye_crops = _teacher_clips("objects", _read_manifest(manifest))
    console.print(f"Objects teacher on {len(only)} clips; {eye_crops} eye crops skipped")
    teacher = ObjectsTeacher(size)
    console.print(f"Wrote {run_teacher('objects', teacher.clip, sorted(shards.glob(pattern)), labels, only=only, limit=limit)}")


@label_app.command("vlm")
def label_vlm(
    shards: ShardsOption = Path("data/shards"),
    labels: LabelsOption = Path("data/labels"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    model: Annotated[str, typer.Option("--model", help="MLX vision-language model id")] = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    limit: LimitOption = 0,
    pattern: PatternOption = "*.tar",
) -> None:
    """Qwen3-VL (MLX) teacher, only for clips with no ground-truth activity; skips eye-crop images."""
    from bulwark.label.run import run_teacher
    from bulwark.label.vlm_teacher import VlmTeacher

    only, eye_crops = _teacher_clips("vlm", _read_manifest(manifest), lambda r: "activity" not in r.labels_gt)
    console.print(f"VLM teacher on {len(only)} clips without ground-truth activity; {eye_crops} eye crops skipped")
    teacher = VlmTeacher(model)
    console.print(f"Wrote {run_teacher('vlm', teacher.clip, sorted(shards.glob(pattern)), labels, only=only, limit=limit)}")


def _teacher_clips(teacher: str, records: list[ClipRecord], wanted: Callable[[ClipRecord], bool] | None = None) -> tuple[set[str], int]:
    """Clip ids a teacher runs on: the `wanted` records, less those whose output `bulwark
    label merge` drops (eye crops, for the teachers that need a face or a scene), and the
    number of those left out."""
    from bulwark.label.run import teacher_applies

    candidates = [r for r in records if wanted is None or wanted(r)]
    only = {r.clip_id for r in candidates if teacher_applies(teacher, r)}
    return only, len(candidates) - len(only)


@label_app.command("merge")
def label_merge(
    manifest: ManifestOption = Path("data/manifest.parquet"),
    labels: LabelsOption = Path("data/labels"),
) -> None:
    """Fold teacher labels into labels_teacher (ground truth always wins; of the face teacher's eyes_closed, blink and yawn only eyes_closed 0 is kept, and only on cameras that face the person, so the face teacher labels neither blink nor yawn, though a synthetic clip's prompted labels still can: FACE_KEEP in bulwark/label/run.py); the manifest keeps its split policy record."""
    from bulwark.label.run import merge_teacher_labels

    records = merge_teacher_labels(read_manifest(manifest), labels)
    write_manifest(records, manifest, metadata=read_manifest_metadata(manifest))
    covered = Counter(k for r in records for k in r.labels_teacher)
    console.print(f"Merged teacher labels into {manifest}: {dict(covered.most_common(12))}")


@label_app.command("agreement")
def label_agreement(
    manifest: ManifestOption = Path("data/manifest.parquet"),
    labels: LabelsOption = Path("data/labels"),
    out: Annotated[Path | None, typer.Option("--out", help='Also write the full table as JSON to this file: {"splits": [...] or "all", "report": {teacher: {label: {dataset: stats}}}}')] = None,
    split: Annotated[list[str] | None, typer.Option("--split", help="Only clips of this split, e.g. --split train (repeatable or comma-separated; default: every split)")] = None,
) -> None:
    """How well each teacher agrees with ground truth, on clips that have both.

    For binary labels, precision is the share of the teacher's positives that ground truth confirms, and npv (negative predictive value) the share of the teacher's negatives that ground truth confirms. Datasets the split policy excludes (zenodo_driver, by a maintainer decision) are left out, so their ground truth is never scored or published here. The face-teacher values that `bulwark label merge` keeps (FACE_KEEP in bulwark/label/run.py: eyes_closed 0, and no blink or yawn value) were chosen with `--split train` alone, so val and test ground truth never shape training labels.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.eval.report import manifest_policy
    from bulwark.label.agreement import teacher_agreement

    splits = _split_filter(split)
    records = _read_manifest(manifest)
    try:
        excluded = set(manifest_policy(manifest).excluded_datasets)
    except ValueError as exc:
        raise _fail(exc) from exc
    report = teacher_agreement([r for r in records if r.dataset not in excluded], labels, splits=splits)
    console.print(f"Splits: {', '.join(sorted(splits)) if splits else 'all'}")
    table = Table("teacher", "label", "dataset", "n", "agreement")
    for teacher, by_label in report.items():
        for name, by_dataset in by_label.items():
            for dataset, stats in by_dataset.items():
                if "f1" in stats:
                    text = (
                        f"F1 {_num(stats['f1'])} (precision {_num(stats['precision'])}, "
                        f"npv {_num(stats['npv'])}, recall {_num(stats['recall'])})"
                    )
                elif "accuracy" in stats:
                    text = f"accuracy {_num(stats['accuracy'])}"
                else:
                    text = f"MAE {_num(stats['mae'])}"
                table.add_row(teacher, name, dataset, str(stats["n"]), text)
    console.print(table if report else "No clip has both a teacher label and ground truth for the same label.")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        # The splits scored go in the file, so a train-only table can be told from an all-split one.
        payload = {"splits": sorted(splits) if splits else "all", "report": report}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        console.print(f"Wrote {out}")


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _split_filter(values: list[str] | None) -> set[str] | None:
    """--split values, repeatable or comma-separated; None (every split) when none are given.
    A --split that names no split (an empty shell variable, say) is an error, not every split."""
    names = {name.strip() for value in values or () for name in value.split(",") if name.strip()}
    if values and not names:
        raise typer.BadParameter("names no split; leave --split out to score every split", param_hint="--split")
    unknown = names - SPLITS
    if unknown:
        raise typer.BadParameter(f"unknown split(s) {sorted(unknown)}; choose from {sorted(SPLITS)}", param_hint="--split")
    return names or None


def _read_manifest(manifest: Path) -> list[ClipRecord]:
    if not manifest.exists():
        console.print(f"[red]{manifest} does not exist; build it with `bulwark split`.[/red]")
        raise typer.Exit(1)
    return read_manifest(manifest)


@app.command()
def coverage(
    manifest: ManifestOption = Path("data/manifest.parquet"),
    split_: Annotated[str, typer.Option("--split", help="Split to check for labels that cannot be learned")] = "train",
    markdown: Annotated[Path | None, typer.Option("--markdown", help="Write the full coverage table (label x dataset x split) as Markdown")] = None,
) -> None:
    """Pre-flight check: labels whose targets in a split cover one class only, or miss classes.

    A head trained on one class learns a constant. Counts are per media type, since videos and images train in separate batches.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.data.coverage import (
        coverage_markdown,
        label_coverage,
        missing_classes,
        single_class_labels,
    )

    records = _read_manifest(manifest)
    single = single_class_labels(records, split_)
    missing = missing_classes(records, split_)
    table = Table("label", "media", "problem")
    for (name, media), reason in single.items():
        table.add_row(name, media, reason)
    for (name, media), classes in missing.items():
        if (name, media) not in single:
            table.add_row(name, media, f"no {split_} targets for {', '.join(classes)}")
    console.print(table if single or missing else f"Every label with {split_} targets covers all its classes.")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(coverage_markdown(label_coverage(records)), encoding="utf-8")
        console.print(f"Wrote {markdown}")


def _train_config(config: Path | None, overrides: list[str], *, fresh: bool = False, allow_mismatch: bool = False, overwrite: bool = False) -> TrainConfig:
    from bulwark.train.common import load_config

    parsed: dict[str, Any] = {}
    for item in overrides:
        key, _, value = item.partition("=")
        parsed[key] = yaml.safe_load(value)
    for key, flag in (("fresh", fresh), ("allow_mismatch", allow_mismatch), ("overwrite", overwrite)):
        if flag:
            parsed[key] = True
    try:
        return load_config(config, **parsed)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _train(run: Callable[[TrainConfig], Path], config: TrainConfig) -> Path:
    """Run a training job; refusals to resume or to mix runs are reported without a traceback."""
    from bulwark.train.common import ResumeError
    from bulwark.train.provenance import ProvenanceError

    try:
        return run(config)
    except (ResumeError, ProvenanceError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


ConfigOption = Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False, help="Training YAML config; defaults are used if omitted")]
SetOption = Annotated[list[str], typer.Option("--set", help="Override a config key, e.g. --set batch_size=8")]
FreshOption = Annotated[bool, typer.Option("--fresh", help="Move an earlier run in `out` aside (out/stale-<time>/) and start over (= --set fresh=true)")]
MismatchOption = Annotated[bool, typer.Option("--allow-mismatch", help="Resume a last.pt written with another config or data, never across commercial_only or release_only (= --set allow_mismatch=true)")]
OverwriteOption = Annotated[bool, typer.Option("--overwrite", help="Clear a features dir cached by a different encoder, init or precision, or one without meta.json (= --set overwrite=true); a dir of this run resumes")]


@train_app.command("cache")
def train_cache(config: ConfigOption = None, set_: SetOption = [], overwrite: OverwriteOption = False) -> None:  # noqa: B006
    """Encode every shard once and store summary tokens (and teacher logits if init is set).

    The job resumes shard by shard and has no last.pt, so fresh and allow_mismatch do not apply; --overwrite starts a dir of another run over.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.train.cache_features import check_config, run

    settings = _train_config(config, set_, overwrite=overwrite)
    try:
        check_config(settings)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Features in {_train(run, settings)}")


@train_app.command("heads")
def train_heads(config: ConfigOption = None, set_: SetOption = [], fresh: FreshOption = False, allow_mismatch: MismatchOption = False) -> None:  # noqa: B006
    """Train multi-task heads on cached features."""
    from bulwark.train.train_heads import run

    console.print(f"Checkpoints in {_train(run, _train_config(config, set_, fresh=fresh, allow_mismatch=allow_mismatch))}")


@train_app.command("lora")
def train_lora(config: ConfigOption = None, set_: SetOption = [], fresh: FreshOption = False, allow_mismatch: MismatchOption = False) -> None:  # noqa: B006
    """End-to-end LoRA fine-tune of the encoder + heads from shards."""
    from bulwark.train.finetune_lora import run

    console.print(f"Checkpoints in {_train(run, _train_config(config, set_, fresh=fresh, allow_mismatch=allow_mismatch))}")


@train_app.command("distill")
def train_distill(config: ConfigOption = None, set_: SetOption = [], fresh: FreshOption = False, allow_mismatch: MismatchOption = False) -> None:  # noqa: B006
    """Distil Bulwark-B1 into the Bulwark-E1 edge student."""
    from bulwark.train.distill import run

    console.print(f"Checkpoints in {_train(run, _train_config(config, set_, fresh=fresh, allow_mismatch=allow_mismatch))}")


def _parity_clips(shards: Path, frames: int, input_size: int, count: int = 4) -> np.ndarray | None:
    """A few real video clips with the shape the exported graph takes, for the parity check."""
    from bulwark.data.shards import read_shard

    clips = []
    for path in sorted(shards.glob("*.tar")):
        for _, stack in read_shard(path):
            if stack.shape[:3] == (frames, input_size, input_size):
                clips.append(stack)
                if len(clips) >= count:
                    return np.stack(clips)
    return np.stack(clips) if clips else None


@app.command()
def export(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, help="Distilled Bulwark-E1 checkpoint (best.pt or final.pt)")],
    out: Annotated[Path, typer.Option("--out", help="ONNX file to write")] = Path("checkpoints/bulwark-e1.onnx"),
    frames: Annotated[int, typer.Option("--frames", help="Frames per clip in the exported graph (fixed; 1 for still images)")] = 16,
    size: Annotated[int | None, typer.Option("--size", help="Student resolution; defaults to the img_size saved in the checkpoint")] = None,
    input_size: Annotated[int, typer.Option("--input-size", help="Side of the letterboxed uint8 frames the graph takes (the shards use 256)")] = 256,
    shards: Annotated[Path | None, typer.Option("--shards", help="Shard directory; real clips from it are used for the parity check")] = None,
    embeddings: Annotated[bool, typer.Option("--embeddings/--no-embeddings", help="Add the `tokens` and `state` outputs (embeddings for adapters); --no-embeddings exports the label logits only")] = True,
    manifest: Annotated[Path | None, typer.Option("--manifest", exists=True, dir_okay=False, help="Training manifest; records which outputs had ground truth, only teacher labels, or no targets (supervised_labels stay the checkpoint's own when it records them)")] = None,
    calibration: Annotated[Path | None, typer.Option("--calibration", "--temperature", exists=True, dir_okay=False, help="calibration.json from `bulwark evaluate` of this checkpoint; stores its temperatures and val-tuned thresholds for the graph's media type: video for --frames above 1, image for --frames 1 (--temperature is the same option, as in predict and release)")] = None,
    weights_licence: Annotated[str, typer.Option("--weights-licence", help="Licence stored in the graph; one that does not restrict use to non-commercial research needs a commercial_only run cleared by `bulwark release-check --commercial`")] = RESEARCH_LICENCE,
    bench: Annotated[bool, typer.Option("--bench/--no-bench", help="Report ONNX Runtime CPU latency after export")] = True,
    runs: Annotated[int, typer.Option("--runs", help="Timed runs per latency measurement")] = 100,
) -> None:
    """Export Bulwark-E1 to ONNX, check it against PyTorch, and report CPU latency.

    The graph returns the label logits and, unless --no-embeddings, the `tokens` and `state` embeddings that adapters read. Its metadata carries the input/output contract, the weights licence, the model card, the checkpoint's training record and its supervised and masked labels, with --manifest the training targets each output had (and the supervised labels, for a checkpoint that records none), and with --calibration the temperatures and thresholds fitted for the graph's media type (one per label; the contract's `calibration` entry says which media type). Latency results are also written to <out>.bench.json, with the graph's sha256.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    import os

    from bulwark.eval.provenance import sha256_file
    from bulwark.eval.report import read_calibration
    from bulwark.export.onnx import benchmark, export_edge, read_contract, write_benchmark
    from bulwark.inference import media_type_for
    from bulwark.inference import read_calibration as read_fitted

    clips = _parity_clips(shards, frames, input_size) if shards else None
    if shards and clips is None:
        console.print(f"[yellow]No {frames}-frame {input_size} px clips in {shards}; the parity check uses random frames.[/yellow]")
    media = media_type_for(frames)
    try:
        temperature = thresholds = record = chosen = None
        if calibration:
            read_calibration(calibration, sha256_file(checkpoint))  # refuses a file fitted on other weights
            chosen = read_fitted(calibration).for_media(media)  # one value per label: the graph's media type
            temperature, thresholds = chosen.temperature, chosen.thresholds
            record = {"media": media, "pooled": sorted(chosen.pooled), "no_val_clips": sorted(chosen.no_val_clips)}
        path = export_edge(
            checkpoint, out, frames=frames, size=size, input_size=input_size, parity_frames=clips,
            embeddings=embeddings, manifest=manifest, temperature=temperature, thresholds=thresholds,
            calibration=record, licence=weights_licence,
        )
    except (AssertionError, RuntimeError, *LOAD_ERRORS) as exc:
        raise _fail(exc) from exc
    terms = read_contract(path)
    console.print(f"Wrote {path}: input {terms['input']['shape']} uint8, student at {terms['model_img_size']} px")
    parity = terms["parity"]
    extra = "".join(f", {k} {parity[k]:.2e}" for k in ("max_abs_state_diff", "max_rel_token_diff") if k in parity)
    console.print(f"Parity with PyTorch on {parity['batch']}: max |Δlogit| {parity['max_abs_logit_diff']:.2e}{extra}")
    status = terms.get("label_status")
    if status:
        console.print(f"Outputs: {len(status['ground_truth'])} trained with ground truth, {len(status['teacher_only'])} teacher-only, untrained: {', '.join(status['untrained']) or 'none'}")
        if status.get("not_trained_by_checkpoint"):
            console.print(f"Not in the checkpoint's supervised_labels, so not returned by default: {', '.join(status['not_trained_by_checkpoint'])}")
        if terms.get("supervised_labels_from", "checkpoint") != "checkpoint":
            console.print("[yellow]The checkpoint records no supervised_labels; the graph lists every output with a training target in --manifest, including labels training may have masked.[/yellow]")
        if status["from"]["matches_training_manifest"] is False:
            console.print("[yellow]--manifest is not the manifest the checkpoint was trained with (sha256 differs); the label status may not match its training data.[/yellow]")
    if chosen is not None:
        console.print(f"Calibration for {media} clips: {len(chosen.temperature)} temperatures, {len(chosen.thresholds or {})} thresholds")
        if chosen.pooled:
            console.print(
                f"[yellow]{calibration} has no {media} entries for {', '.join(sorted(chosen.pooled))} (written before "
                "calibration.json had entries per media type), so the graph carries their values fitted on all val clips. "
                "Re-score the evaluation's predictions with `bulwark report` to fit them per media type.[/yellow]"
            )
        if chosen.no_val_clips:
            console.print(
                f"[yellow]Val had no {media} clips for {', '.join(sorted(chosen.no_val_clips))}, so the graph carries no "
                "threshold or temperature for them.[/yellow]"
            )
    if not terms["release_check"]["cleared"]:
        console.print(f"[yellow]Not cleared for publication: {' '.join(terms['release_check']['problems'])}[/yellow]")
    if bench:
        table = Table("threads", "p50 ms", "p95 ms", "windows/s (p50)")
        results = []
        for threads in sorted({1, os.cpu_count() or 1}):
            result = benchmark(path, runs=runs, threads=threads)
            results.append(result)
            table.add_row(str(threads), f"{result['ms_p50']:.1f}", f"{result['ms_p95']:.1f}", f"{result['windows_per_s_p50']:.2f}")
        console.print(table)
        console.print(f"{result['cpu']}, onnxruntime {result['onnxruntime']}, input {result['input']}; {result['scope']}.")
        console.print(f"Wrote {write_benchmark(path, results)}")


def _splits(value: str) -> list[str]:
    from bulwark.eval.report import EVAL_SPLITS

    parts = [s.strip() for s in value.split(",") if s.strip()]
    unknown = set(parts) - set(EVAL_SPLITS)
    if unknown:
        raise typer.BadParameter(f"Unknown split(s) {sorted(unknown)}; choose from {EVAL_SPLITS}")
    return parts


BootstrapOption = Annotated[int, typer.Option("--bootstrap", help="Subject-level bootstrap draws for 95% intervals (0 = none)")]
ReferenceOption = Annotated[Path | None, typer.Option("--reference", exists=True, file_okay=False, help="Predictions dir of another model (e.g. B1) to compare against")]


@app.command()
def evaluate(
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", exists=True, help="Heads, B1 or E1 checkpoint (.pt), or an exported E1 graph (.onnx)")] = None,
    model: Annotated[str, typer.Option("--model", help="auto, heads, logits, b1, e1 or onnx; auto reads the checkpoint")] = "auto",
    features: Annotated[Path, typer.Option("--features", help="Cached features dir: frozen-encoder tokens (heads) or B1 logits (logits). The default is the dir configs/train/cache.yaml writes and heads.yaml reads; for logits pass the dir cache-b1.yaml writes, data/features/b1-lora-v0")] = Path("data/features/frozen"),
    shards: ShardsOption = Path("data/shards"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    out: Annotated[Path, typer.Option("--out", help="Directory for predictions/, metrics.json, calibration.json and report.html")] = Path("runs/eval"),
    splits: Annotated[str, typer.Option("--splits", help="Comma-separated splits to predict; val only tunes thresholds and temperatures")] = "val,test,holdout",
    reference: ReferenceOption = None,
    bootstrap: BootstrapOption = 1000,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Clips per forward pass (b1, e1, onnx)")] = 16,
    device: Annotated[str, typer.Option("--device", help="auto, cpu, mps or cuda (b1, e1)")] = "auto",
    precision: Annotated[str, typer.Option("--precision", help="auto, fp32, fp16 or bf16 autocast for B1")] = "auto",
    limit: Annotated[int, typer.Option("--limit", help="At most N clips per split, for smoke checks (0 = all); recorded in the report")] = 0,
) -> None:
    """Score a model on test and holdout splits; writes predictions, metrics.json, calibration.json and report.html.

    heads and logits read cached features; b1, e1 and onnx stream the tar shards.
    """
    from bulwark.eval.predict import MODEL_KINDS
    from bulwark.eval.report import evaluate as run_evaluation

    if model not in MODEL_KINDS:
        raise typer.BadParameter(f"--model must be one of {MODEL_KINDS}")
    if checkpoint is None and model not in ("auto", "logits"):
        raise typer.BadParameter(f"--model {model} needs --checkpoint")
    try:
        path = run_evaluation(
            model, checkpoint, manifest=manifest, out=out, features=features, shards=shards, splits=_splits(splits),
            reference=reference, bootstrap=bootstrap, batch_size=batch_size, device=device, precision=precision, limit=limit,
        )
    except LOAD_ERRORS as exc:
        raise _fail(exc) from exc
    console.print(f"Wrote {path}")


@app.command()
def report(
    predictions: Annotated[Path, typer.Option("--predictions", exists=True, file_okay=False, help="Predictions dir written by `bulwark evaluate`")],
    manifest: ManifestOption = Path("data/manifest.parquet"),
    out: Annotated[Path, typer.Option("--out", help="Directory for metrics.json, calibration.json and report.html")] = Path("runs/eval"),
    reference: ReferenceOption = None,
    bootstrap: BootstrapOption = 1000,
) -> None:
    """Re-score saved predictions (e.g. against a new manifest, or E1 against B1)."""
    from bulwark.eval.predict import load_predictions, read_prediction_meta
    from bulwark.eval.provenance import public_path
    from bulwark.eval.report import build_report

    try:
        meta = read_prediction_meta(predictions)
        loaded = load_predictions(predictions)
        reference_loaded = load_predictions(reference) if reference else None
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if reference:
        meta["reference"] = public_path(reference)
    try:
        path = build_report(loaded, manifest, out, meta=meta, reference=reference_loaded, bootstrap=bootstrap)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Wrote {path}")


@app.command()
def probe(
    protocol: Annotated[str, typer.Argument(help="cv, drive_and_act_midlevel or uta_rldd_folds")],
    features: Annotated[Path, typer.Option("--features", help="Features cached with the frozen encoder (no init)")] = Path("data/features/frozen"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    out: Annotated[Path, typer.Option("--out", help="Directory for probe-<protocol>.json and .html")] = Path("runs/probe"),
    datasets_: Annotated[str, typer.Option("--datasets", help="cv: comma-separated datasets to cross-validate")] = "drive_and_act,mrl_eye",
    folds: Annotated[int, typer.Option("--folds", help="cv: number of subject folds")] = 5,
    root: Annotated[Path | None, typer.Option("--root", help="drive_and_act_midlevel: dataset root with activities_3s/; defaults to its root in configs/data.yaml")] = None,
    tokens: Annotated[str, typer.Option("--tokens", help="global (one 768-d token) or all (25 flattened summary tokens)")] = "global",
) -> None:
    """Linear probes on frozen features: subject k-fold CV or a dataset's official protocol."""
    from bulwark.eval.probe import PROTOCOLS, run_probe

    if protocol not in PROTOCOLS:
        raise typer.BadParameter(f"protocol must be one of {PROTOCOLS}")
    if protocol == "drive_and_act_midlevel":
        root = root or _roots().get("drive_and_act")
    names = [d.strip() for d in datasets_.split(",") if d.strip()]
    try:
        path = run_probe(protocol, features, manifest, out, datasets=names, folds=folds, root=root, pool=tokens)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Wrote {path}")



# ---- removals ------------------------------------------------------------------------------


def _feature_dirs(root: Path) -> list[Path]:
    """Every directory under `root` that holds cached features or logits (*.npz)."""
    return sorted({p.parent for p in root.rglob("*.npz")}) if root.exists() else []


@app.command()
def purge(
    subjects: Annotated[list[str] | None, typer.Argument(help="'<dataset>:<subject>' ids listed in bulwark/data/removed_subjects.yaml; default: all of them")] = None,
    clips: Annotated[Path | None, typer.Option("--clips", exists=True, dir_okay=False, help="Purge the clip ids listed in this file (one per line) instead of subjects; manifests are left alone")] = None,
    manifests: Annotated[Path, typer.Option("--manifests", help="Directory of per-dataset manifests")] = Path("data/manifests"),
    manifest: ManifestOption = Path("data/manifest.parquet"),
    shards: Annotated[list[Path] | None, typer.Option("--shards", help="Shard directory (repeatable); default data/shards and data/shards-sample")] = None,
    labels: LabelsOption = Path("data/labels"),
    features: Annotated[Path, typer.Option("--features", help="Every directory of cached features or logits under this one is purged")] = Path("data/features"),
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation")] = False,
) -> None:
    """Delete removed subjects' clips from local manifests, shards, teacher labels and feature caches.

    A subject must be listed in bulwark/data/removed_subjects.yaml (with date and source) before it is purged, so that later `bulwark ingest` and `split` runs keep it out. Saved evaluation predictions (runs/*/predictions) and reports also hold the clips' ids and outputs; delete them, or evaluate again. Copies outside this machine are not touched: afterwards, run `scripts/purge_cloud_copies.sh <dataset>`, which replaces the Azure Blob copies with the purged local data and deletes the Kaggle datasets named in KAGGLE_SLUGS (recreate them with infra/kaggle/push_data.sh). Released weights are not touched. Each purge is logged with its date in purges.jsonl next to the merged manifest.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from datetime import datetime, timezone

    from bulwark.data.removals import (
        REMOVED_PATH,
        load_removed_subjects,
        purge_clips,
        purge_subjects,
    )

    shard_dirs = shards or [Path("data/shards"), Path("data/shards-sample")]
    feature_dirs = _feature_dirs(features)
    targets = [*shard_dirs, labels, *feature_dirs]
    if clips is not None:
        if subjects:
            raise typer.BadParameter("Pass subjects or --clips, not both")
        ids = {line.strip() for line in clips.read_text(encoding="utf-8").splitlines() if line.strip()}
        what = f"{len(ids)} clip ids from {clips.name}"
    else:
        listed = load_removed_subjects()
        names = list(subjects or listed)
        if not names:
            console.print("No subjects to purge: removed_subjects.yaml lists none and none were given.")
            return
        unlisted = sorted(set(names) - set(listed))
        if unlisted:
            console.print(
                f"[red]Not listed in {REMOVED_PATH.name}: {unlisted}. Add each with its date and source first, "
                "so that `bulwark ingest` and `split` keep them out of later manifests; then purge again.[/red]"
            )
            raise typer.Exit(1)
        manifest_paths = [*sorted(manifests.glob("*.parquet")), manifest]
        removed = set(names)
        ids = {r.clip_id for path in manifest_paths if path.exists() for r in read_manifest(path) if r.subject_id in removed}
        what = f"{len(names)} subjects"
    console.print(f"Purging {what} from: {', '.join(str(t) for t in targets)}" + ("" if clips else f", {len(manifest_paths)} manifests"))
    if not yes:
        typer.confirm("This deletes data in place. Continue?", abort=True)
    counts: Counter[str] = Counter()
    for directory in shard_dirs:
        counts.update(purge_clips(ids, shards=directory))
    counts.update(purge_clips(ids, labels=labels, features=feature_dirs))
    if clips is None:
        # Manifests last: they are where the clip ids come from, so an interrupted purge can run again.
        counts.update(purge_subjects(names, manifests=manifest_paths))
    console.print(f"Removed: {dict(counts) or 'nothing (no matching clips)'}")
    log = manifest.parent / "purges.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purged": what, "counts": dict(counts)}
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    console.print(
        f"Logged to {log}. Delete saved predictions (runs/*/predictions) that hold these clips, and carry the "
        "purge over to Azure Blob and Kaggle with scripts/purge_cloud_copies.sh <dataset>."
    )


# ---- releases ------------------------------------------------------------------------------


ModelArgument = Annotated[Path, typer.Argument(exists=True, help="B1, heads or E1 checkpoint (.pt), exported E1 graph (.onnx) or release directory")]
# `bulwark predict` also takes a Hugging Face repository id (bulwark.inference.resolve_model).
HubModelArgument = Annotated[str, typer.Argument(help="B1, heads or E1 checkpoint (.pt), exported E1 graph (.onnx), release directory, or a Hugging Face repository id such as Inferensys/bulwark-b1-driver-monitoring")]


@app.command("release-check")
def release_check(
    model: ModelArgument,
    commercial: Annotated[bool, typer.Option("--commercial", help="Check for a licence that does not restrict use to non-commercial research: also needs a commercial_only run on commercial_ok datasets")] = False,
) -> None:
    """Whether these weights may be published: every training dataset cleared, clean code, and the licence that applies.

    Prints on what each dataset's clearance rests (its terms, its owners' written permission, or the maintainers' own decision, which is not a statement by the dataset's owners). Exits 1 when they may not. The registry flags are our reading of each dataset's terms or our own dated decision, not legal advice.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.export.release import check_release, training_record

    try:
        check = check_release(training_record(model), commercial=commercial)
    except LOAD_ERRORS as exc:
        raise _fail(exc) from exc
    record = check.record
    console.print(f"{record.source}: commercial_only={record.commercial_only}, code {record.commit or 'not recorded'}{' (uncommitted changes)' if record.dirty else ''}")
    table = Table("dataset", "via", "licence", "commercial", "weights release", "train clips")
    for name, entry in check.datasets.items():
        release = {True: "yes", False: "no", None: "not determined"}[entry["weights_release_ok"]]
        if not entry["in_registry"]:
            release += " (not in the registry)"
        table.add_row(
            name, ", ".join(entry["via"]), entry["licence"], "yes" if entry["commercial_ok"] else "no", release, str(entry["train_clips"] or "-")
        )
    console.print(table)
    if any(v != "training" for entry in check.datasets.values() for v in entry["via"]):
        console.print("via: training = the model's own training clips; teacher, init, features = an upstream model it learned from.")
    if not commercial:
        console.print(f"Licence for these weights: {RESEARCH_LICENCE}. The training datasets' terms apply as well:")
        for name, entry in check.datasets.items():
            console.print(f"  {name}: {entry['terms']}")
    bases: dict[str, list[str]] = {}
    for name, entry in check.datasets.items():
        if entry["weights_release_basis"]:
            bases.setdefault(entry["weights_release_basis"], []).append(name)
    if bases:
        console.print("Weights release decided on these grounds (registry weights_release_basis):")
        for basis, names in bases.items():
            console.print(f"  {', '.join(names)}: {basis}")
    if check.problems:
        for problem in check.problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Cleared for release{', on the grounds above' if bases else ''}.[/green]")


def _labels(value: str) -> list[str] | None:
    return [name.strip() for name in value.split(",") if name.strip()] or None


def _adapter_paths(values: list[str] | None) -> dict[str, Path]:
    """--adapter name=path values, checked before any model is loaded."""
    found: dict[str, Path] = {}
    for item in values or []:
        name, sep, path = (part.strip() for part in item.partition("="))
        if not sep or not name or not path:
            raise typer.BadParameter(f"--adapter takes name=path, got {item!r}")
        found[name] = Path(path)
    return found


CalibrationOption = Annotated[
    Path | None,
    typer.Option("--temperature", "--calibration", exists=True, dir_okay=False, help="calibration.json from `bulwark evaluate` of these weights: temperatures and val-tuned thresholds, fitted per media type (image or video) (--calibration is the same option, as in export)"),
]


@app.command()
def predict(
    model: HubModelArgument,
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Video file, or an image: one 1-frame clip, as the image datasets were trained (an ONNX graph needs --frames 1 at export)")],
    out: Annotated[Path | None, typer.Option("--out", help="Write the timeline as JSON here (default: print one line per window)")] = None,
    labels: Annotated[str, typer.Option("--labels", help="Comma-separated outputs to keep (default: the labels the model records)")] = "",
    adapter: Annotated[list[str] | None, typer.Option("--adapter", help="name=path of an adapter to add (repeatable)")] = None,
    calibration: CalibrationOption = None,
    window: Annotated[float, typer.Option("--window", help="Seconds per clip (16 frames)")] = 2.0,
    hop: Annotated[float, typer.Option("--hop", help="Seconds between clip starts")] = 1.0,
    perclos_seconds: Annotated[float, typer.Option("--perclos-seconds", help="Span of the rolling PERCLOS added to each window when the outputs include perclos (0 = off)")] = 60.0,
    device: Annotated[str, typer.Option("--device", help="auto, cpu, mps or cuda (PyTorch models)")] = "auto",
) -> None:
    """Per-window outputs over a video, a thin wrapper over bulwark.inference (see examples/predict_video.py).

    --temperature applies the calibration.json `bulwark evaluate` wrote for these weights (a file fitted on other weights is refused) and adds its val-tuned thresholds of binary labels to the JSON: the temperatures and thresholds of the input's media type (image for an image file or a --frames 1 graph, else video), which the JSON names under "media". When the outputs include perclos, each video window also gets PERCLOS over the last --perclos-seconds of consecutive windows, under "rolling" (null until that much video has been seen); not when --hop exceeds --window, or --perclos-seconds is shorter than one window.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.adapters import Adapter
    from bulwark.eval.provenance import public_path
    from bulwark.inference import IMAGE_SUFFIXES, RollingMean, load, predict_image, predict_video

    adapter_paths = _adapter_paths(adapter)
    try:
        loaded = load(model, device=device, labels=_labels(labels), temperature=calibration)
        adapters = {name: Adapter.load(path) for name, path in adapter_paths.items()}
        image = video.suffix.lower() in IMAGE_SUFFIXES
        # A graph exported with --frames 1 sees single frames of a video too, as images were trained.
        media = "image" if image or loaded.clip_frames == 1 else "video"
        items: Iterable[dict[str, Any]]
        if image:
            items = [predict_image(loaded, video, adapters=adapters)]
        else:
            items = predict_video(loaded, video, window_seconds=window, hop_seconds=hop, adapters=adapters)
        rolling = None
        if perclos_seconds > 0 and "perclos" in loaded.labels_for(media) and not image:
            if hop > window:
                console.print(f"Windows every {hop:g} s do not cover the video ({window:g} s each), so no rolling PERCLOS.")
            elif perclos_seconds < window:
                console.print(f"--perclos-seconds {perclos_seconds:g} is shorter than one {window:g} s window, so no rolling PERCLOS.")
            else:
                rolling = RollingMean("perclos", seconds=perclos_seconds)
        key = f"perclos_{perclos_seconds:g}s"
        windows = []
        for item in items:
            shown = [_headline(n, value) for n, value in list(item["outputs"].items())[:6]]  # masked labels are not there
            if rolling is not None:
                item["rolling"] = {key: rolling.update(item)}
                if item["rolling"][key] is not None:
                    shown.append(_headline(key, item["rolling"][key]))
            windows.append(item)
            if out is None:
                console.print(f"{item['start']:8.1f}s  " + "  ".join(shown))
    except (NotImplementedError, *LOAD_ERRORS) as exc:
        raise _fail(exc) from exc
    if not loaded.labels_known:
        console.print("[yellow]The model records no supervised labels, so every head is returned, including heads without training targets; pass --labels.[/yellow]")
    temperature = loaded.temperature_for(media)
    if not temperature:
        console.print("No temperatures: outputs are uncalibrated.")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        timeline = {  # "model": a local path as public_path gives it, or a Hugging Face repository id
            "model": public_path(model) if Path(model).exists() else model, "kind": loaded.kind, "video": public_path(video), "window_s": window, "hop_s": hop,
            "labels": list(loaded.labels_for(media)), "labels_known": loaded.labels_known, "media": media,
            "masked_labels": {name: list(kinds) for name, kinds in loaded.masked_labels.items()}, "temperature": temperature,
            "thresholds": loaded.thresholds_for(media),
            **({"perclos_seconds": perclos_seconds} if rolling is not None else {}),
            "windows": windows,
        }
        out.write_text(json.dumps(timeline, indent=1) + "\n", encoding="utf-8")
        console.print(f"Wrote {len(windows)} windows to {out}")


def _calibration_summary(model: BulwarkModel) -> str:
    """The temperatures and thresholds a model applies, as counts: per media type when it
    has values per media type."""
    if not model.calibration.media:
        return f"{len(model.temperature)} temperatures, {len(model.thresholds)} thresholds, the same on images and video"
    return "; ".join(
        f"{media} {len(model.temperature_for(media))} temperatures, {len(model.thresholds_for(media))} thresholds"
        for media in sorted(model.calibration.media)
    )


def _headline(name: str, value: object) -> str:
    if isinstance(value, dict):
        top = max(value, key=lambda cls: value[cls])
        return f"{name}={top} ({value[top]:.2f})"
    return f"{name}={value:.2f}"


@app.command()
def release(
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="B1, heads or E1 checkpoint (.pt)")],
    out: Annotated[Path, typer.Argument(help="Release directory to write (model.safetensors and config.json)")],
    weights_licence: Annotated[str, typer.Option("--weights-licence", help="Licence in config.json; one that does not restrict use to non-commercial research needs a commercial_only run cleared by `bulwark release-check --commercial`")] = RESEARCH_LICENCE,
    labels: Annotated[str, typer.Option("--labels", help="Comma-separated supervised labels (default: as the checkpoint records them)")] = "",
    calibration: CalibrationOption = None,
    backbone_checkpoint: Annotated[Path | None, typer.Option("--backbone-checkpoint", exists=True, dir_okay=False, help="Pretrained V-JEPA file for B1 (default: checkpoints/pretrained/)")] = None,
) -> None:
    """Package a checkpoint as a release directory (safetensors weights, LoRA merged, and config.json).

    Packaging is not publishing: config.json records the release check, and `bulwark release-check` says whether the weights may be published.
    """  # noqa: E501 - one line, so --help does not break it mid-sentence
    from bulwark.export.release import prepare_release
    from bulwark.inference import TorchModel, load, release_source, save_release

    try:
        model = load(checkpoint, device="cpu", labels=_labels(labels), temperature=calibration, backbone_checkpoint=backbone_checkpoint)
        if not isinstance(model, TorchModel):
            raise ValueError(f"{checkpoint} is not a PyTorch checkpoint")
        check, extra = prepare_release(checkpoint, weights_licence, kind=model.kind)
        if not model.labels_known:
            raise ValueError(
                f"{checkpoint} records no supervised labels. Pass --labels with the labels its training data covered "
                "(keys of labels_gt or labels_teacher in its training clips)."
            )
        path = save_release(model, out, weights_licence=weights_licence, extra=extra)
    except LOAD_ERRORS as exc:
        raise _fail(exc) from exc
    console.print(
        f"Wrote {model.kind} release to {path}: {len(model.labels)} labels, "
        f"commercial_only={release_source(model)['commercial_only']} (None: not recorded)"
    )
    console.print(f"Calibration: {_calibration_summary(model)}")
    if not check.ok:
        console.print(f"[yellow]Not cleared for publication: {' '.join(check.problems)}[/yellow]")


if __name__ == "__main__":
    app()
