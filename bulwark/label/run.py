"""Run teachers over shards and merge their labels into the manifest.

Teacher output goes to `<labels_dir>/<teacher>.jsonl` (one line per clip), and runs are
resumable: clips already in the file are skipped. Merging rebuilds `labels_teacher` from
scratch each time, so a re-run picks up changed teacher files. Rules:
  * ground truth always wins, and teacher output never becomes ground truth;
  * derived > face > objects > vlm > prompted labels when sources disagree. Derived labels
    (`meta["derived_labels"]`) are computed by an adapter from the dataset's own
    annotations with a rule of this project's, so they are not ground truth (DMD's
    microsleep); prompted labels (`meta["prompted_labels"]`) come from a synthetic clip's
    prompt;
  * face, objects and VLM teacher output is dropped for eye crops (there is no face to
    find, no hands or scene in which to rule an object out, and no person or scene to
    describe); `teacher_applies` says which clips a teacher's output is kept for, so the
    CLI can skip the rest. The face teacher's camera-relative `attention_direction` (and a
    missed face) is kept only for cameras that roughly face the person
    (FRONTAL_FACE_VIEWS);
  * the face teacher's eyes_closed, blink and yawn keep only the values in FACE_KEEP:
    eyes_closed 0, and no blink or yawn value at all. A value is kept when ground truth
    confirmed it at least 90% of the time on the train split and the class that keeping
    only that value leaves out is rare in the clips it labels, so the one-sided labels it
    then gives there are close to the truth. The face teacher thus labels neither blink
    nor yawn; a synthetic clip's prompted labels still can (v0 has none, so there both
    train on DMD's ground truth only). Its eye_openness and perclos
    (EYE_CLOSURE_LABELS, from the same closure measure) follow its eyes_closed:
    on a clip with ground-truth eyes_closed they stay only when the teacher agrees with
    it, elsewhere they go where FACE_KEEP drops its eyes_closed. These were measured on a
    camera that faces the person, so on other cameras all of them go. The kept
    eyes_closed is one class, so on clips without ground truth the teacher gives that
    head negatives only;
  * event labels (blink, head nod, yawn) are dropped for still images;
  * keys starting with "_" (e.g. the VLM caption) go to record.meta.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from bulwark.data.manifest import ClipRecord
from bulwark.data.shards import read_shard, shard_index
from bulwark.data.taxonomy import default_taxonomy

console = Console()
PRIORITY = ("vlm", "objects", "face")  # later entries override earlier ones

# (dataset -> views) whose camera roughly faces the person, so camera-frame head pose
# approximates direction relative to the task. None = every view of that dataset. Views
# not listed (side cameras such as State Farm's, Drive&Act's cabin cameras) are left out
# until their geometry is checked.
FRONTAL_FACE_VIEWS: dict[str, frozenset[str] | None] = {
    "dmd": frozenset({"face"}),
    "nthu_ddd": None,
    "uta_rldd": None,
    "yawdd": frozenset({"dash"}),
}
EVENT_LABELS = ("blink", "head_nod", "yawn")
# Face-teacher values kept, per binary label. A value is kept only when
#   1. ground truth confirmed it at least 90% of the time on the train split (DMD face
#      camera, clips where the teacher found a face; `bulwark label agreement --split
#      train`, measured 2026-10-29), and
#   2. the class that keeping only that value leaves out is rare in the clips it labels
#      (train, video, cameras in FRONTAL_FACE_VIEWS, clips without ground truth for the
#      label: all DMD face camera in v0), so the one-sided labels it then gives there are
#      close to the truth.
# Val and test ground truth play no part, so they never shape training labels.
#   eyes_closed 0 right 98.9% (tn 1587, fn 18); 1 right 66.8% (tp 147, fp 73): the
#     teacher (MediaPipe's eyeBlink blendshape) reads looking down (texting, reading) as
#     closed eyes. 0 is kept: it leaves out closed eyes, which are rare in the clips it
#     labels (it calls 710 of 9,054 closed, 481 of them clips DMD marks as texting or not
#     looking at the road);
#   blink 1 right 97.3% (tp 1621, fp 45); 0 right 55.1% (tn 1127, fn 920): short blinks
#     fall between sampled frames. Neither is kept: 1 leaves out "no blink", which is
#     common (it says no blink on 4,127 of the 7,166 clips it labels and finds one on the
#     other 3,039), so its blinks alone would pull the blink head towards "blink" on
#     every such clip;
#   yawn 1 right 100% (tp 64, fp 0); 0 right 83.2% (tn 1469, fn 297): covered or partial
#     yawns are missed. Neither is kept: 1 leaves out "no yawn", which is common. On v0
#     dropping 1 changes nothing, since it finds no yawn in the 9,049 clips it labels;
#     it stops frontal datasets added later (YawDD's dash camera, which has no yawn
#     ground truth) from getting positives-only yawn rows.
# The face teacher therefore labels neither blink nor yawn. A synthetic clip's prompted
# labels still reach labels_teacher (scripts/synth_wan.py prompts yawn 1 in one scene,
# positives only); v0 has none, so there blink and yawn train on DMD's ground truth only.
FACE_KEEP: dict[str, tuple[float, ...]] = {"eyes_closed": (0.0,), "blink": (), "yawn": ()}
# Face-teacher labels read from the same eye-closure measure as eyes_closed, dropped with
# it: their perclos MAE against ground truth on the train split is 0.192 where the teacher
# says closed and 0.103 where it says open (0.113 overall; `bulwark label agreement`
# reports the overall MAE, the split is the same pairs grouped by the teacher's
# eyes_closed). Where ground truth confirms the teacher's closed eyes they are kept, since
# no dataset gives video eye_openness ground truth and they are its only closed-eye targets.
EYE_CLOSURE_LABELS = ("eye_openness", "perclos")
# Face-teacher labels whose reliability was measured only on a camera that faces the
# person (DMD's face camera): dropped on cameras outside FRONTAL_FACE_VIEWS.
FRONTAL_ONLY_FACE_LABELS = frozenset({*FACE_KEEP, *EYE_CLOSURE_LABELS, "attention_direction"})
# Teachers that need a face, hands or a scene in view. On an eye crop their output has
# nothing behind it, the objects teacher's negatives included.
NOT_FOR_EYE_CROPS = frozenset({"face", "objects", "vlm"})


def teacher_applies(teacher: str, record: ClipRecord) -> bool:
    """False when merge_teacher_labels drops `teacher`'s output for `record` (an eye crop
    and a teacher in NOT_FOR_EYE_CROPS), so running the teacher there is wasted work."""
    return not (teacher in NOT_FOR_EYE_CROPS and record.meta.get("crop_kind") == "eye")


def faces_camera(record: ClipRecord) -> bool:
    """True when the record's camera is in FRONTAL_FACE_VIEWS."""
    if record.dataset not in FRONTAL_FACE_VIEWS:
        return False
    views = FRONTAL_FACE_VIEWS[record.dataset]
    return views is None or record.view in views


def trusted_face_labels(record: ClipRecord, labels: dict[str, Any]) -> dict[str, Any]:
    """The face teacher's labels for `record` without the values FACE_KEEP leaves out.

    Its EYE_CLOSURE_LABELS follow its eyes_closed. Where the clip has ground-truth
    eyes_closed, that decides: they stay when the teacher's eyes_closed agrees with it
    (closed eyes confirmed by ground truth keep their low eye_openness) and go when it
    does not. Elsewhere they go when FACE_KEEP drops the teacher's eyes_closed."""
    drop = {k for k, keep in FACE_KEEP.items() if k in labels and labels[k] not in keep}
    closed, truth = labels.get("eyes_closed"), record.labels_gt.get("eyes_closed")
    if closed is not None and truth is not None:
        if (float(closed) >= 0.5) != (float(truth) >= 0.5):
            drop.update(EYE_CLOSURE_LABELS)
    elif "eyes_closed" in drop:
        drop.update(EYE_CLOSURE_LABELS)
    return {k: v for k, v in labels.items() if k not in drop}


def _face_labels(record: ClipRecord, labels: dict[str, Any]) -> dict[str, Any]:
    labels = trusted_face_labels(record, labels)
    if faces_camera(record):
        return labels
    # Off-axis cameras: the direction is camera-relative, the eye, blink and yawn values
    # were measured only on a camera facing the person, and a missed face is more often a
    # profile or a small face than a face out of view.
    return {
        k: v
        for k, v in labels.items()
        if k not in FRONTAL_ONLY_FACE_LABELS and not (k == "face_visible" and not v)
    }


def run_teacher(
    name: str,
    teacher: Callable[[np.ndarray], dict[str, Any]],
    shards: Iterable[Path],
    labels_dir: Path,
    *,
    only: set[str] | None = None,
    limit: int = 0,
) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    out = labels_dir / f"{name}.jsonl"
    done: set[str] = set()
    if out.exists():
        done = {
            json.loads(line)["clip_id"]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line
        }
    count = 0
    with out.open("a", encoding="utf-8") as handle:
        for shard in shards:
            # Pick the clips from the index, so clips already labelled are never decoded.
            wanted = set(shard_index(shard).values()) - done
            if only is not None:
                wanted &= only
            if not wanted:
                continue
            for clip_id, frames in read_shard(shard, only=wanted):
                try:
                    labels = teacher(frames)
                except Exception as exc:  # one bad clip must not stop a multi-hour run
                    console.print(f"[yellow]{name} failed on {clip_id}: {exc}[/yellow]")
                    continue
                handle.write(json.dumps({"clip_id": clip_id, "labels": labels}) + "\n")
                count += 1
                if count % 500 == 0:
                    handle.flush()
                    console.print(f"{name}: {count} clips")
                if limit and count >= limit:
                    return out
    return out


def load_teacher_labels(labels_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{teacher: {clip_id: labels}}"""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(labels_dir.glob("*.jsonl")):
        table: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                table[row["clip_id"]] = row["labels"]
        found[path.stem] = table
    return found


def merge_teacher_labels(records: list[ClipRecord], labels_dir: Path) -> list[ClipRecord]:
    taxonomy = default_taxonomy()
    tables = load_teacher_labels(labels_dir)
    order = sorted(tables, key=lambda t: PRIORITY.index(t) if t in PRIORITY else -1)
    for record in records:
        # Weak labels from a synthetic clip's prompt sit below every real teacher.
        merged: dict[str, Any] = dict(record.meta.get("prompted_labels", {}))
        for teacher in order:
            if not teacher_applies(teacher, record):
                continue
            labels = tables[teacher].get(record.clip_id, {})
            if teacher == "face":
                labels = _face_labels(record, labels)
            for key, value in labels.items():
                if key.startswith("_"):
                    record.meta[key.lstrip("_")] = value
                elif key in taxonomy.labels:
                    merged[key] = value
        # Rules over the dataset's own annotations outrank model teachers.
        merged.update(record.meta.get("derived_labels", {}))
        if record.media_type == "image":
            for key in EVENT_LABELS:
                merged.pop(key, None)
        if record.meta.get("yawn_video"):
            # Older merges copied the face teacher's yawn into labels_gt for YawDD "Yawning"
            # videos; the adapter never writes one there, so any value is that copy.
            record.labels_gt.pop("yawn", None)
        record.labels_teacher = {k: v for k, v in merged.items() if k not in record.labels_gt}
        taxonomy.validate(record.labels_teacher)
    return records
