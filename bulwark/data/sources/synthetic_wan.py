"""Synthetic clips from scripts/synth_wan.py: <name>.mp4 + <name>.json {prompt, labels}.

The labels say what the prompt asked for, not what the generator drew, so they are never
ground truth: they go to `meta["prompted_labels"]`, and label merging turns them into
teacher-weight labels that real teachers can override (bulwark/label/run.py). The event a
prompt describes happens once per clip, so each clip is one sample spanning its full
length. The dataset is train-only (bulwark/data/splits.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import SourceAdapter, probe_video
from bulwark.data.taxonomy import default_taxonomy


class SyntheticWanAdapter(SourceAdapter):
    name = "synthetic_wan"

    def records(self) -> Iterator[ClipRecord]:
        taxonomy = default_taxonomy()
        for sidecar in sorted(self.root.glob("*.json")):
            video = sidecar.with_suffix(".mp4")
            if not video.exists():
                continue
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            prompted = taxonomy.validate(dict(info["labels"]))
            fps, total = probe_video(video)
            if fps <= 0 or total <= 0:
                continue
            yield from self.video_windows(
                video,
                info.get("subject_group", sidecar.stem),
                {},
                clip_seconds=total / fps,
                meta={
                    "label_quality": "prompted",
                    "prompt": info["prompt"][:300],
                    "prompted_labels": prompted,
                },
            )
