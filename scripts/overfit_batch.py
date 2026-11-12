"""Check the pipeline before a long GPU run: overfit one batch of real clips end to end.

Cuts clips from a video, labels them with the MediaPipe face teacher, then trains
Bulwark-B1 (frozen V-JEPA + LoRA on the top blocks + heads) on that single batch. The
loss must fall sharply; if it does not, something between pixels and loss is broken.

    python scripts/overfit_batch.py --video path/to/cabin.mp4 --steps 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bulwark.data.clips import load_clip
from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import probe_video
from bulwark.data.taxonomy import default_taxonomy
from bulwark.label.face_teacher import FaceTeacher
from bulwark.model.bulwark import BulwarkBase
from bulwark.model.losses import MultiTaskLoss, encode_targets
from bulwark.train.common import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--face-model", type=Path, default=Path("checkpoints/pretrained/face_landmarker.task"))
    parser.add_argument("--out", type=Path, default=Path("runs/overfit"))
    args = parser.parse_args()

    fps, total = probe_video(args.video)
    span = round(2 * fps)
    starts = [round(i * (total - span) / max(1, args.clips - 1)) for i in range(args.clips)]
    records = [
        ClipRecord(
            clip_id=f"overfit/{args.video.name}@{s}", dataset="overfit", subject_id="overfit:demo",
            media_path=args.video.name, media_type="video", licence="internal", commercial_ok=False,
            start_frame=s, num_frames=16, fps=fps, frame_stride=max(1, span // 16),
        )
        for s in starts
    ]
    clips = [load_clip(r, args.video.parent, args.size) for r in records]
    teacher = FaceTeacher(args.face_model)
    labels = [teacher.clip(c) for c in clips]
    for record, label in zip(records, labels, strict=True):
        print(record.start_frame, json.dumps(label))

    device = pick_device()
    taxonomy = default_taxonomy()
    model = BulwarkBase(taxonomy, lora_blocks=4).to(device).train()
    loss_fn = MultiTaskLoss(taxonomy, learn_weights=False).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    frames = torch.from_numpy(np.stack(clips)).to(device)
    with torch.no_grad():
        model.heads.fit_input_stats(model.tokens(frames))
    targets = encode_targets(labels, [{}] * len(labels), taxonomy, teacher_weight=1.0).to(device)

    history = []
    started = time.time()
    for step in range(args.steps):
        loss, parts = loss_fn(model(frames), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss))
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d} loss {float(loss):.4f}  ({time.time() - started:.0f}s)")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "device": str(device),
        "clips": len(clips),
        "labels_per_clip": [len(label) for label in labels],
        "loss_first": history[0],
        "loss_last": history[-1],
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if history[-1] > 0.2 * history[0]:
        raise SystemExit("Loss did not drop by 5x: check the pipeline before training for real.")


if __name__ == "__main__":
    main()
