"""Train an adapter for your own labels from a CSV of clips, using a Bulwark model's embeddings.

    PYTHONPATH=. python examples/train_adapter.py MODEL clips.csv --out adapters/unsafe_reach.pt

clips.csv has a header and the columns `path,label`, plus optional `subject` (who is on
camera) and `start` (seconds into a video; default 0). Paths may be videos or images and
are relative to the CSV. Each row becomes one clip of --window seconds.

With a `subject` column, about --holdout of the subjects are held out first and the
adapter is scored on them, so the number reflects people it has not seen; the saved
adapter is then refitted on every row. The adapter records which model produced its
embeddings, and `predict_video` and `predict_image` refuse to run it on another one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from bulwark.adapters import fit_adapter
from bulwark.inference import CLIP_FRAMES, load, read_clip


def held_out(subject: str, fraction: float) -> bool:
    """Stable per-subject assignment, so reruns hold out the same people."""
    return int(hashlib.sha1(subject.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("clips", type=Path, help="CSV with path,label[,subject][,start]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding", default="tokens", choices=["tokens", "pooled", "state"])
    parser.add_argument("--hidden", type=int, default=0, help="0 = linear adapter, >0 = MLP width")
    parser.add_argument("--window", type=float, default=2.0, help="seconds per clip")
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of subjects to score on")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with args.clips.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    model = load(args.model, device=args.device)
    frames = model.clip_frames or CLIP_FRAMES
    embeddings = []
    for row in rows:
        clip = read_clip(
            args.clips.parent / row["path"], start=float(row.get("start") or 0), window_seconds=args.window,
            clip_frames=frames, size=model.input_size,
        )
        embeddings.append(model.embed(clip, kind=args.embedding)[0])
    if len({e.shape for e in embeddings}) > 1:
        # E1 gives one token per frame, so an image (1 frame) and a video clip (16) differ in
        # length. The adapter mean-pools tokens anyway, so pool here to the same vectors.
        embeddings = [e.mean(0) for e in embeddings]
    x = np.stack(embeddings)
    labels = [row["label"] for row in rows]
    print(f"Embedded {len(rows)} clips with {model.kind}: {dict(Counter(labels))}")

    fit = {
        "hidden": args.hidden,
        "embedding": args.embedding,
        "source": f"{model.kind}:{args.model.name}",
        "model": model.embedding_source(args.embedding),
    }
    classes = tuple(sorted(set(labels)))
    if rows and "subject" in rows[0] and args.holdout > 0:
        test = np.array([held_out(row["subject"], args.holdout) for row in rows])
        train_labels = [label for label, t in zip(labels, test, strict=True) if not t]
        truth = np.array(labels)[test]
        if not test.any() or len(set(train_labels)) < 2:
            print("Not enough subjects or classes to hold any out; skipping the held-out score.")
        elif len(set(truth)) < 2:
            # A constant prediction would score perfectly on a single class.
            print(f"The held-out subjects only have class {str(truth[0])!r}; skipping the held-out score.")
        else:
            adapter = fit_adapter(x[~test], train_labels, classes=classes, **fit)
            pred = np.array(adapter.predict(x[test]))
            recall = {str(c): float((pred[truth == c] == c).mean()) for c in sorted(set(truth))}
            subjects = len({row["subject"] for row, t in zip(rows, test, strict=True) if t})
            print(f"Held-out subjects: {subjects}, clips: {int(test.sum())}; recall per class {recall}; "
                  f"balanced accuracy {np.mean(list(recall.values())):.3f} (chance {1 / len(set(truth)):.3f})")

    adapter = fit_adapter(x, labels, classes=classes, **fit)
    print(f"Wrote {adapter.save(args.out)}")


if __name__ == "__main__":
    main()
