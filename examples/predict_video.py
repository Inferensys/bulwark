"""Run a Bulwark model over a video and write a per-second timeline as JSON.

    PYTHONPATH=. python examples/predict_video.py MODEL VIDEO --out runs/timeline.json \\
        --labels activity,eyes_closed,perclos,yawn \\
        --temperature runs/eval/calibration.json --adapter unsafe_reach=adapters/unsafe_reach.pt

MODEL is a B1 checkpoint (`bulwark train lora`), a heads checkpoint (`bulwark train
heads`, frozen encoder), an E1 checkpoint (`bulwark train distill`), an exported E1
`.onnx` file or a release directory. Each window covers --window seconds (16 frames);
windows start every --hop seconds. An image (.jpg, .jpeg, .png, .bmp) is one 1-frame
clip, as the image datasets were trained, and gives a single entry. Labels whose training
targets were masked on video (or, for an image, on images) are left out (`masked_labels`);
an ONNX graph exported with --frames 1 cuts 1-frame windows, which count as images.
--labels selects outputs in place of the labels the model records; a label outside those
comes from a head that may have had no training targets.
Values are model outputs, not calibrated probabilities, unless the model records
temperatures or --temperature points at the calibration.json that `bulwark evaluate`
wrote for this model; that file also gives the val-tuned thresholds of binary labels,
which the JSON lists next to the timeline. Both are those fitted for the input's media
type (image for an image, or for a --frames 1 graph; video otherwise), which the JSON
names under "media". `bulwark predict` is the command-line equivalent
(`--perclos-seconds`, `--calibration`, `--out`).

When the outputs include `perclos` (the fraction of a clip's frames with the eyes
closed), each window also gets PERCLOS over the last --perclos-seconds (default 60; at
least one --window) of consecutive windows, under "rolling"; it is null until that much
video has been seen.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bulwark.adapters import Adapter
from bulwark.inference import (
    IMAGE_SUFFIXES,
    PERCLOS_SECONDS,
    RollingMean,
    load,
    predict_image,
    predict_video,
)


def headline(name: str, value: Any) -> str:
    if isinstance(value, dict):
        top = max(value, key=lambda cls: value[cls])
        return f"{name}={top} ({value[top]:.2f})"
    return f"{name}={value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--labels", default="", help="comma-separated labels to return (default: those the model records)")
    parser.add_argument("--adapter", action="append", default=[], help="name=path of an adapter to add (repeatable)")
    parser.add_argument("--temperature", type=Path, default=None, help="calibration.json from `bulwark evaluate`")
    parser.add_argument("--window", type=float, default=2.0, help="seconds per clip")
    parser.add_argument("--hop", type=float, default=1.0, help="seconds between clip starts")
    parser.add_argument("--perclos-seconds", type=float, default=PERCLOS_SECONDS, help="rolling PERCLOS span (0 = off)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    labels = [name for name in args.labels.split(",") if name] or None
    try:
        model = load(args.model, device=args.device, labels=labels, temperature=args.temperature)
    except ValueError as error:  # e.g. a calibration.json fitted on other weights
        raise SystemExit(str(error)) from None
    adapters = {name: Adapter.load(path) for name, _, path in (item.partition("=") for item in args.adapter)}
    image = args.video.suffix.lower() in IMAGE_SUFFIXES
    media = "image" if image or model.clip_frames == 1 else "video"  # a 1-frame graph cuts 1-frame windows
    returned = model.labels_for(media)
    temperature = model.temperature_for(media)
    print(f"{model.kind} from {args.model}; {len(returned)} {media} outputs, {model.input_size}px input")
    if not model.labels_known:
        print("The model does not record which labels it was trained on, so every head is returned, including "
              "heads that had no training targets. Pass --labels with the labels its training data covered.")
    if not temperature:
        print("No temperatures: outputs are uncalibrated.")

    perclos = None
    if args.perclos_seconds > 0 and not image:
        if "perclos" not in returned:
            print(f"The {media} outputs do not include perclos, so no rolling PERCLOS.")
        elif args.hop > args.window:
            print(f"Windows every {args.hop:g} s do not cover the video ({args.window:g} s each), so no rolling PERCLOS.")
        elif args.perclos_seconds < args.window:
            print(f"--perclos-seconds {args.perclos_seconds:g} is shorter than one {args.window:g} s window, so no rolling PERCLOS.")
        else:
            perclos = RollingMean("perclos", seconds=args.perclos_seconds)
    key = f"perclos_{args.perclos_seconds:g}s"

    windows: Iterable[dict[str, Any]]
    if image:
        try:
            windows = [predict_image(model, args.video, adapters=adapters)]
        except ValueError as error:  # e.g. a 16-frame ONNX graph
            raise SystemExit(str(error)) from None
    else:
        windows = predict_video(model, args.video, window_seconds=args.window, hop_seconds=args.hop, adapters=adapters)
    timeline = []
    for window in windows:
        shown = [headline(name, value) for name, value in list(window["outputs"].items())[:4]]
        shown += [headline(name, probs) for name, probs in window.get("adapters", {}).items()]
        if perclos is not None:
            window["rolling"] = {key: perclos.update(window)}
            if window["rolling"][key] is not None:
                shown.append(headline(key, window["rolling"][key]))
        timeline.append(window)
        print(f"{window['start']:7.1f}s  " + "  ".join(shown))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "model": str(args.model), "kind": model.kind, "video": str(args.video), "window_s": args.window, "hop_s": args.hop,
            "labels_known": model.labels_known, "labels": list(returned), "media": media,
            "masked_labels": {name: list(kinds) for name, kinds in model.masked_labels.items()},
            "temperature": temperature, "thresholds": model.thresholds_for(media),
        }
        if perclos is not None:
            meta["perclos_seconds"] = args.perclos_seconds
        args.out.write_text(json.dumps({**meta, "windows": timeline}, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {len(timeline)} windows to {args.out}")


if __name__ == "__main__":
    main()
