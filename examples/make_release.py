"""Package a trained checkpoint as a release directory: safetensors weights plus config.json.

    PYTHONPATH=. python examples/make_release.py CHECKPOINT releases/bulwark-e1-v0 \\
        --temperature runs/eval/calibration.json

CHECKPOINT is a B1 checkpoint (`bulwark train lora`), a heads checkpoint (`bulwark train
heads`) or an E1 checkpoint (`bulwark train distill`). The release loads with
`bulwark.inference.load(OUT)` and needs no pretrained V-JEPA file; LoRA is merged into the
encoder. A B1 release also carries Meta's V-JEPA licence notice as LICENSE-vjepa2.

--weights-licence defaults to the research licence (CC BY-NC 4.0 plus the training
datasets' terms; `bulwark.export.release.RESEARCH_LICENCE`). A licence that does not
restrict use to non-commercial research is refused unless the checkpoint's provenance
shows commercially usable data only (commercial_only, and every dataset commercial_ok,
including those of the upstream models it learned from) and `bulwark release-check
--commercial` clears it: v0 weights trained on research-only datasets cannot be released
for commercial use. --labels names the outputs the release advertises, the labels its
training data supervised; it is needed only for a checkpoint that does not record them
(the training jobs record `supervised_labels`), and replaces that record. --temperature
adds the val-fitted temperatures and thresholds from the calibration.json that `bulwark
evaluate` wrote for this checkpoint (a file fitted on other weights is refused): those
fitted on all val clips, and under `media_calibration` those fitted per media type, which
the release applies to images and video.
config.json records the checkpoint's file name, sha256 and provenance, the labels masked
per media type, a link to the model card, the model release and the verdict of `bulwark
release-check`, but no local paths (docs/inference-and-adapters.md, "Releases").

`bulwark release` is the command-line equivalent and writes the same config.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bulwark.export.release import RESEARCH_LICENCE, prepare_release
from bulwark.inference import TorchModel, load, release_source, save_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--weights-licence", default=RESEARCH_LICENCE, help="terms the weights are released under (default: the research licence)"
    )
    parser.add_argument("--labels", default="", help="comma-separated supervised labels (default: as recorded)")
    parser.add_argument("--temperature", type=Path, default=None, help="calibration.json from `bulwark evaluate`")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None, help="pretrained V-JEPA file for B1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    labels = [name for name in args.labels.split(",") if name] or None
    try:
        model = load(
            args.checkpoint, device=args.device, labels=labels, temperature=args.temperature,
            backbone_checkpoint=args.backbone_checkpoint,
        )
        if not isinstance(model, TorchModel):
            raise SystemExit(f"{args.checkpoint} is an ONNX graph; a release is built from a PyTorch checkpoint")
        # The verdict of `bulwark release-check`, and the card of the model's own repository.
        check, extra = prepare_release(args.checkpoint, args.weights_licence, kind=model.kind)
    except ValueError as error:  # e.g. a calibration.json fitted on other weights
        raise SystemExit(str(error)) from None
    if not model.labels_known:
        raise SystemExit(
            f"{args.checkpoint} records no supervised labels. Pass --labels with the labels its training data "
            "covered (keys of labels_gt or labels_teacher in its training clips)."
        )
    try:
        out = save_release(model, args.out, weights_licence=args.weights_licence, extra=extra)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(
        f"Wrote {model.kind} release to {out}: {len(model.labels)} labels, "
        f"commercial_only={release_source(model)['commercial_only']} (None: not recorded)"
    )
    if not model.calibration.media:
        print(f"Calibration: {len(model.temperature)} temperatures, {len(model.thresholds)} thresholds, the same on images and video")
    for media in sorted(model.calibration.media):
        print(f"Calibration for {media} clips: {len(model.temperature_for(media))} temperatures, {len(model.thresholds_for(media))} thresholds")
    for path in sorted(out.iterdir()):
        print(f"  {path.name}")
    if not check.ok:
        print(f"Not cleared for publication: {' '.join(check.problems)}")


if __name__ == "__main__":
    main()
