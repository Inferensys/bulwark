"""Generate synthetic gap-fill clips with Wan 2.2 TI2V-5B (Apache-2.0 model; the card claims
no rights over outputs, subject to its use restrictions).

Targets what the public data lacks: microsleeps at night, IR-looking cabins, operator cabs
on forklifts and cranes. Each clip gets a sidecar JSON with the labels its prompt intends.
Those are weak labels, never ground truth: the adapter stores them in
`meta["prompted_labels"]`, they train at teacher weight, and labels from the real teachers
override them. All takes of one prompt share a subject group (`scene<NN>`), since they
show the same scripted event.

Needs a CUDA GPU (docs/training.md lists the memory it needs); its generation speed has
not been measured for this project:
    pip install "diffusers>=0.35" transformers accelerate imageio[ffmpeg]
    python scripts/synth_wan.py --out data/raw/synthetic_wan --per-prompt 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SUBJECTS = [
    "a middle-aged South Asian man", "a young East Asian woman", "an older white man with a grey beard",
    "a Black woman in her thirties", "a Latino man in his twenties wearing glasses", "a Middle Eastern woman wearing a headscarf",
]

SCENES: list[tuple[str, dict[str, object]]] = [
    ("driving a car at night on a highway, dashboard-mounted camera facing the driver, infrared night-vision look, "
     "monochrome, the driver's eyes slowly close and the head drops forward, then jerks back up",
     {"alertness": "microsleep", "head_nod": 1.0, "modality": "nir", "low_light": 1.0, "context": "vehicle_cabin"}),
    ("driving a truck at night, camera on the windshield facing the driver, infrared monochrome, "
     "the driver yawns widely and rubs their eyes",
     {"alertness": "drowsy", "yawn": 1.0, "modality": "nir", "low_light": 1.0, "context": "vehicle_cabin"}),
    ("driving a car in daylight, camera facing the driver, the driver looks down at a phone held in their lap and types",
     {"activity": "texting", "phone": 1.0, "attention_direction": "down", "context": "vehicle_cabin", "modality": "rgb"}),
    ("operating a forklift in a warehouse, camera mounted in the cab facing the operator, the operator looks over their "
     "shoulder while reversing, alert",
     {"alertness": "alert", "context": "industrial_vehicle", "modality": "rgb"}),
    ("sitting in a crane cab high above a construction site, camera facing the operator, the operator's eyelids are heavy "
     "and they blink slowly, tired",
     {"alertness": "drowsy", "context": "industrial_vehicle", "modality": "rgb"}),
    ("working at a factory control desk late at night, webcam view, the worker's head nods forward as they fall asleep",
     {"alertness": "microsleep", "head_nod": 1.0, "context": "workplace", "low_light": 1.0, "modality": "rgb"}),
    ("driving a bus, camera facing the driver, the driver is attentive, eyes on the road, both hands on the wheel",
     {"alertness": "alert", "activity": "attentive", "hands_on_controls": "both", "context": "vehicle_cabin", "modality": "rgb"}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic_wan"))
    parser.add_argument("--per-prompt", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    fps = 24
    frames = int(args.seconds * fps) // 4 * 4 + 1  # Wan expects 4k+1 frames
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene_index, (scene, labels) in enumerate(SCENES):
        for take in range(args.per_prompt):
            name = f"scene{scene_index:02d}_take{take:02d}"
            if (args.out / f"{name}.mp4").exists():
                continue
            subject = rng.choice(SUBJECTS)
            prompt = f"Realistic in-cabin safety camera footage of {subject} {scene}. Natural lighting, no text, no logos."
            video = pipe(
                prompt=prompt,
                negative_prompt="cartoon, illustration, text, watermark, distorted face, extra limbs",
                height=704, width=1280, num_frames=frames, guidance_scale=5.0,
                generator=torch.Generator("cuda").manual_seed(args.seed * 1000 + scene_index * 100 + take),
            ).frames[0]
            export_to_video(video, str(args.out / f"{name}.mp4"), fps=fps)
            sidecar = {"prompt": prompt, "labels": labels, "subject_group": f"scene{scene_index:02d}"}
            (args.out / f"{name}.json").write_text(json.dumps(sidecar))
            print(name)


if __name__ == "__main__":
    main()
