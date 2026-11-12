"""Qwen3-VL (Apache-2.0) through MLX on Apple Silicon, for clips with no activity labels.

Asks for a small JSON verdict over up to 4 frames of a clip (one for a still image). Only
fields that map cleanly onto the taxonomy are kept; anything unparseable is dropped rather
than guessed. Objects are kept as positives only: leaving an object out of a short list is
no evidence it is absent.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from bulwark.data.taxonomy import Taxonomy, default_taxonomy

DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
OBJECTS = (
    "phone",
    "drink",
    "food",
    "cigarette",
    "sunglasses",
    "eyeglasses",
    "face_mask",
    "helmet",
    "safety_vest",
    "seatbelt",
)

PROMPT = """You are labelling {shown} of a person, for a safety-monitoring dataset.
Answer with ONLY a JSON object with these keys:
  "context": one of {context}
  "activity": one of {activity}
  "attention_direction": one of {direction}
  "alertness": one of {alertness}
  "objects": list from {objects}
  "caption": one short sentence describing what the person is doing
If you cannot see the person's face, use "alert" only if their posture clearly shows it, otherwise "unknown"."""


class VlmTeacher:
    def __init__(
        self, model: str = DEFAULT_MODEL, frames_per_clip: int = 4, max_tokens: int = 160
    ) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self._generate = generate
        self._template = apply_chat_template
        self.model, self.processor = load(model)
        self.config = load_config(model)
        self.frames_per_clip = frames_per_clip
        self.max_tokens = max_tokens
        self.taxonomy = default_taxonomy()

    def clip(self, frames: np.ndarray) -> dict[str, Any]:
        from PIL import Image

        picks = np.linspace(0, len(frames) - 1, min(self.frames_per_clip, len(frames))).astype(int)
        images = [Image.fromarray(frames[i]) for i in picks]
        question = build_prompt(self.taxonomy, len(images))
        prompt = self._template(self.processor, self.config, question, num_images=len(images))
        result = self._generate(
            self.model, self.processor, prompt, images, max_tokens=self.max_tokens, verbose=False
        )
        text = getattr(result, "text", result)
        return parse_answer(str(text), self.taxonomy)


def build_prompt(taxonomy: Taxonomy, frames: int) -> str:
    """The prompt for `frames` images of one clip (1 = a still image)."""
    shown = "a still image" if frames == 1 else f"a short video clip ({frames} frames in time order)"
    return PROMPT.format(
        shown=shown,
        context=list(taxonomy["context"].classes),
        activity=list(taxonomy["activity"].classes),
        direction=list(taxonomy["attention_direction"].classes),
        alertness=[*taxonomy["alertness"].classes, "unknown"],
        objects=list(OBJECTS),
    )


def parse_answer(text: str, taxonomy: Any) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    labels: dict[str, Any] = {}
    for key in ("context", "activity", "attention_direction", "alertness"):
        value = str(answer.get(key, "")).strip().lower()
        if value in taxonomy[key].classes:
            labels[key] = value
    objects = answer.get("objects", [])
    named = (
        {str(o).strip().lower() for o in objects if isinstance(o, str)}
        if isinstance(objects, list)
        else set()
    )
    for name in sorted(named & set(OBJECTS)):
        labels[name] = 1.0
    caption = answer.get("caption")
    if isinstance(caption, str) and caption:
        labels["_caption"] = caption[:200]
    return labels
