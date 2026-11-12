"""V-JEPA 2.1 ViT-B/16 encoder, MIT (V-JEPA 2 code licence; see NOTICE), built from
Meta's code vendored in bulwark/model/vjepa21 (from VJEPA_COMMIT).

Meta's torch.hub entry currently points its weights URL at localhost, so we build the
encoder from their module directly and load the official checkpoint ourselves:
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
Fetch it with `scripts/fetch_pretrained.sh`; training and B1 checkpoints need it, a
release directory does not. The checkpoint's sha256 is pinned (VJEPA_SHA256, the same
value as in that script) and checked before it is loaded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from bulwark.model.vjepa21.vision_transformer import vit_base

REPO_ROOT = Path(__file__).resolve().parents[2]
# The facebookresearch/vjepa2 commit the vendored code comes from, and Meta's MIT licence
# notice that came with it (a B1 release ships it as LICENSE-vjepa2).
VJEPA_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_LICENSE = Path(__file__).resolve().parent / "vjepa21" / "LICENSE"
VJEPA_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
VJEPA_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "pretrained" / "vjepa2_1_vitb_dist_vitG_384.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    img_size: int = 256
    num_frames: int = 16
    tubelet_size: int = 2
    patch_size: int = 16
    activation_checkpointing: bool = False


@lru_cache(maxsize=4)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: Path, expected: str = VJEPA_SHA256) -> str:
    """The checkpoint's sha256, after checking it against the pinned value."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Download it from {VJEPA_URL} (scripts/fetch_pretrained.sh).")
    stat = path.stat()
    actual = _sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if actual != expected:
        raise ValueError(
            f"{path} has sha256 {actual}, not the pinned {expected} of {VJEPA_URL}: the download is incomplete "
            "or it is a different file. Delete it and run scripts/fetch_pretrained.sh."
        )
    return actual


class VJepaEncoder(nn.Module):
    """Wraps the V-JEPA 2.1 encoder; returns tokens reshaped to (B, T', H', W', D)."""

    mean: torch.Tensor  # buffers registered in __init__
    std: torch.Tensor

    def __init__(self, config: BackboneConfig | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        self.encoder = vit_base(
            patch_size=self.config.patch_size,
            img_size=(self.config.img_size, self.config.img_size),
            num_frames=self.config.num_frames,
            tubelet_size=self.config.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            use_activation_checkpointing=self.config.activation_checkpointing,
        )
        self.embed_dim = int(self.encoder.embed_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        # Set once pretrained weights are loaded (provenance).
        self.weights_path: Path | None = None
        self.weights_sha256: str | None = None
        if pretrained:
            self.load_pretrained(Path(self.config.checkpoint))

    def load_pretrained(self, path: Path) -> None:
        sha256 = verify_weights(path)
        # weights_only: the official checkpoint holds only tensors and plain values, and a
        # tampered file must not be able to run code on load.
        state = torch.load(path, map_location="cpu", weights_only=True)
        weights = state["ema_encoder"]
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in weights.items()}
        self.encoder.load_state_dict(cleaned, strict=True)
        self.weights_path, self.weights_sha256 = path, sha256

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, H, W, 3) uint8 or float in [0, 1]. T == 1 uses the image path."""
        x = frames.float()
        if frames.dtype == torch.uint8:
            x = x / 255.0
        x = x.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
        x = (x - self.mean) / self.std
        batch, _, time, height, width = x.shape
        tokens = self.encoder(x)
        t_steps = 1 if time == 1 else time // self.config.tubelet_size
        h_steps, w_steps = height // self.config.patch_size, width // self.config.patch_size
        return tokens.reshape(batch, t_steps, h_steps, w_steps, self.embed_dim)
