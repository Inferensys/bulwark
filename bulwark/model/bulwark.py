"""Bulwark-B1 (base: V-JEPA encoder + summary tokens + heads) and Bulwark-E1 (edge student)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.model.backbone import BackboneConfig, VJepaEncoder
from bulwark.model.heads import BulwarkHeads
from bulwark.model.lora import apply_lora
from bulwark.model.pooling import NUM_SUMMARY_TOKENS, summary_tokens


class BulwarkBase(nn.Module):
    """Bulwark-B1. frames (B, T, H, W, 3) -> logits per label (+ state embedding)."""

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        *,
        backbone: BackboneConfig | None = None,
        pretrained: bool = True,
        lora_blocks: int = 0,
        lora_rank: int = 16,
    ) -> None:
        super().__init__()
        self.taxonomy = taxonomy or default_taxonomy()
        self.encoder = VJepaEncoder(backbone, pretrained=pretrained)
        self.lora_layers: list[str] = []
        if lora_blocks:
            self.lora_layers = apply_lora(self.encoder.encoder, last_blocks=lora_blocks, rank=lora_rank)
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.heads = BulwarkHeads(self.encoder.embed_dim, self.taxonomy, input_tokens=NUM_SUMMARY_TOKENS)

    def tokens(self, frames: torch.Tensor) -> torch.Tensor:
        return summary_tokens(self.encoder(frames))

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.tokens(frames))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable or k.startswith("heads.")}


class BulwarkEdge(nn.Module):
    """Bulwark-E1. A per-frame CNN/ViT from timm -> frame tokens -> heads of the same
    design as B1's, narrower (width 256, 4 attention heads).

    The heads' small transformer does the temporal modelling over frame tokens, so the
    whole thing exports to a single ONNX graph with input (B, T, H, W, 3) uint8.
    """

    mean: torch.Tensor  # buffers registered in __init__
    std: torch.Tensor

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        *,
        backbone: str = "efficientvit_b1.r224_in1k",
        pretrained: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        import timm

        self.taxonomy = taxonomy or default_taxonomy()
        self.img_size = img_size
        self.frame_encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        config = cast("dict[str, Any]", self.frame_encoder.pretrained_cfg)  # a dict on every timm model
        self.register_buffer("mean", torch.tensor(config.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(config.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1), persistent=False)
        # timm's num_features is the pre-head width; some models (EfficientViT) widen it in
        # their head, so measure the pooled output instead.
        with torch.no_grad():
            feature_dim = int(self.frame_encoder(torch.zeros(1, 3, img_size, img_size)).shape[-1])
        self.heads = BulwarkHeads(feature_dim, self.taxonomy, width=256, depth=2, heads=4)

    def tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-frame features (B, T, D): the heads' input tokens, as `BulwarkBase.tokens`."""
        batch, time = frames.shape[:2]
        # The permute alone leaves channels-last strides, which every later op keeps; in fp16
        # under DataParallel on a T4 (cuDNN 9.19) EfficientViT's depthwise convs then fail
        # with "misaligned address". A contiguous NCHW copy (made while still uint8) does not.
        x = frames.flatten(0, 1).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if x.shape[-1] != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.frame_encoder(x).view(batch, time, -1)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.tokens(frames))


def save_checkpoint(path: Path, model: nn.Module, extra: dict[str, Any] | None = None, *, full: bool = False) -> Path:
    """Write {"model": weights, **extra} atomically. B1 keeps only its trainable state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict() if full or not isinstance(model, BulwarkBase) else model.trainable_state()
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": state, **(extra or {})}, tmp)
    tmp.replace(path)
    return path


def load_payload(path: Path) -> dict[str, Any]:
    """Load a Bulwark checkpoint without unpickling arbitrary objects.

    Bulwark payloads hold only tensors and plain Python values, so `weights_only=True`
    loads all of them and refuses a file that would run code when unpickled.
    """
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a Bulwark checkpoint (no 'model' entry)")
    return payload


def load_into(model: nn.Module, path: Path, *, strict: bool = False, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a checkpoint's weights into `model`; pass `payload` to reuse an already loaded file."""
    payload = payload if payload is not None else load_payload(path)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if unexpected:
        raise KeyError(f"Unexpected keys in {path}: {unexpected[:5]}")
    return payload
