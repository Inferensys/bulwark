"""The pretrained encoder pin and the E1 token interface."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from bulwark.model import backbone
from bulwark.model.bulwark import BulwarkEdge


def test_vjepa_weights_are_checked_against_the_pinned_hash(tmp_path: Path) -> None:
    weights = tmp_path / "vjepa.pt"
    weights.write_bytes(b"not the checkpoint")
    digest = hashlib.sha256(b"not the checkpoint").hexdigest()
    assert backbone.verify_weights(weights, digest) == digest
    with pytest.raises(ValueError, match=r"not the pinned.*fetch_pretrained"):
        backbone.verify_weights(weights)
    with pytest.raises(FileNotFoundError):
        backbone.verify_weights(tmp_path / "missing.pt")
    assert backbone.VJEPA_SHA256 == "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"


def test_edge_tokens_are_the_heads_input() -> None:
    torch.manual_seed(0)
    model = BulwarkEdge(pretrained=False, backbone="resnet10t.c3_in1k", img_size=64).eval()
    frames = torch.randint(0, 255, (2, 3, 32, 32, 3), dtype=torch.uint8)
    with torch.no_grad():
        tokens = model.tokens(frames)
        assert tokens.shape[:2] == (2, 3)
        for name, value in model(frames).items():
            assert torch.equal(value, model.heads(tokens)[name]), name


@pytest.mark.parametrize("size", [64, 72])  # at the encoder's size, and resized to it
def test_edge_frame_encoder_gets_contiguous_nchw_frames(size: int) -> None:
    # (B, T, H, W, 3) frames permuted to NCHW keep channels-last strides unless copied, and
    # fp16 depthwise convs under DataParallel on a T4 fail on those ("misaligned address").
    model = BulwarkEdge(pretrained=False, backbone="resnet10t.c3_in1k", img_size=64).eval()
    seen: list[torch.Tensor] = []
    model.frame_encoder.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    with torch.no_grad():
        model.tokens(torch.randint(0, 255, (2, 3, size, size, 3), dtype=torch.uint8))
    assert seen[0].shape == (6, 3, 64, 64)
    assert seen[0].is_contiguous()
