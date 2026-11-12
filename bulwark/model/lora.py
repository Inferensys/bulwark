"""Minimal LoRA for nn.Linear.

By default it wraps the attention (qkv, proj) and MLP (fc1, fc2) linears of the last N
encoder blocks, and also trains the encoder's final output norm.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(self.dropout(x))) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A plain Linear with the adapter folded in (for export)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + (self.up.weight @ self.down.weight) * self.scale)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(
    encoder: nn.Module,
    *,
    last_blocks: int = 4,
    rank: int = 16,
    alpha: float = 32.0,
    targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
) -> list[str]:
    """Freeze the encoder and add LoRA to `targets` inside its last `last_blocks` blocks."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(f"apply_lora needs an encoder whose `blocks` is an nn.ModuleList, got {type(blocks).__name__}")
    wrapped: list[str] = []
    for index in range(len(blocks) - last_blocks, len(blocks)):
        for name, module in list(blocks[index].named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in targets and isinstance(module, nn.Linear):
                parent = blocks[index].get_submodule(name.rsplit(".", 1)[0]) if "." in name else blocks[index]
                setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha))
                wrapped.append(f"blocks.{index}.{name}")
    # The final output norm is cheap and helps adaptation. V-JEPA 2.1 keeps one norm per
    # hierarchical output, but inference uses only the last; the others get no gradient.
    norms = getattr(encoder, "norms_block", None)
    if norms is not None and len(norms):
        norms[-1].requires_grad_(True)
    return wrapped


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """The LoRA adapters and the final output norm: everything `apply_lora` makes trainable."""
    state = module.state_dict()
    norms = sorted({int(k.split("norms_block.", 1)[1].split(".", 1)[0]) for k in state if "norms_block." in k})
    final = f"norms_block.{norms[-1]}." if norms else None
    return {k: v for k, v in state.items() if ".down." in k or ".up." in k or (final is not None and final in k)}
