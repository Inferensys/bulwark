"""Multi-task heads over summary tokens (or per-frame tokens for the edge student)."""

from __future__ import annotations

import torch
from torch import nn

from bulwark.data.taxonomy import Taxonomy, default_taxonomy

MAX_TOKENS = 64


class BulwarkHeads(nn.Module):
    """Small transformer over input tokens -> CLS state embedding -> one output per label.

    `forward` returns raw logits per label: (B,) for binary/regression, (B, C) for
    categorical. Use `predict` for probabilities and de-normalised regression values.

    `input_tokens` fixes the token count and enables per-position input standardisation
    (fit with `fit_input_stats`): encoder summary tokens share a large constant component
    across clips, which the standardisation removes.
    """

    in_mean: torch.Tensor  # buffers registered in __init__
    in_std: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        taxonomy: Taxonomy | None = None,
        *,
        width: int = 512,
        depth: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        input_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.taxonomy = taxonomy or default_taxonomy()
        self.in_dim = in_dim
        self.width = width
        stats_shape = (1, input_tokens or 1, in_dim)
        self.register_buffer("in_mean", torch.zeros(stats_shape))
        self.register_buffer("in_std", torch.ones(stats_shape))
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, MAX_TOKENS + 1, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.label_names = list(self.taxonomy.labels)
        self.widths = [self.taxonomy[name].width for name in self.label_names]
        self.out = nn.Linear(width, sum(self.widths))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    @torch.no_grad()
    def fit_input_stats(self, tokens: torch.Tensor) -> None:
        """Set the standardisation from a sample of training tokens (N, P, D)."""
        tokens = tokens.detach().cpu().double()  # float64 for stable sums; MPS has no float64
        dims = (0, 1) if self.in_mean.shape[1] == 1 else (0,)
        self.in_mean.copy_(tokens.mean(dims, keepdim=True).float())
        self.in_std.copy_(tokens.std(dims, keepdim=True).clamp_min(1e-3).float())

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        if count > MAX_TOKENS:
            raise ValueError(f"At most {MAX_TOKENS} tokens, got {count}")
        x = self.proj((tokens - self.in_mean) / self.in_std)
        x = torch.cat([self.cls.expand(batch, -1, -1), x], dim=1) + self.pos[:, : count + 1]
        return self.norm(self.blocks(x)[:, 0])

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.split(self.out(self.embed(tokens)))

    def split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name, chunk in zip(self.label_names, flat.split(self.widths, dim=-1), strict=True):
            outputs[name] = chunk if self.taxonomy[name].kind == "categorical" else chunk.squeeze(-1)
        return outputs

    @torch.no_grad()
    def predict(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return activate(self(tokens), self.taxonomy)


def activate(logits: dict[str, torch.Tensor], taxonomy: Taxonomy) -> dict[str, torch.Tensor]:
    """Logits -> probabilities (binary, categorical) or values in the label's range."""
    out: dict[str, torch.Tensor] = {}
    for name, value in logits.items():
        spec = taxonomy[name]
        if spec.kind == "categorical":
            out[name] = value.softmax(-1)
        elif spec.kind == "binary":
            out[name] = value.sigmoid()
        else:
            out[name] = regression_value(value, spec.value_range)
    return out


def regression_value(raw: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    low, high = value_range
    return low + (high - low) * raw.sigmoid()
