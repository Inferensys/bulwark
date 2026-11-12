"""Summary tokens: the fixed-size interface between the encoder and the heads.

From a token grid (B, T', H', W', D) we keep:
    1 global token          mean over everything
    8 temporal tokens       mean per time step (blinks, nods, yawns need timing)
    16 spatial tokens       4x4 grid of spatial means over time (hands vs. face vs. lap)
= 25 tokens. These are what `cache_features` stores, so heads trained on cached
features and heads trained end-to-end see exactly the same input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TEMPORAL_TOKENS = 8
SPATIAL_GRID = 4
NUM_SUMMARY_TOKENS = 1 + TEMPORAL_TOKENS + SPATIAL_GRID * SPATIAL_GRID


def summary_tokens(grid: torch.Tensor) -> torch.Tensor:
    batch, time, height, width, dim = grid.shape
    global_token = grid.mean(dim=(1, 2, 3)).unsqueeze(1)

    temporal = grid.mean(dim=(2, 3))  # B, T', D
    if time != TEMPORAL_TOKENS:
        temporal = F.interpolate(temporal.transpose(1, 2), size=TEMPORAL_TOKENS, mode="nearest").transpose(1, 2)

    spatial = grid.mean(dim=1).permute(0, 3, 1, 2)  # B, D, H', W'
    spatial = F.adaptive_avg_pool2d(spatial, SPATIAL_GRID).flatten(2).transpose(1, 2)  # B, 16, D

    return torch.cat([global_token, temporal, spatial], dim=1)
