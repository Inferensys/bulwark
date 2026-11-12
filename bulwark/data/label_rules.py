"""Label definitions that more than one source computes, implemented once.

Dataset adapters and teachers both turn per-frame readings into clip labels. When they
use different rules, the same clip gets contradictory targets, so shared rules live here.
"""

from __future__ import annotations

from collections.abc import Sequence


def eyes_closed_from_mask(closed: Sequence[bool]) -> float:
    """`eyes_closed` as taxonomy.yaml defines it: closed in most of the clip's last half.

    `closed` is one flag per observed frame, in time order. Using the last half makes a
    clip where the eyes are closing (the onset of a microsleep) count as closed, and a
    clip where they reopen count as open.
    """
    tail = list(closed)[len(closed) // 2 :]
    return float(bool(tail) and sum(bool(c) for c in tail) * 2 > len(tail))
