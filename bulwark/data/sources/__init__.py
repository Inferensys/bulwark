from __future__ import annotations

from pathlib import Path
from typing import Any

from bulwark.data.sources.auc_v2 import AucV2Adapter
from bulwark.data.sources.base import SourceAdapter
from bulwark.data.sources.dmd import DmdAdapter
from bulwark.data.sources.drive_and_act import DriveAndActAdapter
from bulwark.data.sources.images import CewAdapter, Sh17Adapter, ZenodoDriverAdapter
from bulwark.data.sources.mrl_eye import MrlEyeAdapter
from bulwark.data.sources.nthu_ddd import NthuDddAdapter
from bulwark.data.sources.state_farm import StateFarmAdapter
from bulwark.data.sources.synthetic_wan import SyntheticWanAdapter
from bulwark.data.sources.uta_rldd import UtaRlddAdapter
from bulwark.data.sources.yawdd import YawddAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    adapter.name: adapter
    for adapter in (
        AucV2Adapter,
        CewAdapter,
        DmdAdapter,
        DriveAndActAdapter,
        MrlEyeAdapter,
        NthuDddAdapter,
        Sh17Adapter,
        StateFarmAdapter,
        SyntheticWanAdapter,
        UtaRlddAdapter,
        YawddAdapter,
        ZenodoDriverAdapter,
    )
}


def get_adapter(name: str, root: Path, **kwargs: Any) -> SourceAdapter:
    try:
        adapter = ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"No adapter for {name!r}. Available: {sorted(ADAPTERS)}") from exc
    return adapter(root, **kwargs)
