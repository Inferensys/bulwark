#!/usr/bin/env python3
"""Compare the manifest with the shards on disk, dataset by dataset.

Shards cut from a partial download, an older adapter or an interrupted run can disagree
with the manifest without any error. This prints, per dataset, the clips in the manifest,
the clips in its shards, how many are missing or stale, and how many of the missing ones
`bulwark shard` could not decode. It exits 1 when a dataset has stale clips (sharded but
no longer in the manifest, e.g. after an adapter change), a `.tar` shard without an index
or that cannot be read (written by an older, interrupted `bulwark shard`, or damaged),
sharded clips the manifest now cuts differently under the same clip ids (see the
fingerprint below), or more missing clips than --max-missing allows. The current writer
renames a shard to `.tar` only once its index is written; an interrupted one leaves
`<dataset>-NNNNN.tar.partial`, which is not read here.

`bulwark shard` adds missing clips to new `<dataset>-NNNNN.tar` files numbered after the
existing ones, and lists the clips that fail to decode in `<dataset>.shards.json`. Those
stay missing however often it runs, so they are counted on their own and do not count
against --max-missing; fix their source files or drop them from the manifest. The same
file holds a fingerprint of how the sharded clips are cut (media path, start frame, frame
count, stride, crop), which is compared with the manifest when it covers the same clips.

    scripts/check_shards.py [--manifest data/manifest.parquet] [--shards data/shards]
        [--max-missing 0.005]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# <dataset>-NNNNN.tar. `bulwark shard` numbers clips it adds after the existing shards.
# <dataset>-rNNN-NNNNN.tar is a legacy name an earlier version of it wrote for added clips;
# such shards still count (as they do in `bulwark shard`). Same pattern as
# infra/kaggle/stage_data.py (a test keeps them equal).
SHARD_NAME = re.compile(r"^(?P<dataset>.+?)(?:-r\d{3})?-\d{5}\.tar$")
# Manifest columns that say how a clip is cut from its source (`bulwark.cli.shard_fingerprint`).
CUT_COLUMNS = ("media_path", "start_frame", "num_frames", "frame_stride", "crop")


@dataclass
class Coverage:
    in_manifest: set[str] = field(default_factory=set)
    in_shards: set[str] = field(default_factory=set)
    broken: list[str] = field(default_factory=list)  # .tar shards without index.json, or unreadable
    failed: set[str] = field(default_factory=set)  # clips `bulwark shard` could not decode
    recut: bool = False  # the manifest cuts the sharded clips differently than when sharded

    @property
    def missing(self) -> set[str]:
        return self.in_manifest - self.in_shards

    @property
    def undecodable(self) -> set[str]:
        return self.missing & self.failed

    @property
    def stale(self) -> set[str]:
        return self.in_shards - self.in_manifest

    @property
    def ok(self) -> bool:
        return not (self.missing or self.stale or self.broken or self.recut)

    def fails(self, max_missing: float) -> bool:
        """Stale or re-cut clips and broken shards always fail; missing clips only above the
        fraction, not counting clips that are known to fail to decode."""
        unexplained = self.missing - self.failed
        too_many = len(unexplained) > max_missing * len(self.in_manifest)
        return bool(self.stale or self.broken or self.recut) or too_many


def fingerprint(cuts: dict[str, tuple[Any, ...]]) -> dict[str, str]:
    """sha256 of the clip ids and of how each clip is cut, as `bulwark.cli.shard_fingerprint`
    computes them; `cuts` maps clip id -> (media_path, start_frame, num_frames, frame_stride,
    crop) with crop parsed from the manifest's JSON."""
    ids, digest = hashlib.sha256(), hashlib.sha256()
    for clip_id in sorted(cuts):
        media_path, start, frames, stride, crop = cuts[clip_id]
        ids.update(f"{clip_id}\n".encode())
        line = [clip_id, media_path, int(start), int(frames), int(stride), crop]
        digest.update((json.dumps(line) + "\n").encode())
    return {"clip_ids": ids.hexdigest(), "cuts": digest.hexdigest()}


def shard_clip_ids(path: Path) -> list[str] | None:
    """Clip ids listed in a shard's index.json, or None if the shard has no index or cannot
    be read (as `bulwark shard` treats it)."""
    try:
        with tarfile.open(path, "r") as tar:
            for member in tar:
                if member.name == "index.json":
                    handle = tar.extractfile(member)
                    return list(json.loads(handle.read()).values()) if handle else None
    except (tarfile.TarError, json.JSONDecodeError):
        return None
    return None


def coverage(manifest: Path, shards: Path) -> dict[str, Coverage]:
    names = set(pq.read_schema(manifest).names)
    with_cuts = all(column in names for column in CUT_COLUMNS)
    table = pq.read_table(manifest, columns=["clip_id", "dataset", *(CUT_COLUMNS if with_cuts else ())])
    found: dict[str, Coverage] = defaultdict(Coverage)
    clip_ids, datasets = table.column("clip_id").to_pylist(), table.column("dataset").to_pylist()
    for clip_id, dataset in zip(clip_ids, datasets, strict=True):
        found[dataset].in_manifest.add(clip_id)
    cuts: dict[str, tuple[Any, ...]] = {}
    if with_cuts:
        columns = [table.column(c).to_pylist() for c in CUT_COLUMNS]
        for clip_id, *values in zip(clip_ids, *columns, strict=True):
            values[-1] = None if values[-1] is None else json.loads(values[-1])
            cuts[clip_id] = tuple(values)
    for path in sorted(shards.glob("*.tar")):
        match = SHARD_NAME.match(path.name)
        if not match:
            continue
        ids = shard_clip_ids(path)
        if ids is None:
            found[match["dataset"]].broken.append(path.name)
        else:
            found[match["dataset"]].in_shards.update(ids)
    for record in sorted(shards.glob("*.shards.json")):
        dataset = record.name.removesuffix(".shards.json")
        if dataset not in found:
            continue
        cov = found[dataset]
        data = json.loads(record.read_text(encoding="utf-8"))
        cov.failed.update(data.get("failed") or {})
        recorded = data.get("fingerprint") or {}
        if recorded and cuts:
            # Compared only over the same clips: a purge or new clips change the set, not the cuts.
            now = fingerprint({c: cuts[c] for c in cov.in_shards & cov.in_manifest})
            cov.recut = recorded.get("clip_ids") == now["clip_ids"] and recorded.get("cuts") != now["cuts"]
    return dict(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--shards", type=Path, default=Path("data/shards"))
    parser.add_argument(
        "--max-missing", type=float, default=0.005,
        help="fraction of a dataset's clips that may be missing from its shards, not counting the "
        "clips <dataset>.shards.json lists as undecodable",
    )  # fmt: skip
    args = parser.parse_args(argv)
    report = coverage(args.manifest, args.shards)
    print(
        f"{'dataset':24s} {'manifest':>9s} {'sharded':>9s} {'missing':>9s} "
        f"{'no decode':>9s} {'stale':>7s}"
    )
    for name, cov in sorted(report.items()):
        flag = ""
        if cov.fails(args.max_missing):
            flag = "  <- differs"
        elif cov.missing - cov.failed:
            flag = "  (a few missing)"
        elif cov.undecodable:
            flag = f"  (see {name}.shards.json)"
        print(
            f"{name:24s} {len(cov.in_manifest):9d} {len(cov.in_shards):9d} "
            f"{len(cov.missing):9d} {len(cov.undecodable):9d} {len(cov.stale):7d}{flag}"
        )
        for shard in cov.broken:
            print(f"  {shard}: no index.json or unreadable (incomplete or damaged shard)")
        if cov.recut:
            print("  the manifest cuts the sharded clips differently than when they were sharded")
    differing = sorted(name for name, cov in report.items() if cov.fails(args.max_missing))
    if differing:
        shard = f"bulwark shard --manifest {args.manifest} --out {args.shards} --dataset <dataset>"
        print(
            "\nShards do not match the manifest for: " + ", ".join(differing) + ".\n"
            f"Missing clips: run `{shard}`. Clips it cannot decode are listed in\n"
            "<dataset>.shards.json and do not count here.\n"
            f"Stale or re-cut clips, or incomplete shards: rm {args.shards}/<dataset>-*.tar && {shard}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
