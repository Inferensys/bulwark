"""Cross-dataset near-duplicate images (64-bit dHash, Hamming distance), so the same
photo never sits in train under one dataset and in test under another. The Zenodo
driver set, for example, appears to contain upscaled, re-encoded State Farm images.

A match also needs the same aspect ratio: copies keep their shape, while a 9x8 dHash of
a 16:9 cabin photo can land within a few bits of a square eye crop by chance (12 Zenodo
photos matched MRL Eye crops before this check).

Only images of different datasets are matched. Consecutive frames of one dataset are
within a few bits of each other, so matching within a dataset would chain them into huge
groups (2,799 of AUC Distracted Driver v2's 2,800 camera-2 images form one), and a single
cross-dataset match would then drop every image of the group's other datasets. This way
every image dropped is itself within the distance of an image in another dataset.

A copy found in another dataset is also licence evidence: when any member of a group is
not cleared for commercial use, the copy that is kept is not either.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from bulwark.data.clips import read_image
from bulwark.data.manifest import ClipRecord

MAX_DISTANCE = 6  # of 64 bits; re-encoding and upscaling flip a few
MAX_ASPECT_GAP = 0.05  # |log(aspect ratio)| difference; resizing keeps the shape
MAX_UNHASHED = 0.01  # share of a dataset's images that may be unreadable before we stop


def dhash(path: Path, crop: tuple[int, int, int, int] | None = None) -> tuple[int, float] | None:
    """64-bit difference hash and width / height of the (cropped) image, or None if the
    file cannot be read."""
    try:
        image = read_image(path, gray=True)
    except FileNotFoundError:
        return None
    if crop is not None:
        x, y, w, h = crop
        image = image[y : y + h, x : x + w]
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0]), image.shape[1] / image.shape[0]


def near_duplicate_groups(
    hashes: Sequence[int],
    max_distance: int = MAX_DISTANCE,
    aspects: Sequence[float] | None = None,
    sources: Sequence[str] | None = None,
) -> list[list[int]]:
    """Index groups of hashes within `max_distance` bits (transitively), and, given
    `aspects`, of the same aspect ratio. Given `sources` (one per hash, the dataset), only
    hashes of different sources are joined, so a group never chains through one source's
    near-identical images.

    Two 64-bit hashes within 7 bits must share at least one of their 8 bytes exactly
    (pigeonhole), so candidates come from byte buckets instead of all pairs.
    """

    def same_shape(a: int, b: int) -> bool:
        return aspects is None or abs(math.log(aspects[a] / aspects[b])) <= MAX_ASPECT_GAP

    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        for byte in range(8):
            buckets[(byte, (value >> (8 * byte)) & 0xFF)].append(index)
    for members in buckets.values():
        if len(members) < 2 or len(members) > 5000:  # giant buckets = flat images; skip
            continue
        for a, b in _candidate_pairs(members, sources):
            if (
                find(a) != find(b)
                and (hashes[a] ^ hashes[b]).bit_count() <= max_distance
                and same_shape(a, b)
            ):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(hashes)):
        groups[find(index)].append(index)
    return list(groups.values())


def _candidate_pairs(
    members: list[int], sources: Sequence[str] | None
) -> Iterator[tuple[int, int]]:
    """Pairs of a bucket's members: all of them, or with `sources` those of different sources."""
    if sources is None:
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1 :]:
                yield a, b
        return
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in members:
        by_source[sources[index]].append(index)
    for first, second in combinations(by_source.values(), 2):
        for a in first:
            for b in second:
                yield a, b


def drop_cross_dataset_duplicates(
    records: Sequence[ClipRecord],
    roots: dict[str, Path],
    *,
    prefer_evaluable: Sequence[str] = (),
    max_unhashed: float = MAX_UNHASHED,
) -> tuple[list[ClipRecord], int]:
    """Remove image records that near-duplicate an image in another dataset.

    Groups join images of different datasets only (see the module docstring). In each group
    the copies from the first dataset in `prefer_evaluable` (list datasets with real subject
    ids first) are kept; otherwise the first record's dataset wins. Kept copies record the
    other datasets in `meta["near_duplicate_of"]`, and lose `commercial_ok` when any copy in
    the group lacks it. Videos pass through.

    Raises ValueError when more than `max_unhashed` of a dataset's images cannot be read
    (usually a missing or wrong root): dedup would otherwise do nothing, silently.
    """
    keep: list[ClipRecord] = []
    images: list[ClipRecord] = []
    hashes: list[int] = []
    aspects: list[float] = []
    total: Counter[str] = Counter()
    unhashed: Counter[str] = Counter()
    for record in records:
        if record.media_type != "image":
            keep.append(record)
            continue
        total[record.dataset] += 1
        root = roots.get(record.dataset)
        value = dhash(root / record.media_path, record.crop) if root is not None else None
        if value is None:
            unhashed[record.dataset] += 1
            keep.append(record)
        else:
            images.append(record)
            hashes.append(value[0])
            aspects.append(value[1])
    failing = {
        d: f"{unhashed[d]}/{total[d]}"
        for d in sorted(unhashed)
        if unhashed[d] > max_unhashed * total[d]
    }
    if failing:
        raise ValueError(
            f"Could not read images to de-duplicate (unreadable/total): {failing}. Check the "
            "dataset roots, or skip de-duplication explicitly."
        )
    dropped = 0
    sources = [r.dataset for r in images]
    for group in near_duplicate_groups(hashes, aspects=aspects, sources=sources):
        members = [images[i] for i in group]
        datasets = {r.dataset for r in members}
        if len(datasets) == 1:
            keep.extend(members)
            continue
        winner = next((d for d in prefer_evaluable if d in datasets), members[0].dataset)
        kept = [r for r in members if r.dataset == winner]
        restricted = sorted({r.dataset for r in members if not r.commercial_ok})
        for record in kept:
            record.meta["near_duplicate_of"] = sorted(datasets - {winner})
            if restricted and record.commercial_ok:
                record.commercial_ok = False
                record.licence = f"{record.licence} (near-duplicate of {', '.join(restricted)})"
        keep.extend(kept)
        dropped += len(members) - len(kept)
    return keep, dropped
