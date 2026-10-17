from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpatialRegion:
    split: str
    y0: int
    x0: int
    y1: int
    x1: int
    block_id: str

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def expand(self, amount: int, *, height: int, width: int) -> "SpatialRegion":
        return SpatialRegion(
            split=self.split,
            y0=max(0, self.y0 - int(amount)),
            x0=max(0, self.x0 - int(amount)),
            y1=min(int(height), self.y1 + int(amount)),
            x1=min(int(width), self.x1 + int(amount)),
            block_id=self.block_id,
        )

    def contains_patch(self, y: np.ndarray, x: np.ndarray, patch_height: int, patch_width: int):
        return (
            (y >= self.y0)
            & (x >= self.x0)
            & (y + patch_height <= self.y1)
            & (x + patch_width <= self.x1)
        )


@dataclass(frozen=True)
class SpatialSplitManifest:
    height: int
    width: int
    block_size: int
    guard_band: int
    seed: int
    split_fractions: tuple[float, float, float]
    assignments: tuple[SpatialRegion, ...]

    def regions(self, split: str) -> list[SpatialRegion]:
        return [region for region in self.assignments if region.split == split]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["split_fractions"] = list(self.split_fractions)
        payload["assignments"] = [asdict(region) for region in self.assignments]
        return payload

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SpatialSplitManifest":
        payload = json.loads(Path(path).read_text())
        payload["split_fractions"] = tuple(float(v) for v in payload["split_fractions"])
        payload["assignments"] = tuple(SpatialRegion(**item) for item in payload["assignments"])
        return cls(**payload)


def _split_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * total
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[index] += 1

    nonzero = np.flatnonzero(fractions > 0)
    if total >= len(nonzero):
        for missing in nonzero[counts[nonzero] == 0]:
            donors = np.argsort(-counts)
            donor = next(index for index in donors if counts[index] > 1)
            counts[donor] -= 1
            counts[missing] += 1
    return counts


def build_spatial_split(
    *,
    height: int,
    width: int,
    block_size: int = 4096,
    guard_band: int = 512,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 1337,
) -> SpatialSplitManifest:
    height = int(height)
    width = int(width)
    block_size = int(block_size)
    guard_band = int(guard_band)
    split_fractions = np.asarray(fractions, dtype=np.float64)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if block_size <= 0 or guard_band < 0 or 2 * guard_band >= block_size:
        raise ValueError("guard_band must be non-negative and smaller than half a block")
    if split_fractions.shape != (3,) or np.any(split_fractions < 0):
        raise ValueError("fractions must contain three non-negative values")
    if not np.isclose(split_fractions.sum(), 1.0):
        raise ValueError("fractions must sum to 1")

    blocks: list[tuple[int, int, int, int, str]] = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            inner = (y0 + guard_band, x0 + guard_band, y1 - guard_band, x1 - guard_band)
            if inner[2] > inner[0] and inner[3] > inner[1]:
                blocks.append((*inner, f"y{y0:06d}_x{x0:06d}"))
    if not blocks:
        raise ValueError("no usable blocks remain after applying guard_band")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(blocks))
    counts = _split_counts(len(blocks), split_fractions)
    names = ("train", "validation", "test")
    assignments: list[SpatialRegion] = []
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        for block_index in order[cursor : cursor + int(count)]:
            y0, x0, y1, x1, block_id = blocks[int(block_index)]
            assignments.append(SpatialRegion(name, y0, x0, y1, x1, block_id))
        cursor += int(count)

    assignments.sort(key=lambda region: (region.split, region.y0, region.x0))
    return SpatialSplitManifest(
        height=height,
        width=width,
        block_size=block_size,
        guard_band=guard_band,
        seed=int(seed),
        split_fractions=tuple(float(v) for v in split_fractions),
        assignments=tuple(assignments),
    )
