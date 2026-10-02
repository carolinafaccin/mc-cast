"""Cellular automaton that allocates urban demand in space."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import ndimage

from .config import URBAN


def simulate(
    lulc: np.ndarray,
    allowed: np.ndarray,
    suitability: Callable[[np.ndarray], np.ndarray],
    new_urban_cells: int,
    steps: int,
    radius: int = 2,
    noise: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Convert `new_urban_cells` cells to urban over `steps` annual steps.

    Each step, a cell's potential is suitability x share of urban cells in its
    (2r+1)^2 neighborhood x a random factor; the highest-potential candidate
    cells (not urban, allowed, touching urban area) are converted. Only urban
    expansion is allocated; other classes change only where urban replaces them.

    `suitability(urban)` returns the score grid for the current urban mask
    (it depends on distance to urban area, which changes every step).
    """
    rng = np.random.default_rng(seed)
    out = lulc.copy()
    urban = out == URBAN
    size = 2 * radius + 1
    quota = np.full(steps, new_urban_cells // steps)
    quota[: new_urban_cells % steps] += 1

    carry = 0  # demand the fringe could not absorb in an earlier step
    for k in quota:
        want = int(k) + carry
        if want == 0:
            continue
        neigh = ndimage.uniform_filter(urban.astype(np.float32), size=size, mode="constant")
        cand = allowed & ~urban & (neigh > 0)
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            carry = want
            continue
        score = suitability(urban).ravel()[idx] * neigh.ravel()[idx]
        score *= 1 - noise + 2 * noise * rng.random(idx.size, dtype=np.float32)
        k = min(want, idx.size)
        chosen = idx[np.argpartition(score, -k)[-k:]]
        urban.ravel()[chosen] = True
        out.ravel()[chosen] = URBAN
        carry = want - chosen.size
    if carry:
        warnings.warn(f"{carry} cells of urban demand could not be allocated (no candidate cells)")
    return out
