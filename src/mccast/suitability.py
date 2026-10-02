"""Minimal, data-driven suitability for urban expansion.

For each factor (distance to urban area, distance to main roads) the share of
candidate cells that became urban during the training period is measured per
quantile bin. A cell's score is the base rate times the product of each bin's
lift over the base rate (a naive-Bayes style combination: factors are assumed
independent). No weights are chosen by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def distance_to(mask: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Distance (m) from each cell to the nearest True cell of `mask`."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return (ndimage.distance_transform_edt(~mask) * cell_size_m).astype(np.float32)


@dataclass
class Suitability:
    edges: dict[str, np.ndarray]   # per factor: interior bin edges
    lifts: dict[str, np.ndarray]   # per factor: lift per bin
    base_rate: float

    @classmethod
    def fit(
        cls,
        urban_t0: np.ndarray,
        urban_t1: np.ndarray,
        allowed: np.ndarray,
        factors: dict[str, np.ndarray],
        bins: int = 12,
    ) -> "Suitability":
        candidates = allowed & ~urban_t0
        became = urban_t1 & candidates
        base = became.sum() / max(candidates.sum(), 1)
        edges, lifts = {}, {}
        for name, f in factors.items():
            vals = f[candidates]
            e = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)[1:-1]))
            b = np.digitize(vals, e)
            n = np.bincount(b, minlength=len(e) + 1)
            k = np.bincount(b, weights=became[candidates], minlength=len(e) + 1)
            rate = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
            edges[name], lifts[name] = e, rate / base if base > 0 else np.ones_like(rate)
        return cls(edges, lifts, float(base))

    def predict(self, factors: dict[str, np.ndarray]) -> np.ndarray:
        """Probability-like score in [0, 1] for each cell."""
        score = np.full(next(iter(factors.values())).shape, self.base_rate, dtype=np.float32)
        for name, f in factors.items():
            score *= self.lifts[name][np.digitize(f, self.edges[name])].astype(np.float32)
        return np.clip(score, 0, 1)

    def to_dict(self) -> dict:
        return {
            "base_rate": self.base_rate,
            "factors": {
                k: {"edges": self.edges[k].tolist(), "lifts": self.lifts[k].tolist()}
                for k in self.edges
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Suitability":
        f = d["factors"]
        return cls(
            {k: np.array(v["edges"]) for k, v in f.items()},
            {k: np.array(v["lifts"]) for k, v in f.items()},
            d["base_rate"],
        )
