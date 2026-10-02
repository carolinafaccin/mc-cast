"""End-to-end steps: train -> validate -> project."""
from __future__ import annotations

import json

import numpy as np

from . import ca, data, markov
from . import config as C
from .suitability import Suitability, distance_to
from .validate import urban_change_metrics

CELL = data.CELL_SIZE_M


def _write(arr: np.ndarray, profile: dict, name: str) -> None:
    import rasterio

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    with rasterio.open(C.OUTPUTS / name, "w", **profile) as o:
        o.write(arr, 1)


def train(cfg: C.Config) -> None:
    """Fit the Markov matrix and the suitability model on start -> train_end."""
    y0, y1 = cfg["years"]["start"], cfg["years"]["train_end"]
    a, profile = data.read_lulc(y0)
    b, _ = data.read_lulc(y1)

    P = markov.transition_matrix(a, b, cfg.n_classes)
    urban_a, urban_b = a == C.URBAN, b == C.URBAN
    allowed = (a > 0) & (b > 0) & (a != C.WATER)
    factors = {"dist_urban": distance_to(urban_a, CELL), "dist_road": data.roads_distance(cfg, profile)}
    suit = Suitability.fit(urban_a, urban_b, allowed, factors, cfg["suitability"]["bins"])

    data.save_json(
        {"from": y0, "to": y1, "classes": {k: v["name"] for k, v in cfg.classes.items()}, "matrix": P.tolist()},
        C.OUTPUTS / "markov_matrix.json",
    )
    data.save_json(suit.to_dict(), C.OUTPUTS / "suitability.json")
    print(f"Trained on {y0}-{y1}: base urbanization rate {suit.base_rate:.4%} of candidate cells.")


def _load_model(cfg: C.Config) -> tuple[np.ndarray, int, Suitability]:
    m = json.loads((C.OUTPUTS / "markov_matrix.json").read_text())
    suit = Suitability.from_dict(json.loads((C.OUTPUTS / "suitability.json").read_text()))
    return np.array(m["matrix"]), m["to"] - m["from"], suit


def run_period(cfg: C.Config, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Simulate t0 -> t1 starting from the observed t0 map."""
    P, train_years, suit = _load_model(cfg)
    lulc0, profile = data.read_lulc(t0)
    roads = data.roads_distance(cfg, profile)

    counts0 = markov.class_counts(lulc0, cfg.n_classes)
    counts1 = markov.project_counts(counts0, markov.rescale(P, train_years, t1 - t0))
    demand = max(0, int(round(counts1[C.URBAN - 1] - counts0[C.URBAN - 1])))

    sim = ca.simulate(
        lulc0,
        allowed=(lulc0 > 0) & (lulc0 != C.WATER),
        suitability=lambda urban: suit.predict({"dist_urban": distance_to(urban, CELL), "dist_road": roads}),
        new_urban_cells=demand,
        steps=t1 - t0,
        radius=cfg["ca"]["neighborhood_radius"],
        noise=cfg["ca"]["noise"],
        seed=cfg["ca"]["seed"],
    )
    return lulc0, sim, profile


def validate(cfg: C.Config) -> dict:
    """Simulate train_end -> validation_end and score it against the observed map."""
    t0, t1 = cfg["years"]["train_end"], cfg["years"]["validation_end"]
    lulc0, sim, profile = run_period(cfg, t0, t1)
    observed, _ = data.read_lulc(t1)
    metrics = {"from": t0, "to": t1, **urban_change_metrics(lulc0, observed, sim)}
    _write(sim, profile, f"lulc_{t1}_sim.tif")
    data.save_json(metrics, C.OUTPUTS / "metrics.json")
    print(json.dumps(metrics, indent=2))
    return metrics


def project(cfg: C.Config) -> None:
    """Project validation_end -> horizon."""
    t0, t1 = cfg["years"]["validation_end"], cfg["years"]["horizon"]
    _, sim, profile = run_period(cfg, t0, t1)
    _write(sim, profile, f"lulc_{t1}_sim.tif")
    print(f"Projection {t0}->{t1} written to outputs/lulc_{t1}_sim.tif")
