"""Compare a simulated map with the observed one (urban expansion only)."""
from __future__ import annotations

import numpy as np

from .config import URBAN


def urban_change_metrics(t0: np.ndarray, observed: np.ndarray, simulated: np.ndarray) -> dict:
    """Spatial agreement of new urban cells, plus binary kappa on urban / non-urban."""
    valid = (t0 > 0) & (observed > 0)
    u0 = (t0 == URBAN) & valid
    obs = (observed == URBAN) & valid
    sim = (simulated == URBAN) & valid

    obs_new, sim_new = obs & ~u0, sim & ~u0
    hits = int((obs_new & sim_new).sum())
    misses = int((obs_new & ~sim_new).sum())
    false_alarms = int((~obs_new & sim_new).sum())
    denom = hits + misses + false_alarms
    fom = hits / denom if denom else float("nan")

    n = valid.sum()
    po = ((obs == sim) & valid).sum() / n
    pe = (obs.sum() * sim.sum() + (valid & ~obs).sum() * (valid & ~sim).sum()) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    return {
        "observed_new_urban_cells": int(obs_new.sum()),
        "simulated_new_urban_cells": int(sim_new.sum()),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "figure_of_merit": fom,
        "producer_accuracy": hits / max(int(obs_new.sum()), 1),
        "user_accuracy": hits / max(int(sim_new.sum()), 1),
        "kappa_urban": float(kappa),
        "no_change_figure_of_merit": 0.0,
    }
