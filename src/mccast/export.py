"""Figures and tables for the website."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import config as C
from . import data

COLORS = {0: "#ffffff", 1: "#2f6b3a", 2: "#e3c76b", 3: "#c0392b", 4: "#5aa9d6", 5: "#9b9b9b"}


def _read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as s:
        return s.read(1)


def area_table(cfg: C.Config) -> Path:
    """Area (km²) per class for every observed year and the simulated ones found in outputs/."""
    _, profile = data.read_lulc(cfg["years"]["train_end"])
    km2 = data.cell_area_km2(profile)
    rows = []
    sources = [(int(p.stem.split("_")[1]), "observed", p) for p in sorted(C.PROCESSED.glob("lulc_*.tif"))]
    sources += [(int(p.stem.split("_")[1]), "simulated", p) for p in sorted(C.OUTPUTS.glob("lulc_*_sim.tif"))]
    for year, kind, p in sorted(sources):
        counts = np.bincount(_read(p).ravel(), minlength=cfg.n_classes + 1)
        for cid, spec in cfg.classes.items():
            rows.append({"year": year, "type": kind, "class": spec["name"], "area_km2": round(counts[cid] * km2, 2)})
    out = C.OUTPUTS / "area_by_class.csv"
    C.OUTPUTS.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "type", "class", "area_km2"])
        w.writeheader()
        w.writerows(rows)
    return out


def _paint(ax, arr, title):
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for cid, hexc in COLORS.items():
        rgb[arr == cid] = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def maps(cfg: C.Config) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    y = cfg["years"]
    panels = [
        (_read(C.PROCESSED / f"lulc_{y['start']}.tif"), f"{y['start']} (observed)"),
        (_read(C.PROCESSED / f"lulc_{y['train_end']}.tif"), f"{y['train_end']} (observed)"),
        (_read(C.PROCESSED / f"lulc_{y['validation_end']}.tif"), f"{y['validation_end']} (observed)"),
    ]
    proj = C.OUTPUTS / f"lulc_{y['horizon']}_sim.tif"
    if proj.exists():
        panels.append((_read(proj), f"{y['horizon']} (projected)"))

    out = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    for ax, (arr, title) in zip(axes.ravel(), panels):
        _paint(ax, arr, title)
    fig.legend(
        handles=[Patch(color=COLORS[c], label=s["label"]) for c, s in cfg.classes.items()],
        loc="lower center", ncol=len(cfg.classes), frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    p = C.OUTPUTS / "maps_land_cover.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out.append(p)

    sim_path = C.OUTPUTS / f"lulc_{y['validation_end']}_sim.tif"
    if sim_path.exists():
        t0, obs, sim = panels[1][0], panels[2][0], _read(sim_path)
        u0, uo, us = t0 == C.URBAN, obs == C.URBAN, sim == C.URBAN
        cmp = np.zeros(t0.shape, dtype=np.uint8)
        cmp[(t0 > 0)] = 1                       # study area
        cmp[u0] = 2                             # urban at start
        cmp[uo & us & ~u0] = 3                  # hit
        cmp[uo & ~us & ~u0] = 4                 # miss
        cmp[~uo & us & ~u0] = 5                 # false alarm
        colors = ["#ffffff", "#e8e8e8", "#7f7f7f", "#2e8b57", "#d62728", "#ff9f1c"]
        labels = ["Urban in %d" % y["train_end"], "Hit", "Missed (observed only)", "False alarm (simulated only)"]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(cmp, cmap=ListedColormap(colors), vmin=0, vmax=5, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Validation {y['train_end']}→{y['validation_end']}: new urban area")
        ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors[2:], labels)], loc="lower left", frameon=False)
        p = C.OUTPUTS / "map_validation.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def suitability_plot(cfg: C.Config) -> Path | None:
    """Lift of urbanization probability per distance bin, from the fitted suitability."""
    f = C.OUTPUTS / "suitability.json"
    if not f.exists():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = json.loads(f.read_text())["factors"]
    labels = {"dist_urban": "Distance to urban area", "dist_road": "Distance to main roads"}
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 3.6), sharey=True)
    for ax, (name, d) in zip(np.atleast_1d(axes), factors.items()):
        ticks = [f"{e / 1000:.1f}" for e in d["edges"]] + [f">{d['edges'][-1] / 1000:.1f}"]
        ax.bar(range(len(d["lifts"])), d["lifts"], color="#c0392b")
        ax.axhline(1, color="k", lw=0.6)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=60, fontsize=8)
        ax.set_xlabel(f"{labels.get(name, name)}, bin upper limit (km)")
    np.atleast_1d(axes)[0].set_ylabel("Lift over base rate (1 = average)")
    fig.suptitle("Where cells urbanized in 1985–2015, relative to the average")
    p = C.OUTPUTS / "suitability_lift.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def export(cfg: C.Config) -> None:
    files = [area_table(cfg), *maps(cfg)]
    if (p := suitability_plot(cfg)) is not None:
        files.append(p)
    for f in (C.OUTPUTS / "metrics.json", C.OUTPUTS / "suitability.json", C.OUTPUTS / "markov_matrix.json"):
        if f.exists():
            files.append(f)
    print("Outputs:", *[f.name for f in files], sep="\n  ")
