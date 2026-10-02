"""Figures and tables for the website."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import config as C
from . import data
from . import style as S



def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S.setup()
    return plt


def _save(fig, name: str) -> Path:
    p = C.OUTPUTS / name
    S.save(fig, p)
    return p


def _aspect(profile: dict) -> float:
    """Height/width of one cell in metres, so maps on the degree grid are not stretched."""
    t = profile["transform"]
    lat = t.f + t.e * profile["height"] / 2
    return abs(t.e) * 110_574.0 / (abs(t.a) * 111_320.0 * np.cos(np.radians(lat)))


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


def _rgb(arr: np.ndarray, colors: dict[int, str]) -> np.ndarray:
    from matplotlib.colors import to_rgb

    rgb = np.full((*arr.shape, 3), 255, dtype=np.uint8)
    for cid, c in colors.items():
        rgb[arr == cid] = [round(v * 255) for v in to_rgb(c)]
    return rgb


def _map(ax, arr, colors, aspect, title=None):
    ax.imshow(_rgb(arr, colors), interpolation="nearest", aspect=aspect)
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="semibold", color=S.INK, pad=4)
    ax.axis("off")


def _legend(fig, items, inches=0.55, ncol=None):
    """Swatch legend in one row, a fixed distance (in inches) above the bottom edge (over the footer)."""
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=c, edgecolor="none", label=l) for c, l in items],
        loc="lower left", ncol=ncol or len(items), bbox_to_anchor=(0.03, S.bottom(fig, inches)),
        handlelength=1.0, handleheight=1.0, columnspacing=1.4, handletextpad=0.5, fontsize=8.5, labelcolor=S.INK,
    )


def _value_axis(ax):
    """Recessive value axis: horizontal grid only, no left spine or ticks."""
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def maps(cfg: C.Config) -> list[Path]:
    plt = _plt()
    y = cfg["years"]
    _, profile = data.read_lulc(y["train_end"])
    aspect = _aspect(profile)
    panels = [
        (_read(C.PROCESSED / f"lulc_{y['start']}.tif"), f"{y['start']} observed"),
        (_read(C.PROCESSED / f"lulc_{y['train_end']}.tif"), f"{y['train_end']} observed"),
        (_read(C.PROCESSED / f"lulc_{y['validation_end']}.tif"), f"{y['validation_end']} observed"),
    ]
    proj = C.OUTPUTS / f"lulc_{y['horizon']}_sim.tif"
    if proj.exists():
        panels.append((_read(proj), f"{y['horizon']} projected"))

    out = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), gridspec_kw={"hspace": 0.1, "wspace": 0.03})
    for ax, (arr, title) in zip(axes.ravel(), panels):
        _map(ax, arr, S.CLASS_COLORS, aspect, title)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    S.header(fig, f"Land cover, {cfg['region']['name']}",
             "MapBiomas Collection 10, simplified to 5 classes. 2040 is a model projection, not a forecast.")
    fig.subplots_adjust(left=0.03, right=0.97, top=S.frac(fig, 1.4), bottom=S.bottom(fig, 0.95))
    _legend(fig, [(S.CLASS_COLORS[c], s["label"]) for c, s in cfg.classes.items()])
    S.footer(fig)
    out.append(_save(fig, "maps_land_cover.png"))

    sim_path = C.OUTPUTS / f"lulc_{y['validation_end']}_sim.tif"
    if sim_path.exists():
        out.append(_validation_map(plt, cfg, panels[1][0], panels[2][0], _read(sim_path), aspect))
    return out


def _validation_map(plt, cfg, t0, obs, sim, aspect) -> Path:
    from matplotlib.patches import Rectangle
    from scipy.ndimage import uniform_filter

    y = cfg["years"]
    u0, uo, us = t0 == C.URBAN, obs == C.URBAN, sim == C.URBAN
    cmp = np.zeros(t0.shape, dtype=np.uint8)
    cmp[t0 > 0] = 1                       # study area
    cmp[u0] = 2                           # urban at start
    cmp[uo & us & ~u0] = 3                # hit
    cmp[uo & ~us & ~u0] = 4               # miss
    cmp[~uo & us & ~u0] = 5               # false alarm
    hit, miss, false = S.CAT[:3]
    colors = {0: "white", 1: S.LAND, 2: S.GREY, 3: hit, 4: miss, 5: false}

    # Zoom on the window with the most new urban cells (observed or simulated).
    h = max(t0.shape[0] // 6, 200)
    w = int(h * aspect * 1.25)
    density = uniform_filter((cmp >= 3).astype(np.float32), size=(h, w), mode="constant")
    r, c = np.unravel_index(np.argmax(density), density.shape)
    r0, c0 = max(r - h // 2, 0), max(c - w // 2, 0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.05})
    _map(ax0, cmp, colors, aspect, "Whole region")
    ax0.add_patch(Rectangle((c0, r0), w, h, fill=False, edgecolor=S.INK, lw=1))
    _map(ax1, cmp[r0:r0 + h, c0:c0 + w], colors, aspect, "Detail: area with the most new urban cells")
    for s in ax1.spines.values():
        s.set_visible(True)
        s.set_edgecolor(S.INK)
    m = json.loads((C.OUTPUTS / "metrics.json").read_text()) if (C.OUTPUTS / "metrics.json").exists() else None
    sub = "New urban area: where the model placed it vs. where it actually appeared."
    if m:
        sub += f"\nFigure of merit {m['figure_of_merit']:.1%}."
    S.header(fig, f"Validation, {y['train_end']}–{y['validation_end']}", sub)
    fig.subplots_adjust(left=0.03, right=0.97, top=S.frac(fig, 1.75), bottom=S.bottom(fig, 0.95))
    _legend(fig, [
        (S.GREY, f"Urban in {y['train_end']}"),
        (hit, "Hit (observed and simulated)"),
        (miss, "Missed (observed only)"),
        (false, "False alarm (simulated only)"),
    ])
    S.footer(fig)
    return _save(fig, "map_validation.png")


def suitability_plot(cfg: C.Config) -> Path | None:
    """Lift of urbanization probability per distance bin, from the fitted suitability."""
    f = C.OUTPUTS / "suitability.json"
    if not f.exists():
        return None
    plt = _plt()
    from matplotlib.ticker import FixedLocator, NullLocator

    factors = json.loads(f.read_text())["factors"]
    labels = {"dist_urban": "Distance to urban area", "dist_road": "Distance to main roads"}
    floor = 1e-3  # log axis: lifts of 0 are drawn at the floor and labelled
    fig, axes = plt.subplots(1, len(factors), figsize=(11, 5.4), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (name, d) in zip(np.atleast_1d(axes), factors.items()):
        lifts = np.array(d["lifts"])
        x = np.arange(len(lifts))
        top = np.maximum(lifts, floor)
        ax.bar(x, np.where(lifts > 0, top - 1, 0), bottom=1, width=0.6, color=S.ORANGE, linewidth=0)
        ax.axhline(1, color=S.INK, lw=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.5, 20)
        ax.yaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0.001×", "0.01×", "0.1×", "1× (avg)", "10×"])
        _value_axis(ax)
        ticks = [f"<{d['edges'][0] / 1000:.1f}"] + [f"{e / 1000:.1f}" for e in d["edges"][1:]] + [f">{d['edges'][-1] / 1000:.0f}"]
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=90, fontsize=7.5)
        ax.set_xlabel("Distance bin, upper limit (km)")
        ax.set_title(labels.get(name, name), loc="left", fontsize=10, fontweight="semibold", color=S.INK)
        ax.annotate(f"{lifts[0]:.1f}×", (0, lifts[0]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color=S.INK)
        for i in np.flatnonzero(lifts == 0):
            ax.annotate("none", (i, 1), xytext=(0, -4), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color=S.MUTED, rotation=90)
    np.atleast_1d(axes)[0].set_ylabel("Urbanization rate vs. regional average")
    y = cfg["years"]
    S.header(fig, f"Where land urbanized, {y['start']}–{y['train_end']}",
             "Share of cells that became urban in each distance bin,\nrelative to the regional average (log scale).")
    fig.subplots_adjust(left=0.11, right=0.97, top=S.frac(fig, 1.6), bottom=S.bottom(fig, 1.35))
    S.footer(fig)
    return _save(fig, "suitability_lift.png")


def urban_area_plot(cfg: C.Config) -> Path | None:
    """Urban area over time: observed, the 2015-2023 validation run and the projection."""
    f = C.OUTPUTS / "area_by_class.csv"
    if not f.exists():
        return None
    plt = _plt()
    import pandas as pd
    from matplotlib.lines import Line2D

    y = cfg["years"]
    urban = cfg.classes[C.URBAN]["name"]
    df = pd.read_csv(f).query("`class` == @urban")
    obs = df[df.type == "observed"].set_index("year")["area_km2"]
    sim = df[df.type == "simulated"].set_index("year")["area_km2"]
    if obs.empty:
        return None

    observed, model_c = S.INK, S.ORANGE
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(obs.index, obs.values, color=observed, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter(obs.index, obs.values, s=44, color=observed, edgecolor="white", linewidth=2, zorder=4)
    model = []
    if y["validation_end"] in sim.index and y["train_end"] in obs.index:
        model.append(([y["train_end"], y["validation_end"]], [obs[y["train_end"]], sim[y["validation_end"]]]))
    if y["horizon"] in sim.index and y["validation_end"] in obs.index:
        model.append(([y["validation_end"], y["horizon"]], [obs[y["validation_end"]], sim[y["horizon"]]]))
    for xs, ys in model:
        ax.plot(xs, ys, color=model_c, lw=2, ls=(0, (4, 2)), dash_capstyle="round", zorder=2)
        ax.scatter(xs[1:], ys[1:], s=44, color=model_c, edgecolor="white", linewidth=2, zorder=4)

    def label(x, v, text, dy=8, va="bottom"):
        ax.annotate(text, (x, v), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                    fontsize=8.5, color=S.INK, path_effects=S.halo())

    label(obs.index[0], obs.iloc[0], f"{obs.iloc[0]:,.0f}")
    if y["validation_end"] in obs.index:
        label(y["validation_end"], obs[y["validation_end"]], f"{obs[y['validation_end']]:,.0f} observed", dy=-10, va="top")
    if y["validation_end"] in sim.index:
        label(y["validation_end"], sim[y["validation_end"]], f"{sim[y['validation_end']]:,.0f} simulated")
    if y["horizon"] in sim.index:
        label(y["horizon"], sim[y["horizon"]], f"{sim[y['horizon']]:,.0f}")

    ax.set_ylim(0, max(df.area_km2.max() * 1.18, 1))
    ax.set_xlim(y["start"] - 2, y["horizon"] + 2)
    ax.set_xticks(sorted({*obs.index, *sim.index}))
    _value_axis(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_ylabel("km²")
    ax.legend(handles=[
        Line2D([], [], color=observed, lw=2, marker="o", markersize=6, markeredgecolor="white", label="Observed (MapBiomas)"),
        Line2D([], [], color=model_c, lw=2, ls=(0, (4, 2)), marker="o", markersize=6, markeredgecolor="white",
               label=f"Model: validation from {y['train_end']},\nprojection from {y['validation_end']}"),
    ], loc="lower right", fontsize=8.5, labelcolor=S.INK)
    S.header(fig, "Urban area", f"{cfg['region']['name']}, km²")
    fig.subplots_adjust(left=0.1, right=0.96, top=S.frac(fig, 1.2), bottom=S.bottom(fig, 0.95))
    S.footer(fig)
    return _save(fig, "urban_area.png")


def export(cfg: C.Config) -> None:
    files = [area_table(cfg), *maps(cfg)]
    for p in (suitability_plot(cfg), urban_area_plot(cfg)):
        if p is not None:
            files.append(p)
    for f in (C.OUTPUTS / "metrics.json", C.OUTPUTS / "suitability.json", C.OUTPUTS / "markov_matrix.json"):
        if f.exists():
            files.append(f)
    print("Outputs:", *[f.name for f in files], sep="\n  ")
