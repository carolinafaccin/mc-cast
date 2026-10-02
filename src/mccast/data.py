"""Download, clip and reclassify MapBiomas land cover; load the study area."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C

CELL_SIZE_M = 30.0  # MapBiomas native resolution


def study_area(cfg: C.Config):
    """Dissolved boundary (EPSG:4326) of the configured municipalities. Cached."""
    import geobr
    import geopandas as gpd

    cache = C.RAW / f"{cfg['region']['slug']}_boundary.gpkg"
    if cache.exists():
        return gpd.read_file(cache)

    munis = geobr.read_municipality(code_muni=cfg["region"]["state"], year=2022)
    wanted = {int(m["code"]): m["name"] for m in cfg["region"]["municipalities"]}
    area = munis[munis["code_muni"].astype(int).isin(wanted)]
    missing = sorted(set(wanted) - set(area["code_muni"].astype(int)))
    print(f"{len(area)}/{len(wanted)} municipalities found.")
    if missing:
        raise SystemExit("Not found in geobr: " + ", ".join(f"{c} {wanted[c]}" for c in missing))
    out = gpd.GeoDataFrame(geometry=[area.to_crs(4326).geometry.union_all()], crs=4326)
    C.RAW.mkdir(parents=True, exist_ok=True)
    out.to_file(cache, driver="GPKG")
    area[["code_muni", "name_muni"]].to_csv(C.RAW / f"{cfg['region']['slug']}_municipalities.csv", index=False)
    return out


def download_year(cfg: C.Config, year: int, overwrite: bool = False) -> Path:
    """Read only the study-area window of the national MapBiomas GeoTIFF."""
    import rasterio
    from rasterio.windows import from_bounds

    dst = C.RAW / f"mapbiomas_{year}.tif"
    if dst.exists() and not overwrite:
        return dst
    bounds = study_area(cfg).total_bounds
    url = "/vsicurl/" + cfg["mapbiomas"]["url_template"].format(year=year)
    with rasterio.open(url) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        profile = src.profile | {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": src.window_transform(win),
            "compress": "deflate",
        }
    C.RAW.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as out:
        out.write(arr, 1)
    return dst


def reclassify(arr: np.ndarray, cfg: C.Config) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for cid, spec in cfg.classes.items():
        lut[spec["mapbiomas"]] = cid
    return lut[arr]


def prepare_year(cfg: C.Config, year: int, overwrite: bool = False) -> Path:
    """Reclassify to model classes and mask everything outside the study area."""
    import rasterio
    from rasterio.features import geometry_mask

    dst = C.PROCESSED / f"lulc_{year}.tif"
    if dst.exists() and not overwrite:
        return dst
    with rasterio.open(C.RAW / f"mapbiomas_{year}.tif") as src:
        arr, profile = src.read(1), src.profile
    inside = geometry_mask(
        study_area(cfg).geometry, arr.shape, profile["transform"], invert=True
    )
    out = np.where(inside, reclassify(arr, cfg), 0).astype(np.uint8)
    profile |= {"dtype": "uint8", "nodata": 0, "compress": "deflate"}
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as o:
        o.write(out, 1)
    return dst


def read_lulc(year: int) -> tuple[np.ndarray, dict]:
    import rasterio

    with rasterio.open(C.PROCESSED / f"lulc_{year}.tif") as src:
        return src.read(1), src.profile


OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def _download_roads(bounds, tiles: int = 6, retries: int = 6):
    """Main roads from OpenStreetMap (Overpass API), fetched in bbox tiles to keep each query small."""
    import time

    import geopandas as gpd
    import numpy as np
    import requests
    from shapely.geometry import LineString

    xs, ys = np.linspace(bounds[0], bounds[2], tiles + 1), np.linspace(bounds[1], bounds[3], tiles + 1)
    lines = {}
    for i in range(tiles):
        for j in range(tiles):
            q = (
                '[out:json][timeout:180];way["highway"~"^(motorway|trunk|primary|secondary)$"]'
                f"({ys[j]},{xs[i]},{ys[j + 1]},{xs[i + 1]});out geom;"
            )
            for attempt in range(retries):
                try:
                    r = requests.post(
                        OVERPASS[attempt % len(OVERPASS)], data={"data": q},
                        headers={"User-Agent": "mccast-study-project/0.1"}, timeout=200,
                    )
                    r.raise_for_status()
                    for el in r.json()["elements"]:
                        if el.get("geometry") and len(el["geometry"]) > 1:
                            lines[el["id"]] = LineString([(p["lon"], p["lat"]) for p in el["geometry"]])
                    break
                except Exception as e:  # Overpass is often busy: back off and retry
                    print(f"  tile {i},{j} attempt {attempt + 1}: {type(e).__name__}")
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"Overpass tile {i},{j} failed after {retries} attempts")
        print(f"  roads: column {i + 1}/{tiles} done ({len(lines)} ways)")
    return gpd.GeoDataFrame(geometry=list(lines.values()), crs=4326)


def roads_distance(cfg: C.Config, profile: dict) -> np.ndarray:
    """Distance (m) to main roads (OpenStreetMap, current network). Cached."""
    import geopandas as gpd
    from rasterio.features import rasterize

    from .suitability import distance_to

    cache = C.RAW / f"{cfg['region']['slug']}_roads.gpkg"
    if cache.exists():
        roads = gpd.read_file(cache)
    else:
        roads = _download_roads(study_area(cfg).total_bounds)
        roads.to_file(cache, driver="GPKG")
    mask = rasterize(
        ((g, 1) for g in roads.geometry),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    return distance_to(mask, CELL_SIZE_M)


def cell_area_km2(profile: dict) -> float:
    """Approximate area of one cell (the grid is in degrees)."""
    t = profile["transform"]
    lat = t.f + t.e * profile["height"] / 2
    m_per_deg = 111_320.0
    return abs(t.a) * m_per_deg * np.cos(np.radians(lat)) * abs(t.e) * 110_574.0 / 1e6


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
