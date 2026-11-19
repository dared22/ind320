"""Notebook-only GIS preparation; the app reads all sampled points from JSON."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy

from ind320.config import ROOT


def load_weather_areas() -> gpd.GeoDataFrame:
    """Load five electricity areas and three approximate NVE region groups."""
    electricity = gpd.read_file(next((ROOT / "Budomraader").glob("*.shp")))
    # The DBF has UTF-8 text but no .cpg file; specify encoding to preserve RØ.
    regions = gpd.read_file(
        next((ROOT / "NVE_Region").glob("*.shp")), encoding="utf-8"
    ).to_crs(electricity.crs)
    electricity["area"] = electricity["budomr"].str.replace(" ", "", regex=False)
    expected = {"RS", "RØ", "RV", "RM", "RN"}
    if set(electricity["area"]) != {f"NO{i}" for i in range(1, 6)}:
        raise ValueError("Expected electricity areas NO1–NO5.")
    if set(regions.REGION) != expected:
        raise ValueError("Expected NVE regions RS, RØ, RV, RM, and RN.")
    groups = {"East": ["RS", "RØ"], "West": ["RV"], "North": ["RM", "RN"]}
    geometries = dict(zip(electricity["area"], electricity.geometry))
    for area, codes in groups.items():
        geometries[area] = regions.loc[regions.REGION.isin(codes)].geometry.union_all()
    return gpd.GeoDataFrame(
        {"area": list(geometries), "geometry": list(geometries.values())},
        crs=electricity.crs,
    ).set_index("area")


def sample_weather_points(
    areas: gpd.GeoDataFrame, count: int = 1000, seed: int = 320
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Rejection-sample projected polygons and choose one candidate per area.

    Sampling is uniform in the source projected coordinate system, EPSG:25833.
    Returned candidates and chosen points use WGS84 longitude and latitude.
    """
    if count < 1:
        raise ValueError("Each area needs at least one candidate.")
    rng = np.random.default_rng(seed)
    candidates = []
    for area, polygon in areas.geometry.items():
        xmin, ymin, xmax, ymax = polygon.bounds
        accepted = []
        while len(accepted) < count:
            x = rng.uniform(xmin, xmax, count)
            y = rng.uniform(ymin, ymax, count)
            inside = contains_xy(polygon, x, y)
            accepted.extend(zip(x[inside], y[inside]))
        points = gpd.GeoDataFrame(
            {"area": [area] * count},
            geometry=gpd.points_from_xy(*zip(*accepted[:count])),
            crs=areas.crs,
        )
        candidates.append(points)
    candidates = gpd.GeoDataFrame(pd.concat(candidates, ignore_index=True)).to_crs(4326)
    selected = [i * count + int(rng.integers(count)) for i in range(len(areas))]
    return candidates, candidates.iloc[selected].reset_index(drop=True)


def export_reservoir_geojson(path: Path) -> None:
    """Write the three map views while keeping GIS dependencies in Jupyter."""
    areas = load_weather_areas()
    price = areas.loc[[f"NO{i}" for i in range(1, 6)], "geometry"].simplify_coverage(250)
    water = areas.loc[["East", "West", "North"], "geometry"].simplify_coverage(250)
    geometry = [price.union_all(), *price, *water]
    features = gpd.GeoDataFrame({
        "area_type": ["NO", *["EL"] * 5, *["VASS"] * 3],
        "area_code": [0, *range(1, 6), *range(1, 4)],
        "name": ["Norway", *price.index, *water.index],
        "geometry": geometry,
    }, crs=areas.crs)
    features["id"] = features.area_type + ":" + features.area_code.astype(str)
    features = features.set_index("id").to_crs(4326)
    path.write_text(features.to_json(separators=(",", ":")) + "\n")
