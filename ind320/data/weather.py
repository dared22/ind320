"""Hourly ERA5 averages over all 1,000 reproducibly sampled points per area."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ind320.config import ROOT

COORDINATES = ROOT / "data" / "weather_coordinates.json"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLUMNS = ["precipitation", "temperature_2m"]


def load_weather_coordinates() -> pd.DataFrame:
    """Read every sampled point without adding GIS dependencies to the app."""
    points = pd.DataFrame(json.loads(COORDINATES.read_text())["points"])
    counts = points.groupby("area").size()
    expected = {f"NO{i}" for i in range(1, 6)} | {"East", "West", "North"}
    if set(counts.index) != expected or not counts.eq(1000).all():
        raise ValueError("Expected 1,000 sampled weather points in each of eight areas.")
    values = points[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (
        points.latitude.between(-90, 90) & points.longitude.between(-180, 180)
    ).all():
        raise ValueError("Weather coordinates must be finite WGS84 locations.")
    return points.set_index("area")


def weather_grid_cells(points: pd.DataFrame) -> pd.DataFrame:
    """Count sampled points per nearest 0.25° ERA5 cell, preserving their weight.

    With nearest-cell selection and elevation downscaling disabled, points in
    the same cell share a series. A cell containing 20 sampled points must
    contribute 20 times as much as one containing a single sampled point.
    """
    cells = (points[["latitude", "longitude"]] * 4).round() / 4
    return cells.groupby(["latitude", "longitude"]).size().rename("weight").reset_index()


def _hourly_frame(payload: dict, expected: pd.DatetimeIndex) -> pd.DataFrame:
    """Reject an incomplete cell instead of biasing the spatial average."""
    try:
        hourly = pd.DataFrame(payload["hourly"]).set_index("time")[WEATHER_COLUMNS]
        hourly.index = pd.to_datetime(hourly.index, utc=True)
    except (KeyError, TypeError) as error:
        raise ValueError("ERA5 response is missing the requested hourly variables.") from error
    hourly.index.name = "time"
    if not hourly.index.equals(expected) or not np.isfinite(hourly.to_numpy(dtype=float)).all():
        raise ValueError("ERA5 did not return a complete, finite hourly timeline for every cell.")
    if (hourly.precipitation < 0).any():
        raise ValueError("ERA5 returned negative precipitation.")
    return hourly


def fetch_weather(
    area: str, start_date: str, end_date: str, *,
    batch_size: int = 20, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download grid cells in batches and return the sampled-area hourly mean.

    Cache validated cell responses locally so retrying after a rate limit resumes
    completed cells. Public weather caches stay out of git. Precipitation is an
    hourly amount in mm; temperature is in degrees Celsius.
    """
    coordinates = load_weather_coordinates()
    if area not in coordinates.index:
        raise ValueError(f"Unknown weather area: {area}")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
            or start > end or start != start.normalize() or end != end.normalize()):
        raise ValueError("Use ordered calendar dates without a time component.")
    if not 1 <= batch_size <= 100:
        raise ValueError("Batch size must be between 1 and 100.")
    expected = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h", tz="UTC")
    cells = weather_grid_cells(coordinates.loc[[area]])
    cache = Path(cache_dir) if cache_dir is not None else ROOT / ".local" / "weather"
    # Version the cache so old single-point data cannot be reused as area data.
    cache = cache / f"era5-nearest-v1_{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    cache.mkdir(parents=True, exist_ok=True)
    total = np.zeros((len(expected), len(WEATHER_COLUMNS)))
    pending = []
    for cell in cells.itertuples(index=False):
        path = cache / f"{cell.latitude:.2f}_{cell.longitude:.2f}.json"
        if path.exists():
            try:
                frame = _hourly_frame(json.loads(path.read_text()), expected)
            except (ValueError, TypeError):
                pending.append((cell, path))
            else:
                total += frame.to_numpy(dtype=float) * cell.weight
        else:
            pending.append((cell, path))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        response = requests.get(
            WEATHER_API,
            params={
                "latitude": ",".join(str(cell.latitude) for cell, _ in batch),
                "longitude": ",".join(str(cell.longitude) for cell, _ in batch),
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "hourly": ",".join(WEATHER_COLUMNS), "models": "era5",
                "timezone": "UTC", "cell_selection": "nearest", "elevation": "nan",
            },
            timeout=(10, 120),
        )
        response.raise_for_status()
        payload = response.json()
        records = payload if isinstance(payload, list) else [payload]
        if len(records) != len(batch):
            raise ValueError("ERA5 returned a different number of grid cells than requested.")
        # Multi-location responses identify their original request order.
        if all("location_id" in record for record in records):
            records = sorted(records, key=lambda record: record["location_id"])
            if [r["location_id"] for r in records] != list(range(len(batch))):
                raise ValueError("ERA5 returned unexpected location identifiers.")
        for (cell, path), record in zip(batch, records):
            frame = _hourly_frame(record, expected)
            if not np.allclose(
                [record.get("latitude", np.nan), record.get("longitude", np.nan)],
                [cell.latitude, cell.longitude], atol=1e-4, rtol=0,
            ):
                raise ValueError("ERA5 returned an unexpected grid cell; sampling weights cannot be applied.")
            # Atomic replacement prevents half-written JSON after interruption.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record))
            temporary.replace(path)
            total += frame.to_numpy(dtype=float) * cell.weight
    hourly = pd.DataFrame(total / cells.weight.sum(), index=expected, columns=WEATHER_COLUMNS)
    hourly.index.name = "time"
    hourly.attrs.update(area=area, sampled_points=int(cells.weight.sum()), grid_cells=len(cells),
                        model="ERA5", spatial_method="sample-weighted nearest-grid mean")
    return hourly


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average 24 hourly amounts/temperatures per UTC day, never rainfall totals."""
    if not (hourly[WEATHER_COLUMNS].resample("D").count() == 24).all().all():
        raise ValueError("Daily averages require 24 nonmissing observations per day.")
    return hourly[WEATHER_COLUMNS].resample("D").mean()
