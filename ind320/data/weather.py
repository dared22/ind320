"""Hourly ERA5 reanalysis at the eight reproducibly sampled locations."""

import pandas as pd
import requests

from ind320.config import ROOT

COORDINATES = ROOT / "data" / "weather_coordinates.csv"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"


def load_weather_coordinates() -> pd.DataFrame:
    """Read the notebook's committed points without any GIS dependencies."""
    return pd.read_csv(COORDINATES).set_index("area")


def fetch_weather(area: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Request inclusive UTC dates and reject incomplete hourly ERA5 responses.

    Precipitation is an hourly amount in mm; temperature is in degrees Celsius.
    Reanalysis estimates weather at a model grid cell, not a weather station.
    """
    coordinates = load_weather_coordinates()
    if area not in coordinates.index:
        raise ValueError(f"Unknown weather area: {area}")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end or start != start.normalize() or end != end.normalize():
        raise ValueError("Use ordered calendar dates without a time component.")
    point = coordinates.loc[area]
    response = requests.get(
        WEATHER_API,
        params={
            "latitude": point.latitude,
            "longitude": point.longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": "precipitation,temperature_2m",
            "models": "era5",
            "timezone": "UTC",
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    payload = response.json()
    hourly = pd.DataFrame(payload["hourly"]).set_index("time")
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    hourly.index.name = "time"
    expected = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h", tz="UTC")
    if not hourly.index.equals(expected) or hourly.isna().any().any():
        raise ValueError("ERA5 did not return a complete, nonmissing hourly timeline.")
    hourly.attrs.update(
        area=area,
        latitude=float(point.latitude),
        longitude=float(point.longitude),
        grid_latitude=payload["latitude"],
        grid_longitude=payload["longitude"],
        model="ERA5",
    )
    return hourly


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average 24 hourly amounts/temperatures per UTC day, never rainfall totals."""
    columns = ["precipitation", "temperature_2m"]
    if not (hourly[columns].resample("D").count() == 24).all().all():
        raise ValueError("Daily averages require 24 nonmissing observations per day.")
    return hourly[columns].resample("D").mean()
