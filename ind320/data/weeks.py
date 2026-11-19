"""Align observations and source availability on Norwegian ISO weeks."""

from datetime import date

import pandas as pd

from ind320.data.transfers import clip_intervals
from ind320.data.weather import fetch_weather

OSLO = "Europe/Oslo"
WEEK = pd.Timedelta(weeks=1)
WEATHER_FEATURES = ("temperature_2m", "precipitation")


def cutoff_time(day: date | str | pd.Timestamp) -> pd.Timestamp:
    """Use local end of day, capped at now for today's running archive."""
    end = (pd.Timestamp(day).normalize() + pd.Timedelta(days=1)).tz_localize(OSLO)
    return min(end - pd.Timedelta(nanoseconds=1), pd.Timestamp.now(tz=OSLO))


def reservoir_weeks(source: pd.DataFrame) -> pd.DataFrame:
    """Separate the Sunday observation label from the scheduled release."""
    rows = source.copy()
    if {"Iso Year", "Iso Week"} <= set(rows):
        # ISO years can differ from the calendar year around New Year.
        labels = (rows["Iso Year"].astype(str) + "-W"
                  + rows["Iso Week"].astype(str).str.zfill(2) + "-7")
        rows["week"] = pd.to_datetime(labels, format="%G-W%V-%u")
    else:
        rows["week"] = rows["Observation Date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    release = rows.week + pd.Timedelta(days=3, hours=13)
    if "Next Publication Date" in rows:
        # A row's next release applies to the following week, never to itself.
        next_release = rows[["Area Type", "Area Code", "week", "Next Publication Date"]].copy()
        next_release["week"] += WEEK
        lookup = next_release.set_index(["Area Type", "Area Code", "week"])["Next Publication Date"]
        if lookup.index.is_unique:
            scheduled = pd.Series(rows.set_index(lookup.index.names).index.map(lookup), index=rows.index)
            valid = scheduled.between(rows.week + pd.Timedelta(days=1), rows.week + WEEK)
            release = scheduled.where(valid, release)
    rows["available_at"] = release.dt.tz_localize(OSLO)
    return rows


def _labels(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    local = index.tz_convert(OSLO).tz_localize(None)
    return local.to_period("W-SUN").end_time.normalize().rename("week")


def _week_end(weeks: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (weeks + pd.Timedelta(days=1)).tz_localize(OSLO)


def weekly_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Respect DST and assign precipitation to the hour in which it fell."""
    temperature = hourly.temperature_2m.groupby(_labels(hourly.index)).agg(["mean", "count"])
    rainfall = hourly.precipitation.groupby(_labels(hourly.index - pd.Timedelta(hours=1))).agg(["sum", "count"])
    weeks = temperature.index.intersection(rainfall.index)
    end = _week_end(weeks)
    start = _week_end(weeks - WEEK)
    hours = (end - start) / pd.Timedelta(hours=1)
    complete = (temperature.loc[weeks, "count"] == hours) & (rainfall.loc[weeks, "count"] == hours)
    return pd.DataFrame({
        "temperature_2m": temperature.loc[weeks, "mean"],
        "precipitation": rainfall.loc[weeks, "sum"],
        "available_at": end + pd.Timedelta(days=5),
    }).loc[complete]


def weather_week_range(area: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Load UTC days around a block of local weeks, including boundary hours."""
    start = first - pd.Timedelta(days=7)
    # Oslo Monday midnight falls on Sunday UTC, including the final rainfall sample.
    end = last
    return weekly_weather(fetch_weather(area, str(start.date()), str(end.date()))).loc[first:last]


def weekly_fx(rates: pd.DataFrame) -> pd.DataFrame:
    """Use the final business-day quote and its local publication hour."""
    rows = rates.sort_values("date").copy()
    rows["week"] = rows.date.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    rows["available_at"] = (rows.date + pd.Timedelta(hours=16)).dt.tz_localize(OSLO)
    last = rows.drop_duplicates(["week", "pair"], keep="last")
    return last.set_index(["week", "pair"])[["nok_per_unit", "available_at"]]


def weekly_flows(data: pd.DataFrame) -> pd.DataFrame:
    """Sum imports minus exports on one border with full interval coverage."""
    columns = ["net_import_mwh", "available_at"]
    if data.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="week"))
    if data.connection.nunique() != 1:
        raise ValueError("Choose one connection for the net-import input.")
    result = []
    for week in pd.date_range(_labels(pd.DatetimeIndex(data.start)).min(),
                              _labels(pd.DatetimeIndex(data.end)).max(), freq="W-SUN"):
        end = _week_end(pd.DatetimeIndex([week]))[0]
        start = _week_end(pd.DatetimeIndex([week - WEEK]))[0]
        block = clip_intervals(data, start, end)
        energy = {}
        for direction, groups in block.groupby("direction"):
            group = groups.sort_values("start")
            contiguous = (group.start.iloc[0] == start and group.end.iloc[-1] == end
                          and (group.start.iloc[1:].array == group.end.iloc[:-1].array).all())
            if contiguous:
                energy[direction] = group.energy_mwh.sum()
        if {"import", "export"} <= energy.keys():
            result.append((week, energy["import"] - energy["export"], end + pd.Timedelta(hours=1)))
    return pd.DataFrame(result, columns=["week", *columns]).set_index("week")


def rolling_weather_correlation(
    filling: pd.Series, weather: pd.Series, lag: int, window: int
) -> pd.Series:
    """Correlate weather leading weekly percentage-point filling changes."""
    filling = filling.sort_index().asfreq("W-SUN")
    change = filling.diff()
    lagged_weather = weather.reindex(filling.index).shift(lag)
    return change.rolling(window, min_periods=window).corr(lagged_weather).clip(-1, 1)
