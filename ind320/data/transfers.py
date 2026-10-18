import xml.etree.ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ind320.config import get_setting

API_URL = "https://web-api.tp.entsoe.eu/api"
SOURCE = "ENTSO-E Transparency Platform, A11 cross-border physical flows"
# Bidding-zone pairs only: no Norwegian internal borders or country aggregates.
# EIC area codes: ENTSO-E API reference, Area list. NO2–GB uses the GB domain.
ZONES = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "FI": "10YFI-1--------U",
    "DK1": "10YDK-1--------W",
    "NL": "10YNL----------L",
    "DE_LU": "10Y1001A1001A82H",
    "GB": "10YGB----------A",
    "RU": "10Y1001A1001A49F",
}
CONNECTIONS = {
    f"{a}–{b}": (a, b)
    for a, b in [
        ("NO1", "SE3"),
        ("NO2", "DK1"),
        ("NO2", "NL"),
        ("NO2", "DE_LU"),
        ("NO2", "GB"),
        ("NO3", "SE2"),
        ("NO4", "SE1"),
        ("NO4", "SE2"),
        ("NO4", "FI"),
        ("NO4", "RU"),
    ]
}
COLUMNS = ["connection", "direction", "start", "end", "power_mw", "energy_mwh"]


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Select Norwegian calendar months, then convert both boundaries to UTC."""
    start = pd.Timestamp(year=year, month=month, day=1, tz="Europe/Oslo")
    return start.tz_convert("UTC"), (start + pd.DateOffset(months=1)).tz_convert("UTC")


def parse_flows(xml: str, connection: str, direction: str) -> pd.DataFrame:
    """Expand A01 points or A03 step changes without inventing missing A01 data."""
    root = ET.fromstring(xml)
    # Ignore namespace versions while keeping the source element names.
    for element in root.iter():
        element.tag = element.tag.split("}")[-1]
    rows = []
    for series in root.findall("TimeSeries"):
        curve = series.findtext("curveType", "A01")
        for period in series.findall("Period"):
            start = pd.Timestamp(period.findtext("timeInterval/start"))
            end = pd.Timestamp(period.findtext("timeInterval/end"))
            step = pd.Timedelta(period.findtext("resolution"))
            points = sorted(
                (int(p.findtext("position")), float(p.findtext("quantity")))
                for p in period.findall("Point")
            )
            for i, (position, power) in enumerate(points):
                left = start + (position - 1) * step
                right = left + step
                if curve == "A03":
                    # Step changes last until the next point or period end.
                    right = end
                    if i + 1 < len(points):
                        right = start + (points[i + 1][0] - 1) * step
                rows.append(
                    (
                        connection,
                        direction,
                        left,
                        right,
                        power,
                        power * (right - left).total_seconds() / 3600,
                    )
                )
    return prepare_intervals(pd.DataFrame(rows, columns=COLUMNS))


def prepare_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the shared fields, parse UTC dates, and remove identical records."""
    data = frame[COLUMNS].copy()
    for column in ("start", "end"):
        data[column] = pd.to_datetime(data[column], utc=True)
    return (
        data.drop_duplicates()
        .sort_values(["connection", "direction", "start"])
        .reset_index(drop=True)
    )


def clip_intervals(
    data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Clip boundary intervals and prorate energy, including local year boundaries."""
    result = data.loc[(data.end > start) & (data.start < end)].copy()
    result["start"] = result.start.clip(lower=start)
    result["end"] = result.end.clip(upper=end)
    result["energy_mwh"] = (
        result.power_mw * (result.end - result.start).dt.total_seconds() / 3600
    )
    return result


def fetch_month(
    session: requests.Session,
    connection: str,
    direction: str,
    year: int,
    month: int,
    token: str,
) -> pd.DataFrame:
    """Request one month without printing URLs that contain the API token."""
    start, end = month_bounds(year, month)
    no, foreign = CONNECTIONS[connection]
    origin, destination = (foreign, no) if direction == "import" else (no, foreign)
    try:
        response = session.get(
            API_URL,
            params={
                "securityToken": token,
                "documentType": "A11",
                "out_Domain": ZONES[origin],
                "in_Domain": ZONES[destination],
                "periodStart": start.strftime("%Y%m%d%H%M"),
                "periodEnd": end.strftime("%Y%m%d%H%M"),
            },
            timeout=(10, 60),
        )
    except requests.RequestException:
        # Request exceptions can contain the API token in their URL.
        print("ENTSO-E request failed. Check the connection and API token.")
        return prepare_intervals(pd.DataFrame(columns=COLUMNS))
    if response.status_code == 400:
        # No-data responses use HTTP 400; the XML contains no observations.
        return parse_flows(response.text, connection, direction)
    if not response.ok:
        print(f"ENTSO-E request returned HTTP {response.status_code}.")
        return prepare_intervals(pd.DataFrame(columns=COLUMNS))
    return clip_intervals(parse_flows(response.text, connection, direction), start, end)


def ingest(
    years: tuple[int, ...] = (2024, 2025), connections: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download monthly chunks and report every gap, without substituting zeros."""
    token = get_setting("ENTSOE_API_TOKEN")
    frames, reports = [], []
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        for connection in sorted(CONNECTIONS if connections is None else connections):
            for year in years:
                for direction in ("import", "export"):
                    for month in range(1, 13):
                        start, end = month_bounds(year, month)
                        data = fetch_month(
                            session, connection, direction, year, month, token
                        )
                        frames.append(data)
                        hours = (data.end - data.start).dt.total_seconds().sum() / 3600
                        message = "No observations returned." if data.empty else ""
                        expected = (end - start).total_seconds() / 3600
                        reports.append(
                            {
                                "connection": connection,
                                "year": year,
                                "month": month,
                                "direction": direction,
                                "observed_hours": hours,
                                "expected_hours": expected,
                                "complete": hours == expected,
                                "message": message,
                            }
                        )
                # Long notebook runs show safe progress without request URLs.
                print(
                    f"ENTSO-E monthly requests finished: {connection}, {year}",
                    flush=True,
                )
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=COLUMNS)
    )
    return prepare_intervals(result), pd.DataFrame(reports)


def annual_totals(
    data: pd.DataFrame, years: tuple[int, ...] = (2024, 2025)
) -> pd.DataFrame:
    """Calculate observed energy and duration; absent years remain explicitly incomplete."""
    rows = []
    for connection in sorted(CONNECTIONS):
        for year in years:
            start, _ = month_bounds(year, 1)
            end, _ = month_bounds(year + 1, 1)
            selected = clip_intervals(
                data.loc[data.connection == connection], start, end
            )
            expected = (end - start).total_seconds() / 3600
            for direction in ("import", "export"):
                part = selected.loc[selected.direction == direction]
                hours = (part.end - part.start).dt.total_seconds().sum() / 3600
                rows.append(
                    {
                        "connection": connection,
                        "year": year,
                        "direction": direction,
                        "energy_mwh": float(part.energy_mwh.sum())
                        if len(part)
                        else None,
                        "observed_hours": float(hours),
                        "expected_hours": expected,
                        "complete": abs(hours - expected) < 1e-6,
                    }
                )
    return pd.DataFrame(rows)
