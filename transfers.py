"""ENTSO-E physical flows: monthly retrieval, UTC intervals, and energy coverage."""

import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import require_setting

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


class TransferError(RuntimeError):
    """A safe user-facing transfer error without private request details."""


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Select Norwegian calendar months, then convert both boundaries to UTC."""
    start = pd.Timestamp(year=year, month=month, day=1, tz="Europe/Oslo")
    return start.tz_convert("UTC"), (start + pd.DateOffset(months=1)).tz_convert("UTC")


def parse_flows(xml: str, connection: str, direction: str) -> pd.DataFrame:
    """Expand A01 points or A03 step changes without inventing missing A01 data."""
    try:
        root = ET.fromstring(xml)
        # Ignore namespace versions while keeping the source element names.
        for element in root.iter():
            element.tag = element.tag.split("}")[-1]
        if root.tag == "Acknowledgement_MarketDocument":
            reason = (root.findtext("Reason/text") or "").lower()
            if "no matching data" in reason or "no data" in reason:
                raise TransferError(
                    "ENTSO-E has no published observations for this request."
                )
            raise TransferError("ENTSO-E reports unavailable data or denied access.")
        rows = []
        no, foreign = CONNECTIONS[connection]
        origin, destination = (foreign, no) if direction == "import" else (no, foreign)
        for series in root.findall("TimeSeries"):
            if series.findtext("quantity_Measure_Unit.name") != "MAW":
                raise ValueError("Unexpected power unit")
            if (
                series.findtext("out_Domain.mRID") != ZONES[origin]
                or series.findtext("in_Domain.mRID") != ZONES[destination]
            ):
                raise ValueError("Unexpected domains")
            curve = series.findtext("curveType", "A01")
            if curve not in ("A01", "A03"):
                raise ValueError("Unsupported curve")
            for period in series.findall("Period"):
                start = pd.Timestamp(period.findtext("timeInterval/start"))
                end = pd.Timestamp(period.findtext("timeInterval/end"))
                if start.tzinfo is None or end.tzinfo is None:
                    raise ValueError("Missing timezone")
                resolution = period.findtext("resolution")
                if not re.fullmatch(r"PT(?:\d+H)?(?:\d+M)?(?:\d+S)?", resolution or ""):
                    raise ValueError("Unsupported resolution")
                step = pd.Timedelta(resolution)
                if step <= pd.Timedelta(0) or (end - start) % step:
                    raise ValueError("Invalid resolution")
                points = sorted(
                    (int(p.findtext("position")), float(p.findtext("quantity")))
                    for p in period.findall("Point")
                )
                positions = [p[0] for p in points]
                if len(set(positions)) != len(positions):
                    raise ValueError("Duplicate positions")
                for i, (position, power) in enumerate(points):
                    left = start + (position - 1) * step
                    right = (
                        start + (points[i + 1][0] - 1) * step
                        if curve == "A03" and i + 1 < len(points)
                        else end
                        if curve == "A03"
                        else left + step
                    )
                    if position < 1 or left >= end or right > end or power < 0:
                        raise ValueError("Invalid point")
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
        return validate_intervals(pd.DataFrame(rows, columns=COLUMNS))
    except (ET.ParseError, ValueError, TypeError, KeyError):
        raise TransferError(
            "ENTSO-E returned invalid intervals, domains, or units."
        ) from None


def validate_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject conflicting keys and overlaps; identical repeated points are harmless."""
    if not set(COLUMNS).issubset(frame.columns):
        raise TransferError("Transfer records are missing required fields.")
    data = frame[COLUMNS].copy()
    for col in ("start", "end"):
        data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    if data[["start", "end"]].isna().any().any():
        raise TransferError("Transfer records contain missing or invalid timestamps.")
    if data.empty:
        return data
    if (
        not data.direction.isin(["import", "export"]).all()
        or not data.connection.isin(CONNECTIONS).all()
        or not data[["power_mw", "energy_mwh"]]
        .map(lambda x: pd.notna(x) and 0 <= x < float("inf"))
        .all()
        .all()
        or (data.end <= data.start).any()
    ):
        raise TransferError("Invalid transfer values or direction.")
    expected = data.power_mw * (data.end - data.start).dt.total_seconds() / 3600
    if ((expected - data.energy_mwh).abs() > 1e-6).any():
        raise TransferError("Transfer energy does not match MW × interval hours.")
    data = data.drop_duplicates().sort_values(["connection", "direction", "start"])
    if data.duplicated(["connection", "direction", "start"]).any():
        raise TransferError("Conflicting transfer intervals share a stable key.")
    previous = data.groupby(["connection", "direction"]).end.shift()
    if (data.start < previous).any():
        raise TransferError("Overlapping transfer intervals would double-count energy.")
    return data.reset_index(drop=True)


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
    """Request one month; exceptions never include the token-bearing URL."""
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
        if response.status_code in (401, 403):
            raise TransferError(
                "ENTSO-E denied API access; check the token and permissions."
            )
        if response.status_code == 400:
            # Acknowledgements describe unavailable data without exposing URLs.
            parse_flows(response.text, connection, direction)
        response.raise_for_status()
    except requests.RequestException:
        raise TransferError(
            "ENTSO-E request failed; check access, token, and availability."
        ) from None
    return clip_intervals(parse_flows(response.text, connection, direction), start, end)


def ingest(
    years: tuple[int, ...] = (2024, 2025), connections: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download monthly chunks and report every gap, without substituting zeros."""
    token = require_setting("ENTSOE_API_TOKEN")
    frames, reports = [], []
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        for connection in sorted(CONNECTIONS if connections is None else connections):
            for year in years:
                for direction in ("import", "export"):
                    for month in range(1, 13):
                        start, end = month_bounds(year, month)
                        message = ""
                        try:
                            data = fetch_month(
                                session, connection, direction, year, month, token
                            )
                            frames.append(data)
                            hours = (
                                data.end - data.start
                            ).dt.total_seconds().sum() / 3600
                        except TransferError as exc:
                            hours, message = 0, str(exc)
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
    return validate_intervals(result), pd.DataFrame(reports)


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
