from typing import Any

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"
INTERVAL_NOTICE = (
    "NVE's public endpoint returns the full history. The app caches that download "
    "for one hour and displays the selected months. The API has no date filters, "
    "so the month slider cannot limit the download."
)

# Keep the Part 1 English labels. Source fractions remain unchanged; the wide
# fill-level table alone uses percentages, avoiding accidental double conversion.
COLUMN_NAMES = {
    "dato_Id": "Observation Date",
    "omrType": "Area Type",
    "omrnr": "Area Code",
    "iso_aar": "Iso Year",
    "iso_uke": "Iso Week",
    "fyllingsgrad": "Fill Level",
    "kapasitet_TWh": "Capacity TWh",
    "fylling_TWh": "Stored Energy TWh",
    "neste_Publiseringsdato": "Next Publication Date",
    "fyllingsgrad_forrige_uke": "Previous Week Fill Level",
    "endring_fyllingsgrad": "Weekly Fill Level Change",
}


def _request_json(endpoint: str) -> Any:
    """Request public JSON with timeouts and bounded retries for transient errors."""
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(
            f"{API_BASE}/{endpoint}",
            timeout=(10, 60),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()


@st.cache_data(ttl=3600, show_spinner="Loading NVE reservoir data...")
def fetch_reservoir_history() -> pd.DataFrame:
    """Download and prepare history once per hour, shared by all pages."""
    records = _request_json("HentOffentligData")
    return prepare_reservoir_data(records)


def prepare_area_metadata(payload: Any) -> pd.DataFrame:
    """Flatten the API's land/elspot/vassdrag descriptions without fixed codes."""
    groups = payload if isinstance(payload, list) else [payload]
    rows = [
        area
        for group in groups
        for key in ("land", "elspot", "vassdrag")
        for area in (group.get(key) or [])
    ]
    result = pd.DataFrame(rows).rename(
        columns={
            "omrType": "Area Type",
            "omrnr": "Area Code",
            "navn": "Name",
            "navn_langt": "Long Name",
            "beskrivelse": "Description",
        }
    )
    result = result[["Area Type", "Area Code", "Name", "Long Name", "Description"]]
    result["Area Code"] = pd.to_numeric(result["Area Code"]).astype(int)
    return result.sort_values(["Area Type", "Area Code"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_area_metadata() -> pd.DataFrame:
    """Return current API area descriptions, including areas without observations."""
    return prepare_area_metadata(_request_json("HentOmr%C3%A5der"))


def prepare_reservoir_data(records: list[dict]) -> pd.DataFrame:
    """Parse source units and dates; preserve unusual readings for inspection."""
    source = pd.DataFrame(records).rename(columns=COLUMN_NAMES)
    source = source[list(COLUMN_NAMES.values())].copy()
    source["Observation Date"] = pd.to_datetime(
        source["Observation Date"], format="ISO8601", utc=True
    ).dt.tz_convert(None)
    # NVE uses year 0001 when the next publication date is unavailable.
    source["Next Publication Date"] = pd.to_datetime(
        source["Next Publication Date"], format="ISO8601", errors="coerce", utc=True
    ).dt.tz_convert(None)
    for column in COLUMN_NAMES.values():
        if column not in ("Observation Date", "Next Publication Date", "Area Type"):
            source[column] = pd.to_numeric(source[column])
    source["Area Code"] = source["Area Code"].astype(int)
    return source.sort_values(
        ["Observation Date", "Area Type", "Area Code"]
    ).reset_index(drop=True)


def select_reservoir_data(
    source: pd.DataFrame,
    start_month: str | None = None,
    end_month: str | None = None,
    area_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter inclusive months and build percentage series for observed areas."""
    start = pd.Period(start_month, freq="M") if start_month else None
    end = pd.Period(end_month, freq="M") if end_month else None
    selected = source.copy()
    months = selected["Observation Date"].dt.to_period("M")
    if start is not None:
        selected = selected.loc[months >= start]
    if end is not None:
        selected = selected.loc[months <= end]
    if area_type is not None:
        selected = selected.loc[selected["Area Type"] == area_type]
    levels = (
        selected.pivot(
            index="Observation Date",
            columns=["Area Type", "Area Code"],
            values="Fill Level",
        )
        .sort_index()
        .dropna(axis=1, how="all")
    )
    # Preserve the original national-first display, then observed EL/VASS codes.
    order = sorted(levels.columns, key=lambda key: (key[0] != "NO", key[0], key[1]))
    levels = levels.loc[:, order] * 100
    levels.columns = [f"{kind} {code}" for kind, code in levels.columns]
    return selected.reset_index(drop=True), levels


def load_reservoir_data(
    start_month: str | None = None,
    end_month: str | None = None,
    area_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select months and areas from the shared, prepared history."""
    source = fetch_reservoir_history()
    return select_reservoir_data(source, start_month, end_month, area_type)


def unusual_fill_levels(source: pd.DataFrame) -> pd.DataFrame:
    """Show readings outside 0–100% without clipping or dropping them."""
    return source.loc[(source["Fill Level"] < 0) | (source["Fill Level"] > 1)].copy()
