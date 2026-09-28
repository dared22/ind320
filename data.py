"""Load and prepare the course's weekly reservoir CSV for the app."""

from pathlib import Path

import pandas as pd
import streamlit as st


CSV_PATH = Path(__file__).resolve().parent / "reservoirs.csv"

# Match the understandable English headers used in the project notebook.
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


@st.cache_data(show_spinner="Loading reservoir data...")
def load_reservoir_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned source rows and nine weekly fill-level series.

    The source has one row per date and area. Pivoting makes each area's
    numeric fill level a column, ready for the required row-wise sparklines.
    """
    source = pd.read_csv(CSV_PATH)
    missing_columns = COLUMN_NAMES.keys() - source.columns
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")

    source = source.rename(columns=COLUMN_NAMES)
    source["Observation Date"] = pd.to_datetime(
        source["Observation Date"], format="%Y-%m-%d", errors="raise"
    )
    # Year 0001 in the CSV marks an unavailable publication date.
    source["Next Publication Date"] = pd.to_datetime(
        source["Next Publication Date"], format="ISO8601", errors="coerce"
    )
    source = source.sort_values(
        ["Observation Date", "Area Type", "Area Code"]
    ).reset_index(drop=True)

    area_key = ["Observation Date", "Area Type", "Area Code"]
    if source.duplicated(area_key).any():
        raise ValueError("The CSV contains more than one fill level for a date and area.")

    # Every column of this wide table is one numeric reservoir-area series.
    fill_levels = source.pivot(
        index="Observation Date",
        columns=["Area Type", "Area Code"],
        values="Fill Level",
    )
    fill_levels.columns = [
        f"{area_type} {area_code}" for area_type, area_code in fill_levels.columns
    ]
    area_order = [
        "NO 0",
        *(f"EL {code}" for code in range(1, 6)),
        *(f"VASS {code}" for code in range(1, 4)),
    ]
    fill_levels = fill_levels[area_order].sort_index() * 100

    return source, fill_levels
