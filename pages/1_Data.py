"""Show a row-wise sparkline table and the imported reservoir records."""

import pandas as pd
import streamlit as st

from data import load_reservoir_data


st.title("Data table")
source, fill_levels = load_reservoir_data()

# The earliest calendar month has four weekly readings for every area.
first_month = fill_levels.index.min().to_period("M")
first_month_levels = fill_levels.loc[
    fill_levels.index.to_period("M") == first_month
]

st.write(
    "Each row below represents one numeric fill-level series imported from "
    "the CSV. The line shows every weekly observation in the first month."
)
summary = pd.DataFrame(
    {
        "Area": fill_levels.columns,
        "First month trend": [
            first_month_levels[area].dropna().tolist() for area in fill_levels.columns
        ],
        "First reading (%)": first_month_levels.iloc[0].round(1).to_numpy(),
        "Latest reading (%)": fill_levels.iloc[-1].round(1).to_numpy(),
    }
)
st.caption(f"First available month: {first_month.strftime('%B %Y')}")
st.dataframe(
    summary,
    column_config={
        "First month trend": st.column_config.LineChartColumn(
            "First month trend (%)"
        )
    },
    hide_index=True,
    width="stretch",
)

st.subheader("Imported CSV records")
st.caption(
    f"The original {len(source):,} records, sorted by date and shown with English headers."
)
st.dataframe(source, hide_index=True, height=360, width="stretch")
