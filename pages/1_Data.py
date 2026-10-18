"""Keep source rows and sparklines synchronized with the selected months/type."""

import pandas as pd
import streamlit as st

from ind320.data.reservoirs import unusual_fill_levels
from ind320.ui.reservoir_controls import reservoir_selection

st.title("Data table")
source, levels = reservoir_selection()
# Each sparkline uses every available observation in the chosen interval.
summary = pd.DataFrame(
    {
        "Area": levels.columns,
        "Selected interval trend": [levels[c].dropna().tolist() for c in levels],
        "First reading (%)": [levels[c].dropna().iloc[0] for c in levels],
        "Latest reading (%)": [levels[c].dropna().iloc[-1] for c in levels],
    }
)
st.dataframe(
    summary,
    column_config={
        "Selected interval trend": st.column_config.LineChartColumn("Fill level (%)")
    },
    hide_index=True,
    width="stretch",
)
st.subheader("NVE API records")
st.dataframe(source, hide_index=True, height=360, width="stretch")
unusual = unusual_fill_levels(source)
if not unusual.empty:
    st.warning(f"{len(unusual)} readings outside 0–100%; original values preserved.")
    with st.expander("Inspect unusual readings"):
        st.dataframe(unusual, hide_index=True)
