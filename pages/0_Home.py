"""Introduce the reservoir dashboard and link to its main views."""

import streamlit as st

from data import ReservoirDataError, load_reservoir_data

st.title("Norwegian reservoir filling")
st.caption("IND320 · Project work, part 2 · Reservoirs and cross-border transfers")
st.write(
    "Explore the available weekly reservoir area series from the NVE API. Use the sidebar to open the data table, interactive plot, or "
    "Transfers or information page. Transfers show ENTSO-E physical flows for 2024–2025, published from Spark/Cassandra to MongoDB."
)

# The summary uses the same cached data as the table and plot pages.
try:
    source, fill_levels = load_reservoir_data()
except ReservoirDataError as exc:
    st.error(str(exc))
    st.stop()
first_date = fill_levels.index.min().strftime("%d %b %Y")
last_date = fill_levels.index.max().strftime("%d %b %Y")

rows_col, areas_col, dates_col = st.columns(3)
rows_col.metric("Source records", f"{len(source):,}")
areas_col.metric("Area series", len(fill_levels.columns))
dates_col.metric("Date range", f"{first_date}–{last_date}")

st.subheader("Start here")
st.page_link(
    "pages/1_Data.py", label="View first-month trends and source data", icon="📋"
)
st.page_link("pages/2_Plots.py", label="Choose areas and months to plot", icon="📈")

st.page_link(
    "pages/3_Transfers.py", label="Explore import and export energy", icon="⚡"
)
