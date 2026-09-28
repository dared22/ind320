"""Introduce the reservoir dashboard and link to its main views."""

import streamlit as st

from data import load_reservoir_data


st.title("Norwegian reservoir filling")
st.caption("IND320 · Project work, part 1")
st.write(
    "Explore weekly reservoir filling levels for the nine areas in the course "
    "dataset. Use the sidebar to open the data table, interactive plot, or "
    "information page."
)

# The summary uses the same cached data as the table and plot pages.
source, fill_levels = load_reservoir_data()
first_date = fill_levels.index.min().strftime("%d %b %Y")
last_date = fill_levels.index.max().strftime("%d %b %Y")

rows_col, areas_col, dates_col = st.columns(3)
rows_col.metric("Source records", f"{len(source):,}")
areas_col.metric("Area series", len(fill_levels.columns))
dates_col.metric("Date range", f"{first_date}–{last_date}")

st.subheader("Start here")
st.page_link("pages/1_Data.py", label="View first-month trends and source data", icon="📋")
st.page_link("pages/2_Plots.py", label="Choose areas and months to plot", icon="📈")
