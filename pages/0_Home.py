"""Introduce the reservoir dashboard and link to its main views."""

import streamlit as st

st.title("Norwegian reservoir filling")
st.caption("IND320 · Project work, part 2 · Reservoirs and cross-border transfers")
st.write(
    "Explore the available weekly reservoir area series from the NVE API. Use the sidebar to open the data table, interactive plot, or "
    "Transfers or information page. Transfers show ENTSO-E physical flows for 2024–2025, published from Spark/Cassandra to MongoDB."
)

st.subheader("Start here")
st.page_link(
    "pages/1_Data.py", label="View reservoir trends and source data", icon="📋"
)
st.page_link("pages/2_Plots.py", label="Choose areas and months to plot", icon="📈")

st.page_link(
    "pages/3_Transfers.py", label="Explore import and export energy", icon="⚡"
)
