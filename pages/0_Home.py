"""Introduce the reservoir dashboard and link to its main views."""

import streamlit as st

st.title("Norwegian reservoir filling")
st.caption("IND320 · Project work, part 3 · Reservoirs, electricity transfers and weather quality")
st.write(
    "Explore the available weekly reservoir area series from the NVE API. Use the sidebar to open the data table, interactive plot, or "
    "Transfers or information page. Transfers show ENTSO-E physical flows for 2024–2025, published from Spark/Cassandra to MongoDB. "
    "Weather quality uses 1,000 sampled locations per area, DCT/SPC precipitation checks and temperature LOF. "
    "Reservoir analysis provides STL decomposition and an interactive spectrogram. Area selections carry across the weather and reservoir pages."
)

st.subheader("Start here")
st.page_link(
    "pages/1_Data.py", label="View reservoir trends and source data", icon="📋"
)
st.page_link("pages/2_Plots.py", label="Choose areas and months to plot", icon="📈")

st.page_link(
    "pages/3_Transfers.py", label="Explore import and export energy", icon="⚡"
)
st.page_link("pages/5_Reservoir_analysis.py", label="Explore STL and spectrogram settings", icon="🔎")
st.page_link("pages/6_Weather.py", label="Inspect sampled-area weather quality", icon="🌦️")
