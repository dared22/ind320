"""Explain the live API source and its area descriptions."""

import streamlit as st

from ind320.data.reservoirs import INTERVAL_NOTICE, load_area_metadata

st.title("About this dashboard")
st.write(
    "The IND320 dashboard retrieves weekly reservoir history from NVE. "
    "The notebook and app share data preparation and numeric percentage series. "
    "ENTSO-E physical flows pass through local Spark → Cassandra → Spark, "
    "then publish curated intervals and annual energy summaries to MongoDB. The Transfers page queries MongoDB directly; app reruns never ingest data."
)
st.caption(INTERVAL_NOTICE)
st.subheader("Area labels from NVE")
st.write(
    "NO identifies the national series, EL the electricity market areas, and "
    "VASS the watercourse areas. Descriptions below come directly from NVE; "
    "an area appears in a plot only when observations are available."
)
# Cache public descriptions across reruns, while showing network failures clearly.
st.dataframe(load_area_metadata(), hide_index=True, width="stretch")
st.link_button(
    "NVE API documentation", "https://biapi.nve.no/magasinstatistikk/swagger/index.html"
)
