"""Explain the data and leave a clear starting point for later course work."""

import streamlit as st


st.title("About this dashboard")
st.write(
    "This is the first part of the IND320 Data to Decision project. "
    "The app reads the course's local reservoirs.csv file and shows weekly "
    "reservoir filling as a percentage of capacity."
)

st.subheader("How to read the area labels")
st.write(
    "NO 0 is the national series. EL 1–5 and VASS 1–3 are the area codes "
    "provided by the source file. The plots preserve those codes rather than "
    "assigning unverified geographic names."
)

# This fourth page can be extended when later project parts add online data.
st.subheader("Next project step")
st.info("Later IND320 project work will replace the local CSV with an online database.")
