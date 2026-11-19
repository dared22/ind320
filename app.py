"""The three IND320 Part 4 pages share an area and a forecast timeframe."""

import streamlit as st

st.set_page_config(page_title="Norwegian reservoir filling", page_icon="💧", layout="wide")
st.navigation([
    st.Page("pages/0_Home.py", title="Map & Overview", default=True),
    st.Page("pages/1_Explore.py", title="Explore & Quality", url_path="explore"),
    st.Page("pages/2_Forecast.py", title="Forecast", url_path="forecast"),
]).run()
