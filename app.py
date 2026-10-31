import streamlit as st

st.set_page_config(
    page_title="Reservoir filling | IND320", page_icon="💧", layout="wide"
)

# Streamlit shows these separate page files as the sidebar menu.
current_page = st.navigation(
    [
        st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/1_Data.py", title="Data table", icon="📋"),
        st.Page("pages/2_Plots.py", title="Plots", icon="📈"),
        st.Page("pages/3_Transfers.py", title="Transfers", icon="⚡"),
        st.Page("pages/5_Reservoir_analysis.py", title="Reservoir analysis", icon="🔎"),
        st.Page("pages/6_Weather.py", title="Weather quality", icon="🌦️"),
        st.Page("pages/4_About.py", title="About", icon="ℹ️"),
    ],
    position="sidebar",
)
current_page.run()
