"""Register the four Streamlit pages with clear sidebar labels."""

import streamlit as st

st.set_page_config(page_title="Reservoir filling | IND320", page_icon="💧", layout="wide")

# Streamlit shows these four separate page files as the sidebar menu.
current_page = st.navigation(
    [
        st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/1_Data.py", title="Data table", icon="📋"),
        st.Page("pages/2_Plots.py", title="Plots", icon="📈"),
        st.Page("pages/3_About.py", title="About", icon="ℹ️"),
    ],
    position="sidebar",
)
current_page.run()
