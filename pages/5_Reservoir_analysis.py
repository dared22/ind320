"""Explore annual structure and changing frequencies across full NVE history."""

import matplotlib.pyplot as plt
import requests
import streamlit as st

from ind320.analysis.reservoirs import filling_spectrogram, stl_analysis, weekly_filling
from ind320.data.reservoirs import fetch_reservoir_history

st.title("Reservoir analysis")
st.write("Explore seasonal patterns and changing frequencies using the full weekly history.")

# This page owns its area control; the date controls on other pages do not apply.
try:
    source = fetch_reservoir_history()
except requests.RequestException:
    st.error("NVE could not be reached. Reload this page to try again.")
    st.stop()
areas = list(source[["Area Type", "Area Code"]].drop_duplicates().itertuples(index=False, name=None))
areas.sort(key=lambda key: (key[0] != "EL", key[0], key[1]))
area_type, area_code = st.selectbox(
    "Reservoir area", areas, index=areas.index(("EL", 3)),
    format_func=lambda key: f"NO{key[1]} (electricity area)" if key[0] == "EL" else f"{key[0]} {key[1]}",
    key="analysis_reservoir_area",
)
try:
    weekly = weekly_filling(source, area_type, area_code)
except ValueError as error:
    st.error(str(error))
    st.stop()
if len(weekly) < 104:
    st.info("This area needs at least 104 weeks of history for these analyses.")
    st.stop()
st.caption(
    f"{len(weekly):,} weeks, {weekly.index.min():%Y-%m-%d} to {weekly.index.max():%Y-%m-%d}. "
    f"{weekly.attrs['interpolated_weeks']} missing weeks interpolated in time. Source: NVE."
)

stl_tab, spectrum_tab = st.tabs(["STL", "Spectrogram"])
with stl_tab:
    st.write("Separate filling into trend, a 52-week seasonal component, and a residual.")
    left, middle, right = st.columns(3)
    seasonal = left.select_slider("Seasonal smoother", options=list(range(7, 54, 2)), value=13,
                                  help="Larger values make the seasonal pattern change more slowly across years.")
    trend = middle.select_slider("Trend smoother (weeks)", options=list(range(53, 210, 2)), value=105,
                                 help="Larger values produce a smoother long-term trend.")
    robust = right.checkbox("Robust fitting", value=True, help="Reduce the influence of extreme residuals.")
    components, summary, fig = stl_analysis(weekly, seasonal, trend, robust)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"Residual standard deviation: {summary['residual_sd_pp']:.2f} percentage points. "
        f"Seasonal range: {summary['seasonal_range_pp']:.2f} percentage points."
    )

with spectrum_tab:
    st.write("Show how the strength of short and long cycles changes over the history.")
    left, right = st.columns(2)
    window = left.slider("Window length (weeks)", 26, min(260, len(weekly)), 104,
                         help="Longer windows separate nearby frequencies but blur changes over time.")
    overlap = right.slider("Overlap (weeks)", 0, window - 1, min(52, window - 1),
                           help="More overlap places window centers closer together.")
    spectrum, summary, fig = filling_spectrogram(weekly, window, overlap)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"{summary['windows']} windows. Frequency resolution: {summary['frequency_step']:.4f} cycles/week. "
        f"Strongest nonzero frequency corresponds to {summary['peak_period_weeks']:.1f} weeks. "
        "One sample/week; the dashed line marks 1/52 cycles/week. PSD uses percentage points squared per cycles/week."
    )
