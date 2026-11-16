"""Inspect sampled-area daily ERA5 averages with shared notebook analyses."""

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from ind320.analysis.weather import precipitation_analysis, temperature_analysis
from ind320.data.weather import daily_weather, fetch_weather, load_weather_coordinates
from ind320.ui.area_controls import WEATHER_AREAS, area_selection


@st.cache_data(ttl=86400, show_spinner="Downloading hourly ERA5 weather...")
def weather_for_year(area: str, year: int) -> pd.DataFrame:
    """Cache only by area/year; plot controls never change the download key."""
    return fetch_weather(area, f"{year}-01-01", f"{year}-12-31")


st.title("Weather quality")
st.write(
    "Each series averages weather over 1,000 reproducibly sampled locations in the selected area. "
    "East combines NVE regions RS and RØ, West uses RV, and North combines RM and RN. "
    "These administrative polygons approximate precipitation areas; they are not exact catchment or VASS boundaries."
)
coordinates = load_weather_coordinates()
left, right = st.columns(2)
with left:
    selected = area_selection("weather_area", [*WEATHER_AREAS, ("NO", 0)])
if selected not in WEATHER_AREAS:
    st.info("Weather sampling covers NO1–NO5 and the three grouped NVE regions. Select one of those areas to analyse weather; the national reservoir selection is retained across pages.")
    st.stop()
area = WEATHER_AREAS[selected]
# Complete historical calendar years avoid silently plotting a partial current year.
year = right.selectbox("Weather year", list(range(date.today().year - 1, 1939, -1)),
                       index=date.today().year - 1 - 2024, key="weather_year")
st.caption(f"{len(coordinates.loc[[area]]):,} sampled locations. Model: ERA5. Dates: UTC.")
try:
    hourly = weather_for_year(area, year)
    daily = daily_weather(hourly)
except (requests.RequestException, ValueError) as error:
    st.error(f"Weather could not be loaded: {error}")
    st.info("Completed grid cells are cached. If the API is rate-limited, wait before retrying; partial area averages are never shown.")
    st.stop()
st.caption(
    f"{hourly.attrs['sampled_points']:,} sampled points represented by {hourly.attrs['grid_cells']:,} ERA5 cells. "
    f"{len(hourly):,} hourly area averages reduced to {len(daily)} daily values. "
    "Precipitation is the average hourly amount in mm, not a daily rainfall total. "
    "Source: Open-Meteo Historical Weather API / Copernicus ERA5 reanalysis (CC BY 4.0)."
)

precipitation_tab, temperature_tab = st.tabs(["Precipitation: DCT and SPC", "Temperature: LOF"])
with precipitation_tab:
    left, right = st.columns(2)
    cutoff = left.slider("Low-frequency DCT coefficients removed", 0, 40, 8,
                         help="Zero keeps the original series; larger values remove more slow variation.")
    multiplier = right.slider("SPC sigma multiplier", 1.0, 6.0, 3.0, 0.25,
                               help="Median ± multiplier × normal-scaled MAD of filtered precipitation.")
    data, summary, fig = precipitation_analysis(daily.precipitation, cutoff, multiplier)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"{summary['outliers']} of {summary['days']} days flagged. "
        f"Robust sigma: {summary['sigma']:.4f} mm ({summary['scale_method']}). "
        "Flags indicate unusual variation; they do not establish measurement errors."
    )
    with st.expander("Flagged precipitation days"):
        st.dataframe(data.loc[data.outlier], width="stretch")

with temperature_tab:
    contamination_percent = st.slider("Expected outlier share (%)", 0.5, 10.0, 1.0, 0.5,
                                       help="LOF uses temperature alone with 20 neighbors. This sets a score threshold, not a probability of error.")
    data, summary, fig = temperature_analysis(daily.temperature_2m, contamination_percent / 100)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"{summary['outliers']} of {summary['days']} days flagged. Mean temperature: {summary['mean_c']:.2f} °C. "
        "Univariate LOF finds sparse temperature values; it does not model seasonality or use dates as a feature."
    )
    with st.expander("Flagged temperature days"):
        st.dataframe(data.loc[data.outlier], width="stretch")

st.link_button("Weather source and license", "https://open-meteo.com/en/docs/historical-weather-api")
