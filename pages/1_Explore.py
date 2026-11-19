"""Explore reservoir quality, sampled weather, correlation and transfers."""

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from ind320.analysis.reservoirs import filling_spectrogram, stl_analysis, weekly_filling
from ind320.analysis.weather import lof_statistics, satv_statistics
from ind320.data.reservoirs import fetch_reservoir_history
from ind320.data.weather import daily_weather
from ind320.data.weeks import reservoir_weeks, rolling_weather_correlation, weekly_weather
from ind320.ui.area_controls import AREAS, WEATHER_AREAS, area_selection
from ind320.ui.source_cache import weather_for_year
from ind320.ui.transfers import render_transfers

PROPERTIES = {"temperature_2m": "Temperature (°C)", "precipitation": "Precipitation (mm)"}


def _reservoir_quality(area: tuple[str, int]) -> None:
    """Render STL and the spectrogram using the same calculations as Part 3."""
    weekly = weekly_filling(fetch_reservoir_history(), *area)
    if len(weekly) < 104:
        st.info("This area needs 104 weekly readings for STL and the spectrogram.")
        return
    st.caption(f"{len(weekly):,} weeks. {weekly.attrs['interpolated_weeks']} gaps interpolated for these analyses.")
    stl_tab, spectrum_tab = st.tabs(["STL", "Spectrogram"])
    with stl_tab:
        left, middle, right = st.columns(3)
        seasonal = left.select_slider("Seasonal smoother", list(range(7, 54, 2)), value=13)
        trend = middle.select_slider("Trend smoother (weeks)", list(range(53, 210, 2)), value=105)
        robust = right.checkbox("Robust fitting", value=True)
        components, summary, _ = stl_analysis(weekly, seasonal, trend, robust, plot=False)
        long = components.rename_axis("Week").reset_index().melt(id_vars="Week", var_name="Component")
        chart = px.line(long, x="Week", y="value", facet_row="Component", height=600)
        chart.update_yaxes(matches=None, title=None)
        chart.for_each_annotation(lambda label: label.update(text=label.text.split("=")[-1].title()))
        st.plotly_chart(chart)
        st.caption(f"Residual standard deviation {summary['residual_sd_pp']:.2f} percentage points.")
    with spectrum_tab:
        left, right = st.columns(2)
        window = left.slider("Window length (weeks)", 26, min(260, len(weekly)), 104)
        overlap = right.slider("Overlap (weeks)", 0, window - 1, min(52, window - 1))
        spectrum, summary, _ = filling_spectrogram(weekly, window, overlap, plot=False)
        power = 10 * np.log10(np.maximum(spectrum.to_numpy(), 1e-12))
        chart = go.Figure(go.Heatmap(z=power, x=spectrum.columns, y=spectrum.index.to_numpy(),
                                       colorbar={"title": "PSD (dB)"}))
        chart.add_hline(y=1 / 52, line_dash="dash")
        chart.update_layout(xaxis_title="Window center", yaxis_title="Cycles per week")
        st.plotly_chart(chart)
        st.caption(f"{summary['windows']} windows. Frequency resolution: "
                   f"{summary['frequency_step']:.4f} cycles/week. Peak period "
                   f"{summary['peak_period_weeks']:.1f} weeks.")


def _quality_chart(data: pd.DataFrame, label: str) -> go.Figure:
    """Show original observations, optional control band and flagged days."""
    chart = go.Figure()
    if "lower" in data:
        chart.add_scatter(x=data.index, y=data.lower, line={"width": 0}, showlegend=False)
        chart.add_scatter(x=data.index, y=data.upper, line={"width": 0},
                          fill="tonexty", fillcolor="rgba(121,189,255,0.18)", name="SPC limits")
    chart.add_scatter(x=data.index, y=data.observed, name="Observed")
    flags = data.loc[data.outlier]
    chart.add_scatter(x=flags.index, y=flags.observed, mode="markers",
                      marker={"symbol": "x", "color": "#ffbd79"}, name="Flagged")
    chart.update_layout(xaxis_title="Date (UTC)", yaxis_title=label)
    return chart


def _weather_quality(hourly: pd.DataFrame) -> None:
    """Run SATV and LOF on the daily version of the same downloaded weather."""
    daily = daily_weather(hourly)
    st.caption("Daily averages over 1,000 sampled locations. Precipitation is the mean hourly amount.")
    satv, lof = st.tabs(["SATV and SPC", "LOF"])
    with satv:
        feature = st.selectbox("SATV weather property", list(PROPERTIES), format_func=PROPERTIES.get)
        c1, c2 = st.columns(2)
        cutoff = c1.slider("Low-frequency DCT coefficients removed", 0, 40, 8)
        multiplier = c2.slider("SPC sigma multiplier", 1.0, 6.0, 3.0, 0.25)
        data, summary = satv_statistics(daily[feature], cutoff, multiplier)
        st.plotly_chart(_quality_chart(data, PROPERTIES[feature]))
        st.caption(f"{summary['outliers']} days flagged using {summary['scale_method']}.")
        with st.expander("Flagged SATV days"):
            st.dataframe(data.loc[data.outlier])
    with lof:
        feature = st.selectbox("LOF weather property", list(PROPERTIES), format_func=PROPERTIES.get)
        share = st.slider("Expected outlier share (%)", 0.5, 10.0, 1.0, 0.5)
        data, summary = lof_statistics(daily[feature], share / 100)
        st.plotly_chart(_quality_chart(data, PROPERTIES[feature]))
        st.caption(f"{summary['outliers']} days flagged by univariate LOF.")
        with st.expander("Flagged LOF days"):
            st.dataframe(data.loc[data.outlier])


def _correlation(area: tuple[str, int], hourly: pd.DataFrame) -> None:
    """Show Pearson correlation on complete weeks, with positive weather lead."""
    weather = weekly_weather(hourly)
    rows = reservoir_weeks(fetch_reservoir_history())
    selected = rows.loc[rows["Area Type"].eq(area[0]) & rows["Area Code"].eq(area[1])]
    filling = selected.set_index("week")["Fill Level"] * 100
    left, middle, right = st.columns(3)
    feature = left.selectbox("Weather property", list(PROPERTIES), format_func=PROPERTIES.get)
    lag = middle.slider("Weather lead (weeks)", 0, 8, 1)
    window = right.selectbox("Window (weeks)", [8, 13, 26], index=1)
    correlation = rolling_weather_correlation(filling, weather[feature], lag, window).dropna()
    if correlation.empty:
        st.info("No complete rolling windows for the selected dates, lead and window length.")
        return
    chart = px.line(x=correlation.index, y=correlation, labels={"x": "Week ending", "y": "Pearson r"})
    chart.add_hline(y=0)
    chart.update_yaxes(range=[-1, 1])
    st.plotly_chart(chart)
    aggregation = "weekly total (mm)" if feature == "precipitation" else "weekly mean (°C)"
    st.caption(f"Weather {aggregation} leads weekly filling change (percentage points) by {lag} weeks.")


st.title("Explore & Quality")
area = area_selection("explore_area", AREAS)
panel = st.segmented_control(
    "Analysis", ["Reservoir quality", "Weather quality", "Weather and filling", "Transfers"],
    default="Reservoir quality", selection_mode="single", key="explore_panel",
)
try:
    if panel == "Transfers":
        render_transfers()
    elif panel == "Reservoir quality":
        _reservoir_quality(area)
    elif panel in ("Weather quality", "Weather and filling"):
        if area not in WEATHER_AREAS:
            st.info("Sampled weather covers the price and water areas. Your national selection is saved.")
        else:
            years = list(range(date.today().year - 1, 1939, -1))
            year = st.selectbox("Weather year", years, index=years.index(2024), key="weather_year")
            hourly = weather_for_year(WEATHER_AREAS[area], year)
            if panel == "Weather quality":
                _weather_quality(hourly)
            else:
                _correlation(area, hourly)
            st.caption("Source · Open-Meteo Historical Weather API / Copernicus ERA5")
except (requests.RequestException, ValueError) as error:
    st.error(f"The selected analysis could not be loaded. {error}")
