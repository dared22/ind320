"""Show the map snapshot and published NVE history for the shared area."""

import json

import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st

from ind320.config import ROOT
from ind320.data.reservoirs import (
    INTERVAL_NOTICE, fetch_reservoir_history, load_area_metadata, unusual_fill_levels,
)
from ind320.data.weeks import WEEK, cutoff_time, reservoir_weeks
from ind320.ui.area_controls import AREAS, SAVED_AREA, WEATHER_AREAS, area_label, area_selection
from ind320.ui.source_cache import weather_weeks
from ind320.ui.time_controls import timeframe

LAYER_NAMES = {"NO": "National", "EL": "Price areas", "VASS": "Water areas"}
FEATURES = {"Fill level": ("Fill Level", "%"),
            "Mean temperature": ("temperature_2m", "°C"),
            "Total precipitation": ("precipitation", "mm")}


def _select_polygon() -> None:
    """Read the chart event before widgets are recreated on the next rerun."""
    objects = st.session_state["reservoir_map"].selection.objects
    for picked in objects.values():
        feature = picked[0]["properties"]
        st.session_state[SAVED_AREA] = (feature["area_type"], feature["area_code"])


@st.cache_data
def _map_features() -> dict:
    return json.loads((ROOT / "data" / "reservoir_areas.geojson").read_text())


st.title("Map & Overview")
st.caption("Norwegian reservoir filling · NVE Magasinstatistikk")
area = area_selection("overview_area", AREAS)
# An area changed on another page should be visible on return to the map.
if st.session_state.get("overview_previous_area") != area:
    st.session_state["map_layer"] = area[0]
st.session_state["overview_previous_area"] = area

try:
    history = reservoir_weeks(fetch_reservoir_history())
except requests.RequestException:
    st.error("NVE could not be reached. Reload the page to try again.")
    st.stop()

start, end, _ = timeframe("overview", history)
cutoff = cutoff_time(end)
observed = history.loc[
    history.week.between(pd.Timestamp(start), pd.Timestamp(end)) & history.available_at.le(cutoff)
]
latest = observed.week.max()
left, right = st.columns(2)
kind = left.selectbox("Map view", list(LAYER_NAMES), format_func=LAYER_NAMES.get, key="map_layer")
metrics = list(FEATURES) if kind != "NO" else ["Fill level"]
metric = right.selectbox("Color by", metrics)
column, unit = FEATURES[metric]

geodata = _map_features()
geodata["features"] = [feature for feature in geodata["features"]
                       if feature["properties"]["area_type"] == kind]

weather_week = latest
weather_values = {}
if metric != "Fill level" and pd.notna(weather_week):
    # ERA5 releases after NVE. Step back to the most recent available full week.
    while (weather_week + pd.Timedelta(days=1)).tz_localize("Europe/Oslo") + pd.Timedelta(days=5) > cutoff:
        weather_week -= WEEK
    snapshot = (kind, weather_week)
    saved_snapshot, loaded = st.session_state.get("weather_snapshot", (None, {}))
    if saved_snapshot == snapshot and weather_week >= pd.Timestamp(start):
        weather_values = loaded
    if weather_week < pd.Timestamp(start):
        st.info("No complete ERA5 week in this timeframe was published by the cutoff.")
    elif st.button("Load weather for this week"):
        # Save complete areas and reuse cached grid cells if a later area hits a rate limit.
        for feature in geodata["features"]:
            key = (kind, feature["properties"]["area_code"])
            if key in weather_values:
                continue
            try:
                weather = weather_weeks(WEATHER_AREAS[key], weather_week, weather_week)
                if not weather.empty:
                    weather_values[key] = weather.loc[weather_week].to_dict()
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 429:
                    st.warning("Open-Meteo rate limit reached. Use the load button again to resume.")
                    break
                st.warning(f"{area_label(key)} weather is unavailable for the selected week.")
            except (requests.RequestException, ValueError):
                st.warning(f"{area_label(key)} weather is unavailable for the selected week.")
        st.session_state["weather_snapshot"] = (snapshot, weather_values)
    if not weather_values:
        st.caption("Load the snapshot to color the map from sampled ERA5 weather.")

fill = observed.loc[observed.week.eq(latest)].set_index(["Area Type", "Area Code"])["Fill Level"]
values = {}
for feature in geodata["features"]:
    properties = feature["properties"]
    key = (kind, properties["area_code"])
    if metric == "Fill level":
        values[key] = fill.get(key, float("nan")) * 100
    elif key in weather_values:
        values[key] = weather_values[key][column]

# Use fixed scales so colors remain comparable between dates.
lower, upper = {"Fill level": (0, 100), "Mean temperature": (-20, 25),
                "Total precipitation": (0, 100)}[metric]
palette = px.colors.sample_colorscale("Blues", [0.25 + i / 12 for i in range(9)], colortype="tuple")
snapshot_week = latest if metric == "Fill level" else weather_week
for feature in geodata["features"]:
    properties = feature["properties"]
    key = (kind, properties["area_code"])
    value = values.get(key, float("nan"))
    if pd.notna(value):
        index = round(8 * min(1, max(0, (value - lower) / (upper - lower))))
        color = [round(channel * 255) for channel in palette[index]] + [175]
    else:
        color = [118, 138, 164, 55]
    properties.update(
        fill=color, outline=[255, 203, 126, 255] if key == area else [173, 211, 241, 200],
        thickness=3 if key == area else 1, label=area_label(key),
        value=f"{metric} {value:.1f} {unit}" if pd.notna(value) else "No value for this week",
        period=str(snapshot_week.date()) if pd.notna(snapshot_week) else "No week in range",
    )

layer = pdk.Layer("GeoJsonLayer", geodata, id="regions", pickable=True,
                  get_fill_color="properties.fill", get_line_color="properties.outline",
                  get_line_width="properties.thickness", line_width_units="pixels")
deck = pdk.Deck(layer, map_style=None,
                initial_view_state=pdk.ViewState(latitude=64.8, longitude=13.6, zoom=3.9),
                tooltip={"html": "{properties.label}<br>{properties.value}<br>Week ending {properties.period}"})
st.pydeck_chart(deck, key="reservoir_map", on_select=_select_polygon, height=550)
st.caption(f"{metric} · scale {lower:g} to {upper:g} {unit} · grey indicates missing coverage. "
           f"Selected area · {area_label(area)}.")
if kind == "VASS":
    st.caption("Approximate water-area boundaries from grouped NVE administrative regions.")

st.subheader("Reservoir filling over time")
compare = st.checkbox("Compare areas of the selected type", disabled=area[0] == "NO")
rows = observed.loc[observed["Area Type"].eq(area[0])]
if not compare:
    rows = rows.loc[rows["Area Code"].eq(area[1])]
if rows.empty:
    st.info("There are no published observations for this selection and timeframe.")
else:
    trend = rows.assign(
        Area=[area_label(key) for key in rows[["Area Type", "Area Code"]].itertuples(index=False, name=None)],
        Filling=rows["Fill Level"] * 100,
    ).sort_values("week")
    fig = px.line(trend, x="week", y="Filling", color="Area",
                  labels={"week": "Week ending", "Filling": "Reservoir filling (%)"})
    st.plotly_chart(fig)
    if not unusual_fill_levels(rows).empty:
        st.warning("Some NVE observations fall outside 0–100%. The chart shows the source values.")
    with st.expander("NVE API records"):
        st.dataframe(rows.drop(columns=["available_at"]), hide_index=True)

with st.expander("Sources and coverage"):
    st.caption(INTERVAL_NOTICE)
    if st.button("Show official NVE area descriptions"):
        try:
            st.dataframe(load_area_metadata(), hide_index=True)
        except requests.RequestException:
            st.warning("NVE area descriptions could not be loaded.")
    st.markdown("[NVE Magasinstatistikk](https://www.nve.no/energi/analyser-og-statistikk/magasinstatistikk/) · "
                "[Open-Meteo / Copernicus ERA5](https://open-meteo.com/en/docs/historical-weather-api) · "
                "[Norges Bank](https://www.norges-bank.no/en/topics/statistics/exchange_rates/) · "
                "[Project repository](https://github.com/dared22/ind320)")
