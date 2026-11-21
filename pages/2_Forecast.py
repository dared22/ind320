"""Run the notebook's River SNARIMAX model on the selected area and timeframe."""

import pandas as pd
import requests
import streamlit as st
from pymongo.errors import PyMongoError

from ind320.analysis.forecast import FEATURE_LAGS, MIN_INTERVAL_ERRORS, SEASON, forecast_reservoir
from ind320.analysis.forecast_defaults import DEFAULT_INPUTS, DEFAULT_ORDER
from ind320.data.reservoirs import fetch_reservoir_history
from ind320.data.transfers import FORECAST_CONNECTIONS
from ind320.data.weeks import WEATHER_FEATURES, WEEK, combine_net_flows, cutoff_time, reservoir_weeks
from ind320.ui.area_controls import AREAS, WEATHER_AREAS, area_selection
from ind320.ui.forecast_chart import ANIMATION_OPTIONS, forecast_chart
from ind320.ui.source_cache import exchange_weeks, flow_weeks, weather_weeks
from ind320.ui.time_controls import forecast_horizon, timeframe

FEATURE_NAMES = {"temperature_2m": "Weekly mean temperature", "precipitation": "Weekly precipitation total",
                 "EUR/NOK": "EUR/NOK", "USD/NOK": "USD/NOK"}

st.title("Forecast")
st.caption(f"Weekly reservoir filling · River SNARIMAX with a {SEASON}-week season")
area = area_selection("forecast_area", AREAS)
try:
    history = reservoir_weeks(fetch_reservoir_history())
except requests.RequestException:
    st.error("NVE could not be reached. Reload the page to try again.")
    st.stop()
dates_col, horizon_col = st.columns([2, 1])
with dates_col:
    start, end = timeframe("forecast", history)
with horizon_col:
    horizon = forecast_horizon()
cutoff = cutoff_time(end)
published = history.loc[
    history["Area Type"].eq(area[0]) & history["Area Code"].eq(area[1])
    & history.week.between(pd.Timestamp(start), pd.Timestamp(end))
    & history.available_at.le(cutoff)
]
if published.empty:
    st.info("Choose a period with published observations for this area.")
    st.stop()
first = pd.offsets.Week(weekday=6).rollforward(pd.Timestamp(start))
last = published.week.max()
training_weeks = (last - first).days // 7 + 1
st.caption(f"{training_weeks} training weeks · Last published week ended {last:%d %b %Y}.")
if training_weeks < 78:
    st.info("Select at least 78 weeks of history for the seasonal model.")
    st.stop()

zone = f"NO{area[1]}" if area[0] == "EL" else None
connections = [code for code, endpoints in FORECAST_CONNECTIONS.items() if zone in endpoints]
options = [feature for feature in FEATURE_NAMES
           if feature not in WEATHER_FEATURES or area in WEATHER_AREAS]
with st.form("forecast_settings"):
    p_col, d_col, q_col = st.columns(3)
    p = p_col.selectbox("Autoregressive order (p)", [1, 2, 3, 4], DEFAULT_ORDER[0] - 1)
    d = d_col.selectbox("Differencing (d)", [0, 1], DEFAULT_ORDER[1])
    q = q_col.selectbox("Moving-average order (q)", [0, 1], DEFAULT_ORDER[2])
    features = tuple(st.multiselect("Additional inputs", options, default=list(DEFAULT_INPUTS),
                                    format_func=FEATURE_NAMES.get))
    selected_connections = st.multiselect("ENTSO-E net flow connections", connections,
        help="Select one or more links to sum. Flow into your area is positive. "
             "Each link needs data for the training period and the preceding week.") if connections else []
    st.caption("Inputs use recent published values and hold the last available weekly value fixed "
               "for the rest of the forecast.")
    submitted = st.form_submit_button("Run forecast", type="primary")

# Discard results from an earlier model or a different shared selection.
selection = ("River SNARIMAX", area, start, end, horizon)
if submitted:
    st.session_state.pop("last_forecast", None)
    try:
        sources = {}
        omitted_weather = ()
        weather_features = tuple(feature for feature in features if feature in WEATHER_FEATURES)
        if {"EUR/NOK", "USD/NOK"}.intersection(features):
            fx = exchange_weeks(first, last + WEEK)
            for feature in ("EUR/NOK", "USD/NOK"):
                if feature in features:
                    if feature not in fx.index.get_level_values("pair"):
                        raise ValueError(f"No exchange-rate coverage for {feature}.")
                    sources[feature] = fx.xs(feature, level="pair").rename(columns={"nok_per_unit": feature})
        if weather_features:
            latest_weather = last
            while ((latest_weather + pd.Timedelta(days=1)).tz_localize("Europe/Oslo")
                   + pd.Timedelta(days=5) > cutoff):
                latest_weather -= WEEK
            try:
                weather = weather_weeks(WEATHER_AREAS[area], first - 2 * WEEK, latest_weather)
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 429:
                    raise
                # Keep the requested settings so a later run can try weather again.
                omitted_weather = weather_features
            else:
                sources.update({feature: weather for feature in weather_features})
        if selected_connections:
            flow_start = first - FEATURE_LAGS["net_import_mwh"] * WEEK
            flows = {connection: flow_weeks(connection, flow_start, last) for connection in selected_connections}
            sources["net_import_mwh"] = combine_net_flows(flows, zone)
        selected_inputs = tuple(feature for feature in features if feature not in omitted_weather)
        if selected_connections:
            selected_inputs += ("net_import_mwh",)
        with st.spinner("Fitting the forecast and estimating the interval…"):
            result = forecast_reservoir(history, sources, area=area, cutoff=cutoff,
                                        training_weeks=training_weeks, horizon=horizon,
                                        order=(p, d, q), selected_inputs=selected_inputs)
        result.attrs["omitted_weather"] = omitted_weather
        result.attrs["flow_connections"] = selected_connections
        st.session_state["last_forecast"] = (selection, result)
    except PyMongoError:
        st.error("MongoDB flow data could not be read. Check connectivity or rerun with other inputs.")
    except (requests.RequestException, ValueError) as error:
        st.error(f"The forecast could not run. {error}")

saved = st.session_state.get("last_forecast")
if saved and saved[0] == selection and "interval_samples" in saved[1]:
    result = saved[1]
    if omitted := result.attrs.get("omitted_weather"):
        names = ", ".join(FEATURE_NAMES[feature] for feature in omitted)
        st.warning(f"Open-Meteo reached its request limit. This forecast ran without {names}. "
                   "Run it again after the quota resets to include weather.")
    observed = published.set_index("week")["Fill Level"] * 100
    chart = forecast_chart(observed, result, autoplay=submitted)
    # Plotly's HTML renderer starts the animation after loading the forecast frames.
    chart_html = chart.to_html(
        full_html=False, auto_play=submitted, animation_opts=ANIMATION_OPTIONS,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )
    st.iframe("<style>body { margin: 0; }</style>" + chart_html, height=chart.layout.height)
    pending = result.index[result.index.date < end]
    if len(pending):
        st.caption(f"An NVE reading is pending for the week ending {pending[-1]:%d %b %Y}.")
    st.caption("The nominal 80% band uses past forecast errors at the same lead time "
               "under the selected inputs and persistence scenario.")
    if flow_connections := result.attrs.get("flow_connections"):
        st.caption(f"ENTSO-E net flow into {zone} across {', '.join(flow_connections)}.")
    if result.interval_samples.lt(MIN_INTERVAL_ERRORS).any():
        st.caption(f"Weeks with fewer than {MIN_INTERVAL_ERRORS} historical errors show only the point forecast.")
    with st.expander("Weekly predictions"):
        predictions = result[["lower", "median", "upper"]].rename(columns={"median": "point"}).round(2)
        predictions.attrs.clear()  # Internal timestamps belong in the fit metadata.
        st.dataframe(predictions)
