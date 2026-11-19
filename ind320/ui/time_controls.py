"""Keep the Map and Forecast timeframe in session state across page changes."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from ind320.analysis.forecast_defaults import DEFAULT_HORIZON, DEFAULT_TRAINING_WEEKS
from ind320.data.weeks import OSLO, cutoff_time


def _save_time(widget: str, saved: str) -> None:
    st.session_state[saved] = st.session_state[widget]


def timeframe(page: str, history: pd.DataFrame) -> tuple[date, date, int]:
    """Use separate widget keys and persistent values shared by both pages."""
    today = pd.Timestamp.now(tz=OSLO).date()
    first = history.week.min().date()
    published = history.loc[history.available_at.le(cutoff_time(today)), "week"]
    latest = published.max().date() if not published.empty else first
    default_start = max(first, latest - timedelta(weeks=DEFAULT_TRAINING_WEEKS - 1))

    start_col, end_col, horizon_col = st.columns(3)
    with start_col:
        key = f"{page}_start"
        st.session_state[key] = st.session_state.get("history_start", default_start)
        start = st.date_input("Historical start", min_value=first, max_value=today,
                              key=key, on_change=_save_time, args=(key, "history_start"))
    with end_col:
        key = f"{page}_end"
        st.session_state[key] = st.session_state.get("history_end", today)
        end = st.date_input("Historical end", min_value=first, max_value=today,
                            key=key, on_change=_save_time, args=(key, "history_end"))
    with horizon_col:
        key = f"{page}_horizon"
        st.session_state[key] = st.session_state.get("forecast_horizon", DEFAULT_HORIZON)
        horizon = st.slider("Forecast horizon (weeks)", 4, 26, key=key,
                            on_change=_save_time, args=(key, "forecast_horizon"))
    if start > end:
        st.info("Choose a historical start on or before the end date.")
        st.stop()
    return start, end, horizon
