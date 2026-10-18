import pandas as pd
import streamlit as st

from ind320.data.reservoirs import INTERVAL_NOTICE, load_reservoir_data


def _save_selection(widget_key: str, saved_key: str) -> None:
    """Retain a control value when Streamlit cleans up widgets on another page."""
    st.session_state[saved_key] = st.session_state[widget_key]


def reservoir_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch cached history before controls, then apply both controls locally."""
    _, full = load_reservoir_data()
    months = full.index.to_period("M").unique().astype(str).tolist()
    # Permanent values survive navigation; temporary keys belong to widgets.
    kinds = ["EL", "NO", "VASS"]
    kind = st.radio(
        "Area type",
        kinds,
        index=kinds.index(st.session_state.get("saved_reservoir_type", "EL")),
        horizontal=True,
        key="reservoir_area_type",
        on_change=_save_selection,
        args=("reservoir_area_type", "saved_reservoir_type"),
    )
    start, end = st.select_slider(
        "Months",
        options=months,
        value=st.session_state.get(
            "saved_reservoir_months", (months[0], months[0])
        ),
        key="reservoir_months",
        on_change=_save_selection,
        args=("reservoir_months", "saved_reservoir_months"),
    )
    selected = load_reservoir_data(start, end, kind)
    if selected[1].empty:
        st.info("No observations are available for these months and area type.")
        st.stop()
    st.caption(INTERVAL_NOTICE)
    return selected
