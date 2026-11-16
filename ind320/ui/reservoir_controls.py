import pandas as pd
import streamlit as st

from ind320.data.reservoirs import INTERVAL_NOTICE, load_reservoir_data
from ind320.ui.area_controls import area_selection


def _save_selection(widget_key: str, saved_key: str) -> None:
    """Retain a control value when Streamlit cleans up widgets on another page."""
    st.session_state[saved_key] = st.session_state[widget_key]


def reservoir_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch cached history before controls, then apply both controls locally."""
    source, full = load_reservoir_data()
    months = full.index.to_period("M").unique().astype(str).tolist()
    # Permanent values survive navigation; temporary keys belong to widgets.
    areas = sorted(source[["Area Type", "Area Code"]].drop_duplicates().itertuples(index=False, name=None))
    kind, code = area_selection("reservoir_area", areas)
    compare = st.checkbox("Compare all areas of this type", key="compare_reservoir_areas")
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
    if not compare:
        rows, levels = selected
        selected = (rows.loc[rows["Area Code"] == code],
                    levels.loc[:, [c for c in levels if c == f"{kind} {code}"]])
    if selected[1].empty:
        st.info("No observations are available for these months and area type.")
        st.stop()
    st.caption(INTERVAL_NOTICE)
    return selected
