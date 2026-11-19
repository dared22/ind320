"""Keep one area selection across weather and Magasinstatistikk pages."""

import streamlit as st

# The grouped NVE regions approximate the three VASS areas in the assignment.
WEATHER_AREAS = {
    **{("EL", i): f"NO{i}" for i in range(1, 6)},
    ("VASS", 1): "East", ("VASS", 2): "West", ("VASS", 3): "North",
}
SAVED_AREA = "selected_area"


def area_label(area: tuple[str, int]) -> str:
    """Show the weather/reservoir correspondence in every selector."""
    kind, code = area
    if kind == "EL":
        return f"NO{code} (electricity area)"
    if kind == "VASS":
        return f"VASS {code} ({WEATHER_AREAS.get(area, 'watercourse area')})"
    return "Norway (national reservoir series)"


def _save_area(widget_key: str) -> None:
    """Use a permanent key because page-specific widgets are cleaned up."""
    st.session_state[SAVED_AREA] = st.session_state[widget_key]


def area_selection(widget_key: str, areas: list[tuple[str, int]]) -> tuple[str, int]:
    """Restore the common selection even when returning to an existing page."""
    selected = st.session_state.get(SAVED_AREA, ("EL", 3))
    if selected not in areas:
        # Never silently display a different area when data is unavailable.
        areas = [*areas, selected]
    st.session_state.setdefault(SAVED_AREA, selected)
    st.session_state[widget_key] = selected
    return st.selectbox(
        "Area", areas, format_func=area_label, key=widget_key,
        on_change=_save_area, args=(widget_key,),
    )

# GeoJSON and the common dropdown use the same area identifiers.
AREAS = [*WEATHER_AREAS, ("NO", 0)]
