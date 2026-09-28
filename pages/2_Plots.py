"""Plot selected reservoir areas over an inclusive range of months."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import load_reservoir_data


st.title("Reservoir filling over time")
st.write("Select one area or compare all nine areas on the same percentage scale.")

_, fill_levels = load_reservoir_data()
months = fill_levels.index.to_period("M").unique().astype(str).tolist()

# The first month is the default for both ends of the selection range.
chosen_area = st.selectbox("Area series", ["All areas", *fill_levels.columns])
start_month, end_month = st.select_slider(
    "Months",
    options=months,
    value=(months[0], months[0]),
)

periods = fill_levels.index.to_period("M")
selected = fill_levels.loc[
    (periods >= pd.Period(start_month)) & (periods <= pd.Period(end_month))
]
areas_to_plot = fill_levels.columns if chosen_area == "All areas" else [chosen_area]

# All nine series share a unit, so an unnormalised comparison is meaningful.
fig, ax = plt.subplots(figsize=(10, 5))
for area in areas_to_plot:
    ax.plot(selected.index, selected[area], label=area, linewidth=2)

ax.set_title(f"Weekly filling levels · {start_month} to {end_month}")
ax.set_xlabel("Observation date")
ax.set_ylabel("Reservoir filling (%)")
ax.grid(alpha=0.25)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.legend(title="Area", ncol=3 if chosen_area == "All areas" else 1)
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

st.caption(f"{len(selected)} weekly dates shown. Source: local reservoirs.csv.")
