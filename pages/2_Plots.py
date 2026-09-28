"""Plot selected reservoir areas over an inclusive range of months."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Reshape the numeric series for Seaborn's area grouping. Every series shares
# a percentage unit, so the original values can be compared on one axis.
plot_data = selected.loc[:, areas_to_plot].rename_axis("Observation date").reset_index()
plot_data = plot_data.melt(
    id_vars="Observation date",
    var_name="Area",
    value_name="Reservoir filling (%)",
)

# Seaborn draws the lines; Matplotlib still provides the figure and date ticks.
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(
    data=plot_data,
    x="Observation date",
    y="Reservoir filling (%)",
    hue="Area",
    hue_order=list(areas_to_plot),
    estimator=None,
    errorbar=None,
    linewidth=2,
    ax=ax,
)

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
