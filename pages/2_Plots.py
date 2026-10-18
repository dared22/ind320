"""Plot selected reservoir areas over an inclusive range of months."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from reservoir_controls import reservoir_selection

st.title("Reservoir filling over time")
st.write("Select one area or compare all available areas on the same percentage scale.")

# The table and this plot use the same inclusive interval and area-type controls.
_, selected = reservoir_selection()
chosen_area = st.selectbox("Area series", ["All areas", *selected.columns])
start_month = selected.index.min().strftime("%Y-%m")
end_month = selected.index.max().strftime("%Y-%m")
areas_to_plot = selected.columns if chosen_area == "All areas" else [chosen_area]

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

st.caption(f"{len(selected)} weekly dates shown. Source: NVE reservoir API.")
