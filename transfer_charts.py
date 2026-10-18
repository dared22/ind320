import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def energy_pie(summary: pd.DataFrame) -> Figure:
    """Show observed energy; the caller displays coverage alongside the figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    values = summary.set_index("direction").energy_mwh.reindex(["import", "export"])
    if values.isna().any() or values.sum() <= 0:
        ax.text(0.5, 0.5, "No two-direction energy total available", ha="center")
        ax.axis("off")
    else:
        ax.pie(values, labels=["Import", "Export"], autopct="%1.1f%%")
    complete = len(summary) == 2 and summary.complete.all()
    ax.set_title(
        f"{'Annual' if complete else 'Incomplete: observed'} transferred energy (MWh)"
    )
    return fig


def transfer_lines(data: pd.DataFrame) -> Figure:
    """Split lines at omitted months and gaps; never bridge unavailable intervals."""
    fig, ax = plt.subplots(figsize=(11, 5))
    connections = sorted(data.connection.unique()) if not data.empty else []
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(connections)}
    if not data.empty:
        data = data.sort_values(["connection", "direction", "start"]).copy()
        data["month"] = (
            data.start.dt.tz_convert("Europe/Oslo").dt.tz_localize(None).dt.to_period("M")
        )
        for (connection, direction), group in data.groupby(["connection", "direction"]):
            color = colors[connection]
            if len(connections) == 1:
                color = "tab:blue" if direction == "import" else "tab:orange"
            # A separate segment also stops lines crossing missing source intervals.
            split = (group.start != group.end.shift()) | (
                group.month != group.month.shift()
            )
            for i, (_, segment) in enumerate(group.groupby(split.cumsum())):
                ax.plot(
                    segment.start.dt.tz_convert("Europe/Oslo"),
                    segment.power_mw,
                    label=f"{connection} {direction}" if i == 0 else None,
                    color=color,
                    linestyle="-" if direction == "import" else "--",
                    linewidth=0.8,
                    marker="." if len(segment) == 1 else None,
                    markersize=3,
                )
        if len(connections) > 1:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        else:
            ax.legend()
    ax.set(xlabel="Time (Europe/Oslo)", ylabel="Physical flow (MW)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
