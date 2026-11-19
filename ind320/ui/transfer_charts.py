import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
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


def interactive_energy_pie(summary: pd.DataFrame) -> go.Figure:
    """Show energy from the published months and label partial coverage."""
    values = summary.set_index("direction").energy_mwh.reindex(["import", "export"])
    complete = len(summary) == 2 and summary.complete.all()
    title = "Annual transferred energy (MWh)" if complete else "Observed transferred energy (MWh)"
    fig = go.Figure(layout={"title": title})
    if values.notna().all() and values.sum() > 0 and values.ge(0).all():
        fig.add_trace(go.Pie(labels=["Import", "Export"], values=values))
    else:
        fig.add_annotation(text="No two-direction energy total available", showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


def interactive_transfer_lines(data: pd.DataFrame) -> go.Figure:
    """Show each direction and break lines when consecutive intervals are missing."""
    fig = go.Figure()
    for direction, group in data.sort_values("start").groupby("direction"):
        # Insert a single null value before each gap to keep a single legend entry.
        gaps = group.start.ne(group.end.shift()) & group.end.shift().notna()
        x, y, hover = [], [], []
        for start, power, gap in zip(group.start.dt.tz_convert("UTC"), group.power_mw, gaps):
            oslo = start.tz_convert("Europe/Oslo").strftime("%Y-%m-%d %H:%M %Z")
            if gap:
                x.append(start)
                y.append(None)
                hover.append("")
            x.append(start)
            y.append(power)
            hover.append(oslo)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker={"size": 2},
                                 name=direction.title(), connectgaps=False, text=hover,
                                 hovertemplate="%{text}<br>%{y:.2f} MW"))
    fig.update_layout(xaxis_title="Time (UTC)", yaxis_title="Physical flow (MW)")
    return fig
