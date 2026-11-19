"""Draw a fixed-axis Plotly animation of the River trajectories."""

import pandas as pd
import plotly.graph_objects as go

from ind320.ui.palette import GREEN, PLOT_BAND, WARM_ACCENT


def forecast_chart(observed: pd.Series, forecast: pd.DataFrame) -> go.Figure:
    """Reveal the median and shaded interval one week at a time."""
    fig = go.Figure(go.Scatter(x=observed.index, y=observed, name="Observed",
                               line={"color": GREEN}))

    def traces(data: pd.DataFrame) -> list[go.Scatter]:
        dates = [observed.index[-1], *data.index]
        anchor = observed.iloc[-1]
        return [
            go.Scatter(x=dates, y=[anchor, *data.lower], line={"width": 0},
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=dates, y=[anchor, *data.upper], line={"width": 0},
                       fill="tonexty", fillcolor=PLOT_BAND,
                       name="Nominal 80% interval", hoverinfo="skip"),
            go.Scatter(x=dates, y=[anchor, *data["median"]], name="Median forecast",
                       line={"color": WARM_ACCENT, "dash": "dash"}),
        ]

    # Show the full answer on load, with Play to replay starting at the first week.
    fig.add_traces(traces(forecast))
    fig.frames = [
        go.Frame(name=str(step), data=traces(forecast.iloc[:step]), traces=[1, 2, 3])
        for step in range(1, len(forecast) + 1)
    ]
    animation = {"frame": {"duration": 350, "redraw": False}, "transition": {"duration": 0}}
    fig.update_layout(
        xaxis={"title": "Week ending", "range": [observed.index[0], forecast.index[-1]]},
        yaxis={"title": "Reservoir filling (%)", "range": [min(0, observed.min()), max(100, observed.max())]},
        updatemenus=[{"type": "buttons", "x": 0, "y": 0, "xanchor": "right", "yanchor": "top",
                      "buttons": [
                          {"label": "Play", "method": "animate", "args": [None, animation]},
                          {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate",
                              "frame": {"duration": 0, "redraw": False}}]},
                      ]}],
        sliders=[{"active": len(forecast) - 1, "currentvalue": {"prefix": "Forecast week "},
                  "steps": [{"label": frame.name, "method": "animate",
                             "args": [[frame.name], {"mode": "immediate",
                                                    "frame": {"duration": 0, "redraw": False}}]}
                            for frame in fig.frames]}],
    )
    return fig
