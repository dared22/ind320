"""Draw a fixed-axis Plotly animation of the reservoir forecast and interval."""

import pandas as pd
import plotly.graph_objects as go

from ind320.ui.palette import GREEN, INK, PEACH, PLOT_BAND, WARM_ACCENT

ANIMATION_OPTIONS = {"mode": "immediate", "frame": {"duration": 350, "redraw": False},
                     "transition": {"duration": 0}}


def forecast_chart(observed: pd.Series, forecast: pd.DataFrame, *, autoplay: bool = False) -> go.Figure:
    """Reveal the point forecast and shaded interval one week at a time."""
    fig = go.Figure(go.Scatter(x=observed.index, y=observed, name="Observed",
                               line={"color": GREEN}))

    def traces(data: pd.DataFrame) -> list[go.Scatter]:
        dates = [observed.index[-1], *data.index]
        anchor = observed.iloc[-1]
        return [
            go.Scatter(x=dates, y=[anchor, *data.lower], mode="lines", line={"width": 0},
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=dates, y=[anchor, *data.upper], mode="lines", line={"width": 0},
                       fill="tonexty", fillcolor=PLOT_BAND,
                       name="Nominal 80% interval", hoverinfo="skip",
                       showlegend=bool(data.upper.notna().any())),
            go.Scatter(x=dates, y=[anchor, *data["median"]], name="Point forecast",
                       line={"color": WARM_ACCENT, "dash": "dash"}),
        ]

    # Start fresh fits at week one; show the full forecast when revisiting a result.
    active_week = 1 if autoplay else len(forecast)
    fig.add_traces(traces(forecast.iloc[:active_week]))
    fig.frames = [
        go.Frame(name=str(step), data=traces(forecast.iloc[:step]), traces=[1, 2, 3])
        for step in range(1, len(forecast) + 1)
    ]
    fig.update_layout(
        template="plotly_white", height=480, margin={"l": 60, "r": 12, "t": 44, "b": 166},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": INK},
        xaxis={"title": "Week ending", "range": [observed.index[0], forecast.index[-1]], "showgrid": False},
        yaxis={"title": "Reservoir filling (%)", "range": [min(0, observed.min()), max(100, observed.max())]},
        # Reserve separate rows for the axis title, playback buttons, and slider.
        updatemenus=[{"type": "buttons", "direction": "right", "showactive": False,
                      "x": 0, "y": 0, "xanchor": "left", "yanchor": "top",
                      "pad": {"l": 0, "r": 0, "t": 57, "b": 0}, "bgcolor": PEACH,
                      "buttons": [
                          {"label": "Play", "method": "animate", "args": [None, ANIMATION_OPTIONS]},
                          {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate",
                              "frame": {"duration": 0, "redraw": False}}]},
                      ]}],
        sliders=[{"active": active_week - 1, "pad": {"l": 0, "r": 0, "t": 99},
                  "currentvalue": {"prefix": "Forecast week "},
                  "steps": [{"label": frame.name, "method": "animate",
                             "args": [[frame.name], {"mode": "immediate",
                                                    "frame": {"duration": 0, "redraw": False}}]}
                            for frame in fig.frames]}],
    )
    return fig
