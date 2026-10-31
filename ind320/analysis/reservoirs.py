"""Weekly filling decomposition and frequency analysis, using full history."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.signal import spectrogram
from statsmodels.tsa.seasonal import STL


def weekly_filling(source: pd.DataFrame, area_type: str, area_code: int) -> pd.Series:
    """Select full history in percent; interpolate internal missing weeks visibly.

    NVE dates are weekly observations. A regular seven-day grid is required by
    both methods. Missing internal values use time interpolation, with a count
    in attrs for the notebook and page; no values outside the history are added.
    """
    selected = source.loc[
        (source["Area Type"] == area_type) & (source["Area Code"] == area_code)
    ].sort_values("Observation Date")
    series = selected.set_index("Observation Date")["Fill Level"] * 100
    if series.empty:
        raise ValueError("No filling observations are available for this area.")
    if series.index.has_duplicates:
        raise ValueError("Weekly filling dates must be unique.")
    grid = pd.date_range(series.index.min(), series.index.max(), freq="7D")
    if not series.index.isin(grid).all():
        raise ValueError("Reservoir observations must share a seven-day grid.")
    series = series.reindex(grid)
    missing = int(series.isna().sum())
    series = series.interpolate(method="time", limit_area="inside")
    series.name = f"{area_type} {area_code} filling (%)"
    series.attrs["interpolated_weeks"] = missing
    return series


def _check_weekly(series: pd.Series) -> None:
    """Prevent a gap or invalid value from silently changing frequency units."""
    if len(series) < 104 or not np.isfinite(series.to_numpy()).all():
        raise ValueError("Use at least two years (104 weeks) of finite filling values.")
    if not (series.index.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all():
        raise ValueError("Filling observations must be exactly one week apart.")


def stl_analysis(
    series: pd.Series, seasonal: int = 13, trend: int = 105, robust: bool = True
) -> tuple[pd.DataFrame, dict, Figure]:
    """Decompose weekly percentages with a fixed 52-week annual period."""
    _check_weekly(series)
    if seasonal < 3 or seasonal % 2 != 1 or trend <= 52 or trend % 2 != 1:
        raise ValueError("Seasonal must be odd and >=3; trend must be odd and >52.")
    result = STL(series, period=52, seasonal=seasonal, trend=trend, robust=robust).fit()
    data = pd.DataFrame(
        {"observed": series, "trend": result.trend, "seasonal": result.seasonal, "residual": result.resid}
    )
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, column in zip(axes, data.columns):
        ax.plot(data.index, data[column], linewidth=0.9)
        ax.set_ylabel(f"{column.title()}\n(pp)")
    axes[0].set_ylabel("Filling (%)")
    axes[0].set_title(f"{series.name}: STL, 52-week period")
    axes[-1].set_xlabel("Observation date")
    fig.tight_layout()
    summary = {"weeks": len(data), "residual_sd_pp": float(data.residual.std()),
               "seasonal_range_pp": float(data.seasonal.max() - data.seasonal.min())}
    return data, summary, fig


def filling_spectrogram(
    series: pd.Series, window: int = 104, overlap: int = 52
) -> tuple[pd.DataFrame, dict, Figure]:
    """Estimate density in pp²/(cycles/week), with window centers on the date axis.

    Hann windows and constant detrending remove each window's mean. Sampling is
    one observation/week, giving a Nyquist frequency of 0.5 cycles/week.
    """
    _check_weekly(series)
    if not 4 <= window <= len(series) or not 0 <= overlap < window:
        raise ValueError("Window must fit the series; overlap must be below the window.")
    frequencies, times, power = spectrogram(
        series.to_numpy(), fs=1.0, window="hann", nperseg=window,
        noverlap=overlap, detrend="constant", scaling="density", mode="psd",
    )
    dates = series.index[0] + pd.to_timedelta(times * 7, unit="D")
    spectrum = pd.DataFrame(power, index=frequencies, columns=dates)
    spectrum.index.name = "frequency_cycles_per_week"
    spectrum.columns.name = "window_center"
    # Express power in dB relative to 1 pp²/(cycles/week), not arbitrary colors.
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(dates, frequencies, 10 * np.log10(np.maximum(power, 1e-12)), shading="auto")
    ax.axhline(1 / 52, color="white", linestyle="--", linewidth=1, label="Annual cycle (1/52)")
    ax.set(xlabel="Window center date", ylabel="Frequency (cycles/week)", title=f"{series.name}: spectrogram")
    ax.legend(fontsize="small")
    fig.colorbar(mesh, ax=ax, label="PSD (dB re 1 pp²/(cycles/week))")
    fig.tight_layout()
    # Exclude the DC bin when describing the strongest oscillation.
    peak = float(frequencies[1 + np.argmax(power[1:].mean(axis=1))])
    summary = {"weeks": len(series), "windows": len(times), "frequency_step": 1 / window,
               "peak_period_weeks": 1 / peak}
    return spectrum, summary, fig
