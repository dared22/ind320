"""DCT/SPC precipitation checks and univariate temperature LOF checks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.fft import dct, idct
from scipy.stats import median_abs_deviation
from sklearn.neighbors import LocalOutlierFactor


def precipitation_analysis(
    series: pd.Series, cutoff: int = 8, multiplier: float = 3.0
) -> tuple[pd.DataFrame, dict, Figure]:
    """Remove low DCT coefficients and flag residuals outside robust SPC limits.

    Use the year's median and normal-scaled MAD as robust location and sigma.
    If MAD is zero (for example, a dry series), use sample standard deviation.
    Limits are exploratory flags, not evidence that a measurement is incorrect.
    """
    values = series.to_numpy(dtype=float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Precipitation needs at least three finite daily values.")
    if not 0 <= cutoff < len(values) or multiplier <= 0:
        raise ValueError("Cutoff must be below the series length; multiplier must be positive.")
    coefficients = dct(values, norm="ortho")
    coefficients[:cutoff] = 0
    # A constant nonzero series can leave roundoff-sized DCT residuals. Treat
    # those as zero before MAD scaling, or numerical noise becomes an outlier.
    filtered = values.copy() if cutoff == 0 else idct(coefficients, norm="ortho")
    tolerance = 10 * np.finfo(float).eps * max(1.0, np.abs(values).max())
    if cutoff > 0:
        filtered[np.abs(filtered) <= tolerance] = 0
    baseline = values - filtered
    center = float(np.median(filtered))
    sigma = float(median_abs_deviation(filtered, scale="normal"))
    scale_method = "normal-scaled MAD"
    if sigma == 0:
        sigma = float(np.std(filtered, ddof=1))
        scale_method = "sample standard deviation (zero MAD fallback)"
    lower, upper = center - multiplier * sigma, center + multiplier * sigma
    outlier = (filtered < lower) | (filtered > upper)
    data = pd.DataFrame(
        {"precipitation": values, "baseline": baseline, "filtered": filtered,
         "lower": baseline + lower, "upper": baseline + upper, "outlier": outlier},
        index=series.index,
    )
    summary = {"days": len(data), "outliers": int(outlier.sum()),
               "center": center, "sigma": sigma, "scale_method": scale_method}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(data.index, values, label="Daily average hourly amount")
    axes[0].plot(data.index, baseline, label="Removed low-frequency component", alpha=0.7)
    axes[0].fill_between(data.index, data.lower, data.upper, alpha=0.15, label="SPC limits")
    axes[0].scatter(data.index[outlier], values[outlier], color="firebrick", marker="x", label="Flagged day")
    axes[0].set(ylabel="Average hourly amount (mm)", title="Precipitation: DCT and robust SPC")
    axes[0].legend(fontsize="small")
    axes[1].plot(data.index, filtered, label="DCT high-pass precipitation")
    axes[1].axhline(lower, color="firebrick", linestyle="--")
    axes[1].axhline(upper, color="firebrick", linestyle="--", label="Residual limits")
    axes[1].set(xlabel="UTC date", ylabel="Filtered amount (mm)")
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    return data, summary, fig


def temperature_analysis(
    series: pd.Series, contamination: float = 0.01
) -> tuple[pd.DataFrame, dict, Figure]:
    """Flag rare daily temperatures using LOF on temperature alone (20 neighbors)."""
    values = series.to_numpy(dtype=float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Temperature needs at least three finite daily values.")
    if not 0 < contamination <= 0.5:
        raise ValueError("LOF contamination must be in (0, 0.5].")
    model = LocalOutlierFactor(n_neighbors=min(20, len(values) - 1), contamination=contamination)
    outlier = model.fit_predict(values.reshape(-1, 1)) == -1
    data = pd.DataFrame(
        {"temperature_2m": values, "lof_score": -model.negative_outlier_factor_, "outlier": outlier},
        index=series.index,
    )
    summary = {"days": len(data), "outliers": int(outlier.sum()),
               "mean_c": float(values.mean()), "contamination": contamination}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, values, label="Daily average temperature")
    ax.scatter(data.index[outlier], values[outlier], color="firebrick", marker="x", label="LOF flag")
    ax.set(title="Temperature: Local Outlier Factor", xlabel="UTC date", ylabel="Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    return data, summary, fig
