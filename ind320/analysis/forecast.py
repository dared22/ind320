"""Chronological River forecasts shared by Jupyter and Streamlit."""

from bisect import bisect_left
from collections.abc import Mapping

import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing, time_series

from ind320.data.weeks import OSLO, WEEK

# Weather needs more lead time than the business-day rates and interval flows.
FEATURE_LAGS = {"temperature_2m": 2, "precipitation": 2, "EUR/NOK": 1, "USD/NOK": 1,
                "net_import_mwh": 1}
MIN_INTERVAL_ERRORS = 12


def _future_inputs(
    sources: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, cutoff: pd.Timestamp,
) -> list[dict]:
    """Build the same lagged persistence scenario at live and historical origins."""
    if not sources:
        return [{} for _ in dates]
    inputs = pd.DataFrame(index=dates)
    for feature, source in sources.items():
        known = source.loc[source.available_at.le(cutoff), feature].dropna()
        if known.empty:
            raise ValueError(f"No values for {feature} were available by the forecast cutoff.")
        later = known.reindex(dates - FEATURE_LAGS[feature] * WEEK)
        # Gaps within the historical archive remain missing.
        later = later.mask(later.index > known.index[-1], known.iloc[-1])
        if later.isna().any():
            raise ValueError(f"{feature} has a gap in the lagged inputs for this forecast.")
        inputs[feature] = later.to_numpy()
    return inputs.to_dict("records")


def _error_quantile(errors: np.ndarray, alpha: float) -> float:
    """Find the breakpoint that minimizes River's quantile loss."""
    ordered = np.sort(errors)
    loss = optim.losses.Quantile(alpha)

    def gradient_right_of(value: float) -> float:
        # Evaluate immediately to the right to include all samples equal to this error.
        return float(loss.gradient(ordered, np.nextafter(value, np.inf)).mean())

    return float(ordered[bisect_left(ordered, -1e-12, key=gradient_right_of)])


def forecast_reservoir(
    observations: pd.DataFrame, sources: Mapping[str, pd.DataFrame], *,
    area: tuple[str, int], cutoff: pd.Timestamp, training_weeks: int, horizon: int,
    order: tuple[int, int, int], selected_inputs: tuple[str, ...] = (),
    intervals: bool = True,
) -> pd.DataFrame:
    """Forecast future weekly filling using only information released by cutoff.

    Each selected source is indexed by Sunday labels and provides its feature
    column plus an availability timestamp. No observations after cutoff are
    used for training or recursive prediction.
    """
    if cutoff.tzinfo is None or not 1 <= horizon <= 52 or training_weeks < 78:
        raise ValueError("Use a timezone-aware cutoff, at least 78 training weeks, and 1–52 forecast weeks.")
    if order[0] not in range(1, 5) or order[1] not in (0, 1) or order[2] not in (0, 1):
        raise ValueError("Use 1–4 autoregressive terms and 0–1 differences and moving-average terms.")
    area_type, area_code = area
    selected = observations.loc[
        observations["Area Type"].eq(area_type) & observations["Area Code"].eq(area_code)
        & observations.available_at.le(cutoff)
    ].sort_values("week")
    if selected.empty:
        raise ValueError("There are no published NVE observations for this region and cutoff.")
    selected = selected.set_index("week")
    history = selected["Fill Level"]
    if history.index.has_duplicates:
        raise ValueError("NVE returned duplicate observations for this region and week.")
    last = history.index[-1]
    past = pd.date_range(end=last, periods=training_weeks, freq="W-SUN", name="week")
    actual = history.reindex(past)
    # Allow isolated gaps using only previous observations. Longer gaps require a different window.
    target = actual.ffill(limit=1)
    if target.isna().any():
        raise ValueError("NVE coverage is incomplete for the selected training window.")
    future = pd.date_range(last + WEEK, periods=horizon, freq="W-SUN", name="week")
    train_features = pd.DataFrame(index=past)
    prepared_sources = {}
    for feature in selected_inputs:
        if feature not in sources or feature not in FEATURE_LAGS:
            raise ValueError(f"No source is available for {feature}.")
        source = sources[feature].sort_index()
        if source.empty:
            raise ValueError(f"No weekly coverage is available for {feature}.")
        known = source.loc[source.available_at.le(cutoff), feature].dropna()
        lag = FEATURE_LAGS[feature] * WEEK
        training_labels = past - lag
        historical = known.reindex(training_labels)
        # ENTSO-E publishes shortly after the interval ends. At the target week's
        # closing boundary each lagged aggregate must already be available.
        historical_release = source.available_at.reindex(training_labels)
        target_end = (past + pd.Timedelta(days=1)).tz_localize(OSLO)
        missing = historical.isna().to_numpy() | (historical_release.array > target_end.array)
        if missing.any():
            dates = training_labels[missing]
            raise ValueError(f"{feature} lacks training coverage from {dates[0]:%Y-%m-%d} "
                             f"to {dates[-1]:%Y-%m-%d}. Adjust the dates or inputs.")
        train_features[feature] = historical.to_numpy()
        prepared_sources[feature] = source
    future_inputs = _future_inputs(prepared_sources, future, cutoff)
    result = pd.DataFrame(index=future)
    train_records = train_features.to_dict("records") if selected_inputs else [{} for _ in past]
    regressor = preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.01), loss=optim.losses.Quantile(0.5),
    )
    season = 52
    model = time_series.SNARIMAX(*order, m=season, sp=1, regressor=regressor)
    # Use up to the most recent year of origins after at least one full season of learning.
    calibration_start = max(season, len(past) - season)
    releases = selected.available_at.reindex(past)
    latest_release = releases.ffill().cummax()
    errors = np.full((len(past), horizon), np.nan) if intervals else None
    for row, (value, features) in enumerate(zip(target, train_records)):
        # River adds autoregressive terms to the supplied dictionaries in place.
        model.learn_one(float(value), features.copy())
        if (errors is None or row + 1 < calibration_start
                or pd.isna(releases.iloc[row]) or row + 1 == len(past)):
            continue
        # Score forecasts made before these target observations were published.
        targets = slice(row + 1, row + 1 + horizon)
        dates = past[targets]
        origin = latest_release.iloc[row]
        try:
            features_at_origin = _future_inputs(prepared_sources, dates, origin)
        except ValueError:
            continue  # Some inputs may be missing as of the historical origin.
        prediction = np.clip(model.forecast(len(dates), features_at_origin), 0, 1)
        outcomes = actual.iloc[targets].where(releases.iloc[targets].gt(origin))
        errors[row, :len(dates)] = (outcomes.to_numpy() - prediction) * 100

    result["median"] = np.clip(model.forecast(horizon, future_inputs), 0, 1) * 100
    if intervals:
        samples = [column[np.isfinite(column)] for column in errors.T]
        result["interval_samples"] = [len(sample) for sample in samples]
        for name, alpha, bound in (("lower", 0.10, min), ("upper", 0.90, max)):
            offsets = [bound(0, _error_quantile(sample, alpha)) if len(sample) >= MIN_INTERVAL_ERRORS
                       else np.nan for sample in samples]
            result[name] = (result["median"] + offsets).clip(0, 100)
    result.attrs.update(training_start=past[0], training_end=last, cutoff=cutoff,
                        imputed=int(actual.isna().sum()))
    return result
