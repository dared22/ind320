"""River SNARIMAX forecasts with lead-time prediction intervals."""

from bisect import bisect_left
from collections.abc import Mapping

import numpy as np
import pandas as pd
from river import optim, time_series

from ind320.data.weeks import OSLO, WEEK

# Weather needs more lead time than the business-day rates and interval flows.
FEATURE_LAGS = {"temperature_2m": 2, "precipitation": 2, "EUR/NOK": 1, "USD/NOK": 1,
                "net_import_mwh": 1}
MIN_INTERVAL_ERRORS = 12
SEASON = 52


def _future_inputs(
    sources: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Build the same lagged persistence scenario at live and historical origins."""
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
    return inputs


def _predict(model: time_series.SNARIMAX, inputs: pd.DataFrame) -> np.ndarray:
    """Forecast each week with a separate feature dictionary for River's lag features."""
    weekly_features = list(inputs.to_dict("index").values())
    return np.asarray(model.forecast(horizon=len(inputs), xs=weekly_features))


def _error_quantile(errors: np.ndarray, alpha: float) -> float:
    """Find the breakpoint that minimizes River's quantile loss."""
    ordered = np.sort(errors)
    loss = optim.losses.Quantile(alpha)

    def gradient_right_of(value: float) -> float:
        # Evaluate immediately to the right to include all samples equal to this error.
        return float(loss.gradient(ordered, np.nextafter(value, np.inf)).mean())

    return float(ordered[bisect_left(ordered, -1e-12, key=gradient_right_of)])


def _train_snarimax(
    target: pd.Series, actual: pd.Series, releases: pd.Series, inputs: pd.DataFrame,
    sources: dict[str, pd.DataFrame], order: tuple[int, int, int], horizon: int, intervals: bool,
) -> tuple[time_series.SNARIMAX, np.ndarray | None]:
    """Train in order and optionally score forecasts as each origin is reached."""
    p, d, q = order
    model = time_series.SNARIMAX(p=p, d=d, q=q, m=SEASON, sp=1)
    start = max(SEASON, len(target) - SEASON)
    errors = np.full((len(target) - start, horizon), np.nan) if intervals else None
    latest_release = releases.ffill().cummax()
    # With no selected inputs, to_dict("index") still supplies one dictionary per week.
    weekly_features = inputs.to_dict("index")
    for row, (week, value) in enumerate(target.items()):
        model.learn_one(value, x=weekly_features[week])
        if errors is None or not start - 1 <= row < len(target) - 1 or pd.isna(releases.at[week]):
            continue
        targets = slice(row + 1, row + 1 + horizon)
        dates = target.index[targets]
        origin = latest_release.iloc[row]
        try:
            features_at_origin = _future_inputs(sources, dates, origin)
        except ValueError:
            continue  # This historical origin has incomplete input coverage.
        prediction = _predict(model, features_at_origin)
        if not np.isfinite(prediction).all():
            continue
        outcomes = actual.iloc[targets].where(releases.iloc[targets].gt(origin))
        errors[row - start + 1, :len(dates)] = (outcomes.to_numpy() - prediction.clip(0, 1)) * 100
    return model, errors


def forecast_reservoir(
    observations: pd.DataFrame, sources: Mapping[str, pd.DataFrame], *,
    area: tuple[str, int], cutoff: pd.Timestamp, training_weeks: int, horizon: int,
    order: tuple[int, int, int], selected_inputs: tuple[str, ...] = (),
    intervals: bool = True,
) -> pd.DataFrame:
    """Forecast future weekly filling using only information released by cutoff.

    Each selected source is indexed by Sunday labels and provides its feature
    column plus an availability timestamp. No observations after cutoff are
    used for training or recursive prediction. Point forecasts are returned in
    the existing `median` column.
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
    model, errors = _train_snarimax(target, actual, selected.available_at.reindex(past),
                                    train_features, prepared_sources, order, horizon, intervals)
    point = _predict(model, future_inputs)
    if not np.isfinite(point).all():
        raise ValueError("River SNARIMAX returned non-finite predictions. Try another model order.")
    result["median"] = point.clip(0, 1) * 100
    if errors is not None:
        samples = [column[np.isfinite(column)] for column in errors.T]
        result["interval_samples"] = [len(sample) for sample in samples]
        for name, alpha, bound in (("lower", 0.10, min), ("upper", 0.90, max)):
            offsets = [bound(0, _error_quantile(sample, alpha)) if len(sample) >= MIN_INTERVAL_ERRORS
                       else np.nan for sample in samples]
            result[name] = (result["median"] + offsets).clip(0, 100)
    result.attrs.update(training_start=past[0], training_end=last, cutoff=cutoff,
                        imputed=int(actual.isna().sum()))
    return result
