"""Classical SARIMAX forecasts with River quantile-loss intervals."""

import warnings
from bisect import bisect_left
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from river import optim
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

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


def _fit_sarimax(
    target: pd.Series, inputs: pd.DataFrame, order: tuple[int, int, int],
) -> tuple[SARIMAXResultsWrapper, Callable[[pd.DataFrame], pd.DataFrame | None]]:
    """Fit the state-space SARIMAX and reuse the training scale for new inputs."""
    center = inputs.mean()
    scale = inputs.std(ddof=0).replace(0, 1)

    def scale_inputs(values: pd.DataFrame) -> pd.DataFrame | None:
        return (values - center) / scale if len(values.columns) else None

    model = SARIMAX(
        target, exog=scale_inputs(inputs), order=order, seasonal_order=(1, 0, 0, SEASON),
        trend="n" if order[1] else "c",
    )
    try:
        with warnings.catch_warnings():
            # Statsmodels can replace its starting values with zeros and still converge.
            warnings.simplefilter("ignore", EstimationWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = model.fit(disp=False, cov_type="none", maxiter=75)
            if not fit.mle_retvals["converged"] and np.isfinite(fit.params).all():
                # Retry from the first fit when the gradient optimizer struggles.
                fit = model.fit(start_params=fit.params, method="powell",
                                disp=False, cov_type="none", maxiter=60)
    except np.linalg.LinAlgError as error:
        raise ValueError("SARIMAX could not fit this series. Try another order or training window.") from error
    if not fit.mle_retvals["converged"] or not np.isfinite(fit.params).all():
        raise ValueError("SARIMAX did not converge. Try another order or training window.")
    return fit, scale_inputs


def _error_quantile(errors: np.ndarray, alpha: float) -> float:
    """Find the breakpoint that minimizes River's quantile loss."""
    ordered = np.sort(errors)
    loss = optim.losses.Quantile(alpha)

    def gradient_right_of(value: float) -> float:
        # Evaluate immediately to the right to include all samples equal to this error.
        return float(loss.gradient(ordered, np.nextafter(value, np.inf)).mean())

    return float(ordered[bisect_left(ordered, -1e-12, key=gradient_right_of)])


def _historical_errors(
    target: pd.Series, actual: pd.Series, releases: pd.Series, inputs: pd.DataFrame,
    sources: dict[str, pd.DataFrame], order: tuple[int, int, int], horizon: int,
) -> np.ndarray:
    """Score earlier SARIMAX forecasts using only data known at each origin."""
    start = max(SEASON, len(target) - SEASON)
    errors = np.full((len(target) - start, horizon), np.nan)
    try:
        fit, scale_inputs = _fit_sarimax(target.iloc[:start], inputs.iloc[:start], order)
    except ValueError:
        # An unstable fit on the shorter window still permits a full-window forecast.
        return errors
    latest_release = releases.ffill().cummax()
    for row in range(start - 1, len(target) - 1):
        if row >= start:
            update = slice(row, row + 1)
            fit = fit.extend(target.iloc[update], exog=scale_inputs(inputs.iloc[update]))
        if pd.isna(releases.iloc[row]):
            continue
        targets = slice(row + 1, row + 1 + horizon)
        dates = target.index[targets]
        origin = latest_release.iloc[row]
        try:
            features_at_origin = _future_inputs(sources, dates, origin)
        except ValueError:
            continue  # This historical origin has incomplete input coverage.
        prediction = np.clip(fit.forecast(len(dates), exog=scale_inputs(features_at_origin)), 0, 1)
        outcomes = actual.iloc[targets].where(releases.iloc[targets].gt(origin))
        errors[row - start + 1, :len(dates)] = (outcomes.to_numpy() - prediction.to_numpy()) * 100
    return errors


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
    fit, scale_inputs = _fit_sarimax(target, train_features, order)
    # In the Gaussian SARIMAX model the point prediction is also the conditional median.
    point = fit.forecast(horizon, exog=scale_inputs(future_inputs))
    if not np.isfinite(point).all():
        raise ValueError("SARIMAX returned non-finite predictions. Try another model order.")
    result["median"] = point.clip(0, 1) * 100
    if intervals:
        errors = _historical_errors(target, actual, selected.available_at.reindex(past),
                                    train_features, prepared_sources, order, horizon)
        samples = [column[np.isfinite(column)] for column in errors.T]
        result["interval_samples"] = [len(sample) for sample in samples]
        for name, alpha, bound in (("lower", 0.10, min), ("upper", 0.90, max)):
            offsets = [bound(0, _error_quantile(sample, alpha)) if len(sample) >= MIN_INTERVAL_ERRORS
                       else np.nan for sample in samples]
            result[name] = (result["median"] + offsets).clip(0, 100)
    result.attrs.update(training_start=past[0], training_end=last, cutoff=cutoff,
                        imputed=int(actual.isna().sum()))
    return result
