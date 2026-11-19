"""Chronological River forecasts shared by Jupyter and Streamlit."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing, time_series

from ind320.data.weeks import OSLO, WEEK

# Weather needs more lead time than the business-day rates and interval flows.
FEATURE_LAGS = {"temperature_2m": 2, "precipitation": 2, "EUR/NOK": 1, "USD/NOK": 1,
                "net_import_mwh": 1}


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
    history = selected.set_index("week")["Fill Level"]
    if history.index.has_duplicates:
        raise ValueError("NVE returned duplicate observations for this region and week.")
    last = history.index[-1]
    past = pd.date_range(end=last, periods=training_weeks, freq="W-SUN", name="week")
    target = history.reindex(past)
    # Allow isolated gaps using only previous observations. Longer gaps require a different window.
    target = target.ffill(limit=1) if past[0] in history.index else target
    if target.isna().any():
        raise ValueError("NVE coverage is incomplete for the selected training window.")
    future = pd.date_range(last + WEEK, periods=horizon, freq="W-SUN", name="week")
    train_features = pd.DataFrame(index=past)
    future_features = pd.DataFrame(index=future)
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
        later = known.reindex(future - lag)
        # A missing past week is a coverage issue, while future weeks use persistence.
        beyond = later.index > known.index[-1]
        later = later.mask(beyond, known.iloc[-1])
        if later.isna().any():
            raise ValueError(f"{feature} has a gap in the lagged inputs for this forecast.")
        future_features[feature] = later.to_numpy()
    quantiles = {"median": 0.5, "lower": 0.1, "upper": 0.9} if intervals else {"median": 0.5}
    result = pd.DataFrame(index=future)
    train_records = train_features.to_dict("records") if selected_inputs else [{} for _ in past]
    future_records = future_features.to_dict("records") if selected_inputs else [{} for _ in future]
    for name, quantile in quantiles.items():
        regressor = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(0.01), loss=optim.losses.Quantile(quantile),
        )
        model = time_series.SNARIMAX(*order, m=52, sp=1, regressor=regressor)
        for value, features in zip(target, train_records):
            model.learn_one(float(value), features.copy())
        # River adds autoregressive terms to the supplied dictionaries in place.
        result[name] = np.clip(model.forecast(horizon, [row.copy() for row in future_records]), 0, 1) * 100
    if intervals:
        result["crossed"] = (result.lower > result["median"]) | (result.upper < result["median"])
        bounds = result[["lower", "median", "upper"]]
        result["lower"], result["upper"] = bounds.min(axis=1), bounds.max(axis=1)
    result.attrs.update(training_start=past[0], training_end=last, cutoff=cutoff,
                        imputed=int(history.reindex(past).isna().sum()))
    return result
