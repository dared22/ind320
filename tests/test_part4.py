"""Forecast, timing and navigation regressions for Part 4."""

import json
from datetime import date
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from river.time_series import SNARIMAX
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.testing.v1 import AppTest

from ind320.analysis.forecast import forecast_reservoir
from ind320.data.exchange import fetch_exchange_rates
from ind320.data.weeks import OSLO, combine_net_flows, cutoff_time, reservoir_weeks, weekly_flows, weekly_fx, weekly_weather
from test_feedback_fixes import reservoir_fixture

ROOT = Path(__file__).resolve().parents[1]


class SourceAlignmentTests(unittest.TestCase):
    def test_business_day_csv_preserves_rates_and_dates(self):
        response = Mock(content=b"BASE_CUR;TIME_PERIOD;OBS_VALUE;UNIT_MULT\n"
                                 b"EUR;2024-03-27;11.64;0\nUSD;2024-03-27;10.8;0\n"
                                 b"EUR;2024-04-02;11.67;0\nUSD;2024-04-02;10.89;0")
        with patch("ind320.data.exchange.requests.get", return_value=response) as download:
            rates = fetch_exchange_rates("2024-03-25", "2024-04-03")
        self.assertIn("B.EUR+USD.NOK.SP", download.call_args.args[0])
        self.assertEqual(rates.pair.unique().tolist(), ["EUR/NOK", "USD/NOK"])
        self.assertEqual(len(rates), 4)
        quote = weekly_fx(rates).loc[(pd.Timestamp("2024-03-31"), "EUR/NOK")]
        self.assertEqual(quote.nok_per_unit, 11.64)
        self.assertEqual(quote.available_at, pd.Timestamp("2024-03-27 16:00", tz=OSLO))

    def test_weather_uses_actual_dst_hours_and_preceding_precipitation(self):
        for sunday, hours in [("2024-03-31", 167), ("2024-10-27", 169)]:
            week = pd.Timestamp(sunday)
            ends_at = (week + pd.Timedelta(days=1)).tz_localize(OSLO).tz_convert("UTC")
            index = pd.date_range(end=ends_at, periods=hours + 1, freq="h")
            data = pd.DataFrame({"precipitation": 1.0, "temperature_2m": 12.0}, index=index)
            data.iloc[0, 0] = 1000.0  # This amount belongs to the previous week.
            data.iloc[-1, 1] = 1000.0  # This instant belongs to the following week.
            result = weekly_weather(data).loc[week]
            self.assertEqual(result.precipitation, hours)
            self.assertEqual(result.temperature_2m, 12)

    def test_weekly_flows_clip_intervals_and_reject_missing_direction(self):
        week = pd.Timestamp("2024-03-31")
        start = pd.Timestamp("2024-03-25 00:00", tz=OSLO).tz_convert("UTC")
        end = pd.Timestamp("2024-04-01 00:00", tz=OSLO).tz_convert("UTC")
        periods = pd.DataFrame({
            "start": [start - pd.Timedelta(hours=1)] * 2, "end": [end + pd.Timedelta(hours=1)] * 2,
            "connection": "NO3–SE2", "direction": ["import", "export"], "power_mw": [3, 1],
        })
        result = weekly_flows(periods)
        self.assertEqual(result.loc[week, "net_import_mwh"], 334)
        self.assertTrue(weekly_flows(periods.iloc[:1]).empty)

    def test_combined_flows_use_area_sign_and_complete_weeks(self):
        weeks = pd.date_range("2024-04-07", periods=2, freq="W-SUN")
        published = pd.Timestamp("2024-04-08 01:00", tz=OSLO)
        first = pd.DataFrame({"net_import_mwh": [300, 400], "available_at": published}, index=weeks)
        second = pd.DataFrame({"net_import_mwh": [500], "available_at": published + pd.Timedelta(hours=1)},
                              index=weeks[:1])
        combined = combine_net_flows({"NO1–NO3": first, "NO3–NO5": second}, "NO3")
        self.assertEqual(combined.index.tolist(), weeks[:1].tolist())
        self.assertEqual(combined.iloc[0].net_import_mwh, 200)
        self.assertEqual(combined.iloc[0].available_at, published + pd.Timedelta(hours=1))

    def test_nve_next_release_belongs_to_following_week(self):
        source = pd.DataFrame({
            "Area Type": "EL", "Area Code": 3, "Iso Year": 2024, "Iso Week": [12, 13],
            "Next Publication Date": pd.to_datetime(["2024-04-04 13:00", "2024-04-10 13:00"]),
        })
        rows = reservoir_weeks(source)
        self.assertEqual(rows.available_at.iloc[0], pd.Timestamp("2024-03-27 13:00", tz=OSLO))
        self.assertEqual(rows.available_at.iloc[1], pd.Timestamp("2024-04-04 13:00", tz=OSLO))


class ForecastTests(unittest.TestCase):
    def test_forecast_runs_with_multiple_net_flow_connections(self):
        def flows(connection, first, last):
            weeks = pd.date_range(first, last, freq="W-SUN")
            return pd.DataFrame({
                "net_import_mwh": 50000 + 20000 * np.sin(np.arange(len(weeks)) / 9),
                "available_at": (weeks + pd.Timedelta(days=1, hours=1)).tz_localize(OSLO),
            }, index=weeks)

        with patch("ind320.data.reservoirs.fetch_reservoir_history", return_value=reservoir_fixture()), patch(
            "ind320.ui.source_cache.flow_weeks", side_effect=flows
        ) as download, patch("ind320.analysis.forecast.forecast_reservoir", wraps=forecast_reservoir) as fit:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("pages/2_Forecast.py").run()
            selected = ["NO3–SE2", "NO1–NO3"]
            app.multiselect[1].set_value(selected)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error, [item.value for item in app.error])
            self.assertEqual([call.args[0] for call in download.call_args_list], selected)
            self.assertEqual(fit.call_args.kwargs["selected_inputs"], ("net_import_mwh",))
            self.assertEqual(len(app.session_state["last_forecast"][1]), 13)

    def test_hourly_weather_limit_runs_forecast_with_a_persistent_notice(self):
        limit = requests.Response()
        limit.status_code = 429
        error = requests.HTTPError("Hourly API request limit exceeded.", response=limit)
        weeks = pd.date_range("2022-06-19", periods=13, freq="W-SUN")
        predictions = pd.DataFrame({"median": 50., "lower": 40., "upper": 60.,
                                    "interval_samples": 20}, index=weeks)
        fx = pd.DataFrame({"week": weeks, "pair": "EUR/NOK", "nok_per_unit": 11.0}).set_index(["week", "pair"])
        with patch("ind320.data.reservoirs.fetch_reservoir_history", return_value=reservoir_fixture()), patch(
            "ind320.ui.source_cache.exchange_weeks", return_value=fx
        ), patch(
            "ind320.ui.source_cache.weather_weeks", side_effect=error
        ) as download, patch("ind320.analysis.forecast.forecast_reservoir", return_value=predictions) as fit:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("pages/2_Forecast.py").run()
            app.multiselect[0].set_value(["temperature_2m", "precipitation", "EUR/NOK"])
            app.button[0].click().run()
            self.assertFalse(app.error, [item.value for item in app.error])
            self.assertFalse(app.exception)
            self.assertEqual(fit.call_args.kwargs["selected_inputs"], ("EUR/NOK",))
            self.assertTrue(any("Weekly mean temperature" in item.value for item in app.warning))
            app.run()
            self.assertTrue(any("Weekly precipitation total" in item.value for item in app.warning))
            download.assert_called_once()
            download.side_effect = None
            download.return_value = pd.DataFrame(index=weeks)
            app.button[0].click().run()
            self.assertFalse(app.warning)
            self.assertEqual(fit.call_args.kwargs["selected_inputs"],
                             ("temperature_2m", "precipitation", "EUR/NOK"))

    def test_interval_width_reflects_errors_on_a_stable_seasonal_series(self):
        weeks = pd.date_range("2021-01-03", periods=156, freq="W-SUN")
        season = 0.55 + 0.12 * np.sin(np.arange(len(weeks)) * 2 * np.pi / 52)
        observed = reservoir_weeks(pd.DataFrame({
            "Observation Date": weeks, "Area Type": "EL", "Area Code": 3,
            "Fill Level": season + np.random.default_rng(0).normal(0, 0.01, len(weeks)),
        }))
        options = dict(area=("EL", 3), cutoff=cutoff_time("2024-01-05"),
                       training_weeks=156, horizon=13, order=(2, 1, 1))
        band = forecast_reservoir(observed, {}, **options)
        # This predictable series spans roughly 25 pp, unlike the old 71 pp interval.
        self.assertLess((band.upper - band.lower).iloc[-1], 35)
        point = forecast_reservoir(observed, {}, intervals=False, **options)
        pd.testing.assert_series_equal(band["median"], point["median"])

    def test_long_horizon_with_short_history_only_shows_supported_bounds(self):
        observations = reservoir_weeks(reservoir_fixture())
        forecast = forecast_reservoir(
            observations, {}, area=("EL", 3), cutoff=cutoff_time("2023-01-04"),
            training_weeks=78, horizon=26, order=(2, 1, 1),
        )
        self.assertFalse(forecast["median"].isna().any())
        self.assertTrue(np.isfinite(forecast.lower.iloc[0]))
        self.assertTrue(np.isfinite(forecast.upper.iloc[0]))
        self.assertEqual(forecast.interval_samples.iloc[-1], 1)
        self.assertTrue(pd.isna(forecast.lower.iloc[-1]))
        self.assertTrue(pd.isna(forecast.upper.iloc[-1]))

    def test_future_targets_and_weather_cannot_change_the_interval_forecast(self):
        rows = reservoir_weeks(reservoir_fixture())
        cutoff = cutoff_time("2022-12-14")
        weeks = pd.date_range("2019-12-22", "2023-07-02", freq="W-SUN")
        weather = pd.DataFrame({
            "temperature_2m": 10 + np.sin(np.arange(len(weeks)) / 5),
            "available_at": (weeks + pd.Timedelta(days=6)).tz_localize(OSLO),
        }, index=weeks)
        options = dict(area=("EL", 3), cutoff=cutoff, training_weeks=104,
                       horizon=13, order=(2, 1, 1), selected_inputs=("temperature_2m",))
        before = forecast_reservoir(rows, {"temperature_2m": weather}, **options)
        rows.loc[rows.available_at.gt(cutoff), "Fill Level"] = 500
        weather.loc[weather.available_at.gt(cutoff), "temperature_2m"] = 500
        after = forecast_reservoir(rows, {"temperature_2m": weather}, **options)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(len(after), 13)
        self.assertTrue((after.lower <= after["median"]).all())
        self.assertTrue((after["median"] <= after.upper).all())
        self.assertEqual(after.attrs["training_end"], pd.Timestamp("2022-12-11"))

    def test_map_and_forecast_share_dates_but_horizon_is_forecast_only(self):
        with patch("ind320.data.reservoirs.fetch_reservoir_history", return_value=reservoir_fixture()):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.date_input(key="overview_start").set_value(date(2020, 2, 1)).run()
            app.date_input(key="overview_end").set_value(date(2023, 1, 15)).run()
            self.assertFalse(any(slider.label == "Forecast horizon (weeks)" for slider in app.slider))
            app.switch_page("pages/2_Forecast.py").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.date_input(key="forecast_start").value, date(2020, 2, 1))
            self.assertEqual(app.date_input(key="forecast_end").value, date(2023, 1, 15))
            app.slider(key="forecast_horizon").set_value(9).run()
            app.date_input(key="forecast_end").set_value(date(2023, 1, 29)).run()
            app.switch_page("pages/0_Home.py").run()
            self.assertEqual(app.date_input(key="overview_end").value, date(2023, 1, 29))
            self.assertFalse(any(slider.label == "Forecast horizon (weeks)" for slider in app.slider))
            app.switch_page("pages/2_Forecast.py").run()
            self.assertEqual(app.slider(key="forecast_horizon").value, 9)


    def test_polygon_selection_updates_the_saved_area_and_outline(self):
        with patch("ind320.data.reservoirs.fetch_reservoir_history", return_value=reservoir_fixture()):
            app = AppTest.from_file(str(ROOT / "app.py")).run()
            chart = app.get("deck_gl_json_chart")[0].proto
            layer = json.loads(chart.json)["layers"][0]
            self.assertEqual(layer["lineWidthUnits"], "pixels")
            self.assertEqual(layer["lineWidthMaxPixels"], 1.25)
            feature = layer["data"]["features"][0]
            properties = feature["properties"]
            self.assertEqual((properties["area_type"], properties["area_code"]), ("EL", 1))
            self.assertEqual(properties["thickness"], 0.5)
            states = WidgetStates()
            states.widgets.add(id=chart.id, string_value=json.dumps({
                "selection": {"indices": {"regions": [0]}, "objects": {"regions": [feature]}},
            }))
            # AppTest has no PyDeck click helper, so submit the browser's widget event.
            app._run(states)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="overview_area").value, ("EL", 1))
            current = app.get("deck_gl_json_chart")[0].proto
            selected = json.loads(current.json)["layers"][0]["data"]["features"][0]
            self.assertEqual(selected["properties"]["thickness"], 1.25)

            app.selectbox(key="overview_area").set_value(("NO", 0)).run()
            national = json.loads(app.get("deck_gl_json_chart")[0].proto.json)["layers"][0]
            norway = national["data"]["features"][0]["properties"]
            self.assertEqual(norway["area_type"], "NO")
            self.assertEqual(norway["thickness"], 1.25)

    def test_forecast_fits_river_snarimax_with_selected_fx_inputs(self):
        def rates(start, end):
            dates = pd.date_range(start, end, freq="B")
            return pd.concat([pd.DataFrame({
                "date": dates, "pair": pair,
                "nok_per_unit": 11 + np.sin(np.arange(len(dates)) / (7 + offset)),
            }) for offset, pair in enumerate(("EUR/NOK", "USD/NOK"))], ignore_index=True)

        with patch("ind320.data.reservoirs.fetch_reservoir_history", return_value=reservoir_fixture()), patch(
            "ind320.ui.source_cache.fetch_exchange_rates", side_effect=rates
        ), patch(
            "ind320.analysis.forecast.time_series.SNARIMAX", wraps=SNARIMAX
        ) as fit_model, patch.object(
            SNARIMAX, "forecast", autospec=True, side_effect=SNARIMAX.forecast
        ) as predict:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
            app.switch_page("pages/2_Forecast.py").run()
            app.multiselect[0].set_value(["EUR/NOK", "USD/NOK"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error, [item.value for item in app.error])
            model_settings = fit_model.call_args_list[0].kwargs
            self.assertEqual((model_settings["sp"], model_settings["m"]), (1, 52))
            future_inputs = predict.call_args.kwargs["xs"]
            self.assertEqual(len(future_inputs), 13)
            self.assertTrue(all({"EUR/NOK", "USD/NOK"} <= features.keys() for features in future_inputs))
            result = app.session_state["last_forecast"][1]
            self.assertEqual(len(result), 13)
            self.assertTrue((result.lower <= result["median"]).all())
