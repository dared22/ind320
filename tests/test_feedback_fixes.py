"""Regression checks for the Part 3 feedback, without external services."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ind320-mpl")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ind320.analysis.weather import precipitation_analysis
from ind320.data.weather import daily_weather, fetch_weather, load_weather_coordinates

ROOT = Path(__file__).resolve().parents[1]


def sampled_points():
    """Two cells with deliberately unequal sample weights expose equal-cell bugs."""
    return pd.DataFrame({
        "area": ["NO3"] * 1000,
        "latitude": [60.01] * 900 + [61.01] * 100,
        "longitude": [10.01] * 900 + [11.01] * 100,
    }).set_index("area")


def weather_response(_url, *, params, timeout):
    """Represent the API's multi-location response, including request-order IDs."""
    times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h")
    records = []
    for i, (lat, lon) in enumerate(zip(params["latitude"].split(','), params["longitude"].split(','))):
        rainfall = float(lat) - 59
        records.append({
            "location_id": i, "latitude": float(lat), "longitude": float(lon),
            "hourly": {"time": times.strftime('%Y-%m-%dT%H:%M').tolist(),
                       "precipitation": [rainfall] * len(times),
                       "temperature_2m": [10 * rainfall] * len(times)},
        })
    # Reverse the response to ensure location_id, not arrival order, sets weights.
    return Mock(json=Mock(return_value=records[::-1] if len(records) > 1 else records[0]))


class WeatherSamplingTests(unittest.TestCase):
    def test_all_sampled_points_are_committed(self):
        points = load_weather_coordinates()
        self.assertEqual(len(points), 8000)
        self.assertTrue(points.groupby(level=0).size().eq(1000).all())
        self.assertFalse((ROOT / 'data/weather_coordinates.csv').exists())

    def test_sample_weights_batching_and_cache(self):
        with tempfile.TemporaryDirectory() as cache, patch(
            'ind320.data.weather.load_weather_coordinates', return_value=sampled_points()
        ), patch('ind320.data.weather.requests.get', side_effect=weather_response) as get:
            result = fetch_weather('NO3', '2024-02-29', '2024-02-29', cache_dir=Path(cache))
            np.testing.assert_allclose(result.precipitation, 1.1)
            np.testing.assert_allclose(result.temperature_2m, 11)
            self.assertEqual(result.attrs['sampled_points'], 1000)
            self.assertEqual(result.attrs['grid_cells'], 2)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args.kwargs['params']['cell_selection'], 'nearest')
            self.assertEqual(get.call_args.kwargs['params']['elevation'], 'nan')
            cached = fetch_weather('NO3', '2024-02-29', '2024-02-29', cache_dir=Path(cache))
            pd.testing.assert_frame_equal(cached, result)
            self.assertEqual(get.call_count, 1)
            self.assertAlmostEqual(daily_weather(result).precipitation.iloc[0], 1.1)
        # One-location responses and multiple batches must produce the same mean.
        with tempfile.TemporaryDirectory() as cache, patch(
            'ind320.data.weather.load_weather_coordinates', return_value=sampled_points()
        ), patch('ind320.data.weather.requests.get', side_effect=weather_response) as get:
            result = fetch_weather('NO3', '2024-02-29', '2024-02-29', batch_size=1, cache_dir=Path(cache))
            np.testing.assert_allclose(result.precipitation, 1.1)
            self.assertEqual(get.call_count, 2)

    def test_incomplete_cell_cannot_be_hidden_by_spatial_averaging(self):
        def incomplete(*args, **kwargs):
            response = weather_response(*args, **kwargs)
            response.json.return_value[0]['hourly']['precipitation'][0] = None
            return response
        with tempfile.TemporaryDirectory() as cache, patch(
            'ind320.data.weather.load_weather_coordinates', return_value=sampled_points()
        ), patch('ind320.data.weather.requests.get', side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, 'complete, finite hourly timeline'):
                fetch_weather('NO3', '2024-01-01', '2024-01-01', cache_dir=Path(cache))

    def test_failed_batch_resumes_cached_cells(self):
        with tempfile.TemporaryDirectory() as cache, patch(
            'ind320.data.weather.load_weather_coordinates', return_value=sampled_points()
        ):
            def fail_second(*args, **kwargs):
                if kwargs['params']['latitude'] == '61.0':
                    raise ValueError('simulated interruption')
                return weather_response(*args, **kwargs)
            with patch('ind320.data.weather.requests.get', side_effect=fail_second):
                with self.assertRaisesRegex(ValueError, 'simulated interruption'):
                    fetch_weather('NO3', '2024-01-01', '2024-01-01', batch_size=1, cache_dir=Path(cache))
            with patch('ind320.data.weather.requests.get', side_effect=weather_response) as get:
                result = fetch_weather('NO3', '2024-01-01', '2024-01-01', cache_dir=Path(cache))
                self.assertEqual(get.call_count, 1)
                self.assertEqual(get.call_args.kwargs['params']['latitude'], '61.0')
                np.testing.assert_allclose(result.precipitation, 1.1)

    def test_missing_hour_rejected(self):
        def incomplete(*args, **kwargs):
            response = weather_response(*args, **kwargs)
            for values in response.json.return_value[0]['hourly'].values():
                values.pop()
            return response
        with tempfile.TemporaryDirectory() as cache, patch(
            'ind320.data.weather.load_weather_coordinates', return_value=sampled_points()
        ), patch('ind320.data.weather.requests.get', side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, 'complete, finite hourly timeline'):
                fetch_weather('NO3', '2024-01-01', '2024-01-01', cache_dir=Path(cache))


class PrecipitationPlotTests(unittest.TestCase):
    def test_only_original_precipitation_is_plotted_with_residual_based_flags(self):
        values = pd.Series(np.r_[np.zeros(100), 20., np.zeros(100)],
                           index=pd.date_range('2024-01-01', periods=201))
        data, summary, figure = precipitation_analysis(values)
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].lines), 1)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), values)
        np.testing.assert_array_equal(data.outlier, (values < data.lower) | (values > data.upper))
        self.assertGreater(summary['outliers'], 0)


def reservoir_fixture():
    """Enough full weeks in each area to exercise STL and spectrogram controls."""
    dates = pd.date_range('2020-01-05', periods=180, freq='7D')
    areas = [('EL', i) for i in range(1, 6)] + [('VASS', i) for i in range(1, 4)] + [('NO', 0)]
    return pd.DataFrame([
        {'Observation Date': date, 'Area Type': kind, 'Area Code': code,
         'Fill Level': (.5 + .1 * np.sin(2 * np.pi * i / 52)
                        + .02 * np.sin(2 * np.pi * i / 17) + .000001 * i ** 2)}
        for kind, code in areas for i, date in enumerate(dates)
    ])


def area_weather_fixture(area, start, end):
    """Keep page tests focused on navigation rather than network access."""
    dates = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq='h', tz='UTC')
    result = pd.DataFrame({'precipitation': np.maximum(0, np.sin(np.arange(len(dates)) / 50)),
                           'temperature_2m': 5 + np.sin(np.arange(len(dates)) / 100)}, index=dates)
    result.attrs.update(area=area, sampled_points=1000, grid_cells=100,
                        model='ERA5', spatial_method='sample-weighted nearest-grid mean')
    return result


class SharedAreaPageTests(unittest.TestCase):
    def test_actual_pages_follow_area_in_both_directions(self):
        import streamlit as st
        st.cache_data.clear()
        with patch('ind320.data.reservoirs.fetch_reservoir_history', return_value=reservoir_fixture()), patch(
            'ind320.data.weather.fetch_weather', side_effect=area_weather_fixture
        ):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.switch_page('pages/6_Weather.py').run()
            self.assertFalse(app.exception)
            app.selectbox(key='weather_area').set_value(('EL', 4)).run()
            app.switch_page('pages/5_Reservoir_analysis.py').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='analysis_reservoir_area').value, ('EL', 4))
            # The suggested video demonstration must produce a visible change.
            before = [c.value for c in app.caption if 'Frequency resolution:' in c.value][0]
            next(s for s in app.slider if s.label == 'Window length (weeks)').set_value(156).run()
            self.assertFalse(app.exception)
            after = [c.value for c in app.caption if 'Frequency resolution:' in c.value][0]
            self.assertNotEqual(before, after)
            # Changes made by reservoir analysis must restore on an already visited page.
            app.selectbox(key='analysis_reservoir_area').set_value(('VASS', 2)).run()
            app.switch_page('pages/6_Weather.py').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='weather_area').value, ('VASS', 2))
            app.switch_page('pages/1_Data.py').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='reservoir_area').value, ('VASS', 2))
            app.selectbox(key='reservoir_area').set_value(('EL', 1)).run()
            app.switch_page('pages/2_Plots.py').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='reservoir_area').value, ('EL', 1))
            app.switch_page('pages/6_Weather.py').run()
            self.assertEqual(app.selectbox(key='weather_area').value, ('EL', 1))
            # National weather is unavailable: keep the selection and explain it.
            app.selectbox(key='weather_area').set_value(('NO', 0)).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('national reservoir selection is retained' in x.value for x in app.info))
            app.switch_page('pages/5_Reservoir_analysis.py').run()
            self.assertEqual(app.selectbox(key='analysis_reservoir_area').value, ('NO', 0))


if __name__ == '__main__':
    unittest.main()
