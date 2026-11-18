"""Tests for weather_client — ambient conditions from the Open-Meteo archive API."""
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from thermographic_report_builder.io.weather_client import fetch_ambient_conditions

BASE = "https://archive-api.open-meteo.com/v1/archive"


def _response(radiation=(677.0, 776.0, 914.0), temperature=(26.0, 28.0, 30.0)):
    """Three hourly rows, local time, as the archive API returns them."""
    return {
        "latitude": -20.49,
        "longitude": -54.65,
        "hourly": {
            "time": ["2026-11-10T10:00", "2026-11-10T11:00", "2026-11-10T12:00"],
            "shortwave_radiation": list(radiation),
            "temperature_2m": list(temperature),
            "relative_humidity_2m": [58, 56, 52],
            "wind_speed_10m": [15.0, 13.0, 10.0],
            "cloud_cover": [11, 4, 64],
        },
    }


class TestFetchAmbientConditions:

    def test_picks_the_flight_hour(self):
        calls = []

        def get_json(url, timeout):
            calls.append(url)
            return _response()

        cond = fetch_ambient_conditions(-20.47, -54.62, datetime(2026, 11, 10, 11, 30), base_url=BASE, get_json=get_json)
        # Radiation stamped 12:00 is the mean of 11:00-12:00, the hour of the flight
        assert cond.irradiance_wm2 == 914.0
        assert cond.hour_start == datetime(2026, 11, 10, 11, 0)
        # Instant variables are interpolated at 11:30
        assert cond.air_temperature_c == 29.0
        assert cond.wind_speed_kmh == 11.5
        assert cond.cloud_cover_pct == 34.0
        assert cond.cloud_cover_okta == 3
        assert cond.dataset == "ecmwf_ifs"
        query = parse_qs(urlparse(calls[0]).query)
        assert query["models"] == ["ecmwf_ifs"]
        assert query["timezone"] == ["auto"]
        assert "apikey" not in query

    def test_falls_back_to_era5(self):
        def get_json(url, timeout):
            if "ecmwf_ifs" in url:
                return _response(radiation=(None, None, None), temperature=(None, None, None))
            return _response()

        cond = fetch_ambient_conditions(-20.47, -54.62, datetime(2026, 11, 10, 11, 30), base_url=BASE, get_json=get_json)
        assert cond.dataset == "era5"

    def test_none_when_no_dataset_answers(self):
        def get_json(url, timeout):
            raise OSError("timed out")

        assert fetch_ambient_conditions(0, 0, datetime(2026, 11, 10, 11, 30), base_url=BASE, get_json=get_json) is None

    def test_sends_the_api_key(self):
        calls = []

        def get_json(url, timeout):
            calls.append(url)
            return _response()

        fetch_ambient_conditions(0, 0, datetime(2026, 11, 10, 11, 30), base_url=BASE, api_key="k123", get_json=get_json)
        assert parse_qs(urlparse(calls[0]).query)["apikey"] == ["k123"]
