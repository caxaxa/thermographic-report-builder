"""Tests for goes_irradiance — GOES-East irradiance, cloud mask and the clear-sky reference."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from thermographic_report_builder.io import goes_irradiance as goes


def _key(product, when):
    return f"{product}/{when:%Y}/{when.timetuple().tm_yday:03d}/{when:%H}/OR_{product}-M6_G19_s{when:%Y}{when.timetuple().tm_yday:03d}{when:%H%M%S}2_e0_c0.nc"


class TestScanSelection:

    def test_scan_start_from_file_name(self):
        assert goes._scan_start("OR_ABI-L2-DSRF-M6_G19_s20263211500213_e2026_c2026.nc") == datetime(2026, 11, 17, 15, 0, 21)

    def test_picks_scans_during_the_flight(self):
        scans = [(datetime(2026, 11, 10, 14, m), f"k{m}") for m in range(0, 60, 10)]
        picked = goes._pick(scans, datetime(2026, 11, 10, 14, 28), datetime(2026, 11, 10, 14, 45), timedelta(minutes=10))
        assert [k for _, k in picked] == ["k20", "k30", "k40"]

    def test_caps_long_flights_at_three_scans(self):
        scans = [(datetime(2026, 11, 10, 14, 0) + timedelta(minutes=10 * i), f"k{i}") for i in range(12)]
        picked = goes._pick(scans, datetime(2026, 11, 10, 14, 0), datetime(2026, 11, 10, 15, 50), timedelta(minutes=10))
        assert [k for _, k in picked] == ["k0", "k6", "k11"]

    def test_hourly_product_uses_the_nearest_scan(self):
        scans = [(datetime(2023, 6, 9, 14, 0), "hourly")]
        picked = goes._pick(scans, datetime(2023, 6, 9, 14, 28), datetime(2023, 6, 9, 14, 45), timedelta(minutes=10))
        assert picked == scans

    def test_no_scan_close_enough(self):
        scans = [(datetime(2023, 6, 9, 12, 0), "far")]
        assert goes._pick(scans, datetime(2023, 6, 9, 14, 28), datetime(2023, 6, 9, 14, 45), timedelta(minutes=10)) == []


class TestSolarGeometry:

    def test_sun_overhead_at_equinox_noon(self):
        assert goes.solar_zenith_deg(0.0, 0.0, datetime(2023, 3, 20, 12, 7)) < 2.0

    def test_clear_sky_ghi(self):
        noon = goes.clear_sky_ghi(0.0, 0.0, datetime(2023, 3, 20, 12, 7))
        assert 1000 < noon < 1050
        assert goes.clear_sky_ghi(0.0, 0.0, datetime(2023, 3, 20, 0, 0)) == 0.0

    def test_area_27_flight(self):
        # Caldeirão Grande do Piauí, 9 Jun 2023, 11:36 local (UTC-3)
        value = goes.clear_sky_ghi(-7.35, -40.58, datetime(2023, 6, 9, 14, 36))
        assert 820 < value < 920


class TestLocalToUtc:

    def test_brazilian_timezones(self):
        pytest.importorskip("timezonefinder")
        assert goes.local_to_utc(datetime(2023, 6, 9, 11, 0), -20.47, -54.62) == datetime(2023, 6, 9, 15, 0)
        assert goes.local_to_utc(datetime(2023, 6, 9, 11, 0), -7.35, -40.58) == datetime(2023, 6, 9, 14, 0)


class _FakeS3:
    def __init__(self, keys):
        self.keys = keys
        self.downloads = []

    def get_paginator(self, name):
        keys = self.keys

        class _P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}

        return _P()

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key))
        open(filename, "wb").close()


class TestFetchSatelliteConditions:

    def test_averages_scans_and_cloud_mask(self, tmp_path, monkeypatch):
        monkeypatch.setattr(goes, "local_to_utc", lambda local, lat, lon: local + timedelta(hours=3))
        flight_utc = datetime(2026, 11, 10, 14, 0)
        keys = [_key("ABI-L2-DSRF", flight_utc + timedelta(minutes=m)) for m in (0, 10, 20)]
        keys += [_key("ABI-L2-ACMF", flight_utc + timedelta(minutes=m)) for m in (0, 10, 20)]
        fake = _FakeS3(keys)
        values = iter([900.0, 910.0, 920.0])

        def fake_read(path, variable, lat, lon, half):
            if variable == "DSR":
                return np.full((5, 5), next(values)), {"NC_GLOBAL#spatial_resolution": "2.0km at nadir"}
            if variable == "DQF":
                return np.zeros((5, 5)), {}
            mask = np.zeros((11, 11))
            mask[:2, :] = 1  # 22 of 121 pixels cloudy
            return mask, {}

        monkeypatch.setattr(goes, "read_window", fake_read)
        result = goes.fetch_satellite_conditions(-7.35, -40.58, datetime(2026, 11, 10, 11, 2), datetime(2026, 11, 10, 11, 19), tmp_path, s3=fake)
        assert result.satellite == "GOES-19"
        # The 14:20 UTC scan starts after the flight ended (14:19), so two scans are averaged
        assert result.irradiance_wm2 == 905.0
        assert result.irradiance_resolution == "2 km"
        assert result.irradiance_scans == [datetime(2026, 11, 10, 11, 0), datetime(2026, 11, 10, 11, 10)]
        assert result.cloud_okta == 1
        assert all(bucket == "noaa-goes19" for bucket, _ in fake.downloads)

    def test_flagged_pixels_are_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(goes, "local_to_utc", lambda local, lat, lon: local + timedelta(hours=3))
        flight_utc = datetime(2026, 11, 10, 14, 0)
        fake = _FakeS3([_key("ABI-L2-DSRF", flight_utc)])

        def fake_read(path, variable, lat, lon, half):
            if variable == "DSR":
                return np.array([[900.0, 100.0]]), {}
            return np.array([[0.0, 1.0]]), {}

        monkeypatch.setattr(goes, "read_window", fake_read)
        result = goes.fetch_satellite_conditions(0, 0, datetime(2026, 11, 10, 11, 0), datetime(2026, 11, 10, 11, 5), tmp_path, s3=fake)
        assert result.irradiance_wm2 == 900.0

    def test_no_scans_keeps_the_clear_sky_reference(self, tmp_path, monkeypatch):
        monkeypatch.setattr(goes, "local_to_utc", lambda local, lat, lon: local + timedelta(hours=3))
        result = goes.fetch_satellite_conditions(-7.35, -40.58, datetime(2026, 11, 10, 11, 0), datetime(2026, 11, 10, 11, 20), tmp_path, s3=_FakeS3([]))
        assert result.satellite is None and result.irradiance_wm2 is None
        assert result.clear_sky_wm2 > 800

    def test_unknown_timezone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(goes, "local_to_utc", lambda local, lat, lon: None)
        assert goes.fetch_satellite_conditions(0, 0, datetime(2026, 1, 1), datetime(2026, 1, 1), tmp_path, s3=_FakeS3([])) is None
