"""Tests for measurement — the DJI Thermal SDK parameters and where each one comes from."""
import io
import math
from datetime import datetime

from PIL import Image

from thermographic_report_builder.models.inspection import AmbientConditions, InspectionDetails, SatelliteConditions
from thermographic_report_builder.processing.measurement import dew_point_c, measurement_parameters, sky_temperature_c
from thermographic_report_builder.processing.thermal_extractor import frame_distance

COND = AmbientConditions(dataset="ecmwf_ifs", hour_start=datetime(2023, 6, 9, 11), air_temperature_c=26.0,
                         relative_humidity_pct=51, cloud_cover_pct=90)
BASE = dict(emissivity=0.85, default_reflected_c=20.0, default_humidity_pct=50.0, object_distance_m=25.0, distance_from_frames=True)


def _rh_for(air_c, dew_c):
    a, b = 17.62, 243.12
    return 100 * math.exp(a * dew_c / (b + dew_c) - a * air_c / (b + air_c))


class TestSkyTemperature:

    def test_dew_point(self):
        assert abs(dew_point_c(25.0, 50.0) - 13.9) < 0.2

    def test_energyplus_reference_case(self):
        # EnergyPlus engineering reference: dry bulb 293 K, dew point 283 K, clear sky -> emissivity 0.815
        air, dew = 293 - 273.15, 283 - 273.15
        expected = 0.815 ** 0.25 * 293 - 273.15
        assert abs(sky_temperature_c(air, _rh_for(air, dew), 0) - expected) < 0.3

    def test_clouds_warm_the_sky(self):
        clear = sky_temperature_c(26.0, 51.0, 0)
        overcast = sky_temperature_c(26.0, 51.0, 8)
        assert 10 < clear < 14 and 21 < overcast < 25


class TestMeasurementParameters:

    def test_measured_reflected_temperature_wins(self):
        m = measurement_parameters(**BASE, inspection=InspectionDetails(reflected_temp_c=-18.0), conditions=COND, satellite=None)
        assert (m.reflected_temp_c, m.reflected_source) == (-18.0, "measured")
        assert (m.relative_humidity_pct, m.humidity_source) == (51, "dataset")

    def test_sky_estimate_uses_the_satellite_clouds(self):
        clear_sat = SatelliteConditions(satellite="GOES-16", cloud_fraction=0.0, clear_sky_wm2=888)
        with_sat = measurement_parameters(**BASE, inspection=None, conditions=COND, satellite=clear_sat)
        model_only = measurement_parameters(**BASE, inspection=None, conditions=COND, satellite=None)
        assert with_sat.reflected_source == "sky_estimate"
        assert with_sat.reflected_temp_c < model_only.reflected_temp_c  # 0/8 observed vs 7/8 modelled

    def test_defaults_without_data(self):
        m = measurement_parameters(**BASE, inspection=None, conditions=None, satellite=None)
        assert (m.reflected_temp_c, m.reflected_source, m.relative_humidity_pct, m.humidity_source) == (20.0, "default", 50.0, "default")

    def test_humidity_is_clamped_to_the_sdk_range(self):
        dry = COND.model_copy(update={"relative_humidity_pct": 12})
        assert measurement_parameters(**BASE, inspection=None, conditions=dry, satellite=None).relative_humidity_pct == 20.0


def _frame(tmp_path, xmp_fields):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "gray").save(buf, "JPEG")
    data = buf.getvalue()
    xmp = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:Description " + xmp_fields + b"/></x:xmpmeta>"
    path = tmp_path / "f.JPG"
    path.write_bytes(data[:2] + b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp + data[2:])
    return path


class TestFrameDistance:

    def test_uses_the_laser_rangefinder(self, tmp_path):
        assert frame_distance(_frame(tmp_path, b'drone-dji:LRFStatus="Normal" drone-dji:LRFTargetDistance="24.6"')) == 24.6

    def test_ignores_an_abnormal_laser_reading(self, tmp_path):
        path = _frame(tmp_path, b'drone-dji:LRFStatus="TooClose" drone-dji:LRFTargetDistance="0.3" drone-dji:RelativeAltitude="+30.1"')
        assert frame_distance(path) != 0.3
