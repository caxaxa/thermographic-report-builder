"""Tests for compliance — deviations from IEC TS 62446-3 and per-finding guidance."""
from datetime import date, datetime

from thermographic_report_builder.models.inspection import (
    AmbientConditions,
    FlightMetadata,
    InspectionDetails,
    MeasurementParameters,
    SatelliteConditions,
)
from thermographic_report_builder.report.compliance import guidance_keys, list_deviations

FLIGHT = FlightMetadata(start=datetime(2026, 11, 10, 11, 2), end=datetime(2026, 11, 10, 11, 19), frame_count=511, camera_model="M30T")
COMPLETE = InspectionDetails(
    pilot_name="Ana Souza",
    pilot_qualification="Termografia nível 1 (ISO 9712)",
    engineer_name="Carlos Lima",
    engineer_crea="MS-12345",
    plant_address="Rodovia MS-080, km 12",
    calibration_lab="Lab X",
    calibration_certificate="C-2026-001",
    calibration_date=date(2026, 3, 1),
)
GOOD = AmbientConditions(dataset="ecmwf_ifs", hour_start=datetime(2026, 11, 10, 11), irradiance_wm2=850, wind_speed_kmh=10, cloud_cover_pct=10)
PARAMS = MeasurementParameters(
    emissivity=0.85, reflected_temp_c=12.2, reflected_source="sky_estimate",
    relative_humidity_pct=51, humidity_source="dataset", object_distance_m=25,
)
ALWAYS = ["irradiance_estimated", "weather_estimated", "reflected_estimated", "no_visual_photos", "no_current_reading", "coverage_not_verified"]
SAT = SatelliteConditions(satellite="GOES-19", irradiance_wm2=905, irradiance_scans=[datetime(2026, 11, 10, 11, 0)], cloud_fraction=0.05, clear_sky_wm2=930)


def _keys(*args):
    return [key for key, _ in list_deviations(*args)]


class TestListDeviations:

    def test_nothing_provided(self):
        keys = _keys(None, None, None, None)
        assert keys[:6] == ["no_pilot", "no_engineer", "no_address", "no_flight_time", "no_camera_model", "no_calibration"]
        assert "irradiance_unavailable" in keys and "weather_unavailable" in keys

    def test_complete_record_leaves_only_platform_limits(self):
        assert _keys(COMPLETE, FLIGHT, GOOD, PARAMS) == ALWAYS

    def test_missing_qualification(self):
        details = COMPLETE.model_copy(update={"pilot_qualification": None})
        assert "no_pilot_qualification" in _keys(details, FLIGHT, GOOD, PARAMS)

    def test_calibration_older_than_two_years(self):
        details = COMPLETE.model_copy(update={"calibration_date": date(2024, 1, 15)})
        deviations = dict(list_deviations(details, FLIGHT, GOOD, PARAMS))
        assert deviations["calibration_expired"] == {"date": date(2024, 1, 15)}

    def test_ambient_limits(self):
        bad = GOOD.model_copy(update={"irradiance_wm2": 480, "wind_speed_kmh": 35, "cloud_cover_pct": 60})
        deviations = dict(list_deviations(COMPLETE, FLIGHT, bad, PARAMS))
        assert deviations["irradiance_low"] == {"value": 480, "kind": "estimated"}
        assert deviations["wind_high"] == {"value": 35}
        assert deviations["cloud_high"] == {"okta": 5, "kind": "estimated"}

    def test_onsite_irradiance_replaces_the_estimate(self):
        details = COMPLETE.model_copy(update={"onsite_irradiance_wm2": 590})
        deviations = dict(list_deviations(details, FLIGHT, GOOD, PARAMS))
        assert deviations["irradiance_low"] == {"value": 590, "kind": "measured"}
        assert "irradiance_estimated" not in deviations
        assert deviations["weather_estimated"]["with_cloud"] is True

    def test_satellite_beats_the_weather_model(self):
        cloudy = GOOD.model_copy(update={"irradiance_wm2": 480, "cloud_cover_pct": 60})
        deviations = dict(list_deviations(COMPLETE, FLIGHT, cloudy, PARAMS, SAT))
        assert deviations["irradiance_satellite"] == {"satellite": "GOES-19"}
        assert "irradiance_low" not in deviations and "cloud_high" not in deviations
        assert deviations["weather_estimated"]["with_cloud"] is False

    def test_satellite_limits(self):
        overcast = SAT.model_copy(update={"irradiance_wm2": 410, "cloud_fraction": 1.0})
        deviations = dict(list_deviations(COMPLETE, FLIGHT, GOOD, PARAMS, overcast))
        assert deviations["irradiance_low"] == {"value": 410, "kind": "satellite"}
        assert deviations["cloud_high"] == {"okta": 8, "kind": "satellite"}

    def test_clear_sky_reference_alone_is_not_a_reading(self):
        reference_only = SatelliteConditions(clear_sky_wm2=930)
        deviations = dict(list_deviations(COMPLETE, FLIGHT, GOOD, PARAMS, reference_only))
        assert "irradiance_estimated" in deviations

    def test_measured_reflected_temperature_is_no_deviation(self):
        measured = PARAMS.model_copy(update={"reflected_source": "measured", "reflected_temp_c": -15.0})
        keys = _keys(COMPLETE, FLIGHT, GOOD, measured)
        assert "reflected_estimated" not in keys and "reflected_default" not in keys

    def test_default_parameters_are_declared(self):
        defaults = MeasurementParameters(emissivity=0.85, reflected_temp_c=20, relative_humidity_pct=50, object_distance_m=25)
        deviations = dict(list_deviations(COMPLETE, FLIGHT, None, defaults))
        assert deviations["reflected_default"] == {"value": 20}
        assert deviations["humidity_default"] == {"value": 50}

    def test_blank_strings_count_as_missing(self):
        details = InspectionDetails(pilot_name="  ", engineer_name="", plant_address=" ")
        assert details.pilot_name is None and details.engineer_name is None and details.plant_address is None


class TestGuidanceKeys:

    def test_hotspot_by_severity(self):
        assert guidance_keys("hotspots", "CRITICAL") == ("hotspot_cause", "hotspot_action_urgent", "hotspot_short_urgent")
        assert guidance_keys("hotspots", "LOW") == ("hotspot_cause", "hotspot_action_routine", "hotspot_short_routine")
        assert guidance_keys("hotspots", None)[1] == "hotspot_action_routine"

    def test_other_types(self):
        assert guidance_keys("faulty_diodes", "HIGH")[0] == "diode_cause"
        assert guidance_keys("faultydiodes", None)[2] == "diode_short"
        assert guidance_keys("offlinepanels", None)[0] == "offline_cause"
        assert guidance_keys("offline_panels", None)[2] == "offline_short"
