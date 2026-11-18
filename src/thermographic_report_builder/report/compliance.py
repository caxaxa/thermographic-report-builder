"""What the report must declare against IEC TS 62446-3, and what to do about each finding.

The TS (5.1) asks the report to note every known deviation or limitation of the procedure.
list_deviations() returns them as (text key, values) pairs; the builder localizes them.
"""

from datetime import timedelta
from typing import Optional

from ..models.inspection import (
    AmbientConditions,
    FlightMetadata,
    InspectionDetails,
    MeasurementParameters,
    SatelliteConditions,
)

# Ambient limits for thermography of operating modules (IEC TS 62446-3).
TS_MIN_IRRADIANCE_WM2 = 600
TS_MAX_WIND_KMH = 28
TS_MAX_CLOUD_OKTA = 2
# Table 1 (j): the camera shall be traceably calibrated at least every two years.
CALIBRATION_INTERVAL = timedelta(days=730)

URGENT_SEVERITIES = {"CRITICAL", "HIGH"}

Deviation = tuple[str, dict]


def list_deviations(
    inspection: Optional[InspectionDetails],
    flight: Optional[FlightMetadata],
    conditions: Optional[AmbientConditions],
    measurement: Optional[MeasurementParameters],
    satellite: Optional[SatelliteConditions] = None,
) -> list[Deviation]:
    """Every deviation or limitation known for this inspection, in report order."""
    details = inspection or InspectionDetails()
    flight = flight or FlightMetadata()
    out: list[Deviation] = []

    if not details.pilot_name:
        out.append(("no_pilot", {}))
    elif not details.pilot_qualification:
        out.append(("no_pilot_qualification", {}))
    if not (details.engineer_name and details.engineer_crea):
        out.append(("no_engineer", {}))
    if not details.plant_address:
        out.append(("no_address", {}))
    if not flight.start:
        out.append(("no_flight_time", {}))
    if not flight.camera_model:
        out.append(("no_camera_model", {}))

    if not (details.calibration_certificate and details.calibration_date):
        out.append(("no_calibration", {}))
    elif flight.start and flight.start.date() - details.calibration_date > CALIBRATION_INTERVAL:
        out.append(("calibration_expired", {"date": details.calibration_date}))

    irradiance, source = irradiance_reading(details, satellite, conditions)
    if source == "satellite":
        out.append(("irradiance_satellite", {"satellite": satellite.satellite}))
    elif source == "estimated":
        out.append(("irradiance_estimated", {"dataset": conditions.dataset}))
    elif source is None:
        out.append(("irradiance_unavailable", {}))
    if irradiance is not None and irradiance < TS_MIN_IRRADIANCE_WM2:
        out.append(("irradiance_low", {"value": irradiance, "kind": source}))

    okta, cloud_source = cloud_reading(satellite, conditions)
    if conditions is None:
        out.append(("weather_unavailable", {}))
    else:
        out.append(("weather_estimated", {"dataset": conditions.dataset, "with_cloud": cloud_source == "estimated"}))
    if conditions and conditions.wind_speed_kmh is not None and conditions.wind_speed_kmh > TS_MAX_WIND_KMH:
        out.append(("wind_high", {"value": conditions.wind_speed_kmh}))
    if okta is not None and okta > TS_MAX_CLOUD_OKTA:
        out.append(("cloud_high", {"okta": okta, "kind": cloud_source}))

    if measurement is not None:
        if measurement.reflected_source == "sky_estimate":
            out.append(("reflected_estimated", {"value": measurement.reflected_temp_c}))
        elif measurement.reflected_source == "default":
            out.append(("reflected_default", {"value": measurement.reflected_temp_c}))
        if measurement.humidity_source == "default":
            out.append(("humidity_default", {"value": measurement.relative_humidity_pct}))
    # Not yet supported by the platform; declared until it is.
    out.append(("no_visual_photos", {}))
    out.append(("no_current_reading", {}))
    out.append(("coverage_not_verified", {}))
    return out


def irradiance_reading(
    details: InspectionDetails,
    satellite: Optional[SatelliteConditions],
    conditions: Optional[AmbientConditions],
) -> tuple[Optional[float], Optional[str]]:
    """Best available irradiance: measured on site, then satellite, then the weather model."""
    if details.onsite_irradiance_wm2 is not None:
        return details.onsite_irradiance_wm2, "measured"
    if satellite and satellite.irradiance_wm2 is not None:
        return satellite.irradiance_wm2, "satellite"
    if conditions and conditions.irradiance_wm2 is not None:
        return conditions.irradiance_wm2, "estimated"
    return None, None


def cloud_reading(
    satellite: Optional[SatelliteConditions],
    conditions: Optional[AmbientConditions],
) -> tuple[Optional[int], Optional[str]]:
    """Cloud cover in okta: the satellite's cloud mask first, then the weather model."""
    if satellite and satellite.cloud_okta is not None:
        return satellite.cloud_okta, "satellite"
    if conditions and conditions.cloud_cover_okta is not None:
        return conditions.cloud_cover_okta, "estimated"
    return None, None


def guidance_keys(defect_type: str, severity: Optional[str]) -> tuple[str, str, str]:
    """(likely-cause key, recommended-action key, short-action key) for a finding.

    Accepts both spellings in use: panel attributes (faulty_diodes) and defect types (faultydiodes).
    """
    kind = defect_type.replace("_", "")
    if kind == "hotspots":
        level = "urgent" if severity in URGENT_SEVERITIES else "routine"
        return "hotspot_cause", f"hotspot_action_{level}", f"hotspot_short_{level}"
    if kind == "faultydiodes":
        return "diode_cause", "diode_action", "diode_short"
    return "offline_cause", "offline_action", "offline_short"
