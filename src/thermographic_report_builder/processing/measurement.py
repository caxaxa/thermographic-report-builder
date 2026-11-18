"""Object parameters for the DJI Thermal SDK, from what this inspection measured or observed.

- Emissivity: PV module glass in the camera's 8-14 µm band, about 0.85 at the near-normal viewing
  angles of a drone looking down (literature 0.85-0.90; it falls only at grazing angles).
- Reflected apparent temperature: the sky mirrored in the glass. Measured on site when the client
  enters it; otherwise the effective sky temperature from air temperature, dew point and cloud cover
  (Clark & Allen clear-sky emissivity with Walton's cloud correction, as in EnergyPlus).
- Relative humidity: from the weather dataset when available.
"""

import math
from typing import Optional

from ..models.inspection import AmbientConditions, InspectionDetails, MeasurementParameters, SatelliteConditions

# Input ranges the DJI Thermal SDK accepts
SDK_HUMIDITY = (20.0, 100.0)
SDK_REFLECTED = (-40.0, 500.0)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def dew_point_c(air_c: float, relative_humidity_pct: float) -> float:
    """Dew point from air temperature and humidity (Magnus formula)."""
    a, b = 17.62, 243.12
    gamma = math.log(max(relative_humidity_pct, 1.0) / 100.0) + a * air_c / (b + air_c)
    return b * gamma / (a - gamma)


def sky_temperature_c(air_c: float, relative_humidity_pct: float, cloud_okta: Optional[int]) -> float:
    """Effective sky temperature (EnergyPlus: Clark & Allen clear sky, Walton cloud correction)."""
    dew_point_k = dew_point_c(air_c, relative_humidity_pct) + 273.15
    clear = 0.787 + 0.764 * math.log(dew_point_k / 273.0)
    tenths = (cloud_okta or 0) * 10.0 / 8.0
    emissivity = min(1.0, clear * (1 + 0.0224 * tenths - 0.0035 * tenths ** 2 + 0.00028 * tenths ** 3))
    return emissivity ** 0.25 * (air_c + 273.15) - 273.15


def measurement_parameters(
    *,
    emissivity: float,
    default_reflected_c: float,
    default_humidity_pct: float,
    object_distance_m: float,
    distance_from_frames: bool,
    inspection: Optional[InspectionDetails],
    conditions: Optional[AmbientConditions],
    satellite: Optional[SatelliteConditions],
) -> MeasurementParameters:
    """The parameters every temperature in the report is computed with, and where each came from."""
    humidity, humidity_source = default_humidity_pct, "default"
    if conditions and conditions.relative_humidity_pct is not None:
        humidity, humidity_source = _clamp(conditions.relative_humidity_pct, SDK_HUMIDITY), "dataset"

    if inspection and inspection.reflected_temp_c is not None:
        reflected, reflected_source = inspection.reflected_temp_c, "measured"
    elif conditions and conditions.air_temperature_c is not None and conditions.relative_humidity_pct is not None:
        if satellite and satellite.cloud_okta is not None:
            okta = satellite.cloud_okta
        else:
            okta = conditions.cloud_cover_okta
        reflected = sky_temperature_c(conditions.air_temperature_c, conditions.relative_humidity_pct, okta)
        reflected_source = "sky_estimate"
    else:
        reflected, reflected_source = default_reflected_c, "default"

    return MeasurementParameters(
        emissivity=emissivity,
        reflected_temp_c=round(_clamp(reflected, SDK_REFLECTED), 1),
        reflected_source=reflected_source,
        relative_humidity_pct=round(humidity, 1),
        humidity_source=humidity_source,
        object_distance_m=object_distance_m,
        distance_from_altitude=distance_from_frames,
    )
