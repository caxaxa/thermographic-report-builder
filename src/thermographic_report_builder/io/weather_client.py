"""Ambient conditions at the plant during the flight, from ECMWF data via the Open-Meteo archive API.

IEC TS 62446-3 asks for irradiance measured on site. When the client has no measurement, the report
prints these estimates, names the dataset and lists the gap as a deviation. Air temperature from a
meteorological service is what the TS itself defines (3.3).

Production use needs an Open-Meteo commercial API key (settings.weather_api_key); the keyless
endpoint is for non-commercial use.
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Callable, Optional

from ..models.inspection import AmbientConditions
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Tried in order: the ECMWF IFS analysis (9 km grid, published the next day), then the ERA5
# reanalysis (0.25° grid, about five days behind real time).
DATASETS = ("ecmwf_ifs", "era5")
HOURLY = ("shortwave_radiation", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover")


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "aisol-thermographic-report-builder"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _interpolate(times: list[datetime], values: list, when: datetime) -> Optional[float]:
    """Linear interpolation between the hourly values either side of `when`."""
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= when <= t1:
            v0, v1 = values[i], values[i + 1]
            if v0 is None or v1 is None:
                return v0 if v1 is None else v1
            share = (when - t0) / (t1 - t0)
            return v0 + (v1 - v0) * share
    return None


def fetch_ambient_conditions(
    latitude: float,
    longitude: float,
    when: datetime,
    *,
    base_url: str,
    api_key: str = "",
    timeout: float = 15.0,
    get_json: Callable[[str, float], dict] = _get_json,
) -> Optional[AmbientConditions]:
    """Conditions at (latitude, longitude) at local time `when`, or None if no dataset has them yet."""
    day = when.date()
    # Radiation is the mean of the PRECEDING hour, so the value stamped at the next full hour
    # covers the hour the flight midpoint falls in.
    hour_end = when.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for dataset in DATASETS:
        params = {
            "latitude": f"{latitude:.5f}",
            "longitude": f"{longitude:.5f}",
            "start_date": day.isoformat(),
            "end_date": (day + timedelta(days=1)).isoformat(),
            "hourly": ",".join(HOURLY),
            "models": dataset,
            "timezone": "auto",
            "wind_speed_unit": "kmh",
        }
        if api_key:
            params["apikey"] = api_key
        try:
            data = get_json(f"{base_url}?{urllib.parse.urlencode(params)}", timeout)
            hourly = data["hourly"]
            times = [datetime.fromisoformat(t) for t in hourly["time"]]
        except Exception as e:
            logger.warning(f"Weather dataset {dataset} unavailable: {e}")
            continue

        def at(name: str) -> Optional[float]:
            return _interpolate(times, hourly.get(name, []), when)

        irradiance = None
        if hour_end in times:
            irradiance = hourly.get("shortwave_radiation", [None] * len(times))[times.index(hour_end)]
        temperature = at("temperature_2m")
        if irradiance is None and temperature is None:
            logger.info(f"Weather dataset {dataset} has no data for {when} yet")
            continue
        conditions = AmbientConditions(
            dataset=dataset,
            hour_start=hour_end - timedelta(hours=1),
            irradiance_wm2=irradiance,
            air_temperature_c=temperature,
            relative_humidity_pct=at("relative_humidity_2m"),
            wind_speed_kmh=at("wind_speed_10m"),
            cloud_cover_pct=at("cloud_cover"),
            grid_latitude=data.get("latitude"),
            grid_longitude=data.get("longitude"),
        )
        logger.info(f"Ambient conditions from {dataset}: {conditions.model_dump()}")
        return conditions
    return None
