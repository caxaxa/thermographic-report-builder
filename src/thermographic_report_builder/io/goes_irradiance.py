"""Irradiance and cloud cover observed by NOAA's GOES-East satellite at the plant during the flight.

GOES-East covers all of Brazil. Every 10 minutes NOAA publishes, about 15 minutes after the scan,
the surface downward shortwave radiation (DSR; 2 km since 2024, hourly at 0.5° before) and a 2 km
clear-sky mask (ACM). Both are public domain, in NOAA's open-data buckets on AWS.

A satellite value is an estimate, not the on-site measurement IEC TS 62446-3 asks for, so the report
still lists it as a deviation; but it observes the actual clouds over the plant at flight time. The
clear-sky reference is what date, time and place alone give for a cloudless sky.
"""

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from ..models.inspection import SatelliteConditions
from ..utils.logger import get_logger

logger = get_logger(__name__)

# GOES-19 took over as GOES-East on 7 Apr 2025; GOES-16 holds the earlier archive.
BUCKETS = (("noaa-goes19", "GOES-19", datetime(2025, 4, 7)), ("noaa-goes16", "GOES-16", datetime(2017, 1, 1)))
DSR_PRODUCT = "ABI-L2-DSRF"
ACM_PRODUCT = "ABI-L2-ACMF"
MAX_SCANS = 3
DSR_HALF_WINDOW = 2   # 5 x 5 pixels: about 10 km at 2 km resolution
ACM_HALF_WINDOW = 5   # 11 x 11 pixels: the sky within about 10 km of the plant
_SCAN_START = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def local_to_utc(local: datetime, latitude: float, longitude: float) -> Optional[datetime]:
    """The drone clock is local time; GOES files are named in UTC."""
    try:
        from timezonefinder import TimezoneFinder

        name = TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
        if not name:
            return None
        return local.replace(tzinfo=ZoneInfo(name)).astimezone(timezone.utc).replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Could not find the plant's timezone: {e}")
        return None


def solar_zenith_deg(latitude: float, longitude: float, when_utc: datetime) -> float:
    """Solar zenith angle from the NOAA general solar position equations (about 0.5° accuracy)."""
    day = when_utc.timetuple().tm_yday
    hours = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    g = 2 * math.pi / 365 * (day - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    true_solar_minutes = hours * 60 + eqtime + 4 * longitude
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    lat = math.radians(latitude)
    cos_z = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))


def clear_sky_ghi(latitude: float, longitude: float, when_utc: datetime) -> float:
    """Clear-sky global horizontal irradiance (Haurwitz model): date, time and place only."""
    cos_z = math.cos(math.radians(solar_zenith_deg(latitude, longitude, when_utc)))
    if cos_z <= 0:
        return 0.0
    return 1098.0 * cos_z * math.exp(-0.057 / cos_z)


def _scan_start(key: str) -> Optional[datetime]:
    m = _SCAN_START.search(key)
    if not m:
        return None
    year, doy, hh, mm, ss = (int(x) for x in m.groups())
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def _list_scans(s3, bucket: str, product: str, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
    """(scan start, key) for every file of `product` in the hours spanning start..end, in UTC."""
    scans = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        prefix = f"{product}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                when = _scan_start(obj["Key"])
                if when:
                    scans.append((when, obj["Key"]))
        hour += timedelta(hours=1)
    return sorted(scans)


def _pick(scans: list[tuple[datetime, str]], start: datetime, end: datetime, cadence: timedelta) -> list[tuple[datetime, str]]:
    """Scans overlapping the flight, at most MAX_SCANS spread over it; else the nearest one."""
    inside = [s for s in scans if start - cadence < s[0] <= end]
    if not inside:
        middle = start + (end - start) / 2
        nearest = min(scans, key=lambda s: abs(s[0] - middle), default=None)
        return [nearest] if nearest and abs(nearest[0] - middle) <= timedelta(minutes=45) else []
    if len(inside) <= MAX_SCANS:
        return inside
    step = (len(inside) - 1) / (MAX_SCANS - 1)
    return [inside[round(i * step)] for i in range(MAX_SCANS)]


def read_window(path: str, variable: str, latitude: float, longitude: float, half: int) -> tuple[np.ndarray, dict]:
    """Physical values in a (2·half+1)² window around the plant, NaN where missing or flagged."""
    import rasterio
    from pyproj import Transformer

    source = f"netcdf:{path}:{variable}" if str(path).endswith(".nc") else str(path)
    with rasterio.open(source) as ds:
        tags = ds.tags()
        x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(longitude, latitude)
        col, row = ~ds.transform * (x, y)
        r, c = int(math.floor(row)), int(math.floor(col))
        h = 0 if ds.crs.is_geographic else half  # coarse lat/lon grids (0.5°): just the plant's cell
        window = rasterio.windows.Window(c - h, r - h, 2 * h + 1, 2 * h + 1)
        raw = ds.read(1, window=window, boundless=True, fill_value=-1).astype(float)
    fill = float(tags.get(f"{variable}#_FillValue", "nan"))
    raw[(raw == fill) | (raw < 0)] = np.nan
    values = raw * float(tags.get(f"{variable}#scale_factor", 1)) + float(tags.get(f"{variable}#add_offset", 0))
    return values, tags


def _resolution_label(tags: dict) -> Optional[str]:
    text = tags.get("NC_GLOBAL#spatial_resolution", "")
    m = re.search(r"([\d.]+)\s*km", text)
    if m:
        return f"{float(m.group(1)):g} km"
    m = re.search(r"([\d.]+)\s*deg", tags.get("DSR#resolution", "") or text)
    return f"{float(m.group(1)):g}°" if m else None


def fetch_satellite_conditions(
    latitude: float,
    longitude: float,
    start_local: datetime,
    end_local: datetime,
    work_dir: Path,
    s3=None,
) -> Optional[SatelliteConditions]:
    """GOES-East irradiance and cloud fraction at the plant during the flight.

    None when the plant's timezone is unknown; without scans, only the clear-sky reference is set.
    """
    start = local_to_utc(start_local, latitude, longitude)
    end = local_to_utc(end_local, latitude, longitude)
    if start is None or end is None:
        return None
    offset = start_local - start
    if s3 is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    work_dir.mkdir(parents=True, exist_ok=True)
    middle = start + (end - start) / 2
    clear_sky = round(clear_sky_ghi(latitude, longitude, middle), 1)

    for bucket, satellite, operational_from in BUCKETS:
        if start < operational_from:
            continue
        try:
            dsr = _pick(_list_scans(s3, bucket, DSR_PRODUCT, start - timedelta(hours=1), end), start, end, timedelta(minutes=10))
            acm = _pick(_list_scans(s3, bucket, ACM_PRODUCT, start - timedelta(minutes=10), end), start, end, timedelta(minutes=10))
            if not dsr and not acm:
                continue

            irradiance, resolution = [], None
            for when, key in dsr:
                local_path = work_dir / Path(key).name
                s3.download_file(bucket, key, str(local_path))
                values, tags = read_window(str(local_path), "DSR", latitude, longitude, DSR_HALF_WINDOW)
                quality, _ = read_window(str(local_path), "DQF", latitude, longitude, DSR_HALF_WINDOW)
                values[quality != 0] = np.nan
                if np.isfinite(values).any():
                    irradiance.append(float(np.nanmean(values)))
                resolution = resolution or _resolution_label(tags)
                local_path.unlink(missing_ok=True)

            cloudy = []
            for when, key in acm:
                local_path = work_dir / Path(key).name
                s3.download_file(bucket, key, str(local_path))
                mask, _ = read_window(str(local_path), "BCM", latitude, longitude, ACM_HALF_WINDOW)
                if np.isfinite(mask).any():
                    cloudy.append(float(np.nanmean(mask)))
                local_path.unlink(missing_ok=True)

            conditions = SatelliteConditions(
                satellite=satellite,
                irradiance_wm2=round(sum(irradiance) / len(irradiance), 1) if irradiance else None,
                irradiance_resolution=resolution,
                irradiance_scans=[when + offset for when, _ in dsr],
                cloud_fraction=round(sum(cloudy) / len(cloudy), 3) if cloudy else None,
                cloud_scans=[when + offset for when, _ in acm],
                clear_sky_wm2=clear_sky,
            )
            logger.info(f"Satellite conditions from {satellite}: {conditions.model_dump()}")
            return conditions
        except Exception as e:
            logger.warning(f"GOES data from {bucket} unavailable: {e}")
    # No satellite scan: the clear-sky reference still holds
    return SatelliteConditions(clear_sky_wm2=clear_sky)
