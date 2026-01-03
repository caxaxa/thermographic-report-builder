"""Helpers for converting orthophoto pixel coordinates to geographic latitude/longitude."""

from __future__ import annotations

import urllib.request
import urllib.parse
import json
from typing import Tuple, Any, Optional

from affine import Affine
from pyproj import CRS, Transformer

from .logger import get_logger

logger = get_logger(__name__)


def reverse_geocode(latitude: float, longitude: float) -> str:
    """
    Perform reverse geocoding to get a human-readable location from coordinates.

    Uses OpenStreetMap's Nominatim API for free reverse geocoding.
    Falls back to coordinate string if API call fails.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees

    Returns:
        Location string (e.g., "Campo Grande, MS, Brasil")
    """
    try:
        # Use Nominatim API for reverse geocoding
        url = (
            f"https://nominatim.openstreetmap.org/reverse?"
            f"lat={latitude}&lon={longitude}&format=json&accept-language=pt-BR"
        )

        headers = {
            'User-Agent': 'SolarReportBuilder/1.0 (thermographic-report-builder)',
        }

        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

            address = data.get('address', {})

            # Build location string from address components
            # Priority: city/town/village, state, country
            city = (
                address.get('city') or
                address.get('town') or
                address.get('village') or
                address.get('municipality') or
                address.get('county') or
                ''
            )
            state = address.get('state', '')
            country = address.get('country', '')

            # Format as "City, State, Country" (Brazilian style)
            parts = [p for p in [city, state, country] if p]
            location = ', '.join(parts)

            if location:
                logger.info(f"Reverse geocoded ({latitude:.4f}, {longitude:.4f}) -> {location}")
                return location

    except Exception as e:
        logger.warning(f"Reverse geocoding failed: {e}")

    # Fallback to formatted coordinates
    lat_dir = 'S' if latitude < 0 else 'N'
    lon_dir = 'W' if longitude < 0 else 'E'
    return f"{abs(latitude):.4f}°{lat_dir}, {abs(longitude):.4f}°{lon_dir}"


def get_orthophoto_center(
    transform: Affine,
    width: int,
    height: int,
    crs: Any,
) -> Tuple[float, float]:
    """
    Get the center coordinates (lat, lon) of an orthophoto.

    Args:
        transform: Rasterio affine transform
        width: Image width in pixels
        height: Image height in pixels
        crs: Coordinate reference system

    Returns:
        Tuple of (latitude, longitude) for the center point
    """
    # Calculate center pixel coordinates
    center_x = width / 2
    center_y = height / 2

    # Convert to source CRS coordinates
    lon, lat = transform * (center_x, center_y)

    # Convert to WGS84 if needed
    target_crs = CRS.from_epsg(4326)

    if crs is not None:
        try:
            source_crs = CRS.from_user_input(crs)
            if not source_crs.equals(target_crs):
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                lon, lat = transformer.transform(lon, lat)
        except Exception as e:
            logger.warning(f"CRS conversion failed, assuming WGS84: {e}")

    return lat, lon


class PixelToLatLonConverter:
    """
    Convert orthophoto pixel coordinates to WGS84 latitude/longitude.

    The orthophoto produced by ODM is typically in a projected CRS (e.g., UTM).
    Raw thermal images, however, store GPS coordinates in WGS84. This helper
    bridges the two spaces so that we can accurately match detections against
    the closest raw image captured by the drone.
    """

    def __init__(self, affine_transform: Affine, raster_crs: Any | None):
        """
        Args:
            affine_transform: Rasterio affine transform mapping pixels to source CRS
            raster_crs: CRS definition reported by the orthophoto
        """
        self._affine = affine_transform
        self.transform = affine_transform  # Public alias for camera_projector

        self._target_crs = CRS.from_epsg(4326)
        self._transformer: Transformer | None = None

        if raster_crs is None:
            self._source_crs = self._target_crs
            return

        try:
            self._source_crs = CRS.from_user_input(raster_crs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Could not parse orthophoto CRS %s, defaulting to EPSG:4326: %s",
                raster_crs,
                exc,
            )
            self._source_crs = self._target_crs
            return

        if not self._source_crs.equals(self._target_crs):
            self._transformer = Transformer.from_crs(
                self._source_crs, self._target_crs, always_xy=True
            )

    def pixel_to_lonlat(self, pixel: Tuple[float, float]) -> tuple[float, float]:
        """
        Convert (x, y) pixel coordinates to longitude/latitude.

        Args:
            pixel: Tuple of (column, row) pixel coordinates

        Returns:
            Tuple of (longitude, latitude) in WGS84
        """
        lon, lat = self._affine * pixel

        if self._transformer:
            lon, lat = self._transformer.transform(lon, lat)

        return lon, lat
