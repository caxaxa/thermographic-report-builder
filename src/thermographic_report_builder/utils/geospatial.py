"""Helpers for converting orthophoto pixel coordinates to geographic latitude/longitude."""

from __future__ import annotations

from typing import Tuple, Any

from affine import Affine
from pyproj import CRS, Transformer

from .logger import get_logger

logger = get_logger(__name__)


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
