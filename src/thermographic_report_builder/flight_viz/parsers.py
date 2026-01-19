"""Parsers for extracting data from ODM artifacts."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal
from pyproj import Transformer

from .models import CameraShot, FlightData

logger = logging.getLogger(__name__)


class ReconstructionParser:
    """Parse OpenSfM reconstruction.json to extract flight data."""

    def parse(self, json_path: Path) -> FlightData:
        """Parse reconstruction.json and return FlightData.

        Args:
            json_path: Path to reconstruction.json file

        Returns:
            FlightData object with camera shots and metadata

        Raises:
            FileNotFoundError: If reconstruction.json doesn't exist
            ValueError: If JSON is malformed or missing required fields
        """
        if not json_path.exists():
            raise FileNotFoundError(f"reconstruction.json not found at {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        # OpenSfM reconstruction.json is a list of reconstructions
        # We typically use the first (and usually only) reconstruction
        if isinstance(data, list):
            if not data:
                raise ValueError("reconstruction.json contains empty list")
            reconstruction = data[0]
        else:
            reconstruction = data

        # Extract reference coordinate system FIRST (needed for coordinate conversion)
        reference_lla = self._extract_reference_lla(reconstruction)

        # Parse camera shots
        shots = []
        shots_data = reconstruction.get("shots", {})

        if not shots_data:
            logger.warning("No shots found in reconstruction.json")

        for shot_id, shot_data in shots_data.items():
            try:
                shot = self._parse_shot(shot_id, shot_data, reference_lla)
                shots.append(shot)
            except Exception as e:
                logger.warning(f"Failed to parse shot {shot_id}: {e}")
                continue

        if not shots:
            raise ValueError("No valid shots could be parsed from reconstruction.json")

        # Extract camera calibration (use first camera)
        cameras = reconstruction.get("cameras", {})
        if not cameras:
            raise ValueError("No cameras found in reconstruction.json")

        camera_id = list(cameras.keys())[0]
        camera_data = cameras[camera_id]

        camera_model = camera_data.get("projection_type", "unknown")
        focal_length = camera_data.get("focal", 0.0)
        k1 = camera_data.get("k1", 0.0)
        k2 = camera_data.get("k2", 0.0)

        # Extract reconstruction metadata
        points = reconstruction.get("points", {})
        total_points = len(points) if points else None

        return FlightData(
            shots=shots,
            reference_lla=reference_lla,
            camera_model=camera_model,
            focal_length=focal_length,
            k1=k1,
            k2=k2,
            total_points=total_points,
            reprojection_error=None,  # Not directly available in reconstruction.json
        )

    def _parse_shot(self, shot_id: str, shot_data: Dict, reference_lla: Tuple[float, float, float]) -> CameraShot:
        """Parse a single camera shot.

        Args:
            shot_id: Image filename (e.g., "DJI_0001.JPG")
            shot_data: Shot data from reconstruction.json
            reference_lla: Reference (lat, lon, alt) for local coordinate system

        Returns:
            CameraShot object
        """
        # GPS position in OpenSfM is in local ENU (East-North-Up) coordinates
        # We need to convert to WGS84 lat/lon/alt
        gps_pos = shot_data.get("gps_position")
        if not gps_pos or len(gps_pos) < 3:
            raise ValueError(f"Invalid GPS position for shot {shot_id}")

        # ENU coordinates (meters from reference point)
        easting = float(gps_pos[0])  # East (meters)
        northing = float(gps_pos[1])  # North (meters)
        up = float(gps_pos[2])  # Up (meters)

        # Convert ENU to WGS84 using reference point
        latitude, longitude, altitude = self._enu_to_wgs84(
            easting, northing, up, reference_lla
        )

        # Parse rotation (can be rotation matrix or axis-angle)
        rotation = self._parse_rotation(shot_data.get("rotation", [0, 0, 0]))
        rotation_matrix = np.array(rotation)  # 3x3 matrix

        # Parse translation (transformation parameter, NOT camera position!)
        # In OpenSfM: camera_center = -R^T * t
        translation = shot_data.get("translation", [0.0, 0.0, 0.0])
        t = np.array(translation)

        # Compute actual camera center (optimized position in ENU coordinates)
        # This is the bundle-adjusted camera position
        camera_center = -rotation_matrix.T @ t

        # Store original GPS position in ENU for error calculation
        gps_position_enu = [easting, northing, up]

        # Calculate GPS prediction error (distance between raw GPS and optimized camera position)
        # This is the key quality metric that shows how well bundle adjustment corrected GPS drift
        gps_error = np.sqrt(
            (camera_center[0] - easting)**2 +
            (camera_center[1] - northing)**2 +
            (camera_center[2] - up)**2
        )

        # Parse optional GPS DOP (dilution of precision)
        gps_dop = shot_data.get("gps_dop")

        # Parse optional EXIF timestamp
        timestamp = None
        capture_time = shot_data.get("capture_time")
        if capture_time:
            try:
                # OpenSfM stores timestamps as Unix timestamps
                timestamp = datetime.fromtimestamp(capture_time)
            except (ValueError, TypeError):
                logger.debug(f"Could not parse timestamp for {shot_id}: {capture_time}")

        # GSD (Ground Sample Distance) - not directly in reconstruction.json
        # Will be computed later from altitude and focal length
        gsd = None

        return CameraShot(
            shot_id=shot_id,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            rotation=rotation,
            translation=translation,
            gps_position_enu=gps_position_enu,
            camera_center_enu=camera_center.tolist(),  # Bundle-adjusted position = -R^T * t
            timestamp=timestamp,
            gsd=gsd,
            gps_dop=gps_dop,
            gps_error=gps_error,
            n_features=None,  # Could be extracted from features.json if needed
            n_matches=None,   # Could be extracted from matches.json if needed
        )

    def _parse_rotation(self, rotation_data) -> List[List[float]]:
        """Parse rotation representation to 3x3 rotation matrix.

        OpenSfM can store rotations as:
        - 3-element axis-angle vector
        - 3x3 rotation matrix
        - Quaternion

        Args:
            rotation_data: Rotation data from JSON

        Returns:
            3x3 rotation matrix as list of lists
        """
        if not rotation_data:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Identity matrix

        # Convert to numpy for easier handling
        rot = np.array(rotation_data)

        if rot.shape == (3, 3):
            # Already a 3x3 matrix
            return rot.tolist()
        elif rot.shape == (3,):
            # Axis-angle representation
            # Convert to rotation matrix using Rodrigues' formula
            theta = np.linalg.norm(rot)
            if theta < 1e-10:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

            k = rot / theta
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])

            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            return R.tolist()
        else:
            logger.warning(f"Unknown rotation format with shape {rot.shape}, using identity")
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def _extract_reference_lla(self, reconstruction: Dict) -> Tuple[float, float, float]:
        """Extract reference lat/lon/alt for local coordinate system.

        OpenSfM uses a local coordinate system centered at a reference point.
        This tries to extract that reference point.

        Args:
            reconstruction: Reconstruction data

        Returns:
            Tuple of (latitude, longitude, altitude)
        """
        # Try to get reference from reconstruction metadata
        reference = reconstruction.get("reference_lla")
        if reference:
            # Handle both dict format {'latitude': x, 'longitude': y, 'altitude': z}
            # and list format [lat, lon, alt]
            if isinstance(reference, dict):
                if all(k in reference for k in ['latitude', 'longitude', 'altitude']):
                    return (float(reference['latitude']), float(reference['longitude']), float(reference['altitude']))
            elif isinstance(reference, (list, tuple)) and len(reference) >= 3:
                return (float(reference[0]), float(reference[1]), float(reference[2]))

        # Fallback: use the mean GPS position of all shots as reference
        shots = reconstruction.get("shots", {})
        if shots:
            lats = []
            lons = []
            alts = []

            for shot_data in shots.values():
                gps_pos = shot_data.get("gps_position")
                if gps_pos and len(gps_pos) >= 3:
                    lats.append(gps_pos[0])
                    lons.append(gps_pos[1])
                    alts.append(gps_pos[2])

            if lats:
                return (
                    float(np.mean(lats)),
                    float(np.mean(lons)),
                    float(np.mean(alts))
                )

        logger.warning("Could not determine reference LLA, using default")
        return (0.0, 0.0, 0.0)

    def _enu_to_wgs84(
        self,
        easting: float,
        northing: float,
        up: float,
        reference_lla: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert local ENU coordinates to WGS84 lat/lon/alt.

        Args:
            easting: East offset in meters from reference point
            northing: North offset in meters from reference point
            up: Up offset in meters from reference point
            reference_lla: Reference (latitude, longitude, altitude) in WGS84

        Returns:
            Tuple of (latitude, longitude, altitude) in WGS84
        """
        ref_lat, ref_lon, ref_alt = reference_lla

        # Create a local tangent plane projection centered at reference point
        # Using Transverse Mercator projection for ENU to lat/lon conversion
        # EPSG:4978 is ECEF (Earth-Centered Earth-Fixed)
        # We'll use a simpler approximation for small areas

        # For small distances (< 100km), we can use simple approximations
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters

        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(ref_lat))

        # Convert ENU offsets to lat/lon offsets
        delta_lat = northing / meters_per_degree_lat
        delta_lon = easting / meters_per_degree_lon

        # Add offsets to reference point
        latitude = ref_lat + delta_lat
        longitude = ref_lon + delta_lon
        altitude = ref_alt + up

        return (latitude, longitude, altitude)


class GeoTIFFMetadataParser:
    """Parse GeoTIFF metadata from orthophoto and elevation models."""

    def __init__(self):
        """Initialize GDAL."""
        gdal.UseExceptions()

    def parse_orthophoto(self, tif_path: Path) -> Dict:
        """Extract metadata from orthophoto GeoTIFF.

        Args:
            tif_path: Path to odm_orthophoto.tif

        Returns:
            Dictionary with orthophoto metadata:
                - bounds: (min_x, min_y, max_x, max_y) in CRS units
                - crs: CRS as WKT string
                - gsd: Ground Sample Distance in meters/pixel
                - width: Image width in pixels
                - height: Image height in pixels
        """
        if not tif_path.exists():
            raise FileNotFoundError(f"Orthophoto not found at {tif_path}")

        ds = gdal.Open(str(tif_path))
        if ds is None:
            raise ValueError(f"Failed to open GeoTIFF: {tif_path}")

        # Get geotransform [origin_x, pixel_width, 0, origin_y, 0, -pixel_height]
        gt = ds.GetGeoTransform()

        width = ds.RasterXSize
        height = ds.RasterYSize

        # Calculate bounds
        min_x = gt[0]
        max_y = gt[3]
        max_x = min_x + (width * gt[1])
        min_y = max_y + (height * gt[5])  # gt[5] is negative

        bounds = (min_x, min_y, max_x, max_y)

        # Get CRS
        crs = ds.GetProjection()

        # GSD (Ground Sample Distance) - pixel size in meters
        # Assume gt[1] is in meters (typical for UTM projections)
        gsd = abs(gt[1])

        ds = None  # Close dataset

        return {
            "bounds": bounds,
            "crs": crs,
            "gsd": gsd,
            "width": width,
            "height": height,
        }

    def parse_dsm(self, tif_path: Path) -> Dict:
        """Extract metadata from DSM (Digital Surface Model) GeoTIFF.

        Args:
            tif_path: Path to dsm.tif

        Returns:
            Dictionary with DSM metadata:
                - bounds: (min_x, min_y, max_x, max_y)
                - min_elevation: Minimum elevation in meters
                - max_elevation: Maximum elevation in meters
                - mean_elevation: Mean elevation in meters
        """
        if not tif_path.exists():
            raise FileNotFoundError(f"DSM not found at {tif_path}")

        ds = gdal.Open(str(tif_path))
        if ds is None:
            raise ValueError(f"Failed to open DSM: {tif_path}")

        # Get geotransform
        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize

        # Calculate bounds
        min_x = gt[0]
        max_y = gt[3]
        max_x = min_x + (width * gt[1])
        min_y = max_y + (height * gt[5])

        bounds = (min_x, min_y, max_x, max_y)

        # Read elevation data
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        # Get statistics
        stats = band.GetStatistics(True, True)  # Force recomputation
        min_elev, max_elev, mean_elev, stddev_elev = stats

        ds = None  # Close dataset

        return {
            "bounds": bounds,
            "min_elevation": min_elev,
            "max_elevation": max_elev,
            "mean_elevation": mean_elev,
        }
