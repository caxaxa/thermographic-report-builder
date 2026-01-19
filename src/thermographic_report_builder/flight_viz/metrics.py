"""Metrics calculators for flight statistics and coverage analysis."""

import logging
from typing import Optional

import numpy as np
from pyproj import Geod

from .models import CameraShot, CoverageMetrics, FlightData, FlightMetrics

logger = logging.getLogger(__name__)


class FlightMetricsCalculator:
    """Calculate flight statistics from flight data."""

    def __init__(self):
        """Initialize WGS84 geodesic calculator."""
        self.geod = Geod(ellps="WGS84")

    def calculate(self, flight_data: FlightData) -> FlightMetrics:
        """Calculate comprehensive flight metrics.

        Args:
            flight_data: Parsed flight data from reconstruction

        Returns:
            FlightMetrics object with computed statistics
        """
        shots = flight_data.shots

        if not shots:
            raise ValueError("Cannot calculate metrics from empty shot list")

        # Basic stats
        total_images = len(shots)

        # Extract lists for calculations
        altitudes = [shot.altitude for shot in shots]
        timestamps = [shot.timestamp for shot in shots if shot.timestamp is not None]

        # Altitude stats
        min_altitude = float(np.min(altitudes))
        max_altitude = float(np.max(altitudes))
        mean_altitude = float(np.mean(altitudes))
        median_altitude = float(np.median(altitudes))

        # Flight duration (if timestamps available)
        flight_duration = None
        if len(timestamps) >= 2:
            sorted_timestamps = sorted(timestamps)
            duration = sorted_timestamps[-1] - sorted_timestamps[0]
            flight_duration = duration.total_seconds()

        # Calculate total distance flown
        total_distance = self._calculate_total_distance(shots)

        # Speed stats (if timestamps available)
        min_speed = None
        max_speed = None
        mean_speed = None

        if len(timestamps) >= 2:
            speeds = self._calculate_speeds(shots)
            if speeds:
                min_speed = float(np.min(speeds))
                max_speed = float(np.max(speeds))
                mean_speed = float(np.mean(speeds))

        # GSD stats (if available)
        gsds = [shot.gsd for shot in shots if shot.gsd is not None]
        min_gsd = float(np.min(gsds)) if gsds else None
        max_gsd = float(np.max(gsds)) if gsds else None
        mean_gsd = float(np.mean(gsds)) if gsds else None

        # Bounding box
        lats = [shot.latitude for shot in shots]
        lons = [shot.longitude for shot in shots]

        bbox_min_lat = float(np.min(lats))
        bbox_max_lat = float(np.max(lats))
        bbox_min_lon = float(np.min(lons))
        bbox_max_lon = float(np.max(lons))

        # Coverage area (approximate)
        coverage_area = self._estimate_coverage_area(bbox_min_lon, bbox_min_lat,
                                                       bbox_max_lon, bbox_max_lat)

        # GPS error statistics (componentwise in ENU coordinate system)
        # Error = camera_center (bundle-adjusted) - gps_position (raw GPS)
        # This shows how much the bundle adjustment corrected the raw GPS positions
        gps_errors_x = []  # East component errors
        gps_errors_y = []  # North component errors
        gps_errors_z = []  # Up component errors
        gps_errors_3d = []  # 3D Euclidean distances

        for shot in shots:
            if shot.camera_center_enu and shot.gps_position_enu:
                # Error = optimized_position (camera_center) - gps_position (raw GPS)
                error_x = shot.camera_center_enu[0] - shot.gps_position_enu[0]
                error_y = shot.camera_center_enu[1] - shot.gps_position_enu[1]
                error_z = shot.camera_center_enu[2] - shot.gps_position_enu[2]

                gps_errors_x.append(error_x)
                gps_errors_y.append(error_y)
                gps_errors_z.append(error_z)

                # 3D distance (already correctly computed in parser using camera_center)
                if shot.gps_error is not None:
                    gps_errors_3d.append(shot.gps_error)

        # Compute mean and std dev for each component
        mean_gps_error_x = float(np.mean(gps_errors_x)) if gps_errors_x else None
        mean_gps_error_y = float(np.mean(gps_errors_y)) if gps_errors_y else None
        mean_gps_error_z = float(np.mean(gps_errors_z)) if gps_errors_z else None

        std_gps_error_x = float(np.std(gps_errors_x)) if gps_errors_x else None
        std_gps_error_y = float(np.std(gps_errors_y)) if gps_errors_y else None
        std_gps_error_z = float(np.std(gps_errors_z)) if gps_errors_z else None

        # 3D Euclidean distance statistics
        mean_gps_error = float(np.mean(gps_errors_3d)) if gps_errors_3d else None
        max_gps_error = float(np.max(gps_errors_3d)) if gps_errors_3d else None

        return FlightMetrics(
            total_images=total_images,
            flight_duration=flight_duration,
            total_distance=total_distance,
            min_altitude=min_altitude,
            max_altitude=max_altitude,
            mean_altitude=mean_altitude,
            median_altitude=median_altitude,
            min_speed=min_speed,
            max_speed=max_speed,
            mean_speed=mean_speed,
            min_gsd=min_gsd,
            max_gsd=max_gsd,
            mean_gsd=mean_gsd,
            coverage_area=coverage_area,
            bbox_min_lat=bbox_min_lat,
            bbox_min_lon=bbox_min_lon,
            bbox_max_lat=bbox_max_lat,
            bbox_max_lon=bbox_max_lon,
            mean_gps_error_x=mean_gps_error_x,
            mean_gps_error_y=mean_gps_error_y,
            mean_gps_error_z=mean_gps_error_z,
            std_gps_error_x=std_gps_error_x,
            std_gps_error_y=std_gps_error_y,
            std_gps_error_z=std_gps_error_z,
            mean_gps_error=mean_gps_error,
            max_gps_error=max_gps_error,
        )

    def _calculate_total_distance(self, shots: list[CameraShot]) -> float:
        """Calculate total distance flown along flight path.

        Args:
            shots: List of camera shots

        Returns:
            Total distance in meters
        """
        if len(shots) < 2:
            return 0.0

        # Sort by timestamp if available, otherwise assume shots are in order
        sorted_shots = sorted(shots, key=lambda s: s.timestamp) if shots[0].timestamp else shots

        total_distance = 0.0

        for i in range(len(sorted_shots) - 1):
            shot1 = sorted_shots[i]
            shot2 = sorted_shots[i + 1]

            # Calculate geodesic distance between consecutive shots
            _, _, distance = self.geod.inv(
                shot1.longitude, shot1.latitude,
                shot2.longitude, shot2.latitude
            )

            total_distance += distance

        return total_distance

    def _calculate_speeds(self, shots: list[CameraShot]) -> list[float]:
        """Calculate speed between consecutive shots.

        Args:
            shots: List of camera shots with timestamps

        Returns:
            List of speeds in m/s
        """
        # Filter shots with timestamps
        shots_with_time = [s for s in shots if s.timestamp is not None]

        if len(shots_with_time) < 2:
            return []

        # Sort by timestamp
        sorted_shots = sorted(shots_with_time, key=lambda s: s.timestamp)

        speeds = []

        for i in range(len(sorted_shots) - 1):
            shot1 = sorted_shots[i]
            shot2 = sorted_shots[i + 1]

            # Calculate distance
            _, _, distance = self.geod.inv(
                shot1.longitude, shot1.latitude,
                shot2.longitude, shot2.latitude
            )

            # Calculate time difference
            time_diff = (shot2.timestamp - shot1.timestamp).total_seconds()

            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)

        return speeds

    def _estimate_coverage_area(self, min_lon: float, min_lat: float,
                                  max_lon: float, max_lat: float) -> float:
        """Estimate coverage area from bounding box.

        Args:
            min_lon, min_lat, max_lon, max_lat: Bounding box coordinates

        Returns:
            Approximate area in square meters
        """
        # Calculate geodesic distances
        # Width (at middle latitude)
        mid_lat = (min_lat + max_lat) / 2
        _, _, width = self.geod.inv(min_lon, mid_lat, max_lon, mid_lat)

        # Height (at middle longitude)
        mid_lon = (min_lon + max_lon) / 2
        _, _, height = self.geod.inv(mid_lon, min_lat, mid_lon, max_lat)

        # Approximate area as rectangle
        return width * height


class CoverageMetricsCalculator:
    """Calculate coverage and overlap statistics.

    Note: This requires access to the point cloud data to determine
    how many images see each point. This is a simplified implementation
    that would need enhancement to parse point cloud files.
    """

    def calculate(self, flight_data: FlightData,
                  point_cloud_path: Optional[str] = None) -> CoverageMetrics:
        """Calculate coverage and overlap metrics.

        Args:
            flight_data: Parsed flight data
            point_cloud_path: Optional path to point cloud file (for overlap analysis)

        Returns:
            CoverageMetrics object

        Note:
            Without point cloud data, this returns placeholder values.
            Full implementation would parse the point cloud to determine
            which images see each 3D point.
        """
        if point_cloud_path is None:
            logger.warning(
                "Point cloud path not provided. Coverage metrics will be estimated."
            )
            return CoverageMetrics(
                min_overlap=None,
                max_overlap=None,
                mean_overlap=None,
                median_overlap=None,
                single_coverage_pct=0.0,
                double_coverage_pct=0.0,
                triple_coverage_pct=0.0,
                coverage_gaps=0,
            )

        # TODO: Implement point cloud parsing
        # This would involve:
        # 1. Parse point cloud file (PLY or LAZ format)
        # 2. For each point, count how many cameras see it (using tracks.csv or point colors)
        # 3. Calculate overlap statistics
        # 4. Detect coverage gaps

        logger.info("Point cloud analysis not yet implemented")

        return CoverageMetrics(
            min_overlap=None,
            max_overlap=None,
            mean_overlap=None,
            median_overlap=None,
            single_coverage_pct=0.0,
            double_coverage_pct=0.0,
            triple_coverage_pct=0.0,
            coverage_gaps=0,
        )


class ReconstructionHealthCalculator:
    """Calculate reconstruction quality metrics.

    This would analyze feature detection, matching, and reprojection errors
    from OpenSfM outputs.
    """

    def calculate_from_opensfm_stats(self, stats_path: str):
        """Calculate reconstruction health from OpenSfM stats.

        Args:
            stats_path: Path to OpenSfM stats directory

        Note:
            This is a placeholder. Full implementation would parse:
            - stats/features.json (feature detection stats)
            - stats/matches.json (matching stats)
            - stats/reconstruction.json (reprojection errors)
        """
        # TODO: Implement OpenSfM stats parsing
        logger.info("OpenSfM stats analysis not yet implemented")
        return None
