"""Data models for flight visualization."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class CameraShot:
    """Single camera shot from reconstruction.json."""

    shot_id: str  # Image filename (e.g., "DJI_0001.JPG")
    # GPS coordinates (WGS84)
    latitude: float
    longitude: float
    altitude: float  # meters above sea level

    # Camera orientation (rotation matrix or quaternion)
    rotation: List[List[float]]  # 3x3 rotation matrix

    # OpenSfM local coordinates (for reprojection)
    translation: List[float]  # [x, y, z] OpenSfM translation vector (NOT camera position!)
    gps_position_enu: List[float]  # [east, north, up] original GPS in ENU (for error calculation)
    camera_center_enu: Optional[List[float]] = None  # [east, north, up] bundle-adjusted position = -R^T * t

    # EXIF metadata (if available)
    timestamp: Optional[datetime] = None
    gsd: Optional[float] = None  # Ground Sample Distance in cm/px
    gps_dop: Optional[float] = None  # GPS dilution of precision

    # Reconstruction quality metrics
    n_features: Optional[int] = None  # Number of detected features
    n_matches: Optional[int] = None  # Number of feature matches
    gps_error: Optional[float] = None  # GPS prediction error in meters (distance between GPS and optimized position)


@dataclass
class FlightData:
    """Complete flight data parsed from reconstruction.json."""

    shots: List[CameraShot]

    # Reference coordinate system
    reference_lla: Tuple[float, float, float]  # Reference lat/lon/alt for local coordinates

    # Camera calibration
    camera_model: str  # e.g., "brown", "fisheye", "perspective"
    focal_length: float  # in pixels
    k1: float = 0.0  # Radial distortion coefficients
    k2: float = 0.0

    # Reconstruction metadata
    total_points: Optional[int] = None  # Number of 3D points in reconstruction
    reprojection_error: Optional[float] = None  # RMS reprojection error in pixels


@dataclass
class FlightMetrics:
    """Computed flight statistics."""

    # Basic stats
    total_images: int
    flight_duration: Optional[float] = None  # seconds
    total_distance: float = 0.0  # meters

    # Altitude stats
    min_altitude: float = 0.0
    max_altitude: float = 0.0
    mean_altitude: float = 0.0
    median_altitude: float = 0.0

    # Speed stats (if timestamps available)
    min_speed: Optional[float] = None  # m/s
    max_speed: Optional[float] = None
    mean_speed: Optional[float] = None

    # GSD stats
    min_gsd: Optional[float] = None  # cm/px
    max_gsd: Optional[float] = None
    mean_gsd: Optional[float] = None

    # Coverage area
    coverage_area: Optional[float] = None  # square meters

    # Bounding box (WGS84)
    bbox_min_lat: float = 0.0
    bbox_min_lon: float = 0.0
    bbox_max_lat: float = 0.0
    bbox_max_lon: float = 0.0

    # GPS alignment error (prediction accuracy) - componentwise
    mean_gps_error_x: Optional[float] = None  # meters (East component)
    mean_gps_error_y: Optional[float] = None  # meters (North component)
    mean_gps_error_z: Optional[float] = None  # meters (Up component)
    std_gps_error_x: Optional[float] = None   # meters (std dev East)
    std_gps_error_y: Optional[float] = None   # meters (std dev North)
    std_gps_error_z: Optional[float] = None   # meters (std dev Up)

    # GPS alignment error (3D Euclidean distance)
    mean_gps_error: Optional[float] = None  # meters (mean 3D distance)
    max_gps_error: Optional[float] = None   # meters (max 3D distance)


@dataclass
class CoverageMetrics:
    """Coverage and overlap statistics."""

    # Overlap statistics
    min_overlap: Optional[int] = None  # Minimum number of images seeing a point
    max_overlap: Optional[int] = None
    mean_overlap: Optional[float] = None
    median_overlap: Optional[float] = None

    # Coverage percentages
    single_coverage_pct: float = 0.0  # % of area seen by exactly 1 image
    double_coverage_pct: float = 0.0  # % seen by 2+ images
    triple_coverage_pct: float = 0.0  # % seen by 3+ images

    # Gaps
    coverage_gaps: int = 0  # Number of holes in coverage


@dataclass
class ReconstructionHealthMetrics:
    """Reconstruction quality metrics."""

    # Feature statistics
    total_features_detected: int = 0
    mean_features_per_image: float = 0.0
    min_features_per_image: int = 0
    max_features_per_image: int = 0

    # Matching statistics
    total_matches: int = 0
    mean_matches_per_image: float = 0.0

    # 3D point statistics
    total_3d_points: int = 0
    mean_track_length: Optional[float] = None  # Average number of views per 3D point

    # Reprojection error
    mean_reprojection_error: Optional[float] = None  # pixels
    max_reprojection_error: Optional[float] = None

    # GPS alignment error
    mean_gps_error: Optional[float] = None  # meters
    max_gps_error: Optional[float] = None
