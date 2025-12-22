"""Camera reprojection for mapping orthophoto pixels to raw image pixels.

Uses OpenSfM reconstruction data to project world coordinates back into
source camera frames, enabling precise pixel-level correspondence between
orthophoto defects and raw thermal images.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyproj import CRS, Transformer

from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter

logger = get_logger(__name__)


@dataclass
class RawImageMatch:
    """A match between an orthophoto location and a raw image pixel."""

    image_name: str
    pixel_x: float
    pixel_y: float
    distance_from_center: float  # 0.0 = center, 1.0 = corner

    @property
    def is_central(self) -> bool:
        """True if the point is in the central 50% of the image."""
        return self.distance_from_center < 0.5


class CameraProjector:
    """
    Projects orthophoto coordinates back to raw image pixels using camera poses.

    This uses the OpenSfM reconstruction data which contains:
    - Camera intrinsics (focal length, distortion)
    - Camera extrinsics (position, orientation) for each image

    Falls back to None if reconstruction data is not available,
    allowing the caller to use GPS matching instead.
    """

    def __init__(
        self,
        reconstruction_json: Optional[dict],
        geo_converter: PixelToLatLonConverter,
    ):
        """
        Initialize camera projector.

        Args:
            reconstruction_json: Parsed reconstruction.json from OpenSfM, or None
            geo_converter: Converter for ortho pixels to world coordinates
        """
        self.geo_converter = geo_converter

        # Raw thermal images and reconstruction use the same resolution: 1280x1024
        self.image_width = 1280
        self.image_height = 1024

        if reconstruction_json is None:
            self.available = False
            self.cameras = {}
            self.shots = {}
            logger.info("Camera projector disabled (no reconstruction data)")
            return

        # OpenSfM stores reconstructions as a list (usually just one)
        if isinstance(reconstruction_json, list):
            if len(reconstruction_json) == 0:
                self.available = False
                self.cameras = {}
                self.shots = {}
                logger.warning("Empty reconstruction.json")
                return
            reconstruction = reconstruction_json[0]
        else:
            reconstruction = reconstruction_json

        self.cameras = reconstruction.get("cameras", {})
        self.shots = reconstruction.get("shots", {})
        self.reference_lla = reconstruction.get("reference_lla", None)

        # Set up coordinate transformer from orthophoto CRS to local ENU
        # OpenSfM uses local ENU (East-North-Up) coordinates relative to reference_lla
        self._utm_to_enu_transformer = None
        self._ref_utm_x = None
        self._ref_utm_y = None

        if self.reference_lla:
            ref_lat = self.reference_lla.get("latitude", 0)
            ref_lon = self.reference_lla.get("longitude", 0)

            # Get the orthophoto's CRS from the geo_converter
            # The geo_converter has _source_crs which is the orthophoto's CRS
            ortho_crs = getattr(geo_converter, "_source_crs", None)

            if ortho_crs:
                # Transform reference point from WGS84 to orthophoto CRS (UTM)
                wgs84_to_utm = Transformer.from_crs(
                    CRS.from_epsg(4326), ortho_crs, always_xy=True
                )
                self._ref_utm_x, self._ref_utm_y = wgs84_to_utm.transform(ref_lon, ref_lat)
                logger.info(
                    f"Reference point UTM: ({self._ref_utm_x:.2f}, {self._ref_utm_y:.2f})"
                )

        # Estimate ground elevation from reconstruction 3D points
        self._ground_elevation = 0.0
        points = reconstruction.get("points", {})
        if points:
            # Get median Z coordinate of 3D points as ground elevation estimate
            z_coords = [p.get("coordinates", [0, 0, 0])[2] for p in points.values() if "coordinates" in p]
            if z_coords:
                self._ground_elevation = float(np.median(z_coords))
                logger.info(f"Estimated ground elevation: {self._ground_elevation:.1f}m")

        self.available = len(self.shots) > 0

        if self.available:
            logger.info(
                f"Camera projector initialized with {len(self.shots)} camera poses"
            )
        else:
            logger.warning("No camera shots in reconstruction.json")

    def project_ortho_to_raw(
        self,
        ortho_x: float,
        ortho_y: float,
        elevation: float = 0.0,
        max_gps_distance: float = 15.0,
    ) -> List[RawImageMatch]:
        """
        Project an orthophoto pixel to raw image pixels.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            elevation: Ground elevation at this point (meters, from DSM)
            max_gps_distance: Maximum horizontal distance (meters) from drone to defect.
                             Images where drone was farther than this are excluded.

        Returns:
            List of RawImageMatch objects, sorted by distance_from_center (best first).
            Empty list if projection not available or point not visible in any image.
        """
        if not self.available:
            return []

        # Convert orthophoto pixel to world coordinates (local ENU)
        world_x, world_y = self._ortho_to_world(ortho_x, ortho_y)
        world_point = np.array([world_x, world_y, elevation])
        defect_xy = np.array([world_x, world_y])

        matches = []

        for image_name, shot in self.shots.items():
            try:
                # FIRST: Check GPS distance - only consider images where drone was close enough
                gps_position = shot.get('gps_position')
                if gps_position:
                    drone_xy = np.array([gps_position[0], gps_position[1]])
                    gps_dist = np.linalg.norm(defect_xy - drone_xy)
                    if gps_dist > max_gps_distance:
                        # Drone was too far away - this image cannot contain the defect
                        continue

                # THEN: Project to get precise pixel coordinates
                pixel_coords = self._project_to_camera(world_point, shot)

                if pixel_coords is not None:
                    px, py = pixel_coords

                    # Check if point is within image bounds (1280x1024)
                    if 0 <= px < self.image_width and 0 <= py < self.image_height:
                        # Calculate distance from center (normalized)
                        cx = self.image_width / 2
                        cy = self.image_height / 2
                        max_dist = np.hypot(cx, cy)
                        dist = np.hypot(px - cx, py - cy) / max_dist

                        matches.append(RawImageMatch(
                            image_name=image_name,
                            pixel_x=px,
                            pixel_y=py,
                            distance_from_center=dist,
                        ))
            except Exception as e:
                logger.debug(f"Projection failed for {image_name}: {e}")
                continue

        # Sort by distance from center (most central first)
        matches.sort(key=lambda m: m.distance_from_center)

        return matches

    def get_best_match(
        self,
        ortho_x: float,
        ortho_y: float,
        elevation: float = 0.0,
    ) -> Optional[RawImageMatch]:
        """
        Get the single best raw image match for an orthophoto location.

        The "best" image is the one where the point appears most centrally,
        which typically means best visibility and least distortion.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            elevation: Ground elevation (meters)

        Returns:
            Best RawImageMatch or None if not available
        """
        matches = self.project_ortho_to_raw(ortho_x, ortho_y, elevation)
        return matches[0] if matches else None

    def project_to_specific_image(
        self,
        ortho_x: float,
        ortho_y: float,
        image_name: str,
        elevation: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Project orthophoto coordinates to a specific raw image.

        This is used when the image has already been selected (e.g., by GPS matching)
        and we just need the pixel coordinates for annotation.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            image_name: Name of the target image (e.g., "DJI_20230612111751_0126_T.JPG")
            elevation: Ground elevation (meters), defaults to estimated ground level

        Returns:
            (pixel_x, pixel_y) tuple or None if projection fails
        """
        if not self.available:
            return None

        # Get shot data for this image
        shot = self.shots.get(image_name)
        if shot is None:
            logger.debug(f"Image {image_name} not found in reconstruction")
            return None

        # Use estimated ground elevation if not provided
        if elevation == 0.0:
            elevation = self._ground_elevation

        # Convert orthophoto pixel to world coordinates
        world_x, world_y = self._ortho_to_world(ortho_x, ortho_y)
        world_point = np.array([world_x, world_y, elevation])

        # Project to camera
        try:
            pixel_coords = self._project_to_camera(world_point, shot)
            if pixel_coords is not None:
                px, py = pixel_coords
                # Check if within image bounds
                if 0 <= px < self.image_width and 0 <= py < self.image_height:
                    logger.debug(f"Projected to {image_name}: ({px:.1f}, {py:.1f})")
                    return px, py
                else:
                    logger.debug(f"Projection to {image_name} out of bounds: ({px:.1f}, {py:.1f})")
        except Exception as e:
            logger.debug(f"Projection to {image_name} failed: {e}")

        return None

    def _ortho_to_world(self, ortho_x: float, ortho_y: float) -> Tuple[float, float]:
        """Convert orthophoto pixel to OpenSfM local ENU coordinates.

        The orthophoto is in UTM coordinates. OpenSfM uses local ENU (East-North-Up)
        coordinates relative to reference_lla. We need to:
        1. Convert ortho pixel to UTM using the GeoTIFF transform
        2. Subtract the reference point UTM to get local ENU (X=East, Y=North)
        """
        # Use the GeoTIFF transform to get UTM coordinates
        transform = self.geo_converter.transform
        utm_x = transform.c + ortho_x * transform.a + ortho_y * transform.b
        utm_y = transform.f + ortho_x * transform.d + ortho_y * transform.e

        # Convert to local ENU by subtracting reference point
        if self._ref_utm_x is not None and self._ref_utm_y is not None:
            local_x = utm_x - self._ref_utm_x  # East offset in meters
            local_y = utm_y - self._ref_utm_y  # North offset in meters
            return local_x, local_y
        else:
            # Fallback: assume UTM coordinates directly (won't work correctly)
            logger.warning("No reference point set, using raw UTM coordinates")
            return utm_x, utm_y

    def _project_to_camera(
        self,
        world_point: np.ndarray,
        shot: dict,
    ) -> Optional[Tuple[float, float]]:
        """
        Project a 3D world point into a camera's image plane.

        Args:
            world_point: 3D point in world coordinates [x, y, z]
            shot: Camera shot data from reconstruction.json

        Returns:
            (pixel_x, pixel_y) or None if behind camera
        """
        # Get camera model
        camera_id = shot.get("camera")
        if camera_id not in self.cameras:
            return None
        camera = self.cameras[camera_id]

        # Get camera pose
        # OpenSfM uses axis-angle rotation
        rotation = np.array(shot.get("rotation", [0, 0, 0]))
        translation = np.array(shot.get("translation", [0, 0, 0]))

        # Convert axis-angle to rotation matrix
        R = self._axis_angle_to_rotation_matrix(rotation)

        # Transform world point to camera coordinates
        # In OpenSfM: camera_point = R * (world_point - camera_position)
        # But translation is actually R * (-camera_position), so:
        # camera_point = R * world_point + translation
        camera_point = R @ world_point + translation

        # Check if point is in front of camera
        if camera_point[2] <= 0:
            return None

        # Project to normalized image coordinates
        x = camera_point[0] / camera_point[2]
        y = camera_point[1] / camera_point[2]

        # Apply camera intrinsics
        # OpenSfM brown camera model has focal_x, focal_y, c_x, c_y, k1, k2, p1, p2, k3
        focal_x = camera.get("focal_x", camera.get("focal", 1.0))
        focal_y = camera.get("focal_y", focal_x)
        c_x = camera.get("c_x", 0.0)  # Principal point offset X
        c_y = camera.get("c_y", 0.0)  # Principal point offset Y
        k1 = camera.get("k1", 0.0)
        k2 = camera.get("k2", 0.0)

        # Apply radial distortion
        r2 = x * x + y * y
        distortion = 1.0 + k1 * r2 + k2 * r2 * r2
        x_distorted = x * distortion
        y_distorted = y * distortion

        # Convert to pixel coordinates
        # OpenSfM uses normalized coordinates where focal is relative to max(width, height)
        max_size = max(self.image_width, self.image_height)
        pixel_x = x_distorted * focal_x * max_size + self.image_width / 2 + c_x * max_size
        pixel_y = y_distorted * focal_y * max_size + self.image_height / 2 + c_y * max_size

        return pixel_x, pixel_y

    @staticmethod
    def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle rotation to 3x3 rotation matrix."""
        angle = np.linalg.norm(axis_angle)

        if angle < 1e-10:
            return np.eye(3)

        axis = axis_angle / angle

        # Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R


def load_reconstruction(reconstruction_path: Path) -> Optional[dict]:
    """
    Load reconstruction.json from a file path.

    Args:
        reconstruction_path: Path to reconstruction.json

    Returns:
        Parsed JSON or None if not found/invalid
    """
    if not reconstruction_path.exists():
        logger.info(f"Reconstruction file not found: {reconstruction_path}")
        return None

    try:
        with open(reconstruction_path) as f:
            data = json.load(f)
        logger.info(f"Loaded reconstruction.json ({reconstruction_path.stat().st_size} bytes)")
        return data
    except Exception as e:
        logger.warning(f"Failed to load reconstruction.json: {e}")
        return None
