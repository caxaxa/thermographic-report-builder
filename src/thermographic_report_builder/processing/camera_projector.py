"""Camera reprojection for mapping orthophoto pixels to raw image pixels.

Uses OpenSfM reconstruction data to project world coordinates back into
source camera frames, enabling precise pixel-level correspondence between
orthophoto defects and raw thermal images.

For best accuracy, uses DSM (Digital Surface Model) to get the exact elevation
at each orthophoto pixel - the same elevation ODM used when creating the orthophoto.
"""

import json
import numpy as np
import rasterio
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
        dsm_path: Optional[Path] = None,
    ):
        """
        Initialize camera projector.

        Args:
            reconstruction_json: Parsed reconstruction.json from OpenSfM, or None
            geo_converter: Converter for ortho pixels to world coordinates
            dsm_path: Optional path to DSM (Digital Surface Model) for precise elevation
        """
        self.geo_converter = geo_converter
        self.points: Dict[str, dict] = {}

        # Default resolution; overridden by camera model when available
        self.image_width = 1280
        self.image_height = 1024

        # Load DSM for precise per-pixel elevation
        self._dsm_data = None
        self._dsm_transform = None
        if dsm_path and dsm_path.exists():
            try:
                with rasterio.open(dsm_path) as dsm:
                    self._dsm_data = dsm.read(1)  # Single band elevation data
                    self._dsm_transform = dsm.transform
                    self._dsm_nodata = dsm.nodata
                logger.info(
                    f"Loaded DSM for precise elevation: {self._dsm_data.shape}, "
                    f"range [{np.nanmin(self._dsm_data):.1f}, {np.nanmax(self._dsm_data):.1f}]m"
                )
            except Exception as e:
                logger.warning(f"Failed to load DSM, will use estimated elevation: {e}")
                self._dsm_data = None

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
        self.points = reconstruction.get("points", {})
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
        self._dsm_to_enu_offset = 0.0  # Offset to convert DSM absolute elevation to local ENU Z
        points = reconstruction.get("points", {})
        if points:
            # Get median Z coordinate of 3D points as ground elevation estimate
            z_coords = [p.get("coordinates", [0, 0, 0])[2] for p in points.values() if "coordinates" in p]
            if z_coords:
                self._ground_elevation = float(np.median(z_coords))
                logger.info(f"Estimated ground elevation from reconstruction: {self._ground_elevation:.1f}m (local ENU)")

        # Update default dimensions from the first camera if available
        if self.cameras:
            first_camera = next(iter(self.cameras.values()))
            cam_width = first_camera.get("width")
            cam_height = first_camera.get("height")
            if cam_width and cam_height:
                self.image_width = int(cam_width)
                self.image_height = int(cam_height)

        self.available = len(self.shots) > 0

        if self.available:
            dsm_status = "with DSM" if self._dsm_data is not None else "estimated elevation"
            logger.info(
                f"Camera projector initialized with {len(self.shots)} camera poses ({dsm_status})"
            )
        else:
            logger.warning("No camera shots in reconstruction.json")

    def get_elevation_at_ortho_pixel(self, ortho_x: float, ortho_y: float) -> float:
        """
        Get precise elevation at an orthophoto pixel from DSM using bilinear interpolation.

        The DSM and orthophoto share the same georeferencing, so we can directly
        look up the elevation. This gives us the exact elevation ODM used when
        creating that orthophoto pixel.

        Uses bilinear interpolation instead of nearest neighbor to avoid
        elevation jumps at panel edges which can cause significant projection errors
        (0.5m Z error at 25m altitude = ~15 pixel horizontal shift).

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)

        Returns:
            Elevation in meters (local ENU Z coordinate)
        """
        if self._dsm_data is None:
            # Log warning only once per session to avoid spam
            if not getattr(self, '_dsm_warning_logged', False):
                logger.warning(
                    f"DSM not available, using fallback elevation {self._ground_elevation:.1f}m. "
                    f"Projection accuracy may be reduced by ~15px for 0.5m height errors."
                )
                self._dsm_warning_logged = True
            return self._ground_elevation

        # Convert orthophoto pixel to UTM using orthophoto transform
        # Add 0.5 for pixel center (consistent with _ortho_to_world)
        ortho_transform = self.geo_converter.transform
        utm_x = ortho_transform.c + (ortho_x + 0.5) * ortho_transform.a + (ortho_y + 0.5) * ortho_transform.b
        utm_y = ortho_transform.f + (ortho_x + 0.5) * ortho_transform.d + (ortho_y + 0.5) * ortho_transform.e

        # Convert UTM to DSM pixel coordinates using DSM transform
        # DSM pixel = inverse_transform(UTM)
        dsm_transform = self._dsm_transform
        # Inverse affine: col = (x - c) / a, row = (y - f) / e (for non-rotated)
        dsm_col = (utm_x - dsm_transform.c) / dsm_transform.a
        dsm_row = (utm_y - dsm_transform.f) / dsm_transform.e

        dsm_h, dsm_w = self._dsm_data.shape

        # Check if completely out of bounds
        if dsm_col < 0 or dsm_col >= dsm_w or dsm_row < 0 or dsm_row >= dsm_h:
            return self._ground_elevation

        # Bilinear interpolation
        # Get integer coordinates and fractional parts
        col0 = int(np.floor(dsm_col))
        row0 = int(np.floor(dsm_row))
        col1 = min(col0 + 1, dsm_w - 1)
        row1 = min(row0 + 1, dsm_h - 1)

        # Clamp col0/row0 to valid range
        col0 = max(0, min(dsm_w - 1, col0))
        row0 = max(0, min(dsm_h - 1, row0))

        # Fractional distances
        dx = dsm_col - col0
        dy = dsm_row - row0

        # Get the four corner elevations
        z00 = self._dsm_data[row0, col0]
        z01 = self._dsm_data[row0, col1]
        z10 = self._dsm_data[row1, col0]
        z11 = self._dsm_data[row1, col1]

        # Check for nodata values - if any corner is nodata, fall back to nearest valid
        if self._dsm_nodata is not None:
            corners = [z00, z01, z10, z11]
            valid_corners = [z for z in corners if z != self._dsm_nodata and not np.isnan(z)]
            if len(valid_corners) == 0:
                return self._ground_elevation
            elif len(valid_corners) < 4:
                # Use mean of valid corners as fallback
                elevation_utm = float(np.mean(valid_corners))
            else:
                # Bilinear interpolation: lerp in x, then lerp in y
                z_top = z00 * (1 - dx) + z01 * dx
                z_bot = z10 * (1 - dx) + z11 * dx
                elevation_utm = z_top * (1 - dy) + z_bot * dy
        else:
            # No nodata handling needed
            z_top = z00 * (1 - dx) + z01 * dx
            z_bot = z10 * (1 - dx) + z11 * dx
            elevation_utm = z_top * (1 - dy) + z_bot * dy

        # Convert UTM/absolute elevation to local ENU Z
        # OpenSfM reference_lla includes altitude, but local Z is relative to that
        # The DSM values are in the same vertical datum as the orthophoto (typically ellipsoidal or geoid height)
        #
        # IMPORTANT: When reference_lla.altitude = 0 (common in ODM output), we need to
        # calibrate the DSM-to-ENU offset using known ground points from the reconstruction.
        # Otherwise, DSM absolute elevation (~850m) would be used as local Z, causing
        # massive projection errors.

        if self.reference_lla:
            ref_alt = self.reference_lla.get("altitude", 0)

            if ref_alt == 0 and self._dsm_to_enu_offset == 0.0 and self._ground_elevation != 0.0:
                # Reference altitude is 0 but we have ground elevation from 3D points
                # The DSM elevation at ground level should map to _ground_elevation in local ENU
                # So offset = DSM_value - ground_elevation_enu
                # We compute this once using the current DSM sample
                self._dsm_to_enu_offset = elevation_utm - self._ground_elevation
                logger.info(
                    f"Calibrated DSM-to-ENU offset: {self._dsm_to_enu_offset:.1f}m "
                    f"(DSM={elevation_utm:.1f}m -> ENU ground={self._ground_elevation:.1f}m)"
                )

            if ref_alt != 0:
                # Normal case: reference altitude is set
                elevation_enu = elevation_utm - ref_alt
            else:
                # Reference altitude is 0: use calibrated offset
                elevation_enu = elevation_utm - self._dsm_to_enu_offset
        else:
            elevation_enu = elevation_utm

        return float(elevation_enu)

    def project_ortho_to_raw(
        self,
        ortho_x: float,
        ortho_y: float,
        elevation: Optional[float] = None,
        max_gps_distance: Optional[float] = None,
    ) -> List[RawImageMatch]:
        """
        Project an orthophoto pixel to raw image pixels.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            elevation: Ground elevation at this point (meters). If None, automatically
                      looks up from DSM for precise elevation.
            max_gps_distance: Maximum horizontal distance (meters) from drone to defect.
                             Images where drone was farther than this are excluded.
                             If None, distance filtering is disabled and all shots
                             in reconstruction are considered.

        Returns:
            List of RawImageMatch objects, sorted by distance_from_center (best first).
            Empty list if projection not available or point not visible in any image.
        """
        if not self.available:
            return []

        # Get precise elevation from DSM if not provided
        if elevation is None:
            elevation = self.get_elevation_at_ortho_pixel(ortho_x, ortho_y)

        # Convert orthophoto pixel to world coordinates (local ENU)
        world_x, world_y = self._ortho_to_world(ortho_x, ortho_y)
        world_point = np.array([world_x, world_y, elevation])
        defect_xy = np.array([world_x, world_y])

        # QC logging: track projection pipeline
        logger.debug(
            f"Projection pipeline: ortho_px=({ortho_x:.1f}, {ortho_y:.1f}) -> "
            f"world_enu=({world_x:.2f}, {world_y:.2f}, z={elevation:.2f}m)"
        )

        matches = []
        filtered_by_distance = 0
        filtered_by_bounds = 0
        projection_failed = 0

        for image_name, shot in self.shots.items():
            try:
                # FIRST: Check GPS distance - only consider images where drone was close enough
                if max_gps_distance is not None:
                    gps_position = shot.get('gps_position')
                    if gps_position:
                        drone_xy = np.array([gps_position[0], gps_position[1]])
                        gps_dist = np.linalg.norm(defect_xy - drone_xy)
                        if gps_dist > max_gps_distance:
                            # Drone was too far away - this image cannot contain the defect
                            filtered_by_distance += 1
                            continue

                # THEN: Project to get precise pixel coordinates
                pixel_coords = self._project_to_camera(world_point, shot)

                if pixel_coords is not None:
                    px, py, img_w, img_h = pixel_coords

                    # Check if point is within image bounds
                    if 0 <= px < img_w and 0 <= py < img_h:
                        # Calculate distance from center (normalized)
                        cx = img_w / 2
                        cy = img_h / 2
                        max_dist = np.hypot(cx, cy)
                        dist = np.hypot(px - cx, py - cy) / max_dist

                        matches.append(RawImageMatch(
                            image_name=image_name,
                            pixel_x=px,
                            pixel_y=py,
                            distance_from_center=dist,
                        ))
                    else:
                        filtered_by_bounds += 1
                else:
                    projection_failed += 1
            except Exception as e:
                logger.debug(f"Projection failed for {image_name}: {e}")
                projection_failed += 1
                continue

        # Sort by distance from center (most central first)
        matches.sort(key=lambda m: m.distance_from_center)

        # QC logging: summarize filtering
        logger.debug(
            f"Projection candidates: {len(matches)} matches, "
            f"filtered_by_distance={filtered_by_distance} (>{max_gps_distance}m), "
            f"filtered_by_bounds={filtered_by_bounds}, projection_failed={projection_failed}"
        )
        if matches:
            best = matches[0]
            logger.debug(
                f"Best match: {best.image_name} at raw_px=({best.pixel_x:.1f}, {best.pixel_y:.1f}), "
                f"centrality={1-best.distance_from_center:.2f}"
            )

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
        elevation: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Project orthophoto coordinates to a specific raw image.

        This is used when the image has already been selected (e.g., by GPS matching)
        and we just need the pixel coordinates for annotation.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            image_name: Name of the target image (e.g., "DJI_20230612111751_0126_T.JPG")
            elevation: Ground elevation (meters). If None, automatically looks up from DSM.

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

        # Get precise elevation from DSM if not provided
        if elevation is None:
            elevation = self.get_elevation_at_ortho_pixel(ortho_x, ortho_y)

        # Convert orthophoto pixel to world coordinates
        world_x, world_y = self._ortho_to_world(ortho_x, ortho_y)
        world_point = np.array([world_x, world_y, elevation])

        # Project to camera
        try:
            pixel_coords = self._project_to_camera(world_point, shot)
            if pixel_coords is not None:
                px, py, img_w, img_h = pixel_coords
                # Check if within image bounds
                if 0 <= px < img_w and 0 <= py < img_h:
                    logger.debug(f"Projected to {image_name}: ({px:.1f}, {py:.1f})")
                    return px, py
                else:
                    logger.debug(f"Projection to {image_name} out of bounds: ({px:.1f}, {py:.1f})")
        except Exception as e:
            logger.debug(f"Projection to {image_name} failed: {e}")

        return None

    def project_world_to_image(
        self,
        image_name: str,
        world_point: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """
        Project a 3D world point (local ENU) directly into a specific image.

        Args:
            image_name: Name of the target image (shot key)
            world_point: 3D point in local ENU coordinates [x, y, z]

        Returns:
            (pixel_x, pixel_y) or None if projection fails/out of bounds
        """
        if not self.available:
            return None

        shot = self.shots.get(image_name)
        if shot is None:
            logger.debug(f"Image {image_name} not found in reconstruction")
            return None

        try:
            pixel_coords = self._project_to_camera(world_point, shot)
            if pixel_coords is None:
                return None
            px, py, img_w, img_h = pixel_coords
            if 0 <= px < img_w and 0 <= py < img_h:
                return px, py
            return None
        except Exception as e:
            logger.debug(f"World projection failed for {image_name}: {e}")
            return None

    def ortho_to_local(self, ortho_x: float, ortho_y: float) -> Tuple[float, float]:
        """Public wrapper for ortho pixel -> local ENU XY coordinates."""
        return self._ortho_to_world(ortho_x, ortho_y)

    def _ortho_to_world(self, ortho_x: float, ortho_y: float) -> Tuple[float, float]:
        """Convert orthophoto pixel to OpenSfM local ENU coordinates.

        The orthophoto is in UTM coordinates. OpenSfM uses local ENU (East-North-Up)
        coordinates relative to reference_lla. We need to:
        1. Convert ortho pixel to UTM using the GeoTIFF transform
        2. Subtract the reference point UTM to get local ENU (X=East, Y=North)

        IMPORTANT: The affine transform maps pixel CORNERS, not centers.
        To get the center of a pixel, we add 0.5 to both coordinates.
        This prevents a systematic half-pixel shift that gets amplified
        after reprojection (at ~3cm GSD, half pixel = 1.5cm error).
        """
        # Use the GeoTIFF transform to get UTM coordinates
        # Add 0.5 to convert from pixel corner to pixel CENTER
        transform = self.geo_converter.transform
        utm_x = transform.c + (ortho_x + 0.5) * transform.a + (ortho_y + 0.5) * transform.b
        utm_y = transform.f + (ortho_x + 0.5) * transform.d + (ortho_y + 0.5) * transform.e

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
    ) -> Optional[Tuple[float, float, int, int]]:
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
        img_w, img_h = self._get_camera_dimensions(camera)

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
        k3 = camera.get("k3", 0.0)
        p1 = camera.get("p1", 0.0)  # Tangential distortion
        p2 = camera.get("p2", 0.0)

        # Apply radial distortion (k1, k2, k3)
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Apply tangential distortion (p1, p2)
        x_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        x_distorted = x * radial_distortion + x_tangential
        y_distorted = y * radial_distortion + y_tangential

        # Convert to pixel coordinates
        # OpenSfM uses normalized coordinates where focal is relative to max(width, height)
        # The denormalization formula from OpenSfM docs:
        #   pixel = normalized * max(w,h) + (dimension - 1) / 2
        max_size = max(img_w, img_h)
        pixel_x = x_distorted * focal_x * max_size + (img_w - 1) / 2.0 + c_x * max_size
        pixel_y = y_distorted * focal_y * max_size + (img_h - 1) / 2.0 + c_y * max_size

        return pixel_x, pixel_y, img_w, img_h

    def _get_camera_dimensions(self, camera: dict) -> Tuple[int, int]:
        """Return image width/height for a camera model, with fallbacks."""
        width = camera.get("width", self.image_width)
        height = camera.get("height", self.image_height)
        return int(width), int(height)

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
