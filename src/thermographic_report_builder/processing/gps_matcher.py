"""Match raw thermal images to defects using GPS coordinates or camera reprojection.

Supports two matching strategies:
1. Camera Reprojection (preferred): Uses OpenSfM reconstruction data to precisely
   project orthophoto coordinates back to raw image pixels. Provides exact pixel
   locations and enables temperature extraction.

2. GPS Proximity (fallback): Matches defects to the nearest raw image by GPS
   coordinates. Used when reconstruction data is not available.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.defect import Panel
from ..io.s3_client import S3Client
from ..io.image_loader import load_raw_image_with_exif, save_image
from ..config import settings
from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter
from .camera_projector import CameraProjector, RawImageMatch, load_reconstruction
from .thermal_extractor import ThermalExtractor, TemperatureReading
from .thermal_annotator import ThermalAnnotator

logger = get_logger(__name__)

# Flight direction detection: 'south' means drone flying south, camera looking north
# Images taken while flying south need to be rotated 180° to orient north-up
FlightDirection = str  # 'north', 'south', or None


@dataclass
class DefectMatch:
    """Result of matching a defect to a raw image."""

    image_name: str
    image_path: Path
    method: str  # 'reprojection' or 'gps'
    pixel_x: Optional[float] = None  # Pixel X in raw image (only for reprojection)
    pixel_y: Optional[float] = None  # Pixel Y in raw image (only for reprojection)
    temperature: Optional[TemperatureReading] = None  # Temperature data if available


class GPSMatcher:
    """Match raw thermal images to defects using reprojection or GPS proximity."""

    def __init__(
        self,
        s3_client: S3Client,
        geo_converter: PixelToLatLonConverter,
        reconstruction_data: Optional[dict] = None,
        enable_temperature: bool = True,
    ):
        """
        Initialize image matcher.

        Args:
            s3_client: S3 client for downloading raw images
            geo_converter: Helper to convert pixels into geodetic coordinates
            reconstruction_data: Optional OpenSfM reconstruction.json data.
                                If provided, enables precise camera reprojection.
            enable_temperature: If True, extract temperature data when reprojection available
        """
        self.s3_client = s3_client
        self.geo_converter = geo_converter
        self.image_cache: Dict[str, dict] = {}  # Cache GPS data to avoid re-reading
        self.orientation_map: Dict[str, FlightDirection] = {}  # filename -> flight direction

        # Initialize camera projector if reconstruction data available
        self.camera_projector = CameraProjector(
            reconstruction_json=reconstruction_data,
            geo_converter=geo_converter,
        )

        # Initialize thermal extractor and annotator if enabled
        if enable_temperature:
            self.thermal_extractor = ThermalExtractor()
            self.thermal_annotator = ThermalAnnotator(self.thermal_extractor)
        else:
            self.thermal_extractor = None
            self.thermal_annotator = None

        if self.camera_projector.available:
            logger.info("Image matcher initialized with camera reprojection")
        else:
            logger.info("Image matcher initialized with GPS-only matching (no reconstruction data)")

    def match_images_to_panels(
        self,
        panel_grid: Dict[Tuple[int, int], Panel],
        temp_dir: Path,
        output_dir: Path,
    ) -> Tuple[int, Dict[str, DefectMatch]]:
        """
        Match raw thermal images to panels with defects.

        For each panel with defects, finds the best raw image using either:
        - Camera reprojection (if reconstruction data available)
        - GPS proximity (fallback)

        Args:
            panel_grid: Dictionary of panels
            temp_dir: Temporary directory for downloading images
            output_dir: Directory to save matched images

        Returns:
            Tuple of (number of images matched, dict of panel_id -> DefectMatch)
        """
        method = "reprojection" if self.camera_projector.available else "GPS"
        logger.info(f"Matching raw thermal images to defects via {method}")

        # Download and index all raw images by GPS
        self._index_raw_images(temp_dir)

        if not self.image_cache:
            logger.warning("No raw images with GPS data found")
            return 0, {}

        # Determine flight direction for each image based on GPS trajectory
        self._compute_flight_directions()

        matched_count = 0
        defect_matches: Dict[str, DefectMatch] = {}

        for panel in panel_grid.values():
            if not panel.has_defects:
                continue

            # Process each defect type separately
            for defect_type in ["hotspots", "faulty_diodes", "offline_panels"]:
                defects = getattr(panel, defect_type, [])

                for defect in defects:
                    defect_center_px = defect.bbox.center

                    # Try camera reprojection first, fallback to GPS
                    match = self._find_best_match(
                        ortho_x=defect_center_px[0],
                        ortho_y=defect_center_px[1],
                        temp_dir=temp_dir,
                    )

                    if match:
                        # Use defect type string format for filename (without underscore)
                        defect_type_str = defect_type.replace("_", "")
                        filename = f"{defect_type_str}_({panel.panel_id}).jpg"
                        output_path = output_dir / filename

                        # Load image
                        img, _ = load_raw_image_with_exif(match.image_path)

                        # Get the source filename to check orientation
                        src_filename = match.image_name
                        flight_dir = self.orientation_map.get(src_filename)

                        # Store original (unrotated) pixel coordinates for raw file operations
                        # Temperature extraction and annotation need the ORIGINAL coordinates
                        # because they read directly from the unrotated raw R-JPEG file
                        original_pixel_x = match.pixel_x
                        original_pixel_y = match.pixel_y

                        # Log coordinate tracing for debugging hotspot location
                        logger.info(
                            f"Projecting {panel.panel_id}: ortho=({defect_center_px[0]:.1f}, {defect_center_px[1]:.1f}) "
                            f"-> raw=({original_pixel_x:.1f}, {original_pixel_y:.1f}) in {match.image_name}"
                        )

                        # Rotate 180° if flying south (to orient image north-up)
                        if flight_dir == "south":
                            img = cv2.rotate(img, cv2.ROTATE_180)
                            # Also rotate the pixel coordinates for display
                            if match.pixel_x is not None and match.pixel_y is not None:
                                img_h, img_w = img.shape[:2]
                                match.pixel_x = img_w - match.pixel_x
                                match.pixel_y = img_h - match.pixel_y
                            logger.debug(f"Rotated {src_filename} 180° (flying south)")

                        # Extract temperature if we have precise pixel coordinates
                        # NOTE: Use ORIGINAL coordinates since we read from the unrotated raw file
                        if (
                            original_pixel_x is not None
                            and original_pixel_y is not None
                            and self.thermal_extractor
                            and self.thermal_extractor.available
                        ):
                            # We've already selected the raw image with the highest temperature
                            # at the projected point (in _find_best_match), so no need to search
                            # for a local hotspot - the projection should be accurate
                            match.temperature = self.thermal_extractor.extract_temperature_at_pixel(
                                image_path=match.image_path,
                                pixel_x=int(original_pixel_x),
                                pixel_y=int(original_pixel_y),
                            )
                            if match.temperature:
                                logger.info(
                                    f"Temperature at {panel.panel_id}: "
                                    f"defect={match.temperature.defect_temp_celsius}°C, "
                                    f"panel_avg={match.temperature.panel_avg_celsius}°C, "
                                    f"ΔT={match.temperature.delta_t}°C ({match.temperature.severity})"
                                )

                                # Create annotated thermal image with temperature overlay
                                # NOTE: Use ORIGINAL coordinates since annotator reads from unrotated raw file
                                if self.thermal_annotator:
                                    # Get geospatial coordinates for the defect
                                    defect_lon, defect_lat = self.geo_converter.pixel_to_lonlat(
                                        (defect_center_px[0], defect_center_px[1])
                                    )

                                    annotated = self.thermal_annotator.create_annotated_image(
                                        raw_image_path=match.image_path,
                                        defect_pixel_x=original_pixel_x,
                                        defect_pixel_y=original_pixel_y,
                                        panel_id=panel.panel_id,
                                        panel_row=panel.row,
                                        panel_column=panel.column,
                                        latitude=defect_lat,
                                        longitude=defect_lon,
                                    )

                                    if annotated:
                                        # Save annotated image
                                        annotated_filename = f"{defect_type_str}_({panel.panel_id})_annotated.jpg"
                                        annotated_path = output_dir / annotated_filename
                                        self.thermal_annotator.save_annotated_image(
                                            annotated, annotated_path
                                        )

                        # Resize to half size
                        img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

                        save_image(img_resized, output_path, quality=settings.jpeg_quality)
                        matched_count += 1

                        # Store match info
                        match_key = f"{defect_type}_{panel.panel_id}"
                        defect_matches[match_key] = match

                        # Only save one image per panel per defect type
                        break

        reprojection_count = sum(1 for m in defect_matches.values() if m.method == "reprojection")
        gps_count = sum(1 for m in defect_matches.values() if m.method == "gps")
        temp_count = sum(1 for m in defect_matches.values() if m.temperature is not None)

        logger.info(
            f"Matched {matched_count} images: "
            f"{reprojection_count} via reprojection, {gps_count} via GPS, "
            f"{temp_count} with temperature data"
        )

        return matched_count, defect_matches

    def _find_best_match(
        self,
        ortho_x: float,
        ortho_y: float,
        temp_dir: Path,
    ) -> Optional[DefectMatch]:
        """
        Find the best raw image match for an orthophoto location.

        For defects (hotspots), we need to find the raw image where the defect
        actually appears as a thermal anomaly. This is done by:
        1. Getting all candidate images that can see this location
        2. Checking the temperature at the projected pixel in each
        3. Selecting the image with highest temperature delta (actual hotspot)

        Falls back to GPS matching if reprojection not available.

        Args:
            ortho_x: X coordinate in orthophoto
            ortho_y: Y coordinate in orthophoto
            temp_dir: Directory containing downloaded raw images

        Returns:
            DefectMatch or None
        """
        # Try camera reprojection first
        if self.camera_projector.available:
            # Get ALL candidate matches (not just the "best" by centrality)
            all_matches = self.camera_projector.project_ortho_to_raw(ortho_x, ortho_y)

            if all_matches and self.thermal_extractor and self.thermal_extractor.available:
                # Score each candidate by temperature at projected point
                best_match = None
                best_temp_delta = -999

                for match in all_matches:  # Check ALL candidates that can see this point
                    image_path = temp_dir / match.image_name

                    if not image_path.exists():
                        continue

                    try:
                        # Get temperature at the projected pixel
                        temp_array = self.thermal_extractor.get_full_temperature_array(image_path)
                        if temp_array is None:
                            continue

                        # IMPORTANT: Thermal array is 640x512, visual image is 1280x1024
                        # Scale coordinates from visual to thermal resolution
                        h, w = temp_array.shape
                        scale_x = w / 1280.0  # 640/1280 = 0.5
                        scale_y = h / 1024.0  # 512/1024 = 0.5
                        px = int(max(0, min(w - 1, match.pixel_x * scale_x)))
                        py = int(max(0, min(h - 1, match.pixel_y * scale_y)))

                        temp_at_point = temp_array[py, px]
                        mean_temp = temp_array.mean()
                        temp_delta = temp_at_point - mean_temp

                        if temp_delta > best_temp_delta:
                            best_temp_delta = temp_delta
                            best_match = match

                    except Exception as e:
                        logger.debug(f"Failed to score {match.image_name}: {e}")
                        continue

                logger.info(
                    f"Temperature scoring: checked {len(all_matches)} candidates, "
                    f"best delta={best_temp_delta:.1f}°C"
                )

                if best_match and best_temp_delta > 0:
                    image_path = temp_dir / best_match.image_name
                    logger.info(
                        f"Selected {best_match.image_name} with temp delta +{best_temp_delta:.1f}°C"
                    )
                    return DefectMatch(
                        image_name=best_match.image_name,
                        image_path=image_path,
                        method="reprojection",
                        pixel_x=best_match.pixel_x,
                        pixel_y=best_match.pixel_y,
                    )

            # Fallback to most central match if temperature scoring fails
            if all_matches:
                for match in all_matches:
                    image_path = temp_dir / match.image_name
                    if image_path.exists():
                        return DefectMatch(
                            image_name=match.image_name,
                            image_path=image_path,
                            method="reprojection",
                            pixel_x=match.pixel_x,
                            pixel_y=match.pixel_y,
                        )

        # Fallback to GPS matching
        defect_lon, defect_lat = self.geo_converter.pixel_to_lonlat((ortho_x, ortho_y))
        closest = self._find_closest_image(defect_lat, defect_lon)

        if closest:
            return DefectMatch(
                image_name=Path(closest["path"]).name,
                image_path=Path(closest["path"]),
                method="gps",
                pixel_x=None,
                pixel_y=None,
            )

        return None

    def _index_raw_images(self, temp_dir: Path) -> None:
        """Download and index all raw images with GPS data."""
        logger.info("Indexing raw thermal images")

        raw_image_keys = self.s3_client.list_raw_images()

        for idx, s3_key in enumerate(raw_image_keys, 1):
            filename = Path(s3_key).name
            local_path = temp_dir / filename

            try:
                # Download image
                self.s3_client.download_raw_image(s3_key, local_path)

                # Load and extract GPS
                _, exif = load_raw_image_with_exif(local_path)

                if exif.get("has_gps") and "latitude" in exif and "longitude" in exif:
                    self.image_cache[filename] = exif
                    logger.debug(f"Indexed {filename}: GPS ({exif['latitude']}, {exif['longitude']})")

            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
                continue

            # Log progress every 50 images
            if idx % 50 == 0:
                logger.info(f"Indexed {idx}/{len(raw_image_keys)} images ({len(self.image_cache)} with GPS)")

        logger.info(f"Indexed {len(self.image_cache)} images with GPS data")

    def _find_closest_image(self, target_lat: float, target_lon: float) -> dict | None:
        """
        Find image with GPS coordinates closest to target.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude

        Returns:
            Image metadata dict or None
        """
        min_dist = float("inf")
        closest = None

        for image_meta in self.image_cache.values():
            img_lat = image_meta["latitude"]
            img_lon = image_meta["longitude"]

            # Simple Euclidean distance (good enough for small areas)
            dist = np.hypot(target_lat - img_lat, target_lon - img_lon)

            if dist < min_dist:
                min_dist = dist
                closest = image_meta

        return closest

    def _compute_flight_directions(self) -> None:
        """
        Determine flight direction for each image based on GPS trajectory.

        By comparing consecutive images (sorted by filename), we can determine
        if the drone was flying north or south:
        - If next image has lower latitude -> flying south
        - If next image has higher latitude -> flying north

        Images taken while flying south need to be rotated 180° because
        the thermal camera is pointed down and the drone's nose (north of camera)
        appears at the top of the frame. When flying south, this means
        "south" is at the top, so we rotate to put "north" at top.
        """
        if not self.image_cache:
            return

        # Get images sorted by filename (captures sequential flight order)
        sorted_images: List[Tuple[str, dict]] = sorted(
            [(fname, meta) for fname, meta in self.image_cache.items()
             if meta.get("latitude") is not None],
            key=lambda x: x[0]
        )

        if len(sorted_images) < 2:
            logger.warning("Not enough images to determine flight direction")
            return

        # Compare each image with the next to determine direction
        south_count = 0
        north_count = 0

        for i in range(len(sorted_images) - 1):
            current_fname, current_meta = sorted_images[i]
            next_fname, next_meta = sorted_images[i + 1]

            current_lat = current_meta["latitude"]
            next_lat = next_meta["latitude"]

            if next_lat < current_lat:
                # Latitude decreasing -> flying south
                self.orientation_map[current_fname] = "south"
                south_count += 1
            else:
                # Latitude increasing or same -> flying north
                self.orientation_map[current_fname] = "north"
                north_count += 1

        # Last image: assign same direction as previous (or None)
        if sorted_images:
            last_fname = sorted_images[-1][0]
            if len(sorted_images) >= 2:
                prev_fname = sorted_images[-2][0]
                self.orientation_map[last_fname] = self.orientation_map.get(prev_fname)
            else:
                self.orientation_map[last_fname] = None

        logger.info(
            "Flight direction analysis: %d south-facing, %d north-facing images",
            south_count, north_count
        )
