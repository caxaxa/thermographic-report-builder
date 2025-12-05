"""Match raw thermal images to defects using GPS coordinates."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ..models.defect import Panel
from ..io.s3_client import S3Client
from ..io.image_loader import load_raw_image_with_exif, save_image
from ..config import settings
from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter

logger = get_logger(__name__)

# Flight direction detection: 'south' means drone flying south, camera looking north
# Images taken while flying south need to be rotated 180° to orient north-up
FlightDirection = str  # 'north', 'south', or None


class GPSMatcher:
    """Match raw thermal images to defects based on GPS proximity."""

    def __init__(self, s3_client: S3Client, geo_converter: PixelToLatLonConverter):
        """
        Initialize GPS matcher.

        Args:
            s3_client: S3 client for downloading raw images
            geo_converter: Helper to convert pixels into geodetic coordinates
        """
        self.s3_client = s3_client
        self.geo_converter = geo_converter
        self.image_cache: Dict[str, dict] = {}  # Cache GPS data to avoid re-reading
        self.orientation_map: Dict[str, FlightDirection] = {}  # filename -> flight direction

    def match_images_to_panels(
        self,
        panel_grid: Dict[Tuple[int, int], Panel],
        temp_dir: Path,
        output_dir: Path,
    ) -> int:
        """
        Match raw thermal images to panels with defects.

        For each panel with defects, finds the closest raw image by GPS
        and saves it for the report.

        Args:
            panel_grid: Dictionary of panels
            temp_dir: Temporary directory for downloading images
            output_dir: Directory to save matched images

        Returns:
            Number of images matched
        """
        logger.info("Matching raw thermal images to defects via GPS")

        # Download and index all raw images by GPS
        self._index_raw_images(temp_dir)

        if not self.image_cache:
            logger.warning("No raw images with GPS data found")
            return 0

        # Determine flight direction for each image based on GPS trajectory
        self._compute_flight_directions()

        matched_count = 0

        for panel in panel_grid.values():
            if not panel.has_defects:
                continue

            # Process each defect type separately
            for defect_type in ["hotspots", "faulty_diodes", "offline_panels"]:
                defects = getattr(panel, defect_type, [])

                for defect in defects:
                    # Calculate GPS coordinates from the DEFECT's bbox center
                    # This ensures each defect gets matched to its closest raw image
                    defect_center_px = defect.bbox.center
                    defect_lon, defect_lat = self.geo_converter.pixel_to_lonlat(defect_center_px)

                    # Find closest image for THIS specific defect
                    closest_image = self._find_closest_image(defect_lat, defect_lon)

                    if closest_image:
                        # Use defect type string format for filename (without underscore)
                        defect_type_str = defect_type.replace("_", "")
                        filename = f"{defect_type_str}_({panel.panel_id}).jpg"
                        output_path = output_dir / filename

                        # Load image
                        img_path = Path(closest_image["path"])
                        img, _ = load_raw_image_with_exif(img_path)

                        # Get the source filename to check orientation
                        src_filename = Path(closest_image["path"]).name
                        flight_dir = self.orientation_map.get(src_filename)

                        # Rotate 180° if flying south (to orient image north-up)
                        if flight_dir == "south":
                            img = cv2.rotate(img, cv2.ROTATE_180)
                            logger.debug(f"Rotated {src_filename} 180° (flying south)")

                        # Resize to half size
                        img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

                        save_image(img_resized, output_path, quality=settings.jpeg_quality)
                        matched_count += 1

                        # Only save one image per panel per defect type
                        # (if panel has multiple hotspots, they all share one raw image)
                        break

        logger.info(f"Matched {matched_count} raw images to defects")
        return matched_count

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
