"""Match raw thermal images to defects using GPS coordinates."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from ..models.defect import Panel
from ..io.s3_client import S3Client
from ..io.image_loader import load_raw_image_with_exif, save_image
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GPSMatcher:
    """Match raw thermal images to defects based on GPS proximity."""

    def __init__(self, s3_client: S3Client):
        """
        Initialize GPS matcher.

        Args:
            s3_client: S3 client for downloading raw images
        """
        self.s3_client = s3_client
        self.image_cache: Dict[str, dict] = {}  # Cache GPS data to avoid re-reading

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

        matched_count = 0

        for panel in panel_grid.values():
            if not panel.has_defects:
                continue

            # Get panel geospatial centroid
            panel_lon = panel.hotspots[0].panel_centroid_geospatial.longitude if panel.hotspots else None
            panel_lat = panel.hotspots[0].panel_centroid_geospatial.latitude if panel.hotspots else None

            if panel_lon is None or panel_lat is None:
                continue

            # Find closest image
            closest_image = self._find_closest_image(panel_lat, panel_lon)

            if closest_image:
                # Copy image for each defect type
                for defect_type in ["hotspots", "faultydiodes", "offlinepanels"]:
                    defects = getattr(panel, defect_type, [])
                    if defects:
                        filename = f"{defect_type}_({panel.panel_id}).jpg"
                        output_path = output_dir / filename

                        # Load, resize, and save
                        img_path = Path(closest_image["path"])
                        img, _ = load_raw_image_with_exif(img_path)

                        # Resize to half size
                        img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

                        save_image(img_resized, output_path, quality=settings.jpeg_quality)
                        matched_count += 1

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
