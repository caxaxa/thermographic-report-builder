"""Match raw thermal images to defects using GPS coordinates or camera reprojection.

Supports two matching strategies:
1. Camera Reprojection (preferred): Uses OpenSfM reconstruction data to precisely
   project orthophoto coordinates back to raw image pixels. Provides exact pixel
   locations and enables temperature extraction.

2. GPS Proximity (fallback): Matches defects to the nearest raw image by GPS
   coordinates. Used when reconstruction data is not available.
"""

import cv2
import re
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pattern to match UUID prefix in uploaded filenames (e.g., "01KE5WBPHE280VNG9DJF0B7W1H_DJI_...")
UUID_PREFIX_PATTERN = re.compile(r'^[A-Z0-9]{26}_(.+)$')

from ..models.defect import Panel
from ..models.annotation_manifest import (
    AnnotationPoint,
    AnnotationEntry,
    AnnotationManifest,
    OverrideManifest,
)
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
        dsm_path: Optional[Path] = None,
        enable_temperature: bool = True,
    ):
        """
        Initialize image matcher.

        Args:
            s3_client: S3 client for downloading raw images
            geo_converter: Helper to convert pixels into geodetic coordinates
            reconstruction_data: Optional OpenSfM reconstruction.json data.
                                If provided, enables precise camera reprojection.
            dsm_path: Optional path to DSM (Digital Surface Model) for precise elevation.
                     Using DSM significantly improves projection accuracy.
            enable_temperature: If True, extract temperature data when reprojection available
        """
        self.s3_client = s3_client
        self.geo_converter = geo_converter
        self.image_cache: Dict[str, dict] = {}  # Cache GPS data to avoid re-reading
        self.orientation_map: Dict[str, FlightDirection] = {}  # filename -> flight direction
        self.annotation_entries: List[AnnotationEntry] = []  # Collect annotations for manifest

        # Initialize camera projector if reconstruction data available
        self.camera_projector = CameraProjector(
            reconstruction_json=reconstruction_data,
            geo_converter=geo_converter,
            dsm_path=dsm_path,
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

            # Track if we've saved the main panel image (context, location, drone)
            panel_image_saved = False

            # Process each defect type separately
            for defect_type in ["hotspots", "faulty_diodes", "offline_panels"]:
                defects = getattr(panel, defect_type, [])

                # Only create thermal analysis for hotspots (not faulty_diodes or offline_panels)
                create_thermal_analysis = defect_type == "hotspots"

                # Process ALL defects
                for defect_idx, defect in enumerate(defects):
                    defect_center_px = defect.bbox.center
                    defect_lon, defect_lat = self.geo_converter.pixel_to_lonlat(defect_center_px)

                    # STEP 1: For HOTSPOTS, use temperature-scored matching to find best image
                    # This uses reconstruction.json to find all images that can see the point,
                    # then scores by temperature to find the one with the actual hotspot
                    # For other defect types, use GPS-only matching
                    match = None

                    if create_thermal_analysis and self.thermal_extractor and self.thermal_extractor.available:
                        # Use temperature-based scoring to find image with actual hotspot
                        match = self._find_best_match(
                            ortho_x=defect_center_px[0],
                            ortho_y=defect_center_px[1],
                            temp_dir=temp_dir,
                        )
                        if match:
                            logger.info(
                                f"Temperature-scored match for {panel.panel_id} defect {defect_idx}: "
                                f"{match.image_name} via {match.method}"
                            )

                    # Fallback to GPS matching if temperature scoring didn't work
                    if not match:
                        closest_image = self._find_closest_image(defect_lat, defect_lon)
                        if not closest_image:
                            logger.warning(f"No matching image found for {panel.panel_id} defect {defect_idx}")
                            continue

                        image_path = Path(closest_image["path"])
                        image_name = image_path.name

                        # Try camera reprojection to get pixel coordinates
                        pixel_x, pixel_y = None, None
                        if self.camera_projector.available:
                            projection = self.camera_projector.project_to_specific_image(
                                ortho_x=defect_center_px[0],
                                ortho_y=defect_center_px[1],
                                image_name=image_name,
                            )
                            if projection:
                                pixel_x, pixel_y = projection

                        match = DefectMatch(
                            image_name=image_name,
                            image_path=image_path,
                            method="gps" if pixel_x is None else "reprojection",
                            pixel_x=pixel_x,
                            pixel_y=pixel_y,
                        )

                    if match:
                        # Get the source filename to check orientation
                        src_filename = match.image_name
                        flight_dir = self.orientation_map.get(src_filename)

                        # Store original (unrotated) pixel coordinates for raw file operations
                        original_pixel_x = match.pixel_x
                        original_pixel_y = match.pixel_y

                        # Log coordinate tracing for debugging
                        if original_pixel_x is not None:
                            logger.info(
                                f"Projecting {panel.panel_id} defect {defect_idx}: "
                                f"ortho=({defect_center_px[0]:.1f}, {defect_center_px[1]:.1f}) "
                                f"-> raw=({original_pixel_x:.1f}, {original_pixel_y:.1f}) in {match.image_name}"
                            )

                        # Save the main panel image ONCE per panel (for context, location, drone images)
                        if not panel_image_saved:
                            # Load image
                            img, _ = load_raw_image_with_exif(match.image_path)

                            # Rotate 180° if flying south
                            if flight_dir == "south":
                                img = cv2.rotate(img, cv2.ROTATE_180)
                                logger.debug(f"Rotated {src_filename} 180° (flying south)")

                            # Resize to half size for the main drone image
                            img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

                            # Use defect type string format for filename
                            defect_type_str = defect_type.replace("_", "")
                            filename = f"{defect_type_str}_({panel.panel_id}).jpg"
                            output_path = output_dir / filename
                            save_image(img_resized, output_path, quality=settings.jpeg_quality)

                            panel_image_saved = True
                            matched_count += 1

                            # Store match info for the panel
                            match_key = f"{defect_type}_{panel.panel_id}"
                            defect_matches[match_key] = match

                        # Create thermal analysis image for HOTSPOTS ONLY
                        # Even if reprojection fails, try with image center as fallback
                        if (
                            create_thermal_analysis
                            and self.thermal_extractor
                            and self.thermal_extractor.available
                        ):
                            # Fallback pixel coordinates if projection failed
                            # Use image center (640, 512) - panel is likely visible there
                            if original_pixel_x is None or original_pixel_y is None:
                                original_pixel_x = 640.0  # Center of 1280
                                original_pixel_y = 512.0  # Center of 1024
                                logger.warning(
                                    f"Reprojection failed for {panel.panel_id} defect {defect_idx}, "
                                    f"using image center as fallback"
                                )

                            # Refine to the hottest nearby pixel to avoid edge offsets
                            try:
                                hot_x, hot_y, hot_temp = self.thermal_extractor.find_local_hotspot(
                                    image_path=match.image_path,
                                    initial_x=int(original_pixel_x),
                                    initial_y=int(original_pixel_y),
                                    search_radius=150,  # Increased to catch slight projection drift
                                )
                                if (hot_x, hot_y) != (int(original_pixel_x), int(original_pixel_y)):
                                    logger.info(
                                        f"Refined hotspot for {panel.panel_id} defect {defect_idx}: "
                                        f"({original_pixel_x:.1f},{original_pixel_y:.1f}) -> ({hot_x},{hot_y}) "
                                        f"with temp {hot_temp:.1f}C"
                                    )
                                    original_pixel_x, original_pixel_y = float(hot_x), float(hot_y)
                                    match.pixel_x, match.pixel_y = original_pixel_x, original_pixel_y
                            except Exception as refine_err:
                                logger.debug(f"Hotspot refinement skipped: {refine_err}")

                            # Extract temperature data
                            match.temperature = self.thermal_extractor.extract_temperature_at_pixel(
                                image_path=match.image_path,
                                pixel_x=int(original_pixel_x),
                                pixel_y=int(original_pixel_y),
                            )
                            if match.temperature:
                                logger.info(
                                    f"Temperature at {panel.panel_id} defect {defect_idx}: "
                                    f"defect={match.temperature.defect_temp_celsius}°C, "
                                    f"panel_avg={match.temperature.panel_avg_celsius}°C, "
                                    f"ΔT={match.temperature.delta_t}°C ({match.temperature.severity})"
                                )

                            # Create annotated thermal image for THIS hotspot
                            if self.thermal_annotator:
                                # Project panel bbox corners to raw image coordinates
                                panel_bbox_visual = None
                                if self.camera_projector.available:
                                    # Get panel corners in orthophoto coordinates
                                    ortho_left = panel.bbox.left
                                    ortho_top = panel.bbox.top
                                    ortho_right = panel.bbox.right
                                    ortho_bottom = panel.bbox.bottom

                                    # Project all 4 corners to raw image
                                    corners_raw = []
                                    for ox, oy in [
                                        (ortho_left, ortho_top),
                                        (ortho_right, ortho_top),
                                        (ortho_right, ortho_bottom),
                                        (ortho_left, ortho_bottom),
                                    ]:
                                        proj = self.camera_projector.project_to_specific_image(
                                            ortho_x=ox,
                                            ortho_y=oy,
                                            image_name=match.image_name,
                                        )
                                        if proj:
                                            corners_raw.append(proj)

                                    # Accept 3+ corners (some might be out of frame)
                                    if len(corners_raw) >= 3:
                                        xs = [c[0] for c in corners_raw]
                                        ys = [c[1] for c in corners_raw]
                                        # Clamp to image bounds (1280x1024)
                                        panel_bbox_visual = (
                                            max(0, int(min(xs))),  # left
                                            max(0, int(min(ys))),  # top
                                            min(1280, int(max(xs))),  # right
                                            min(1024, int(max(ys))),  # bottom
                                        )
                                        logger.debug(
                                            f"Panel {panel.panel_id} bbox in raw image: {panel_bbox_visual} "
                                            f"(from {len(corners_raw)} corners)"
                                        )
                                    elif len(corners_raw) > 0:
                                        # Even with 1-2 corners, create a bbox around the defect point
                                        # Use projected defect point as center with padding
                                        padding = 100  # pixels
                                        panel_bbox_visual = (
                                            max(0, int(original_pixel_x - padding)),
                                            max(0, int(original_pixel_y - padding)),
                                            min(1280, int(original_pixel_x + padding)),
                                            min(1024, int(original_pixel_y + padding)),
                                        )
                                        logger.debug(
                                            f"Panel {panel.panel_id} using defect-centered bbox: {panel_bbox_visual}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Panel {panel.panel_id}: No corners projected, using defect-centered bbox"
                                        )
                                        padding = 100
                                        panel_bbox_visual = (
                                            max(0, int(original_pixel_x - padding)),
                                            max(0, int(original_pixel_y - padding)),
                                            min(1280, int(original_pixel_x + padding)),
                                            min(1024, int(original_pixel_y + padding)),
                                        )
                                else:
                                    # Camera projector not available - use defect-centered bbox
                                    logger.debug(
                                        f"Panel {panel.panel_id}: No camera projector, using defect-centered bbox"
                                    )
                                    padding = 100
                                    panel_bbox_visual = (
                                        max(0, int(original_pixel_x - padding)),
                                        max(0, int(original_pixel_y - padding)),
                                        min(1280, int(original_pixel_x + padding)),
                                        min(1024, int(original_pixel_y + padding)),
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
                                    defect_index=defect_idx,  # Pass defect index for labeling
                                    panel_bbox_visual=panel_bbox_visual,  # Constrain search to panel
                                )

                                if annotated:
                                    # Save annotated image with defect index if multiple defects
                                    defect_type_str = defect_type.replace("_", "")
                                    if len(defects) > 1:
                                        # Multiple defects: include index in filename
                                        annotated_filename = f"{defect_type_str}_({panel.panel_id})_defeito{defect_idx + 1}_annotated.jpg"
                                    else:
                                        # Single defect: no index needed
                                        annotated_filename = f"{defect_type_str}_({panel.panel_id})_annotated.jpg"
                                    annotated_path = output_dir / annotated_filename
                                    self.thermal_annotator.save_annotated_image(
                                        annotated, annotated_path
                                    )
                                    logger.info(f"Saved thermal analysis: {annotated_filename}")

                                    # Create annotated raw image with zoom box indicator
                                    # This shows users where the thermal crop comes from
                                    self._save_annotated_raw_image(
                                        raw_image_path=match.image_path,
                                        hot_point_x=annotated.hot_cold.hot_x,
                                        hot_point_y=annotated.hot_cold.hot_y,
                                        zoom_size=200,  # Same as thermal_annotator default
                                        output_dir=output_dir,
                                        defect_type_str=defect_type_str,
                                        panel_id=panel.panel_id,
                                        flight_dir=flight_dir,
                                    )

                                    # Record annotation entry for manifest
                                    defect_id = f"{defect_type_str}_({panel.panel_id})_defeito{defect_idx + 1}"
                                    s3_raw_path = f"s3://{settings.uploads_bucket}/{settings.user_id}/projects/{settings.project_id}/images/{match.image_name}"

                                    # Convert visual coords to thermal coords for manifest
                                    from ..utils.thermal_alignment import visual_to_thermal
                                    hot_thermal_x, hot_thermal_y = visual_to_thermal(
                                        annotated.hot_cold.hot_x, annotated.hot_cold.hot_y
                                    )
                                    cold_thermal_x, cold_thermal_y = visual_to_thermal(
                                        annotated.hot_cold.cold_x, annotated.hot_cold.cold_y
                                    )

                                    entry = AnnotationEntry(
                                        defect_id=defect_id,
                                        panel_id=panel.panel_id,
                                        defect_type=defect_type,
                                        defect_index=defect_idx + 1,
                                        raw_image_path=s3_raw_path,
                                        raw_image_name=match.image_name,
                                        annotated_image=annotated_filename,
                                        hot_point=AnnotationPoint(
                                            x=int(hot_thermal_x),
                                            y=int(hot_thermal_y),
                                            temp=annotated.hot_cold.hot_temp,
                                        ),
                                        cold_point=AnnotationPoint(
                                            x=int(cold_thermal_x),
                                            y=int(cold_thermal_y),
                                            temp=annotated.hot_cold.cold_temp,
                                        ),
                                        delta_t=annotated.hot_cold.hot_temp - annotated.hot_cold.cold_temp,
                                        severity=annotated.severity,
                                    )
                                    self.annotation_entries.append(entry)
                                else:
                                    logger.warning(
                                        f"Failed to create thermal analysis for {panel.panel_id} defect {defect_idx}: "
                                        f"annotator returned None (pixel={original_pixel_x:.1f},{original_pixel_y:.1f}, "
                                        f"bbox={panel_bbox_visual})"
                                    )

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
                    # Find actual file path (handles UUID prefix mismatch)
                    image_path = self._find_local_path_for_reconstruction_name(
                        match.image_name, temp_dir
                    )

                    if image_path is None:
                        continue

                    try:
                        # Get temperature at the projected pixel
                        temp_array = self.thermal_extractor.get_full_temperature_array(image_path)
                        if temp_array is None:
                            continue

                        # Convert from visual (1280x1024) to thermal (640x512) coordinates
                        # This applies any configured alignment offset
                        from ..utils.thermal_alignment import visual_to_thermal, clamp_thermal_coords

                        thermal_x, thermal_y = visual_to_thermal(match.pixel_x, match.pixel_y)
                        px, py = clamp_thermal_coords(thermal_x, thermal_y)

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
                    image_path = self._find_local_path_for_reconstruction_name(
                        best_match.image_name, temp_dir
                    )
                    if image_path:
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
                    image_path = self._find_local_path_for_reconstruction_name(
                        match.image_name, temp_dir
                    )
                    if image_path:
                        return DefectMatch(
                            image_name=match.image_name,
                            image_path=image_path,
                            method="reprojection",
                            pixel_x=match.pixel_x,
                            pixel_y=match.pixel_y,
                        )

        # No GPS fallback here - only use reconstruction.json images
        # GPS matching is handled at a higher level only when reconstruction unavailable
        return None

    def _save_annotated_raw_image(
        self,
        raw_image_path: Path,
        hot_point_x: int,
        hot_point_y: int,
        zoom_size: int,
        output_dir: Path,
        defect_type_str: str,
        panel_id: str,
        flight_dir: str,
    ) -> None:
        """
        Save annotated raw drone image with zoom box indicator.

        Creates a version of the raw thermal image with a blue rectangle
        showing where the thermal analysis crop is taken from. This helps
        users understand the context of the zoomed thermal analysis.

        Args:
            raw_image_path: Path to raw thermal image
            hot_point_x: X coordinate of hottest point (visual coords)
            hot_point_y: Y coordinate of hottest point (visual coords)
            zoom_size: Size of zoom window used in thermal analysis
            output_dir: Directory to save output
            defect_type_str: Defect type string (e.g., "hotspots")
            panel_id: Panel identifier
            flight_dir: Flight direction for rotation
        """
        try:
            # Load raw image
            img, _ = load_raw_image_with_exif(raw_image_path)

            # Rotate if flying south (same as main image)
            if flight_dir == "south":
                img = cv2.rotate(img, cv2.ROTATE_180)
                # Also transform coordinates for rotated image
                img_h, img_w = img.shape[:2]
                hot_point_x = img_w - hot_point_x
                hot_point_y = img_h - hot_point_y

            # Calculate zoom box bounds (same as thermal_annotator)
            half_size = zoom_size // 2
            img_h, img_w = img.shape[:2]
            x1 = max(0, hot_point_x - half_size)
            y1 = max(0, hot_point_y - half_size)
            x2 = min(img_w, hot_point_x + half_size)
            y2 = min(img_h, hot_point_y + half_size)

            # Draw blue rectangle showing zoom area
            # Use thick stroke for visibility
            stroke_thickness = 6
            corner_thickness = 8
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 128, 0),  # BGR: Light blue/cyan color
                stroke_thickness,
            )

            # Add corner markers for extra visibility
            corner_len = 30
            # Top-left corner
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), (255, 128, 0), corner_thickness)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), (255, 128, 0), corner_thickness)
            # Top-right corner
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), (255, 128, 0), corner_thickness)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), (255, 128, 0), corner_thickness)
            # Bottom-left corner
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), (255, 128, 0), corner_thickness)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), (255, 128, 0), corner_thickness)
            # Bottom-right corner
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), (255, 128, 0), corner_thickness)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), (255, 128, 0), corner_thickness)

            # Resize to half size (same as main drone image)
            img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # Save with _zoombox suffix (this will be used for frontend hover)
            filename = f"{defect_type_str}_({panel_id})_zoombox.jpg"
            output_path = output_dir / filename
            save_image(img_resized, output_path, quality=settings.jpeg_quality)
            logger.debug(f"Saved annotated raw image with zoom box: {filename}")

        except Exception as e:
            logger.warning(f"Failed to create annotated raw image for {panel_id}: {e}")

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

    def _find_local_path_for_reconstruction_name(
        self, reconstruction_name: str, temp_dir: Path
    ) -> Optional[Path]:
        """
        Find the actual local file path for a reconstruction.json image name.

        Reconstruction.json uses original DJI filenames (e.g., "DJI_xxx.JPG"),
        but uploaded files may have UUID prefixes (e.g., "01KE5WBPHE_DJI_xxx.JPG").

        Args:
            reconstruction_name: Image name from reconstruction.json (e.g., "DJI_xxx.JPG")
            temp_dir: Directory containing downloaded raw images

        Returns:
            Path to the actual file, or None if not found
        """
        # First try exact match
        exact_path = temp_dir / reconstruction_name
        if exact_path.exists():
            return exact_path

        # Search for file with UUID prefix that ends with the reconstruction name
        for local_file in temp_dir.iterdir():
            if local_file.is_file():
                # Check if filename ends with the reconstruction name (after UUID prefix)
                match = UUID_PREFIX_PATTERN.match(local_file.name)
                if match and match.group(1) == reconstruction_name:
                    return local_file

        return None

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

        for filename, image_meta in self.image_cache.items():
            img_lat = image_meta["latitude"]
            img_lon = image_meta["longitude"]

            # Simple Euclidean distance (good enough for small areas)
            dist = np.hypot(target_lat - img_lat, target_lon - img_lon)

            if dist < min_dist:
                min_dist = dist
                closest = image_meta

        return closest

    def _find_candidate_images(
        self,
        target_lat: float,
        target_lon: float,
        max_distance_deg: float = 0.0003,  # ~25-30m at typical latitudes
        max_candidates: int = 5,
    ) -> List[dict]:
        """
        Find multiple candidate images within distance, sorted by proximity.

        Instead of returning just the closest image, this returns multiple
        candidates so we can score them by temperature to find the one
        that actually shows the hotspot.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            max_distance_deg: Maximum distance in degrees (~0.0003 = ~30m)
            max_candidates: Maximum number of candidates to return

        Returns:
            List of image metadata dicts, sorted by distance (closest first)
        """
        candidates = []

        for image_meta in self.image_cache.values():
            img_lat = image_meta["latitude"]
            img_lon = image_meta["longitude"]

            # Simple Euclidean distance in degrees
            dist = np.hypot(target_lat - img_lat, target_lon - img_lon)

            if dist <= max_distance_deg:
                candidates.append((dist, image_meta))

        # Sort by distance, return top N
        candidates.sort(key=lambda x: x[0])
        return [meta for _, meta in candidates[:max_candidates]]

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

    def export_annotation_manifest(self) -> AnnotationManifest:
        """
        Export all collected annotations as a manifest.

        The manifest can be used by a frontend to display annotations for human review,
        and to generate override files for re-rendering with corrected positions.

        Returns:
            AnnotationManifest with all annotations from this session
        """
        return AnnotationManifest(
            project_id=settings.project_id,
            user_id=settings.user_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            annotations=self.annotation_entries,
        )

    def apply_overrides(
        self,
        overrides: OverrideManifest,
        raw_images_dir: Path,
        output_dir: Path,
    ) -> int:
        """
        Re-render annotations with human-provided override coordinates.

        For each override in the manifest, re-renders the annotated thermal image
        using the specified hot/cold point coordinates instead of auto-detection.

        Args:
            overrides: Override manifest from human review
            raw_images_dir: Directory containing raw thermal images
            output_dir: Directory to save re-rendered images

        Returns:
            Number of annotations successfully re-rendered
        """
        if not self.thermal_annotator:
            logger.error("Cannot apply overrides: thermal annotator not initialized")
            return 0

        re_rendered = 0

        for override in overrides.overrides:
            # Find the original annotation entry
            entry = self._find_annotation_entry(override.defect_id)
            if not entry:
                logger.warning(f"Override for unknown defect: {override.defect_id}")
                continue

            # Determine final hot/cold points (use override if provided, else original)
            hot_point = override.hot_point if override.hot_point else entry.hot_point
            cold_point = override.cold_point if override.cold_point else entry.cold_point

            # Re-render with specified coordinates
            raw_image_path = raw_images_dir / entry.raw_image_name
            if not raw_image_path.exists():
                logger.warning(f"Raw image not found for override: {entry.raw_image_name}")
                continue

            try:
                annotated = self.thermal_annotator.create_annotated_image_with_points(
                    raw_image_path=raw_image_path,
                    hot_point=hot_point,
                    cold_point=cold_point,
                    panel_id=entry.panel_id,
                    defect_index=entry.defect_index,
                )

                if annotated:
                    output_path = output_dir / entry.annotated_image
                    self.thermal_annotator.save_annotated_image(annotated, output_path)
                    logger.info(f"Re-rendered with override: {entry.annotated_image}")
                    re_rendered += 1

                    # Update the entry with new coordinates and temperatures
                    entry.hot_point = AnnotationPoint(
                        x=hot_point.x, y=hot_point.y, temp=annotated.hot_cold.hot_temp
                    )
                    entry.cold_point = AnnotationPoint(
                        x=cold_point.x, y=cold_point.y, temp=annotated.hot_cold.cold_temp
                    )
                    entry.delta_t = annotated.hot_cold.hot_temp - annotated.hot_cold.cold_temp
                    entry.severity = annotated.severity

                    # Also regenerate the zoombox image with updated hot point
                    # Convert thermal coords to visual coords for zoombox
                    from ..utils.thermal_alignment import thermal_to_visual
                    visual_x, visual_y = thermal_to_visual(hot_point.x, hot_point.y)

                    # Get flight direction for this image
                    flight_dir = self.orientation_map.get(entry.raw_image_name)

                    # Create zoombox with updated hot point position
                    defect_type_str = entry.defect_type.replace("_", "")
                    self._save_annotated_raw_image(
                        raw_image_path=raw_image_path,
                        hot_point_x=int(visual_x),
                        hot_point_y=int(visual_y),
                        zoom_size=200,  # Same as thermal_annotator default
                        output_dir=output_dir,
                        defect_type_str=defect_type_str,
                        panel_id=entry.panel_id,
                        flight_dir=flight_dir,
                    )
                    logger.info(f"Re-rendered zoombox with override: {defect_type_str}_({entry.panel_id})_zoombox.jpg")
                else:
                    logger.warning(f"Failed to re-render {entry.defect_id}")
            except Exception as e:
                logger.error(f"Error applying override for {entry.defect_id}: {e}")

        logger.info(f"Applied {re_rendered} annotation overrides")
        return re_rendered

    def _find_annotation_entry(self, defect_id: str) -> Optional[AnnotationEntry]:
        """Find an annotation entry by defect ID."""
        for entry in self.annotation_entries:
            if entry.defect_id == defect_id:
                return entry
        return None
