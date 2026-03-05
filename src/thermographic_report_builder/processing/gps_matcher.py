"""Match raw thermal images to defects using mesh backtracking, camera reprojection, or GPS.

Supports three matching strategies (in order of preference):

1. Mesh Backtracking (primary): Uses ODM mesh + labeling.vec for deterministic
   pixel-to-pixel mapping. Raycasts from orthophoto through mesh to find the face,
   looks up the view from labeling.vec, then projects to raw image. This provides
   100% reproducible results with no heuristics.

2. Camera Reprojection (fallback): Uses OpenSfM reconstruction data to project
   orthophoto coordinates back to raw image pixels. May use temperature scoring
   to pick the best candidate image.

3. GPS Proximity (fallback): Matches defects to the nearest raw image by GPS
   coordinates. Used when reconstruction data is not available.

Coordinate spaces:
- Defect labels are in cropped/rotated orthophoto space (from inference pipeline)
- Mesh backtracker requires uncropped orthophoto space (aligned with mesh geo-coords)
- CropTransform handles conversion between these coordinate spaces
"""

import csv
from collections import defaultdict
import cv2
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from .mesh_backtracker import MeshBacktracker
from .source_map_backtracker import SourceMapBacktracker
from .crop_transform import CropTransform
from .thermal_extractor import ThermalExtractor, TemperatureReading
from .thermal_annotator import ThermalAnnotator

logger = get_logger(__name__)

# Flight direction detection: 'south' means drone flying south, camera looking north
# Images taken while flying south need to be rotated 180° to orient north-up
FlightDirection = str  # 'north', 'south', or None

# Track calibration settings (feature-based offsets from tracks.csv)
_TRACK_COORD_MODES = ("pixel", "norm01", "centered", "centered_max")
_TRACK_MODE_SAMPLE_LIMIT = 20000
_TRACK_MAX_SAMPLES_PER_IMAGE = 5000
_TRACK_MIN_POINTS = 30
@dataclass
class DefectMatch:
    """Result of matching a defect to a raw image."""

    image_name: str
    image_path: Path
    method: str  # 'mesh', 'reprojection', or 'gps'
    pixel_x: Optional[float] = None  # Pixel X in raw image
    pixel_y: Optional[float] = None  # Pixel Y in raw image
    temperature: Optional[TemperatureReading] = None  # Temperature data if available


class GPSMatcher:
    """Match raw thermal images to defects using mesh, reprojection, or GPS."""

    def __init__(
        self,
        s3_client: S3Client,
        geo_converter: PixelToLatLonConverter,
        reconstruction_data: Optional[dict] = None,
        dsm_path: Optional[Path] = None,
        camera_projector: Optional[CameraProjector] = None,
        mesh_backtracker: Optional[MeshBacktracker] = None,
        source_map_backtracker: Optional[SourceMapBacktracker] = None,
        crop_transform: Optional[CropTransform] = None,
        enable_temperature: bool = True,
        tracks_path: Optional[Path] = None,
    ):
        """
        Initialize image matcher.

        Supports four matching strategies (in order of preference):
        1. Source Map (primary): Uses ODM source-map (odm_orthophoto_sources.tif)
           for authoritative ortho->raw pixel mapping created during orthophoto generation.
        2. Mesh Backtracking: Uses ODM mesh + labeling.vec for deterministic
           pixel-to-pixel mapping from orthophoto to raw images.
        3. Camera Reprojection (fallback): Uses OpenSfM reconstruction for projection.
        4. GPS Proximity (fallback): Matches by GPS distance when others unavailable.

        Args:
            s3_client: S3 client for downloading raw images
            geo_converter: Helper to convert pixels into geodetic coordinates (uncropped space)
            reconstruction_data: Optional OpenSfM reconstruction.json data.
            dsm_path: Optional path to DSM for precise elevation.
            camera_projector: Camera projector for reprojection.
            mesh_backtracker: Mesh backtracker for deterministic matching.
            source_map_backtracker: Source-map backtracker for authoritative pixel mapping.
            crop_transform: Transform for converting coordinates between cropped/rotated
                           defect label space and uncropped orthophoto space.
            enable_temperature: If True, extract temperature data when reprojection available
            tracks_path: Optional OpenSfM tracks.csv for feature-based calibration.
        """
        self.s3_client = s3_client
        self.geo_converter = geo_converter
        self.image_cache: Dict[str, dict] = {}  # Cache GPS data to avoid re-reading
        self.orientation_map: Dict[str, FlightDirection] = {}  # filename -> flight direction
        self.annotation_entries: List[AnnotationEntry] = []  # Collect annotations for manifest

        # Flag to force GPS-only matching when reprojection is unsafe (e.g., size mismatch).
        # This is detected during _index_raw_images() based on image dimensions vs reconstruction.
        self._use_gps_fallback = False
        # Track whether raw images are thermal-only (e.g., M3T 640x512) for coordinate alignment.
        self._thermal_only_images = False
        self._detected_image_width = None
        self._detected_image_height = None

        # Initialize camera projector if reconstruction data available
        self.camera_projector = camera_projector or CameraProjector(
            reconstruction_json=reconstruction_data,
            geo_converter=geo_converter,
            dsm_path=dsm_path,
        )
        self.mesh_backtracker = mesh_backtracker
        self.source_map_backtracker = source_map_backtracker
        self.crop_transform = crop_transform
        self.tracks_path = tracks_path
        self._track_calibration_offsets: Dict[str, Tuple[float, float]] = {}
        self._track_coord_mode: Optional[str] = None

        # Initialize thermal extractor and annotator if enabled
        if enable_temperature:
            self.thermal_extractor = ThermalExtractor()
            self.thermal_annotator = ThermalAnnotator(self.thermal_extractor)
        else:
            self.thermal_extractor = None
            self.thermal_annotator = None


        if self.source_map_backtracker and self.source_map_backtracker.available:
            logger.info("Image matcher initialized with source-map backtracking (primary)")
        elif self.camera_projector.available:
            logger.info("Image matcher initialized with camera reprojection")
        else:
            logger.info("Image matcher initialized with GPS-only matching (no reconstruction data)")

        self.stats = {
            "source_map_success": 0,
            "source_map_failed": 0,
            "mesh_backtrack_success": 0,
            "mesh_backtrack_failed": 0,
            "fallback_reprojection": 0,
            "fallback_gps": 0,
        }

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
        # Download and index all raw images by GPS (also detects thermal-only images)
        self._index_raw_images(temp_dir)

        if not self.image_cache:
            logger.warning("No raw images with GPS data found")
            return 0, {}

        # Determine flight direction for each image based on GPS trajectory
        self._compute_flight_directions()

        matched_count = 0
        defect_matches: Dict[str, DefectMatch] = {}
        hotspot_exclusions: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)

        # Determine matching method: source_map > mesh > reprojection > GPS
        use_source_map = (
            self.source_map_backtracker is not None
            and self.source_map_backtracker.available
            and not self._use_gps_fallback
        )
        use_mesh_backtrack = (
            self.mesh_backtracker is not None
            and self.mesh_backtracker.available
            and not self._use_gps_fallback
        )
        use_reprojection = self.camera_projector.available and not self._use_gps_fallback
        method = "source_map" if use_source_map else ("mesh" if use_mesh_backtrack else ("reprojection" if use_reprojection else "GPS"))
        logger.info(f"Matching raw thermal images to defects via {method}")

        if use_source_map:
            logger.info("Fallback chain: source map (primary) -> camera reprojection/GPS -> mesh (last resort)")
        elif use_mesh_backtrack:
            logger.info("Fallback chain: camera reprojection/GPS -> mesh (last resort) - source map disabled")
        else:
            logger.info("Fallback chain: camera reprojection/GPS only - source map and mesh disabled")

        # Compute per-image calibration offsets from tracks.csv if available
        if self.tracks_path:
            self._compute_track_calibration_offsets()

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

                    # Transform defect coordinates from cropped/rotated space to uncropped space
                    # This is needed because defect labels are created on cropped ortho,
                    # but mesh backtracker and geo_converter use uncropped ortho coordinates
                    if self.crop_transform and self.crop_transform.available:
                        uncropped_x, uncropped_y = self.crop_transform.cropped_to_uncropped(
                            defect_center_px[0], defect_center_px[1]
                        )
                    else:
                        uncropped_x, uncropped_y = defect_center_px[0], defect_center_px[1]

                    defect_lon, defect_lat = self.geo_converter.pixel_to_lonlat((uncropped_x, uncropped_y))

                    # FALLBACK CHAIN: source map -> camera reprojection/GPS -> mesh (last resort)
                    # Mesh is demoted to last because:
                    # 1. Redundant with source map (both use centrality scoring)
                    # 2. Requires large artifacts (100MB+) causing bandwidth costs
                    # 3. Disabled by default via ENABLE_MESH_FALLBACK setting
                    match = None
                    mesh_result = None
                    mesh_used = False
                    source_map_result = None

                    # PRIMARY: Source-map backtracking (most authoritative - uses ODM's actual mapping)
                    # Uses uncropped coordinates since source-map is aligned to original orthophoto
                    if use_source_map:
                        source_map_result = self.source_map_backtracker.backtrack_with_search(
                            ortho_x=uncropped_x,
                            ortho_y=uncropped_y,
                            search_radius=10,  # Search nearby pixels if exact pixel has no data
                        )
                        if source_map_result and source_map_result.image_name:
                            # Found authoritative mapping from source-map
                            image_path = temp_dir / source_map_result.image_name
                            if not image_path.exists():
                                image_path = self._ensure_raw_image_downloaded(
                                    source_map_result.image_name,
                                    temp_dir,
                                )
                            if image_path and image_path.exists():
                                resolved_name = image_path.name
                                match = DefectMatch(
                                    image_name=resolved_name,
                                    image_path=image_path,
                                    method="source_map",
                                    pixel_x=float(source_map_result.raw_pixel_x),
                                    pixel_y=float(source_map_result.raw_pixel_y),
                                )
                                self.stats["source_map_success"] += 1
                                logger.info(
                                    f"Source-map match: ortho=({uncropped_x:.0f},{uncropped_y:.0f}) -> "
                                    f"{resolved_name} at ({source_map_result.raw_pixel_x}, {source_map_result.raw_pixel_y})"
                                )
                            else:
                                logger.warning(f"Source-map image not found: {source_map_result.image_name}")
                                self.stats["source_map_failed"] += 1
                        else:
                            self.stats["source_map_failed"] += 1

                    # SECONDARY: Temperature-scored matching for hotspots
                    if (
                        match is None
                        and create_thermal_analysis
                        and self.thermal_extractor
                        and self.thermal_extractor.available
                    ):
                        # Use temperature-based scoring to find image with actual hotspot
                        # Uses uncropped coordinates for projection
                        match = self._find_best_match(
                            ortho_x=uncropped_x,
                            ortho_y=uncropped_y,
                            temp_dir=temp_dir,
                        )
                        if match:
                            logger.info(
                                f"Temperature-scored match for {panel.panel_id} defect {defect_idx}: "
                                f"{match.image_name} via {match.method}"
                            )

                    # TERTIARY: GPS + camera reprojection fallback
                    if not match:
                        # For non-M30T (thermal-only), use temperature-scored GPS
                        # to pick the best candidate from multiple nearby images
                        if (
                            (self._thermal_only_images or self._use_gps_fallback)
                            and self.thermal_extractor
                            and self.thermal_extractor.available
                        ):
                            closest_image = self._find_best_image_by_temperature(defect_lat, defect_lon)
                        else:
                            closest_image = self._find_closest_image(defect_lat, defect_lon)
                        if not closest_image:
                            logger.warning(f"No matching image found for {panel.panel_id} defect {defect_idx}")
                            continue

                        image_path = Path(closest_image["path"])
                        image_name = image_path.name

                        # Try camera reprojection to get pixel coordinates (skip if reprojection disabled)
                        # Uses uncropped coordinates for projection
                        pixel_x, pixel_y = None, None
                        if self.camera_projector.available and not self._use_gps_fallback:
                            projection = self.camera_projector.project_to_specific_image(
                                ortho_x=uncropped_x,
                                ortho_y=uncropped_y,
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
                        if match.method == "reprojection":
                            self.stats["fallback_reprojection"] += 1
                        else:
                            self.stats["fallback_gps"] += 1

                    # LAST RESORT: Mesh backtracking (disabled by default via ENABLE_MESH_FALLBACK=False)
                    # Demoted to last because it's redundant with source map (both use centrality scoring)
                    # and requires large artifacts (100MB+) causing bandwidth costs.
                    if match is None and use_mesh_backtrack:
                        mesh_result = self.mesh_backtracker.backtrack(
                            ortho_x=uncropped_x,
                            ortho_y=uncropped_y,
                        )
                        if mesh_result and mesh_result.image_name:
                            image_path = temp_dir / mesh_result.image_name
                            if image_path.exists():
                                mesh_pixel = self._project_mesh_point_to_image(
                                    mesh_result, mesh_result.image_name
                                )
                                if (
                                    mesh_pixel is None
                                    and mesh_result.pixel_x is not None
                                    and mesh_result.pixel_y is not None
                                ):
                                    mesh_pixel = (
                                        mesh_result.pixel_x,
                                        mesh_result.pixel_y,
                                    )

                                if mesh_pixel is not None:
                                    match = DefectMatch(
                                        image_name=mesh_result.image_name,
                                        image_path=image_path,
                                        method="mesh",
                                        pixel_x=mesh_pixel[0],
                                        pixel_y=mesh_pixel[1],
                                    )
                                    self.stats["mesh_backtrack_success"] += 1
                                    mesh_used = True
                                    logger.info(
                                        f"Mesh fallback (last resort): ortho=({uncropped_x:.0f},{uncropped_y:.0f}) -> "
                                        f"{mesh_result.image_name} at ({mesh_pixel[0]}, {mesh_pixel[1]})"
                                    )
                                else:
                                    self.stats["mesh_backtrack_failed"] += 1
                            else:
                                self.stats["mesh_backtrack_failed"] += 1
                        else:
                            self.stats["mesh_backtrack_failed"] += 1

                    if match:
                        # Get the source filename to check orientation
                        src_filename = match.image_name
                        flight_dir = self.orientation_map.get(src_filename)

                        # If no match found, try looking up with common extensions
                        # This handles the case where NVM strips extensions but orientation_map uses full filenames
                        if flight_dir is None and "." not in src_filename:
                            for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG", ".tif", ".TIF"]:
                                flight_dir = self.orientation_map.get(src_filename + ext)
                                if flight_dir is not None:
                                    logger.debug(f"Found flight direction via extension lookup: {src_filename + ext} -> {flight_dir}")
                                    break

                        # Log if we couldn't find flight direction
                        if flight_dir is None:
                            logger.debug(
                                f"Could not find flight direction for {src_filename}. "
                                f"orientation_map has {len(self.orientation_map)} entries, "
                                f"sample keys: {list(self.orientation_map.keys())[:3]}"
                            )

                        # Store original (unrotated) pixel coordinates for raw file operations
                        original_pixel_x = match.pixel_x
                        original_pixel_y = match.pixel_y

                        # Log coordinate tracing for debugging
                        # Show both cropped and uncropped coords for clarity
                        if original_pixel_x is not None:
                            logger.info(
                                f"Projecting {panel.panel_id} defect {defect_idx}: "
                                f"cropped=({defect_center_px[0]:.1f}, {defect_center_px[1]:.1f}) "
                                f"-> uncropped=({uncropped_x:.1f}, {uncropped_y:.1f}) "
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
                            # Use image center based on detected dimensions
                            if original_pixel_x is None or original_pixel_y is None:
                                fallback_w = self._detected_image_width or 1280
                                fallback_h = self._detected_image_height or 1024
                                original_pixel_x = fallback_w / 2.0
                                original_pixel_y = fallback_h / 2.0
                                logger.warning(
                                    f"Reprojection failed for {panel.panel_id} defect {defect_idx}, "
                                    f"using image center ({original_pixel_x:.0f}, {original_pixel_y:.0f}) as fallback"
                                )

                            # Refine to the hottest nearby pixel to avoid edge offsets
                            # SKIP refinement for mesh backtracking to preserve determinism
                            # Use smaller radius for source_map (already precise) vs GPS/reprojection (less precise)
                            should_refine = match.method != "mesh"
                            if should_refine:
                                try:
                                    from ..utils.thermal_alignment import ImageFormat
                                    image_format = ImageFormat.THERMAL_ONLY if self._thermal_only_images else ImageFormat.VISUAL_THERMAL
                                    visual_size = None
                                    if self._detected_image_width and self._detected_image_height:
                                        visual_size = (self._detected_image_width, self._detected_image_height)

                                    # Source-map backtracking is already very precise (sub-pixel)
                                    # Use small radius to avoid finding hotspots on neighboring panels
                                    # GPS and reprojection need larger radius due to projection drift
                                    if match.method == "source_map":
                                        refine_radius = 30  # Small radius for precise source-map coords
                                    else:
                                        refine_radius = 150  # Larger radius for GPS/reprojection drift

                                    hot_x, hot_y, hot_temp = self.thermal_extractor.find_local_hotspot(
                                        image_path=match.image_path,
                                        initial_x=int(original_pixel_x),
                                        initial_y=int(original_pixel_y),
                                        search_radius=refine_radius,
                                        image_format=image_format,
                                        visual_size=visual_size,
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
                            else:
                                logger.debug(
                                    f"Skipping hotspot refinement for {panel.panel_id} defect {defect_idx} "
                                    f"(mesh backtrack preserves determinism)"
                                )

                            # Extract temperature data
                            from ..utils.thermal_alignment import ImageFormat
                            image_format = ImageFormat.THERMAL_ONLY if self._thermal_only_images else ImageFormat.VISUAL_THERMAL
                            visual_size = None
                            if self._detected_image_width and self._detected_image_height:
                                visual_size = (self._detected_image_width, self._detected_image_height)

                            match.temperature = self.thermal_extractor.extract_temperature_at_pixel(
                                image_path=match.image_path,
                                pixel_x=int(original_pixel_x),
                                pixel_y=int(original_pixel_y),
                                image_format=image_format,
                                visual_size=visual_size,
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
                                # Project panel bbox corners to raw image coordinates (skip for non-M30T)
                                panel_bbox_visual = None
                                if self.camera_projector.available and not self._use_gps_fallback:
                                    # Get panel corners in cropped orthophoto coordinates
                                    cropped_left = panel.bbox.left
                                    cropped_top = panel.bbox.top
                                    cropped_right = panel.bbox.right
                                    cropped_bottom = panel.bbox.bottom

                                    # Transform corners from cropped to uncropped space for projection
                                    # (camera_projector uses uncropped geo-transform)
                                    corners_cropped = [
                                        (cropped_left, cropped_top),
                                        (cropped_right, cropped_top),
                                        (cropped_right, cropped_bottom),
                                        (cropped_left, cropped_bottom),
                                    ]

                                    # Project all 4 corners to raw image
                                    corners_raw = []
                                    for cx, cy in corners_cropped:
                                        # Transform to uncropped space
                                        if self.crop_transform and self.crop_transform.available:
                                            ux, uy = self.crop_transform.cropped_to_uncropped(cx, cy)
                                        else:
                                            ux, uy = cx, cy

                                        proj = self.camera_projector.project_to_specific_image(
                                            ortho_x=ux,
                                            ortho_y=uy,
                                            image_name=match.image_name,
                                        )
                                        if proj:
                                            corners_raw.append(proj)

                                    # Accept 3+ corners (some might be out of frame)
                                    # Use detected image dimensions instead of hardcoded values
                                    img_max_w = self._detected_image_width or 1280
                                    img_max_h = self._detected_image_height or 1024

                                    if len(corners_raw) >= 3:
                                        xs = [c[0] for c in corners_raw]
                                        ys = [c[1] for c in corners_raw]
                                        # Clamp to image bounds
                                        panel_bbox_visual = (
                                            max(0, int(min(xs))),  # left
                                            max(0, int(min(ys))),  # top
                                            min(img_max_w, int(max(xs))),  # right
                                            min(img_max_h, int(max(ys))),  # bottom
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
                                            min(img_max_w, int(original_pixel_x + padding)),
                                            min(img_max_h, int(original_pixel_y + padding)),
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
                                            min(img_max_w, int(original_pixel_x + padding)),
                                            min(img_max_h, int(original_pixel_y + padding)),
                                        )
                                else:
                                    # Camera projector not available - use defect-centered bbox
                                    img_max_w = self._detected_image_width or 1280
                                    img_max_h = self._detected_image_height or 1024
                                    logger.debug(
                                        f"Panel {panel.panel_id}: No camera projector, using defect-centered bbox"
                                    )
                                    padding = 100
                                    panel_bbox_visual = (
                                        max(0, int(original_pixel_x - padding)),
                                        max(0, int(original_pixel_y - padding)),
                                        min(img_max_w, int(original_pixel_x + padding)),
                                        min(img_max_h, int(original_pixel_y + padding)),
                                    )

                                # Use smaller search radius for source-map (already precise)
                                # vs larger radius for GPS/reprojection (more drift)
                                if match.method == "source_map":
                                    hot_search_radius = 30  # Small radius, source-map is precise
                                else:
                                    hot_search_radius = 60  # Default for less precise methods

                                exclusion_key = (panel.panel_id, match.image_name)
                                excluded_hot_points = hotspot_exclusions.get(exclusion_key)

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
                                    flight_direction=flight_dir,  # For image rotation
                                    hot_search_radius=hot_search_radius,  # Match refinement radius
                                    excluded_hot_points=excluded_hot_points,
                                    min_hot_distance=10,
                                )

                                if annotated:
                                    raw_hot_x = annotated.hot_cold.hot_x
                                    raw_hot_y = annotated.hot_cold.hot_y
                                    if flight_dir == "south":
                                        img_w = self._detected_image_width or 1280
                                        img_h = self._detected_image_height or 1024
                                        raw_hot_x = img_w - 1 - raw_hot_x
                                        raw_hot_y = img_h - 1 - raw_hot_y
                                    hotspot_exclusions[exclusion_key].append(
                                        (int(raw_hot_x), int(raw_hot_y))
                                    )

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
                                    # For non-M30T (thermal-only), coords are already thermal - skip transformation
                                    from ..utils.thermal_alignment import visual_to_thermal, ImageFormat
                                    image_format = ImageFormat.THERMAL_ONLY if self._thermal_only_images else ImageFormat.VISUAL_THERMAL
                                    visual_size = None
                                    if self._detected_image_width and self._detected_image_height:
                                        visual_size = (self._detected_image_width, self._detected_image_height)
                                    thermal_size = None
                                    if self.thermal_extractor:
                                        try:
                                            temp_array = self.thermal_extractor.get_full_temperature_array(match.image_path)
                                            if temp_array is not None:
                                                thermal_size = (temp_array.shape[1], temp_array.shape[0])
                                        except Exception:
                                            thermal_size = None

                                    hot_thermal_x, hot_thermal_y = visual_to_thermal(
                                        annotated.hot_cold.hot_x,
                                        annotated.hot_cold.hot_y,
                                        image_format=image_format,
                                        visual_size=visual_size,
                                        thermal_size=thermal_size,
                                    )
                                    cold_thermal_x, cold_thermal_y = visual_to_thermal(
                                        annotated.hot_cold.cold_x,
                                        annotated.hot_cold.cold_y,
                                        image_format=image_format,
                                        visual_size=visual_size,
                                        thermal_size=thermal_size,
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
                                        flight_direction=flight_dir,  # For UI rotation
                                    )
                                    self.annotation_entries.append(entry)
                                else:
                                    logger.warning(
                                        f"Failed to create thermal analysis for {panel.panel_id} defect {defect_idx}: "
                                        f"annotator returned None (pixel={original_pixel_x:.1f},{original_pixel_y:.1f}, "
                                        f"bbox={panel_bbox_visual})"
                                    )

        mesh_count = sum(1 for m in defect_matches.values() if m.method == "mesh")
        reprojection_count = sum(1 for m in defect_matches.values() if m.method == "reprojection")
        gps_count = sum(1 for m in defect_matches.values() if m.method == "gps")
        temp_count = sum(1 for m in defect_matches.values() if m.temperature is not None)

        logger.info(
            f"Matched {matched_count} images: "
            f"{mesh_count} via mesh, {reprojection_count} via reprojection, {gps_count} via GPS, "
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
        # Try camera reprojection first (skip if reprojection disabled)
        if self.camera_projector.available and not self._use_gps_fallback:
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

                        # Convert from visual to thermal coordinates using detected dimensions
                        # For non-M30T (thermal-only), coords are already thermal - skip transformation
                        from ..utils.thermal_alignment import visual_to_thermal, clamp_thermal_coords, ImageFormat
                        image_format = ImageFormat.THERMAL_ONLY if self._thermal_only_images else ImageFormat.VISUAL_THERMAL
                        visual_size = None
                        if self._detected_image_width and self._detected_image_height:
                            visual_size = (self._detected_image_width, self._detected_image_height)
                        thermal_size = (temp_array.shape[1], temp_array.shape[0])

                        thermal_x, thermal_y = visual_to_thermal(
                            match.pixel_x,
                            match.pixel_y,
                            image_format=image_format,
                            visual_size=visual_size,
                            thermal_size=thermal_size,
                        )
                        px, py = clamp_thermal_coords(
                            thermal_x,
                            thermal_y,
                            thermal_size=thermal_size,
                        )

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

        # No GPS fallback here - only use reconstruction.json images
        # GPS matching is handled at a higher level only when reconstruction unavailable
        return None

    def _project_mesh_point_to_image(
        self,
        mesh_result,
        image_name: str,
    ) -> Optional[Tuple[float, float]]:
        """Project a mesh backtrack world point into a specific image."""
        if not self.camera_projector.available:
            return None

        world_point = np.array(
            [mesh_result.world_x, mesh_result.world_y, mesh_result.world_z]
        )
        projection = self.camera_projector.project_world_to_image(
            image_name=image_name, world_point=world_point
        )
        if projection is None:
            return None

        px, py = projection
        return (px, py)

    def _get_track_image_dimensions(self, image_name: str) -> Tuple[int, int]:
        """Return image width/height for track coordinate conversion."""
        width = int(self.camera_projector.image_width)
        height = int(self.camera_projector.image_height)
        if not self.camera_projector.available:
            return width, height
        shot = self.camera_projector.shots.get(image_name)
        if not shot:
            return width, height
        camera_id = shot.get("camera")
        camera = self.camera_projector.cameras.get(camera_id, {})
        cam_width = camera.get("width")
        cam_height = camera.get("height")
        if cam_width and cam_height:
            return int(cam_width), int(cam_height)
        return width, height

    def _normalize_track_image_name(self, raw_name: str) -> Optional[str]:
        """Normalize tracks.csv image names to reconstruction shot keys."""
        if not raw_name:
            return None
        name = Path(raw_name).name
        if name in self.camera_projector.shots:
            return name
        if name.endswith(".JPG"):
            base = name[:-4]
            if not base.endswith("_T"):
                candidate = f"{base}_T.JPG"
                if candidate in self.camera_projector.shots:
                    return candidate
            else:
                candidate = f"{base[:-2]}.JPG"
                if candidate in self.camera_projector.shots:
                    return candidate
        return None

    def _tracks_xy_to_pixels(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        mode: str,
    ) -> Tuple[float, float]:
        """Convert tracks.csv coordinates to pixel coordinates."""
        if mode == "pixel":
            return x, y
        if mode == "norm01":
            return x * width, y * height
        if mode == "centered":
            return (x + 0.5) * width, (y + 0.5) * height
        if mode == "centered_max":
            max_dim = max(width, height)
            return x * max_dim + width / 2.0, y * max_dim + height / 2.0
        return x, y

    def _detect_track_coord_mode(
        self,
        tracks_path: Path,
        points_by_id: Dict[str, dict],
    ) -> Optional[str]:
        """Detect tracks.csv coordinate mode by minimizing reprojection error."""
        if self._track_coord_mode:
            return self._track_coord_mode

        if not self.camera_projector.available:
            return None

        mode_errors: Dict[str, List[float]] = {mode: [] for mode in _TRACK_COORD_MODES}
        sample_count = 0

        try:
            with tracks_path.open(newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 5:
                        continue
                    track_id = row[0].strip()
                    image_name = self._normalize_track_image_name(row[1].strip())
                    if not image_name:
                        continue
                    try:
                        x = float(row[3])
                        y = float(row[4])
                    except ValueError:
                        continue

                    point = points_by_id.get(track_id)
                    if not point:
                        continue
                    coords = point.get("coordinates")
                    if not coords or len(coords) < 3:
                        continue

                    world_point = np.array(coords, dtype=float)
                    projection = self.camera_projector.project_world_to_image(
                        image_name=image_name, world_point=world_point
                    )
                    if projection is None:
                        continue

                    px, py = projection
                    width, height = self._get_track_image_dimensions(image_name)
                    for mode in _TRACK_COORD_MODES:
                        obs_x, obs_y = self._tracks_xy_to_pixels(x, y, width, height, mode)
                        mode_errors[mode].append(float(np.hypot(obs_x - px, obs_y - py)))

                    sample_count += 1
                    if sample_count >= _TRACK_MODE_SAMPLE_LIMIT:
                        break
        except Exception as exc:
            logger.warning(f"Failed to parse tracks.csv for mode detection: {exc}")
            return None

        if sample_count == 0:
            logger.warning("tracks.csv mode detection skipped (no usable samples)")
            return None

        best_mode = None
        best_median = None
        for mode, errors in mode_errors.items():
            if not errors:
                continue
            median_error = float(np.median(errors))
            if best_median is None or median_error < best_median:
                best_median = median_error
                best_mode = mode

        if best_mode:
            self._track_coord_mode = best_mode
            logger.info(
                "tracks.csv coordinate mode detected as %s (median error %.2fpx)",
                best_mode,
                best_median if best_median is not None else -1.0,
            )
        else:
            logger.warning("tracks.csv mode detection failed (no valid errors)")
        return best_mode

    def _compute_track_calibration_offsets(self) -> None:
        """Compute per-image offsets using tracks.csv feature correspondences."""
        if not self.tracks_path:
            return
        if not self.tracks_path.exists():
            logger.info(f"tracks.csv not found: {self.tracks_path}")
            return
        if not self.camera_projector.available:
            logger.warning("tracks.csv calibration skipped (camera projector unavailable)")
            return
        if not self.camera_projector.points:
            logger.warning("tracks.csv calibration skipped (no reconstruction points)")
            return

        points_by_id = {
            str(track_id): point
            for track_id, point in self.camera_projector.points.items()
            if isinstance(point, dict) and point.get("coordinates")
        }
        if not points_by_id:
            logger.warning("tracks.csv calibration skipped (no valid points)")
            return

        coord_mode = self._detect_track_coord_mode(self.tracks_path, points_by_id)
        if not coord_mode:
            logger.warning("tracks.csv calibration skipped (unknown coordinate mode)")
            return

        offsets: Dict[str, List[Tuple[float, float]]] = {}
        try:
            with self.tracks_path.open(newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 5:
                        continue
                    track_id = row[0].strip()
                    image_name = self._normalize_track_image_name(row[1].strip())
                    if not image_name:
                        continue
                    try:
                        x = float(row[3])
                        y = float(row[4])
                    except ValueError:
                        continue

                    point = points_by_id.get(track_id)
                    if not point:
                        continue
                    coords = point.get("coordinates")
                    if not coords or len(coords) < 3:
                        continue

                    world_point = np.array(coords, dtype=float)
                    projection = self.camera_projector.project_world_to_image(
                        image_name=image_name, world_point=world_point
                    )
                    if projection is None:
                        continue

                    width, height = self._get_track_image_dimensions(image_name)
                    obs_x, obs_y = self._tracks_xy_to_pixels(x, y, width, height, coord_mode)
                    pred_x, pred_y = projection
                    dx = obs_x - pred_x
                    dy = obs_y - pred_y

                    pairs = offsets.setdefault(image_name, [])
                    if len(pairs) < _TRACK_MAX_SAMPLES_PER_IMAGE:
                        pairs.append((dx, dy))
        except Exception as exc:
            logger.warning(f"Failed to parse tracks.csv for calibration: {exc}")
            return

        self._track_calibration_offsets = {}
        for image_name, pairs in offsets.items():
            if len(pairs) < _TRACK_MIN_POINTS:
                continue
            dxs = np.array([p[0] for p in pairs], dtype=float)
            dys = np.array([p[1] for p in pairs], dtype=float)
            median_dx = float(np.median(dxs))
            median_dy = float(np.median(dys))

            mad_x = float(np.median(np.abs(dxs - median_dx)))
            mad_y = float(np.median(np.abs(dys - median_dy)))
            if mad_x > 0 or mad_y > 0:
                mask = (np.abs(dxs - median_dx) <= 3 * max(mad_x, 1e-6)) & (
                    np.abs(dys - median_dy) <= 3 * max(mad_y, 1e-6)
                )
                if mask.sum() >= _TRACK_MIN_POINTS:
                    dxs = dxs[mask]
                    dys = dys[mask]
                    median_dx = float(np.median(dxs))
                    median_dy = float(np.median(dys))

            self._track_calibration_offsets[image_name] = (median_dx, median_dy)

        if self._track_calibration_offsets:
            logger.info(
                "tracks.csv calibration offsets loaded for %d images (mode=%s)",
                len(self._track_calibration_offsets),
                coord_mode,
            )

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
                # Coordinates are already in rotated space from find_hot_cold_points()

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

                    # Detect image dimensions on first image to identify thermal-only cameras
                    if self._detected_image_width is None and "width" in exif and "height" in exif:
                        self._detected_image_width = exif["width"]
                        self._detected_image_height = exif["height"]

                        # M30T images are 1280x1024, thermal-only are typically 640x512
                        if self._detected_image_width != 1280 or self._detected_image_height != 1024:
                            self._thermal_only_images = True
                            logger.info(
                                f"Detected thermal-only images ({self._detected_image_width}x{self._detected_image_height}). "
                                f"Using thermal-only alignment."
                            )
                        else:
                            logger.info(
                                f"Detected M30T images ({self._detected_image_width}x{self._detected_image_height})."
                            )

                        # Disable reprojection only if reconstruction camera size mismatches raw images
                        if self.camera_projector.available:
                            camera_model = next(iter(getattr(self.camera_projector, "cameras", {}).values()), {})
                            cam_width = camera_model.get("width")
                            cam_height = camera_model.get("height")

                            if cam_width and cam_height:
                                if cam_width != self._detected_image_width or cam_height != self._detected_image_height:
                                    self._use_gps_fallback = True
                                    logger.warning(
                                        "Reprojection disabled due to size mismatch: "
                                        f"reconstruction {cam_width}x{cam_height} vs raw "
                                        f"{self._detected_image_width}x{self._detected_image_height}. "
                                        "Using GPS-only matching."
                                    )
                                else:
                                    logger.info(
                                        "Reprojection enabled: reconstruction camera size matches raw images."
                                    )
                            else:
                                logger.info(
                                    "Reprojection enabled: reconstruction camera dimensions unavailable for validation."
                                )

            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
                continue

            # Log progress every 50 images
            if idx % 50 == 0:
                logger.info(f"Indexed {idx}/{len(raw_image_keys)} images ({len(self.image_cache)} with GPS)")

        logger.info(f"Indexed {len(self.image_cache)} images with GPS data")

    def _ensure_raw_image_downloaded(self, image_name: str, temp_dir: Path) -> Optional[Path]:
        """Attempt to download a missing raw image by filename."""
        if not image_name:
            return None

        candidates = [image_name]
        suffix = Path(image_name).suffix.lower()
        if not suffix:
            candidates.extend([f"{image_name}.JPG", f"{image_name}.jpg"])
        elif suffix == ".tif":
            base = image_name[:-4]
            candidates.extend([base, f"{base}.JPG", f"{base}.jpg"])

        last_error = None
        for candidate in candidates:
            local_path = temp_dir / candidate
            if local_path.exists():
                return local_path
            s3_key = f"{settings.user_id}/projects/{settings.project_id}/images/{candidate}"
            try:
                self.s3_client.download_raw_image(s3_key, local_path)
                if local_path.exists():
                    logger.info(f"Downloaded missing raw image for source-map match: {candidate}")
                    return local_path
            except Exception as e:
                last_error = e
                continue

        if last_error is not None:
            logger.warning(
                f"Failed to download raw image for source-map match: {image_name}: {last_error}"
            )
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

    def _find_best_image_by_temperature(
        self,
        target_lat: float,
        target_lon: float,
        max_distance_deg: float = 0.0005,  # ~50m radius for candidate search
        max_candidates: int = 10,
    ) -> dict | None:
        """
        Find the best image by scoring GPS candidates by max temperature.

        For thermal-only images (non-M30T), GPS proximity alone may not find the
        correct image. This method finds multiple GPS candidates and scores them
        by maximum temperature in the image - the one with the highest temp is
        most likely to contain the actual hotspot.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            max_distance_deg: Maximum GPS distance in degrees (~0.0005 = ~50m)
            max_candidates: Maximum number of GPS candidates to evaluate

        Returns:
            Best image metadata dict or None if no candidates found
        """
        candidates = self._find_candidate_images(
            target_lat, target_lon, max_distance_deg, max_candidates
        )

        if not candidates:
            # Fall back to closest image if no candidates within distance
            return self._find_closest_image(target_lat, target_lon)

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate by max temperature
        best_image = None
        best_max_temp = float("-inf")

        for candidate in candidates:
            try:
                image_path = Path(candidate["path"])
                temp_array = self.thermal_extractor.get_full_temperature_array(image_path)

                if temp_array is None:
                    continue

                max_temp = float(temp_array.max())

                if max_temp > best_max_temp:
                    best_max_temp = max_temp
                    best_image = candidate

            except Exception as e:
                logger.debug(f"Failed to score candidate {candidate.get('filename', 'unknown')}: {e}")
                continue

        if best_image:
            logger.info(
                f"Temperature-scored GPS: selected image with max_temp={best_max_temp:.1f}°C "
                f"from {len(candidates)} candidates"
            )
            return best_image

        # Fall back to closest if temperature scoring failed
        return candidates[0] if candidates else None

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
                # Get flight direction for this image to apply correct rotation
                flight_dir = self.orientation_map.get(entry.raw_image_name)

                annotated = self.thermal_annotator.create_annotated_image_with_points(
                    raw_image_path=raw_image_path,
                    hot_point=hot_point,
                    cold_point=cold_point,
                    panel_id=entry.panel_id,
                    defect_index=entry.defect_index,
                    flight_direction=flight_dir,
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
                    # For non-M30T (thermal-only), coords are already thermal - skip transformation
                    from ..utils.thermal_alignment import thermal_to_visual, ImageFormat
                    image_format = ImageFormat.THERMAL_ONLY if self._thermal_only_images else ImageFormat.VISUAL_THERMAL
                    visual_size = None
                    thermal_size = None
                    try:
                        img = cv2.imread(str(raw_image_path))
                        if img is not None:
                            img_h, img_w = img.shape[:2]
                            visual_size = (img_w, img_h)
                    except Exception:
                        visual_size = None
                    if self.thermal_extractor:
                        try:
                            temp_array = self.thermal_extractor.get_full_temperature_array(raw_image_path)
                            if temp_array is not None:
                                thermal_size = (temp_array.shape[1], temp_array.shape[0])
                        except Exception:
                            thermal_size = None

                    visual_x, visual_y = thermal_to_visual(
                        hot_point.x,
                        hot_point.y,
                        image_format=image_format,
                        visual_size=visual_size,
                        thermal_size=thermal_size,
                    )

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
