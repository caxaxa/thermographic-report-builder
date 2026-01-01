"""Create annotated thermal images with temperature overlay for reports.

This module generates professional annotated thermal images that show:
- Zoomed view of the defect area in the raw thermal image
- Temperature colorbar overlay
- RED marker on hottest point (detected hotspot)
- BLUE marker on coolest reference point (in outer ring)
- Temperature readings for both points and delta T
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..utils.logger import get_logger
from .thermal_extractor import ThermalExtractor, TemperatureReading

logger = get_logger(__name__)


@dataclass
class HotColdPoints:
    """Hot and cold reference points detected in thermal image."""

    hot_x: int  # X coordinate of hottest point (visual coords)
    hot_y: int  # Y coordinate of hottest point (visual coords)
    hot_temp: float  # Temperature at hottest point
    cold_x: int  # X coordinate of coldest reference point (visual coords)
    cold_y: int  # Y coordinate of coldest reference point (visual coords)
    cold_temp: float  # Temperature at coldest reference point
    delta_t: float  # Temperature difference (hot - cold)

    @property
    def severity(self) -> str:
        """Classify defect severity based on ΔT."""
        if self.delta_t >= 30:
            return "CRITICAL"
        elif self.delta_t >= 20:
            return "HIGH"
        elif self.delta_t >= 10:
            return "MEDIUM"
        elif self.delta_t >= 5:
            return "LOW"
        else:
            return "MINIMAL"


@dataclass
class AnnotatedThermalImage:
    """Result of creating an annotated thermal image."""

    image: np.ndarray  # The annotated BGR image
    defect_temp: float  # Defect temperature in Celsius
    panel_avg_temp: float  # Panel average temperature in Celsius (now coolest reference)
    delta_t: float  # Temperature difference
    severity: str  # Severity classification
    hot_cold: Optional[HotColdPoints] = None  # Hot/cold point details if detected


class ThermalAnnotator:
    """Creates annotated thermal images with temperature overlays."""

    def __init__(self, thermal_extractor: Optional[ThermalExtractor] = None):
        """
        Initialize thermal annotator.

        Args:
            thermal_extractor: Optional thermal extractor instance.
                             Creates new one if not provided.
        """
        self.thermal_extractor = thermal_extractor or ThermalExtractor()

    def find_hot_cold_points(
        self,
        image_path: Path,
        initial_x: int,
        initial_y: int,
        panel_bbox_visual: Optional[Tuple[int, int, int, int]] = None,
        edge_margin: int = 50,
        hot_search_radius: int = 60,
    ) -> Optional[HotColdPoints]:
        """
        Find hottest and coldest points WITHIN the panel bounding box.

        The algorithm:
        1. Search for HOTTEST pixel in a small radius around initial point, constrained to panel
        2. Prefer hot points closer to panel center over edge hot points
        3. Find COLDEST pixel within the panel, excluding edge pixels (to avoid shadows)

        Args:
            image_path: Path to DJI thermal R-JPEG image
            initial_x: Initial X coordinate in VISUAL image (1280x1024)
            initial_y: Initial Y coordinate in VISUAL image (1280x1024)
            panel_bbox_visual: Optional (left, top, right, bottom) in visual coords (1280x1024).
                              If provided, search is constrained to within this panel.
            edge_margin: Pixels to exclude from edges when searching for cold point (visual coords)
            hot_search_radius: Radius around initial point to search for hottest pixel (visual coords)

        Returns:
            HotColdPoints or None if detection fails
        """
        if not self.thermal_extractor or not self.thermal_extractor.available:
            return None

        try:
            temp_array = self.thermal_extractor.get_full_temperature_array(image_path)
            if temp_array is None:
                return None

            height, width = temp_array.shape

            # Convert from visual (1280x1024) to thermal (640x512) coordinates
            # This applies any configured alignment offset
            from ..utils.thermal_alignment import (
                visual_to_thermal, thermal_to_visual, clamp_thermal_coords,
                visual_bbox_to_thermal, SCALE_X
            )

            thermal_init_x_f, thermal_init_y_f = visual_to_thermal(initial_x, initial_y)
            thermal_init_x, thermal_init_y = clamp_thermal_coords(thermal_init_x_f, thermal_init_y_f)

            # Create panel mask in thermal coordinates
            if panel_bbox_visual is not None:
                # Convert panel bbox from visual to thermal coordinates (with alignment)
                left_v, top_v, right_v, bottom_v = panel_bbox_visual
                left_t, top_t, right_t, bottom_t = visual_bbox_to_thermal(
                    left_v, top_v, right_v, bottom_v
                )

                # Create panel mask
                panel_mask = np.zeros((height, width), dtype=bool)
                panel_mask[top_t:bottom_t, left_t:right_t] = True

                # Panel center in thermal coords
                panel_center_x = (left_t + right_t) / 2
                panel_center_y = (top_t + bottom_t) / 2

                # Create eroded mask for cold point search (exclude edges to avoid shadows)
                # Use percentage-based erosion: exclude outer 30% from each edge
                panel_width_t = right_t - left_t
                panel_height_t = bottom_t - top_t

                # Take the larger of: fixed margin OR 30% of panel dimension
                edge_margin_t = int(edge_margin * SCALE_X)
                margin_x = max(edge_margin_t, int(panel_width_t * 0.30))
                margin_y = max(edge_margin_t, int(panel_height_t * 0.30))

                inner_left = left_t + margin_x
                inner_top = top_t + margin_y
                inner_right = right_t - margin_x
                inner_bottom = bottom_t - margin_y

                # Ensure inner region is valid (at least 20% of panel remains)
                if inner_left < inner_right and inner_top < inner_bottom:
                    panel_mask_eroded = np.zeros((height, width), dtype=bool)
                    panel_mask_eroded[inner_top:inner_bottom, inner_left:inner_right] = True
                    logger.debug(
                        f"Cold search area: ({inner_left}, {inner_top}) to ({inner_right}, {inner_bottom}) "
                        f"[margins: x={margin_x}, y={margin_y}]"
                    )
                else:
                    # Panel too small after erosion, use center 40% of panel
                    center_x = (left_t + right_t) // 2
                    center_y = (top_t + bottom_t) // 2
                    half_w = max(1, panel_width_t // 5)  # 40% width
                    half_h = max(1, panel_height_t // 5)  # 40% height
                    panel_mask_eroded = np.zeros((height, width), dtype=bool)
                    panel_mask_eroded[
                        center_y - half_h : center_y + half_h,
                        center_x - half_w : center_x + half_w
                    ] = True
                    logger.debug(f"Using center 40% of panel for cold search")

                logger.debug(
                    f"Panel bbox thermal: ({left_t}, {top_t}) to ({right_t}, {bottom_t}), "
                    f"eroded margin: {edge_margin_t}px"
                )
            else:
                # No panel bbox - fallback to old behavior (search around initial point)
                logger.warning("No panel bbox provided, using fallback circular search")
                y_coords, x_coords = np.ogrid[:height, :width]
                distances = np.sqrt((x_coords - thermal_init_x) ** 2 + (y_coords - thermal_init_y) ** 2)
                panel_mask = distances <= 50  # 50 thermal pixels radius
                panel_mask_eroded = distances <= 40
                panel_center_x = thermal_init_x
                panel_center_y = thermal_init_y
                left_t, top_t, right_t, bottom_t = 0, 0, width, height

            if not panel_mask.any():
                logger.warning("Panel mask is empty, cannot find hot/cold points")
                return None

            # --- Find HOTTEST point: search in radius around initial point, within panel ---
            # Create distance grid from initial point
            y_coords, x_coords = np.ogrid[:height, :width]
            dist_from_initial = np.sqrt(
                (x_coords - thermal_init_x) ** 2 + (y_coords - thermal_init_y) ** 2
            )

            # Scale search radius to thermal coords
            hot_radius_t = int(hot_search_radius * SCALE_X)

            # Inner search: small radius around initial point, within panel
            inner_search_mask = (dist_from_initial <= hot_radius_t) & panel_mask

            # Also create a "prefer inner" mask - points closer to panel center
            dist_from_center = np.sqrt(
                (x_coords - panel_center_x) ** 2 + (y_coords - panel_center_y) ** 2
            )
            # Inner half of panel (closer to center)
            panel_half_diag = np.sqrt(
                ((right_t - left_t) / 2) ** 2 + ((bottom_t - top_t) / 2) ** 2
            )
            prefer_inner_mask = dist_from_center <= (panel_half_diag * 0.6)

            # Try to find hottest in inner search area first, preferring center
            inner_prefer_center = inner_search_mask & prefer_inner_mask
            if inner_prefer_center.any():
                # Found candidates in inner area closer to center
                hot_temps = np.where(inner_prefer_center, temp_array, -np.inf)
                hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                hot_thermal_y, hot_thermal_x = hot_idx
                hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                logger.debug(f"Found hot point in inner+center area: {hot_temp:.1f}°C")
            elif inner_search_mask.any():
                # Found candidates in inner area (but not center-preferred)
                hot_temps = np.where(inner_search_mask, temp_array, -np.inf)
                hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                hot_thermal_y, hot_thermal_x = hot_idx
                hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                logger.debug(f"Found hot point in inner search area: {hot_temp:.1f}°C")
            else:
                # Fallback: search entire panel but prefer center
                panel_prefer_center = panel_mask & prefer_inner_mask
                if panel_prefer_center.any():
                    hot_temps = np.where(panel_prefer_center, temp_array, -np.inf)
                else:
                    hot_temps = np.where(panel_mask, temp_array, -np.inf)
                hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                hot_thermal_y, hot_thermal_x = hot_idx
                hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                logger.debug(f"Found hot point in panel fallback: {hot_temp:.1f}°C")

            # --- Find COLDEST point within panel (excluding edges) ---
            if panel_mask_eroded.any():
                cold_temps = np.where(panel_mask_eroded, temp_array, np.inf)
            else:
                # Fallback to full panel if eroded mask is empty
                cold_temps = np.where(panel_mask, temp_array, np.inf)

            cold_idx = np.unravel_index(np.argmin(cold_temps), cold_temps.shape)
            cold_thermal_y, cold_thermal_x = cold_idx
            cold_temp = float(temp_array[cold_thermal_y, cold_thermal_x])

            # Convert back to visual coordinates (with alignment correction reversed)
            hot_visual_x, hot_visual_y = thermal_to_visual(hot_thermal_x, hot_thermal_y)
            hot_visual_x = int(hot_visual_x)
            hot_visual_y = int(hot_visual_y)
            cold_visual_x, cold_visual_y = thermal_to_visual(cold_thermal_x, cold_thermal_y)
            cold_visual_x = int(cold_visual_x)
            cold_visual_y = int(cold_visual_y)

            delta_t = hot_temp - cold_temp

            logger.info(
                f"Hot/cold detection: HOT=({hot_visual_x}, {hot_visual_y}) {hot_temp:.1f}°C, "
                f"COLD=({cold_visual_x}, {cold_visual_y}) {cold_temp:.1f}°C, ΔT={delta_t:.1f}°C"
            )

            return HotColdPoints(
                hot_x=hot_visual_x,
                hot_y=hot_visual_y,
                hot_temp=round(hot_temp, 1),
                cold_x=cold_visual_x,
                cold_y=cold_visual_y,
                cold_temp=round(cold_temp, 1),
                delta_t=round(delta_t, 1),
            )

        except Exception as e:
            logger.warning(f"Hot/cold point detection failed for {image_path}: {e}")
            return None

    def create_annotated_image(
        self,
        raw_image_path: Path,
        defect_pixel_x: float,
        defect_pixel_y: float,
        panel_id: str,
        panel_row: int,
        panel_column: int,
        latitude: float,
        longitude: float,
        defect_index: int = 0,
        zoom_size: int = 200,
        output_size: Tuple[int, int] = (600, 500),
        panel_bbox_visual: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create an annotated thermal image centered on the defect with hot/cold markers.

        Uses hot/cold point detection to find:
        - RED marker: Hottest point WITHIN the panel
        - BLUE marker: Coldest reference point WITHIN the panel (excluding edges)

        Args:
            raw_image_path: Path to the raw thermal image (DJI R-JPEG)
            defect_pixel_x: X pixel coordinate of defect in raw image
            defect_pixel_y: Y pixel coordinate of defect in raw image
            panel_id: Panel identifier (e.g., "A-01")
            panel_row: Panel row number
            panel_column: Panel column number
            latitude: Latitude of defect location
            longitude: Longitude of defect location
            defect_index: Index of defect within panel (for multiple hotspots)
            zoom_size: Size of zoom window around defect (pixels)
            output_size: Final output image size (width, height)
            panel_bbox_visual: Panel bounding box in VISUAL coordinates (left, top, right, bottom)
                              as projected onto the raw thermal image. Used to constrain
                              hot/cold search to within the panel only.

        Returns:
            AnnotatedThermalImage or None if extraction fails
        """
        if not self.thermal_extractor or not self.thermal_extractor.available:
            logger.warning("Thermal extractor not available, cannot create annotated image")
            return None

        try:
            # Find hot and cold points constrained to panel bounding box
            hot_cold = self.find_hot_cold_points(
                image_path=raw_image_path,
                initial_x=int(defect_pixel_x),
                initial_y=int(defect_pixel_y),
                panel_bbox_visual=panel_bbox_visual,
                edge_margin=50,  # Exclude 50 visual pixels from edges for cold search (avoid shadows)
                hot_search_radius=60,  # Search within 60px radius of initial point for hottest
            )

            # Load the visual image
            img = cv2.imread(str(raw_image_path))
            if img is None:
                logger.warning(f"Could not load image: {raw_image_path}")
                return None

            img_h, img_w = img.shape[:2]

            # REQUIRE hot_cold detection - no fallback, skip thermal analysis if it fails
            if not hot_cold:
                logger.warning(
                    f"Hot/cold detection failed for {panel_id}, skipping thermal analysis"
                )
                return None

            center_pixel_x = hot_cold.hot_x
            center_pixel_y = hot_cold.hot_y
            hot_temp = hot_cold.hot_temp
            cold_temp = hot_cold.cold_temp
            delta_t = hot_cold.delta_t
            severity = hot_cold.severity

            # Calculate zoom window bounds centered on HOT point
            half_size = zoom_size // 2
            x1 = max(0, center_pixel_x - half_size)
            y1 = max(0, center_pixel_y - half_size)
            x2 = min(img_w, center_pixel_x + half_size)
            y2 = min(img_h, center_pixel_y + half_size)

            # Crop the zoomed region
            zoomed = img[y1:y2, x1:x2].copy()

            # Resize to target size (leaving room for colorbar only - no text annotations)
            img_area_height = output_size[1] - 20  # Minimal margin
            img_area_width = output_size[0] - 80  # Reserve space for colorbar

            # Calculate scale factors for marker positioning
            crop_width = x2 - x1
            crop_height = y2 - y1
            scale_to_canvas_x = img_area_width / crop_width if crop_width > 0 else 1
            scale_to_canvas_y = img_area_height / crop_height if crop_height > 0 else 1

            zoomed_resized = cv2.resize(
                zoomed,
                (img_area_width, img_area_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Create the output canvas
            canvas = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

            # Place the zoomed image
            canvas[10 : 10 + img_area_height, 10 : 10 + img_area_width] = zoomed_resized

            # --- Draw HOT marker (RED) ---
            # Convert hot point coordinates to canvas position
            hot_canvas_x = int(10 + (hot_cold.hot_x - x1) * scale_to_canvas_x)
            hot_canvas_y = int(10 + (hot_cold.hot_y - y1) * scale_to_canvas_y)

            # Clamp to image area
            hot_canvas_x = max(10, min(10 + img_area_width - 1, hot_canvas_x))
            hot_canvas_y = max(10, min(10 + img_area_height - 1, hot_canvas_y))

            marker_size = 15

            # Red circle with white outline for HOT point
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size + 2, (255, 255, 255), 3)
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size, (0, 0, 255), 2)
            # Cross inside
            cv2.drawMarker(
                canvas,
                (hot_canvas_x, hot_canvas_y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                marker_size - 2,
                2,
            )

            # --- Draw COLD marker (BLUE) ---
            cold_canvas_x = int(10 + (hot_cold.cold_x - x1) * scale_to_canvas_x)
            cold_canvas_y = int(10 + (hot_cold.cold_y - y1) * scale_to_canvas_y)

            # Only draw if cold point is within the cropped region
            if 10 <= cold_canvas_x <= 10 + img_area_width and 10 <= cold_canvas_y <= 10 + img_area_height:
                # Blue circle with white outline for COLD point
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size + 2, (255, 255, 255), 3)
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size, (255, 128, 0), 2)
                # Cross inside
                cv2.drawMarker(
                    canvas,
                    (cold_canvas_x, cold_canvas_y),
                    (255, 128, 0),
                    cv2.MARKER_CROSS,
                    marker_size - 2,
                    2,
                )

            # Add colorbar on the right
            colorbar_x = img_area_width + 30
            colorbar_width = 30
            colorbar_height = img_area_height

            # Create temperature colorbar (whitehot grayscale - matches DJI thermal display)
            colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
            for i in range(colorbar_height):
                ratio = 1.0 - (i / colorbar_height)
                gray_value = int(255 * ratio)
                colorbar[i, :] = (gray_value, gray_value, gray_value)

            canvas[10 : 10 + colorbar_height, colorbar_x : colorbar_x + colorbar_width] = colorbar

            # Draw colorbar border
            cv2.rectangle(
                canvas,
                (colorbar_x, 10),
                (colorbar_x + colorbar_width, 10 + colorbar_height),
                (0, 0, 0),
                1,
            )

            # Add temperature labels on colorbar
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (0, 0, 0)

            cv2.putText(
                canvas,
                f"{hot_temp:.0f}C",
                (colorbar_x + colorbar_width + 5, 20),
                font,
                font_scale,
                font_color,
                1,
            )
            cv2.putText(
                canvas,
                f"{cold_temp:.0f}C",
                (colorbar_x + colorbar_width + 5, 10 + colorbar_height - 5),
                font,
                font_scale,
                font_color,
                1,
            )

            # Text annotations removed - temperature data will be displayed in LaTeX legend

            logger.info(
                f"Created annotated thermal image for {panel_id}: "
                f"HOT={hot_temp:.1f}C, COLD={cold_temp:.1f}C, "
                f"delta_t={delta_t:.1f}C ({severity})"
            )

            return AnnotatedThermalImage(
                image=canvas,
                defect_temp=hot_temp,
                panel_avg_temp=cold_temp,
                delta_t=delta_t,
                severity=severity,
                hot_cold=hot_cold,
            )

        except Exception as e:
            logger.error(f"Failed to create annotated thermal image: {e}")
            return None

    def save_annotated_image(
        self,
        annotated: AnnotatedThermalImage,
        output_path: Path,
        quality: int = 95,
    ) -> bool:
        """
        Save annotated thermal image to file.

        Args:
            annotated: The annotated image result
            output_path: Path to save the image
            quality: JPEG quality (0-100)

        Returns:
            True if saved successfully
        """
        try:
            cv2.imwrite(
                str(output_path),
                annotated.image,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            logger.info(f"Saved annotated thermal image: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save annotated image: {e}")
            return False

    def create_annotated_image_with_points(
        self,
        raw_image_path: Path,
        hot_point: "AnnotationPoint",
        cold_point: "AnnotationPoint",
        panel_id: str,
        defect_index: int = 1,
        zoom_size: int = 200,
        output_size: Tuple[int, int] = (600, 500),
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create annotated thermal image using manually specified hot/cold points.

        This method bypasses automatic hot/cold detection and uses the provided
        coordinates directly. Used for human-in-the-loop overrides.

        Args:
            raw_image_path: Path to the raw thermal image (DJI R-JPEG)
            hot_point: Hot point coordinates in THERMAL space (640x512)
            cold_point: Cold point coordinates in THERMAL space (640x512)
            panel_id: Panel identifier (e.g., "A-01")
            defect_index: Defect index (1-based)
            zoom_size: Size of zoom window around defect (pixels)
            output_size: Final output image size (width, height)

        Returns:
            AnnotatedThermalImage or None if creation fails
        """
        # Import here to avoid circular imports
        from ..models.annotation_manifest import AnnotationPoint

        if not self.thermal_extractor or not self.thermal_extractor.available:
            logger.warning("Thermal extractor not available, cannot create annotated image")
            return None

        try:
            # Get temperature array to look up temperatures at specified points
            temp_array = self.thermal_extractor.get_full_temperature_array(raw_image_path)
            if temp_array is None:
                logger.warning(f"Could not get temperature array for {raw_image_path}")
                return None

            # Clamp coordinates to valid range
            from ..utils.thermal_alignment import clamp_thermal_coords, thermal_to_visual

            hot_tx, hot_ty = clamp_thermal_coords(hot_point.x, hot_point.y)
            cold_tx, cold_ty = clamp_thermal_coords(cold_point.x, cold_point.y)

            # Get temperatures at the specified thermal coordinates
            hot_temp = float(temp_array[hot_ty, hot_tx])
            cold_temp = float(temp_array[cold_ty, cold_tx])
            delta_t = hot_temp - cold_temp

            # Convert thermal coordinates to visual coordinates for drawing
            hot_vx, hot_vy = thermal_to_visual(hot_tx, hot_ty)
            cold_vx, cold_vy = thermal_to_visual(cold_tx, cold_ty)

            # Create HotColdPoints structure (in visual coords for drawing)
            hot_cold = HotColdPoints(
                hot_x=int(hot_vx),
                hot_y=int(hot_vy),
                hot_temp=round(hot_temp, 1),
                cold_x=int(cold_vx),
                cold_y=int(cold_vy),
                cold_temp=round(cold_temp, 1),
                delta_t=round(delta_t, 1),
            )

            # Load the visual image
            img = cv2.imread(str(raw_image_path))
            if img is None:
                logger.warning(f"Could not load image: {raw_image_path}")
                return None

            img_h, img_w = img.shape[:2]

            center_pixel_x = hot_cold.hot_x
            center_pixel_y = hot_cold.hot_y
            severity = hot_cold.severity

            # Calculate zoom window bounds centered on HOT point
            half_size = zoom_size // 2
            x1 = max(0, center_pixel_x - half_size)
            y1 = max(0, center_pixel_y - half_size)
            x2 = min(img_w, center_pixel_x + half_size)
            y2 = min(img_h, center_pixel_y + half_size)

            # Crop the zoomed region
            zoomed = img[y1:y2, x1:x2].copy()

            # Resize to target size (leaving room for colorbar only - no text annotations)
            img_area_height = output_size[1] - 20  # Minimal margin
            img_area_width = output_size[0] - 80  # Reserve space for colorbar

            crop_width = x2 - x1
            crop_height = y2 - y1
            scale_to_canvas_x = img_area_width / crop_width if crop_width > 0 else 1
            scale_to_canvas_y = img_area_height / crop_height if crop_height > 0 else 1

            zoomed_resized = cv2.resize(
                zoomed,
                (img_area_width, img_area_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Create the output canvas
            canvas = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

            # Place the zoomed image
            canvas[10 : 10 + img_area_height, 10 : 10 + img_area_width] = zoomed_resized

            # --- Draw HOT marker (RED) ---
            hot_canvas_x = int(10 + (hot_cold.hot_x - x1) * scale_to_canvas_x)
            hot_canvas_y = int(10 + (hot_cold.hot_y - y1) * scale_to_canvas_y)
            hot_canvas_x = max(10, min(10 + img_area_width - 1, hot_canvas_x))
            hot_canvas_y = max(10, min(10 + img_area_height - 1, hot_canvas_y))

            marker_size = 15
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size + 2, (255, 255, 255), 3)
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size, (0, 0, 255), 2)
            cv2.drawMarker(canvas, (hot_canvas_x, hot_canvas_y), (0, 0, 255), cv2.MARKER_CROSS, marker_size - 2, 2)

            # --- Draw COLD marker (BLUE) ---
            cold_canvas_x = int(10 + (hot_cold.cold_x - x1) * scale_to_canvas_x)
            cold_canvas_y = int(10 + (hot_cold.cold_y - y1) * scale_to_canvas_y)

            if 10 <= cold_canvas_x <= 10 + img_area_width and 10 <= cold_canvas_y <= 10 + img_area_height:
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size + 2, (255, 255, 255), 3)
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size, (255, 128, 0), 2)
                cv2.drawMarker(canvas, (cold_canvas_x, cold_canvas_y), (255, 128, 0), cv2.MARKER_CROSS, marker_size - 2, 2)

            # Add colorbar on the right
            colorbar_x = img_area_width + 30
            colorbar_width = 30
            colorbar_height = img_area_height

            colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
            for i in range(colorbar_height):
                ratio = 1.0 - (i / colorbar_height)
                gray_value = int(255 * ratio)
                colorbar[i, :] = (gray_value, gray_value, gray_value)

            canvas[10 : 10 + colorbar_height, colorbar_x : colorbar_x + colorbar_width] = colorbar
            cv2.rectangle(canvas, (colorbar_x, 10), (colorbar_x + colorbar_width, 10 + colorbar_height), (0, 0, 0), 1)

            # Add temperature labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (0, 0, 0)
            cv2.putText(canvas, f"{hot_temp:.0f}C", (colorbar_x + colorbar_width + 5, 20), font, font_scale, font_color, 1)
            cv2.putText(canvas, f"{cold_temp:.0f}C", (colorbar_x + colorbar_width + 5, 10 + colorbar_height - 5), font, font_scale, font_color, 1)

            # Text annotations removed - temperature data will be displayed in LaTeX legend

            logger.info(
                f"Created annotated thermal image with manual points for {panel_id}: "
                f"HOT=({hot_tx}, {hot_ty}) {hot_temp:.1f}C, COLD=({cold_tx}, {cold_ty}) {cold_temp:.1f}C, "
                f"delta_t={delta_t:.1f}C ({severity})"
            )

            return AnnotatedThermalImage(
                image=canvas,
                defect_temp=hot_temp,
                panel_avg_temp=cold_temp,
                delta_t=delta_t,
                severity=severity,
                hot_cold=hot_cold,
            )

        except Exception as e:
            logger.error(f"Failed to create annotated thermal image with manual points: {e}")
            return None
