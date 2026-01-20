"""Create annotated thermal images with temperature overlay for reports.

This module generates professional annotated thermal images that show:
- Zoomed view of the defect area in the raw thermal image
- Temperature colorbar overlay (matching detected palette)
- RED marker on hottest point (detected hotspot)
- BLUE marker on coolest reference point (in outer ring)
- Temperature readings for both points and delta T
"""

import cv2
import numpy as np
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from ..utils.logger import get_logger
from .thermal_extractor import ThermalExtractor, TemperatureReading

logger = get_logger(__name__)


def detect_thermal_palette(image_path: Path) -> str:
    """
    Detect the color palette used in a DJI thermal image.

    Reads the EXIF 'Image Description' field which contains the palette name.
    Common DJI palettes: WhiteHot, BlackHot, Iron, IronRed, Rainbow, Fulgurite, Medical

    Args:
        image_path: Path to DJI thermal R-JPEG image

    Returns:
        Palette name (lowercase) or 'whitehot' as default
    """
    try:
        result = subprocess.run(
            ['exiftool', '-ImageDescription', '-s', '-s', '-s', str(image_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            palette = result.stdout.strip().lower()
            logger.debug(f"Detected thermal palette: {palette}")
            return palette
    except Exception as e:
        logger.debug(f"Could not detect palette from {image_path}: {e}")

    return "whitehot"  # Default


def create_colorbar(
    height: int,
    width: int,
    palette: str = "whitehot",
) -> np.ndarray:
    """
    Create a colorbar image matching the thermal palette.

    Args:
        height: Height of colorbar in pixels
        width: Width of colorbar in pixels
        palette: Palette name (whitehot, blackhot, iron, rainbow, fulgurite, medical)

    Returns:
        BGR colorbar image (numpy array)
    """
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)

    palette = palette.lower().strip()

    for i in range(height):
        # ratio goes from 1.0 (top, hot) to 0.0 (bottom, cold)
        ratio = 1.0 - (i / height)

        if palette == "whitehot":
            # White = hot, Black = cold (grayscale)
            gray_value = int(255 * ratio)
            colorbar[i, :] = (gray_value, gray_value, gray_value)

        elif palette == "blackhot":
            # Black = hot, White = cold (inverted grayscale)
            gray_value = int(255 * (1 - ratio))
            colorbar[i, :] = (gray_value, gray_value, gray_value)

        elif palette in ("iron", "ironbow", "ironred"):
            # Iron/Ironbow: black -> blue -> purple -> red -> orange -> yellow -> white
            if ratio < 0.15:
                # Black to blue
                t = ratio / 0.15
                colorbar[i, :] = (int(128 * t), 0, 0)  # BGR
            elif ratio < 0.30:
                # Blue to purple
                t = (ratio - 0.15) / 0.15
                colorbar[i, :] = (128 + int(50 * t), 0, int(80 * t))
            elif ratio < 0.50:
                # Purple to red
                t = (ratio - 0.30) / 0.20
                colorbar[i, :] = (int(178 * (1 - t)), 0, int(80 + 175 * t))
            elif ratio < 0.70:
                # Red to orange
                t = (ratio - 0.50) / 0.20
                colorbar[i, :] = (0, int(100 * t), 255)
            elif ratio < 0.85:
                # Orange to yellow
                t = (ratio - 0.70) / 0.15
                colorbar[i, :] = (0, int(100 + 155 * t), 255)
            else:
                # Yellow to white
                t = (ratio - 0.85) / 0.15
                colorbar[i, :] = (int(255 * t), 255, 255)

        elif palette == "rainbow":
            # Rainbow: blue -> cyan -> green -> yellow -> red
            if ratio < 0.25:
                t = ratio / 0.25
                colorbar[i, :] = (255, int(255 * t), 0)  # Blue to cyan
            elif ratio < 0.50:
                t = (ratio - 0.25) / 0.25
                colorbar[i, :] = (int(255 * (1 - t)), 255, 0)  # Cyan to green
            elif ratio < 0.75:
                t = (ratio - 0.50) / 0.25
                colorbar[i, :] = (0, 255, int(255 * t))  # Green to yellow
            else:
                t = (ratio - 0.75) / 0.25
                colorbar[i, :] = (0, int(255 * (1 - t)), 255)  # Yellow to red

        elif palette == "fulgurite":
            # Fulgurite: dark blue -> light blue -> white -> yellow -> orange
            if ratio < 0.25:
                t = ratio / 0.25
                colorbar[i, :] = (int(100 + 100 * t), int(50 * t), int(50 * t))
            elif ratio < 0.50:
                t = (ratio - 0.25) / 0.25
                colorbar[i, :] = (int(200 + 55 * t), int(50 + 150 * t), int(50 + 150 * t))
            elif ratio < 0.75:
                t = (ratio - 0.50) / 0.25
                colorbar[i, :] = (int(255 * (1 - t * 0.5)), 255, int(200 + 55 * t))
            else:
                t = (ratio - 0.75) / 0.25
                colorbar[i, :] = (0, int(255 * (1 - t * 0.5)), 255)

        elif palette == "medical":
            # Medical: blue to red through green/yellow
            if ratio < 0.33:
                t = ratio / 0.33
                colorbar[i, :] = (255, int(255 * t), 0)  # Blue to cyan
            elif ratio < 0.66:
                t = (ratio - 0.33) / 0.33
                colorbar[i, :] = (int(255 * (1 - t)), 255, int(255 * t))  # Cyan through green to yellow
            else:
                t = (ratio - 0.66) / 0.34
                colorbar[i, :] = (0, int(255 * (1 - t)), 255)  # Yellow to red

        else:
            # Unknown palette - default to whitehot
            gray_value = int(255 * ratio)
            colorbar[i, :] = (gray_value, gray_value, gray_value)

    return colorbar


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
        image_format=None,
        visual_size: Optional[Tuple[int, int]] = None,
        flight_direction: Optional[str] = None,
        excluded_hot_points: Optional[List[Tuple[int, int]]] = None,
        min_hot_distance: int = 10,
    ) -> Optional[HotColdPoints]:
        """
        Find hottest and coldest points WITHIN the panel bounding box.

        The algorithm:
        1. Search for HOTTEST pixel in a small radius around initial point, constrained to panel
        2. Prefer hot points closer to panel center over edge hot points
        3. Find COLDEST pixel within the panel, excluding edge pixels (to avoid shadows)

        Args:
            image_path: Path to DJI thermal R-JPEG image
            initial_x: Initial X coordinate in VISUAL image (1280x1024), already rotated if needed
            initial_y: Initial Y coordinate in VISUAL image (1280x1024), already rotated if needed
            panel_bbox_visual: Optional (left, top, right, bottom) in visual coords (1280x1024).
                              If provided, search is constrained to within this panel.
                              Already rotated if flight_direction is 'south'.
            edge_margin: Pixels to exclude from edges when searching for cold point (visual coords)
            hot_search_radius: Radius around initial point to search for hottest pixel (visual coords)
            flight_direction: 'south' if drone was flying south (thermal array needs 180° rotation)
            excluded_hot_points: Optional list of visual coords to avoid for hot selection.
                                Already rotated if flight_direction is 'south'.
            min_hot_distance: Minimum separation from excluded hot points (visual pixels)

        Returns:
            HotColdPoints or None if detection fails
        """
        if not self.thermal_extractor or not self.thermal_extractor.available:
            return None

        try:
            temp_array = self.thermal_extractor.get_full_temperature_array(image_path)
            if temp_array is None:
                return None

            # Rotate thermal array 180° for south-facing flights
            # This aligns the thermal data with the rotated visual coordinates
            # Without this, we search in the wrong part of the thermal image
            if flight_direction == "south":
                temp_array = np.rot90(temp_array, 2)  # 180° rotation
                logger.debug("Rotated thermal array 180° for south-facing flight")

            height, width = temp_array.shape

            # Convert from visual (1280x1024) to thermal (640x512) coordinates
            # This applies any configured alignment offset
            from ..utils.thermal_alignment import (
                visual_to_thermal,
                thermal_to_visual,
                clamp_thermal_coords,
                visual_bbox_to_thermal,
                ImageFormat,
            )

            if image_format is None:
                image_format = ImageFormat.VISUAL_THERMAL

            thermal_size = (width, height)
            thermal_init_x_f, thermal_init_y_f = visual_to_thermal(
                initial_x,
                initial_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            thermal_init_x, thermal_init_y = clamp_thermal_coords(
                thermal_init_x_f,
                thermal_init_y_f,
                thermal_size=thermal_size,
            )

            # Create panel mask in thermal coordinates
            if panel_bbox_visual is not None:
                # Convert panel bbox from visual to thermal coordinates (with alignment)
                left_v, top_v, right_v, bottom_v = panel_bbox_visual
                left_t, top_t, right_t, bottom_t = visual_bbox_to_thermal(
                    left_v,
                    top_v,
                    right_v,
                    bottom_v,
                    image_format=image_format,
                    visual_size=visual_size,
                    thermal_size=thermal_size,
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
                if image_format == ImageFormat.VISUAL_THERMAL and visual_size:
                    scale_x = thermal_size[0] / visual_size[0]
                else:
                    scale_x = 1.0
                edge_margin_t = int(edge_margin * scale_x)
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

            y_coords, x_coords = np.ogrid[:height, :width]

            hot_exclusion_mask = None
            if excluded_hot_points:
                if image_format == ImageFormat.VISUAL_THERMAL and visual_size:
                    scale_x_dist = thermal_size[0] / visual_size[0]
                    scale_y_dist = thermal_size[1] / visual_size[1]
                else:
                    scale_x_dist = 1.0
                    scale_y_dist = 1.0

                min_distance_t = max(
                    1, int(round(min_hot_distance * ((scale_x_dist + scale_y_dist) / 2.0)))
                )
                hot_exclusion_mask = np.zeros((height, width), dtype=bool)
                for ex_x, ex_y in excluded_hot_points:
                    ex_t_x_f, ex_t_y_f = visual_to_thermal(
                        ex_x,
                        ex_y,
                        image_format=image_format,
                        visual_size=visual_size,
                        thermal_size=thermal_size,
                    )
                    ex_t_x, ex_t_y = clamp_thermal_coords(
                        ex_t_x_f,
                        ex_t_y_f,
                        thermal_size=thermal_size,
                    )
                    dist_from_exclusion = np.sqrt(
                        (x_coords - ex_t_x) ** 2 + (y_coords - ex_t_y) ** 2
                    )
                    hot_exclusion_mask |= dist_from_exclusion <= min_distance_t

                if hot_exclusion_mask.any():
                    logger.debug(
                        f"Excluding {len(excluded_hot_points)} prior hot points within "
                        f"{min_distance_t}px (thermal)"
                    )

            hot_allowed_mask = panel_mask
            if hot_exclusion_mask is not None:
                hot_allowed_mask = panel_mask & ~hot_exclusion_mask
                if not hot_allowed_mask.any():
                    logger.warning(
                        "Hot exclusion removed all candidates; ignoring exclusions for this defect"
                    )
                    hot_allowed_mask = panel_mask

            # --- Find HOTTEST point: search in radius around initial point (DEFECT center), within panel ---
            # Create distance grid from initial point (the individual defect center, NOT panel center)
            # This is critical for panels with multiple hotspots - each defect must find its OWN hot point
            dist_from_defect = np.sqrt(
                (x_coords - thermal_init_x) ** 2 + (y_coords - thermal_init_y) ** 2
            )

            # Scale search radius to thermal coords
            if image_format == ImageFormat.VISUAL_THERMAL and visual_size:
                scale_x = thermal_size[0] / visual_size[0]
            else:
                scale_x = 1.0
            hot_radius_t = int(hot_search_radius * scale_x)

            # Inner search: small radius around defect center, within panel
            inner_search_mask = (dist_from_defect <= hot_radius_t) & hot_allowed_mask

            # Create a "prefer near defect" mask - points closer to the DEFECT center (not panel center)
            # This ensures each defect on a panel finds its OWN hot point, not the panel's hottest overall
            # Use 2x search radius for preference to still find nearby hot spots
            prefer_near_defect_mask = dist_from_defect <= (hot_radius_t * 2)

            # Try to find hottest in search area around DEFECT center (not panel center)
            # This ensures each defect on a panel finds its OWN hot point
            if inner_search_mask.any():
                # Found candidates within search radius of defect center
                hot_temps = np.where(inner_search_mask, temp_array, -np.inf)
                hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                hot_thermal_y, hot_thermal_x = hot_idx
                hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                logger.debug(f"Found hot point near defect center: {hot_temp:.1f}°C at ({hot_thermal_x}, {hot_thermal_y})")
            elif prefer_near_defect_mask.any():
                # Expand to 2x radius if inner search found nothing
                expanded_mask = prefer_near_defect_mask & hot_allowed_mask
                if expanded_mask.any():
                    hot_temps = np.where(expanded_mask, temp_array, -np.inf)
                    hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                    hot_thermal_y, hot_thermal_x = hot_idx
                    hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                    logger.debug(f"Found hot point in expanded search: {hot_temp:.1f}°C")
                else:
                    # Last resort: search within panel mask only
                    hot_temps = np.where(hot_allowed_mask, temp_array, -np.inf)
                    hot_idx = np.unravel_index(np.argmax(hot_temps), hot_temps.shape)
                    hot_thermal_y, hot_thermal_x = hot_idx
                    hot_temp = float(temp_array[hot_thermal_y, hot_thermal_x])
                    logger.debug(f"Found hot point in panel fallback: {hot_temp:.1f}°C")
            else:
                # No valid search area - use panel mask
                hot_temps = np.where(hot_allowed_mask, temp_array, -np.inf)
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
            hot_visual_x, hot_visual_y = thermal_to_visual(
                hot_thermal_x,
                hot_thermal_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            hot_visual_x = int(hot_visual_x)
            hot_visual_y = int(hot_visual_y)
            cold_visual_x, cold_visual_y = thermal_to_visual(
                cold_thermal_x,
                cold_thermal_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
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
        flight_direction: Optional[str] = None,
        hot_search_radius: int = 60,
        excluded_hot_points: Optional[List[Tuple[int, int]]] = None,
        min_hot_distance: int = 10,
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create an annotated thermal image centered on the defect with hot/cold markers.

        Uses hot/cold point detection to find:
        - RED marker: Hottest point WITHIN the panel
        - BLUE marker: Coldest reference point WITHIN the panel (excluding edges)

        Args:
            raw_image_path: Path to the raw thermal image (DJI R-JPEG)
            defect_pixel_x: X pixel coordinate of defect in raw image (unrotated coords)
            defect_pixel_y: Y pixel coordinate of defect in raw image (unrotated coords)
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
            flight_direction: 'south' if drone was flying south (image needs 180° rotation),
                             'north' or None if no rotation needed.
            hot_search_radius: Radius in VISUAL pixels to search for hottest point around defect center.
                              Use smaller values (10-20px) for precise backtracking methods (source-map).
                              Use larger values (60px) for less precise methods (GPS, reprojection).
            excluded_hot_points: Optional list of visual coords to avoid for hot selection.
            min_hot_distance: Minimum separation from excluded hot points (visual pixels).

        Returns:
            AnnotatedThermalImage or None if extraction fails
        """
        if not self.thermal_extractor or not self.thermal_extractor.available:
            logger.warning("Thermal extractor not available, cannot create annotated image")
            return None

        try:
            # Load the visual image
            img = cv2.imread(str(raw_image_path))
            if img is None:
                logger.warning(f"Could not load image: {raw_image_path}")
                return None

            img_h, img_w = img.shape[:2]

            excluded_hot_points_rotated = None
            if excluded_hot_points:
                excluded_hot_points_rotated = [
                    (int(ex_x), int(ex_y)) for ex_x, ex_y in excluded_hot_points
                ]

            # Apply 180° rotation for south-facing images
            # This rotates both the image and transforms the input coordinates
            if flight_direction == "south":
                img = cv2.rotate(img, cv2.ROTATE_180)
                # Transform coordinates for rotated image: (x, y) -> (w-1-x, h-1-y)
                defect_pixel_x = img_w - 1 - defect_pixel_x
                defect_pixel_y = img_h - 1 - defect_pixel_y
                if excluded_hot_points_rotated:
                    excluded_hot_points_rotated = [
                        (img_w - 1 - ex_x, img_h - 1 - ex_y)
                        for ex_x, ex_y in excluded_hot_points_rotated
                    ]
                # Also transform panel bbox if provided
                if panel_bbox_visual is not None:
                    left, top, right, bottom = panel_bbox_visual
                    # After 180° rotation: new_left = w-1-right, new_right = w-1-left, etc.
                    panel_bbox_visual = (
                        img_w - 1 - right,   # new left
                        img_h - 1 - bottom,  # new top
                        img_w - 1 - left,    # new right
                        img_h - 1 - top,     # new bottom
                    )
                logger.debug(f"Rotated image 180° for south-facing flight")

            # Determine image format and size for thermal alignment
            from ..utils.thermal_alignment import ImageFormat
            image_format = ImageFormat.VISUAL_THERMAL
            if img_w != 1280 or img_h != 1024:
                image_format = ImageFormat.THERMAL_ONLY
            visual_size = (img_w, img_h)

            # Find hot and cold points constrained to panel bounding box
            # Note: defect_pixel_x/y and panel_bbox_visual are already rotated if south-facing
            hot_cold = self.find_hot_cold_points(
                image_path=raw_image_path,
                initial_x=int(defect_pixel_x),
                initial_y=int(defect_pixel_y),
                panel_bbox_visual=panel_bbox_visual,
                edge_margin=50,  # Exclude 50 visual pixels from edges for cold search (avoid shadows)
                hot_search_radius=hot_search_radius,  # Use caller-specified radius
                image_format=image_format,
                visual_size=visual_size,
                flight_direction=flight_direction,  # For thermal array rotation
                excluded_hot_points=excluded_hot_points_rotated,
                min_hot_distance=min_hot_distance,
            )

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

            marker_size = 20
            outline_thickness = 5
            marker_thickness = 4

            # Red circle with white outline for HOT point
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size + 3, (255, 255, 255), outline_thickness)
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size, (0, 0, 255), marker_thickness)
            # Cross inside
            cv2.drawMarker(
                canvas,
                (hot_canvas_x, hot_canvas_y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                marker_size - 4,
                marker_thickness,
            )

            # --- Draw COLD marker (BLUE) ---
            cold_canvas_x = int(10 + (hot_cold.cold_x - x1) * scale_to_canvas_x)
            cold_canvas_y = int(10 + (hot_cold.cold_y - y1) * scale_to_canvas_y)

            # Only draw if cold point is within the cropped region
            if 10 <= cold_canvas_x <= 10 + img_area_width and 10 <= cold_canvas_y <= 10 + img_area_height:
                # Blue circle with white outline for COLD point
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size + 3, (255, 255, 255), outline_thickness)
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size, (255, 128, 0), marker_thickness)
                # Cross inside
                cv2.drawMarker(
                    canvas,
                    (cold_canvas_x, cold_canvas_y),
                    (255, 128, 0),
                    cv2.MARKER_CROSS,
                    marker_size - 4,
                    marker_thickness,
                )

            # Add colorbar on the right - detect palette from image
            colorbar_x = img_area_width + 30
            colorbar_width = 30
            colorbar_height = img_area_height

            # Detect palette from raw image and create matching colorbar
            palette = detect_thermal_palette(raw_image_path)
            colorbar = create_colorbar(colorbar_height, colorbar_width, palette)
            logger.debug(f"Created colorbar for palette: {palette}")

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
        flight_direction: Optional[str] = None,
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create annotated thermal image using manually specified hot/cold points.

        This method bypasses automatic hot/cold detection and uses the provided
        coordinates directly. Used for human-in-the-loop overrides.

        Args:
            raw_image_path: Path to the raw thermal image (DJI R-JPEG)
            hot_point: Hot point coordinates in THERMAL space (640x512), in ROTATED space
            cold_point: Cold point coordinates in THERMAL space (640x512), in ROTATED space
            panel_id: Panel identifier (e.g., "A-01")
            defect_index: Defect index (1-based)
            zoom_size: Size of zoom window around defect (pixels)
            output_size: Final output image size (width, height)
            flight_direction: 'south' if drone was flying south (requires 180° rotation)

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

            # Rotate thermal array 180° for south-facing flights
            # Override coordinates are in ROTATED space, so we need to rotate the raw data to match
            if flight_direction == "south":
                temp_array = np.rot90(temp_array, 2)  # 180° rotation
                logger.debug("Rotated thermal array 180° for south-facing flight (override)")

            # Get temperatures using 5x5 area around the clicked point
            # Hot point: take maximum temperature in 5x5 area
            # Cold point: take minimum temperature in 5x5 area
            height, width = temp_array.shape
            thermal_size = (width, height)

            # Load the visual image to determine dimensions
            img = cv2.imread(str(raw_image_path))
            if img is None:
                logger.warning(f"Could not load image: {raw_image_path}")
                return None

            # Rotate visual image 180° for south-facing flights
            if flight_direction == "south":
                img = cv2.rotate(img, cv2.ROTATE_180)
                logger.debug("Rotated visual image 180° for south-facing flight (override)")

            img_h, img_w = img.shape[:2]

            from ..utils.thermal_alignment import clamp_thermal_coords, thermal_to_visual, ImageFormat
            image_format = ImageFormat.VISUAL_THERMAL
            if img_w != 1280 or img_h != 1024:
                image_format = ImageFormat.THERMAL_ONLY
            visual_size = (img_w, img_h)

            # Clamp coordinates to valid range
            hot_tx, hot_ty = clamp_thermal_coords(
                hot_point.x, hot_point.y, thermal_size=thermal_size
            )
            cold_tx, cold_ty = clamp_thermal_coords(
                cold_point.x, cold_point.y, thermal_size=thermal_size
            )

            # Extract 5x5 region around hot point and find max
            hot_y1 = max(0, hot_ty - 2)
            hot_y2 = min(height, hot_ty + 3)
            hot_x1 = max(0, hot_tx - 2)
            hot_x2 = min(width, hot_tx + 3)
            hot_region = temp_array[hot_y1:hot_y2, hot_x1:hot_x2]
            hot_temp = float(np.max(hot_region))

            # Extract 5x5 region around cold point and find min
            cold_y1 = max(0, cold_ty - 2)
            cold_y2 = min(height, cold_ty + 3)
            cold_x1 = max(0, cold_tx - 2)
            cold_x2 = min(width, cold_tx + 3)
            cold_region = temp_array[cold_y1:cold_y2, cold_x1:cold_x2]
            cold_temp = float(np.min(cold_region))
            delta_t = hot_temp - cold_temp

            # Convert thermal coordinates to visual coordinates for drawing
            hot_vx, hot_vy = thermal_to_visual(
                hot_tx,
                hot_ty,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            cold_vx, cold_vy = thermal_to_visual(
                cold_tx,
                cold_ty,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )

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

            marker_size = 20
            outline_thickness = 5
            marker_thickness = 4
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size + 3, (255, 255, 255), outline_thickness)
            cv2.circle(canvas, (hot_canvas_x, hot_canvas_y), marker_size, (0, 0, 255), marker_thickness)
            cv2.drawMarker(canvas, (hot_canvas_x, hot_canvas_y), (0, 0, 255), cv2.MARKER_CROSS, marker_size - 4, marker_thickness)

            # --- Draw COLD marker (BLUE) ---
            cold_canvas_x = int(10 + (hot_cold.cold_x - x1) * scale_to_canvas_x)
            cold_canvas_y = int(10 + (hot_cold.cold_y - y1) * scale_to_canvas_y)

            if 10 <= cold_canvas_x <= 10 + img_area_width and 10 <= cold_canvas_y <= 10 + img_area_height:
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size + 3, (255, 255, 255), outline_thickness)
                cv2.circle(canvas, (cold_canvas_x, cold_canvas_y), marker_size, (255, 128, 0), marker_thickness)
                cv2.drawMarker(canvas, (cold_canvas_x, cold_canvas_y), (255, 128, 0), cv2.MARKER_CROSS, marker_size - 4, marker_thickness)

            # Add colorbar on the right - detect palette from image
            colorbar_x = img_area_width + 30
            colorbar_width = 30
            colorbar_height = img_area_height

            # Detect palette from raw image and create matching colorbar
            palette = detect_thermal_palette(raw_image_path)
            colorbar = create_colorbar(colorbar_height, colorbar_width, palette)

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
