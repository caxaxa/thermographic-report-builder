"""Create annotated thermal images with temperature overlay for reports.

This module generates professional annotated thermal images that show:
- Zoomed view of the defect area in the raw thermal image
- Temperature colorbar overlay (matching detected palette)
- RED marker on hottest point (detected hotspot)
- BLUE marker on the reference point (healthy module surface next to the hotspot)
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
    DJI palettes: WhiteHot, BlackHot, Fulgurite, IronRed, HotIron, Medical,
    Arctic, Rainbow1, Rainbow2, Tint.

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


# DJI Thermal SDK palette LUTs — extracted from libdirp.so via dirp_get_pseudo_color_lut().
# Each entry is a list of (lut_index, R, G, B) control points. Index 0 = coldest, 255 = hottest.
_DJI_PALETTE_CONTROL_POINTS = {
    "whitehot": [
        (0, 0, 0, 0), (128, 128, 128, 128), (255, 255, 255, 255),
    ],
    "blackhot": [
        (0, 255, 255, 255), (128, 127, 127, 127), (255, 0, 0, 0),
    ],
    "fulgurite": [
        (0, 0, 0, 0), (32, 86, 0, 0), (64, 172, 0, 0), (96, 255, 3, 0),
        (128, 255, 88, 0), (160, 255, 173, 0), (192, 255, 255, 4),
        (224, 255, 255, 131), (255, 255, 255, 255),
    ],
    "ironred": [
        (0, 0, 0, 32), (32, 34, 0, 97), (64, 109, 0, 131), (96, 183, 2, 159),
        (112, 220, 34, 77), (128, 255, 66, 0), (160, 255, 129, 0),
        (192, 255, 193, 0), (224, 255, 255, 6), (255, 255, 255, 200),
    ],
    "hotiron": [
        (0, 0, 0, 0), (32, 12, 47, 47), (64, 24, 94, 94), (96, 51, 139, 117),
        (128, 120, 178, 78), (160, 189, 217, 38), (192, 255, 251, 0),
        (224, 255, 124, 0), (255, 224, 0, 0),
    ],
    "medical": [
        (0, 64, 32, 64), (32, 218, 63, 225), (64, 32, 39, 255),
        (96, 19, 238, 235), (128, 6, 185, 75), (160, 106, 194, 7),
        (192, 255, 251, 20), (224, 255, 39, 39), (255, 255, 255, 255),
    ],
    "arctic": [
        (0, 0, 32, 64), (32, 0, 67, 249), (64, 4, 163, 58), (96, 131, 255, 0),
        (128, 255, 251, 0), (160, 255, 124, 0), (192, 255, 0, 8),
        (224, 252, 8, 252), (255, 255, 255, 255),
    ],
    "rainbow1": [
        (0, 128, 0, 0), (32, 250, 0, 199), (64, 96, 0, 255), (96, 0, 77, 255),
        (128, 0, 255, 243), (160, 0, 255, 53), (192, 140, 255, 0),
        (224, 255, 180, 0), (255, 255, 0, 0),
    ],
    "rainbow2": [
        (0, 0, 0, 131), (32, 0, 3, 255), (64, 0, 131, 255), (96, 3, 255, 251),
        (128, 131, 255, 123), (160, 255, 251, 0), (192, 255, 123, 0),
        (224, 251, 0, 0), (255, 127, 0, 0),
    ],
    "tint": [
        (0, 0, 0, 0), (64, 91, 91, 91), (128, 182, 182, 182),
        (176, 251, 251, 251), (192, 255, 204, 204), (224, 254, 95, 95),
        (255, 224, 0, 0),
    ],
}

# Aliases: DJI EXIF strings map to canonical palette names
_PALETTE_ALIASES = {
    "iron": "ironred",
    "ironbow": "ironred",
    "iron_red": "ironred",
    "hot_iron": "hotiron",
    "rainbow": "rainbow1",
    "white_hot": "whitehot",
    "black_hot": "blackhot",
}


def _build_lut_256(palette: str) -> np.ndarray:
    """Build a full 256x3 RGB LUT by interpolating control points."""
    key = _PALETTE_ALIASES.get(palette, palette)
    points = _DJI_PALETTE_CONTROL_POINTS.get(key)
    if points is None:
        logger.warning(f"Unknown palette '{palette}', falling back to whitehot")
        points = _DJI_PALETTE_CONTROL_POINTS["whitehot"]

    indices = np.array([p[0] for p in points], dtype=np.float64)
    r_vals = np.array([p[1] for p in points], dtype=np.float64)
    g_vals = np.array([p[2] for p in points], dtype=np.float64)
    b_vals = np.array([p[3] for p in points], dtype=np.float64)

    x = np.arange(256, dtype=np.float64)
    lut = np.stack([
        np.interp(x, indices, r_vals),
        np.interp(x, indices, g_vals),
        np.interp(x, indices, b_vals),
    ], axis=-1).astype(np.uint8)
    return lut


def create_colorbar(
    height: int,
    width: int,
    palette: str = "whitehot",
) -> np.ndarray:
    """
    Create a colorbar image matching the DJI thermal palette.

    Uses exact LUT data extracted from the DJI Thermal SDK (libdirp.so).
    Supports all 10 DJI palettes: WhiteHot, BlackHot, Fulgurite, IronRed,
    HotIron, Medical, Arctic, Rainbow1, Rainbow2, Tint.

    Args:
        height: Height of colorbar in pixels
        width: Width of colorbar in pixels
        palette: Palette name (case-insensitive, underscores optional)

    Returns:
        BGR colorbar image (numpy array)
    """
    lut = _build_lut_256(palette.lower().strip())

    # Map each row to a LUT index: top = hot (255), bottom = cold (0)
    row_indices = np.linspace(255, 0, height).astype(np.uint8)
    # RGB LUT → BGR for OpenCV
    bgr_lut = lut[row_indices][:, ::-1]
    colorbar = np.tile(bgr_lut[:, np.newaxis, :], (1, width, 1))
    return colorbar


@dataclass
class HotColdPoints:
    """Hot and cold reference points detected in thermal image."""

    hot_x: int  # X coordinate of hottest point (visual coords)
    hot_y: int  # Y coordinate of hottest point (visual coords)
    hot_temp: float  # Temperature at hottest point
    cold_x: int  # X coordinate of reference point on the module surface (visual coords)
    cold_y: int  # Y coordinate of reference point on the module surface (visual coords)
    cold_temp: float  # Module surface temperature at the reference point
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
    panel_avg_temp: float  # Reference temperature in Celsius (module surface next to the defect)
    delta_t: float  # Temperature difference
    severity: str  # Severity classification
    hot_cold: Optional[HotColdPoints] = None  # Hot/cold point details if detected


# Reference point search (the "cold" point of each defect), see find_module_reference_point().
# Distances are thermal pixels, temperatures °C; tuned against reviewer-placed reference points.
_REF_MAX_MODULE_SCALE = 80.0  # per 640 px of frame width; a larger box is not a single module
_REF_MIN_HOT_GAP = 4  # keeps the marker's 5x5 window off the hot point
_REF_CLEAR_HOT_GAP = 16  # keeps the two markers apart in the report image (51 px rings, 3.4 px per thermal pixel)
_REF_LEVEL_TOL = 0.05  # marker pixel must be at the surface level
_REF_FLATNESS_SLACK = 0.55  # 5x5 spread allowed above the flattest candidate (0.5 + half a 0.1 step)
_REF_MAX_DIP = 2.0  # deeper dip on the way to the hot point = gap between modules
_REF_PATH_CANDIDATES = 64  # nearest clear candidates checked for a gap-free path


def _shorth_midpoint(values: np.ndarray) -> float:
    """Median of the shortest interval holding half of the values (densest part of the distribution)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n < 4:
        return float(np.median(v))
    k = max(2, int(np.ceil(n * 0.5)))
    start = int(np.argmin(v[k - 1:] - v[:n - k + 1]))
    return float(np.median(v[start:start + k]))


def find_module_reference_point(
    temp_array: np.ndarray,
    panel_mask: np.ndarray,
    hot_x: int,
    hot_y: int,
) -> Optional[Tuple[int, int, float]]:
    """
    Find the reference point for a hot spot: healthy module surface next to it.

    The coldest pixel of the panel box is not a usable reference: it sits on the gap
    between modules, a shadow or the ground, and inflates ΔT. Instead:

    1. Region: panel pixels within one module scale (sqrt of the panel area) of the hot
       point, plus everything within half a module scale of it. The second term keeps the
       region on the hot module when the projected box is a few pixels off; the scale is
       capped so an oversized box degrades to a disc around the hot point.
    2. Level: midpoint of the shortest interval holding half of the region's temperatures.
       That is the module surface; hot cells, gaps, shadow and ground fall in the tails.
    3. Marker: a pixel at that level on a flat 5x5 patch, preferring one with no cold gap on
       the straight line to the hot point (same module). Among those, the nearest one that
       leaves room for both markers in the report image; on a module too small for that,
       the farthest one.

    The returned temperature is the minimum of the 5x5 window at the marker, which is what
    create_annotated_image_with_points() reads at a cold point, so re-reading the marker
    through the override path gives the same value.

    Args:
        temp_array: Temperatures in Celsius (already rotated for south-facing flights)
        panel_mask: Boolean mask of the panel box in the same frame
        hot_x: X coordinate of the hot point (thermal pixels)
        hot_y: Y coordinate of the hot point (thermal pixels)

    Returns:
        (x, y, temperature) of the reference point in thermal pixels, or None if there is
        no valid temperature around the hot point
    """
    temps = np.asarray(temp_array, dtype=np.float64)
    temps = np.where(np.isfinite(temps), temps, np.inf)  # invalid pixels never win a minimum
    height, width = temps.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_hot = np.hypot(x_coords - hot_x, y_coords - hot_y)

    module_scale = min(float(np.sqrt(panel_mask.sum())), _REF_MAX_MODULE_SCALE * width / 640.0)
    region = (panel_mask & (dist_from_hot <= max(8.0, module_scale))) | (
        dist_from_hot <= max(4.0, 0.5 * module_scale)
    )
    region &= np.isfinite(temps)
    if not region.any():
        return None

    ys, xs = np.nonzero(region)
    region_temps = temps[ys, xs]
    distance = dist_from_hot[ys, xs]
    level = _shorth_midpoint(region_temps)

    # Candidates: pixels at the surface level, clear of the hot point when possible
    deviation = np.abs(region_temps - level)
    at_level = deviation <= max(_REF_LEVEL_TOL, float(deviation.min())) + 1e-6
    clear_of_hot = at_level & (distance >= _REF_MIN_HOT_GAP)
    candidates = np.nonzero(clear_of_hot if clear_of_hot.any() else at_level)[0]

    # Keep the flat 5x5 patches (window minimum close to the level)
    kernel = np.ones((5, 5), np.uint8)
    with np.errstate(invalid="ignore"):  # inf - inf where a whole window is invalid, never at a candidate
        spread = cv2.dilate(temps, kernel) - cv2.erode(temps, kernel)
    spread = spread[ys[candidates], xs[candidates]]
    candidates = candidates[spread <= float(spread.min()) + _REF_FLATNESS_SLACK]
    candidates = candidates[np.argsort(distance[candidates], kind="stable")]

    # Order: clear of the hot marker, nearest first; then the closer ones, farthest first
    clear = distance[candidates] >= _REF_CLEAR_HOT_GAP
    candidates = np.concatenate((candidates[clear][:_REF_PATH_CANDIDATES], candidates[~clear][::-1]))

    # Prefer a marker on the same module: no cold gap on the straight line from the hot point
    best = candidates[0]
    for candidate in candidates:
        steps = int(max(abs(xs[candidate] - hot_x), abs(ys[candidate] - hot_y))) + 1
        line_x = np.rint(np.linspace(hot_x, xs[candidate], steps)).astype(int)
        line_y = np.rint(np.linspace(hot_y, ys[candidate], steps)).astype(int)
        if level - float(temps[line_y, line_x].min()) <= _REF_MAX_DIP:
            best = candidate
            break

    ref_x, ref_y = int(xs[best]), int(ys[best])
    window = temps[max(0, ref_y - 2):min(height, ref_y + 3), max(0, ref_x - 2):min(width, ref_x + 3)]
    return ref_x, ref_y, float(np.min(window))


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
        hot_search_radius: int = 60,
        image_format=None,
        visual_size: Optional[Tuple[int, int]] = None,
        flight_direction: Optional[str] = None,
        excluded_hot_points: Optional[List[Tuple[int, int]]] = None,
        min_hot_distance: int = 10,
        hot_within_panel: bool = True,
    ) -> Optional[HotColdPoints]:
        """
        Find the hot point of a defect and its reference point on the module surface.

        The algorithm:
        1. Search for HOTTEST pixel in a small radius around initial point, constrained to panel
        2. Prefer hot points closer to panel center over edge hot points
        3. Find the REFERENCE point: healthy module surface next to the hot point
           (see find_module_reference_point), reported as the cold point

        Args:
            image_path: Path to DJI thermal R-JPEG image
            initial_x: Initial X coordinate in VISUAL image (1280x1024), already rotated if needed
            initial_y: Initial Y coordinate in VISUAL image (1280x1024), already rotated if needed
            panel_bbox_visual: Optional (left, top, right, bottom) in visual coords (1280x1024).
                              If provided, search is constrained to within this panel.
                              Already rotated if flight_direction is 'south'.
            hot_search_radius: Radius around initial point to search for hottest pixel (visual coords)
            flight_direction: 'south' if drone was flying south (thermal array needs 180° rotation)
            excluded_hot_points: Optional list of visual coords to avoid for hot selection.
                                Already rotated if flight_direction is 'south'.
            min_hot_distance: Minimum separation from excluded hot points (visual pixels)
            hot_within_panel: Keep the hot search inside the projected panel box. Pass False
                              when the initial point itself is precise (source-map): the hot
                              point is then the hottest pixel within hot_search_radius of it.

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

                logger.debug(f"Panel bbox thermal: ({left_t}, {top_t}) to ({right_t}, {bottom_t})")
            else:
                # No panel bbox - fallback to old behavior (search around initial point)
                logger.warning("No panel bbox provided, using fallback circular search")
                y_coords, x_coords = np.ogrid[:height, :width]
                distances = np.sqrt((x_coords - thermal_init_x) ** 2 + (y_coords - thermal_init_y) ** 2)
                panel_mask = distances <= 50  # 50 thermal pixels radius
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

            # Hot search stays a few pixels inside the projected module: the box
            # is only as good as the projection, and the soil just outside it
            # often runs hotter than any cell (seen ~9 % past the module edge).
            hot_panel_mask = panel_mask
            if panel_bbox_visual is not None:
                inset_x = max(1, int(round((right_t - left_t) * 0.10)))
                inset_y = max(1, int(round((bottom_t - top_t) * 0.10)))
                inset_mask = np.zeros_like(panel_mask)
                inset_mask[top_t + inset_y:bottom_t - inset_y, left_t + inset_x:right_t - inset_x] = True
                if inset_mask.any():
                    hot_panel_mask = inset_mask

            hot_allowed_mask = hot_panel_mask
            if hot_exclusion_mask is not None:
                hot_allowed_mask = hot_panel_mask & ~hot_exclusion_mask
                if not hot_allowed_mask.any():
                    logger.warning(
                        "Hot exclusion removed all candidates; ignoring exclusions for this defect"
                    )
                    hot_allowed_mask = hot_panel_mask

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

            if hot_within_panel:
                # Inner search: small radius around defect center, within panel
                inner_search_mask = (dist_from_defect <= hot_radius_t) & hot_allowed_mask
            else:
                # Precise defect point: the labelled hot cell lies within a few pixels of it,
                # often on the module edge the inset drops, and the projected panel box can
                # sit several pixels off the module, so search around the point alone.
                inner_search_mask = dist_from_defect <= hot_radius_t
                if hot_exclusion_mask is not None:
                    inner_search_mask &= ~hot_exclusion_mask

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

            # --- Find REFERENCE point: module surface next to the hot point ---
            reference = find_module_reference_point(
                temp_array, panel_mask, hot_thermal_x, hot_thermal_y
            )
            if reference is None:
                logger.warning("No valid temperatures around the hot point, cannot find reference point")
                return None
            cold_thermal_x, cold_thermal_y, cold_temp = reference

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
        center_on_defect: bool = False,
        hot_within_panel: bool = True,
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create an annotated thermal image centered on the defect with hot/cold markers.

        Uses hot/cold point detection to find:
        - RED marker: Hottest point WITHIN the panel
        - BLUE marker: Reference point on the module surface next to the hot point

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
            hot_within_panel: Keep the hot search inside the panel box (see find_hot_cold_points).

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
                hot_search_radius=hot_search_radius,  # Use caller-specified radius
                image_format=image_format,
                visual_size=visual_size,
                flight_direction=flight_direction,  # For thermal array rotation
                excluded_hot_points=excluded_hot_points_rotated,
                min_hot_distance=min_hot_distance,
                hot_within_panel=hot_within_panel,
            )

            # REQUIRE hot_cold detection - no fallback, skip thermal analysis if it fails
            if not hot_cold:
                logger.warning(
                    f"Hot/cold detection failed for {panel_id}, skipping thermal analysis"
                )
                return None

            # Normally we centre on the detected hot point. When the caller cannot
            # trust the projected pixel (see gps_matcher._untrusted_geometry), the
            # hottest pixel nearby is as likely to be sun-baked soil as the module,
            # so centre on the supplied point and let the wider crop carry the defect.
            if center_on_defect:
                center_pixel_x = int(defect_pixel_x)
                center_pixel_y = int(defect_pixel_y)
            else:
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

            # When the caller could not trust the projected pixel, hot/cold are just
            # the local extremes near the frame centre -- on a site where bare soil
            # runs hotter than the modules that is usually dirt. A confident
            # crosshair on dirt is worse than none; the wide crop is the deliverable.
            show_markers = not center_on_defect

            if show_markers:
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
            if (
                show_markers
                and 10 <= cold_canvas_x <= 10 + img_area_width
                and 10 <= cold_canvas_y <= 10 + img_area_height
            ):
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
