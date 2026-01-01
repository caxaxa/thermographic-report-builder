"""Thermal-Visual coordinate alignment utilities.

The DJI M30T R-JPEG contains both visual (1280x1024) and thermal (640x512) data.
While DJI's onboard processing pre-aligns these, there can be residual offset
due to manufacturing tolerances or temperature drift. These utilities provide
a centralized way to convert between visual and thermal coordinate spaces,
applying any configured alignment corrections.
"""

from typing import Tuple

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Standard DJI M30T resolutions
VISUAL_WIDTH = 1280
VISUAL_HEIGHT = 1024
THERMAL_WIDTH = 640
THERMAL_HEIGHT = 512

# Scale factors (thermal = visual * scale)
SCALE_X = THERMAL_WIDTH / VISUAL_WIDTH  # 0.5
SCALE_Y = THERMAL_HEIGHT / VISUAL_HEIGHT  # 0.5


def visual_to_thermal(
    visual_x: float,
    visual_y: float,
    apply_offset: bool = True,
) -> Tuple[float, float]:
    """
    Convert visual image coordinates to thermal image coordinates.

    Applies the configured alignment offset before scaling.

    Args:
        visual_x: X coordinate in visual image (0-1279)
        visual_y: Y coordinate in visual image (0-1023)
        apply_offset: Whether to apply the alignment offset (default True)

    Returns:
        Tuple of (thermal_x, thermal_y) in thermal image space (0-639, 0-511)
    """
    if apply_offset:
        # Apply offset in visual space first
        # Positive offset = shift the thermal lookup position
        adjusted_x = visual_x + settings.thermal_visual_offset_x
        adjusted_y = visual_y + settings.thermal_visual_offset_y
    else:
        adjusted_x = visual_x
        adjusted_y = visual_y

    # Scale to thermal resolution
    thermal_x = adjusted_x * SCALE_X
    thermal_y = adjusted_y * SCALE_Y

    return thermal_x, thermal_y


def thermal_to_visual(
    thermal_x: float,
    thermal_y: float,
    apply_offset: bool = True,
) -> Tuple[float, float]:
    """
    Convert thermal image coordinates back to visual image coordinates.

    Args:
        thermal_x: X coordinate in thermal image (0-639)
        thermal_y: Y coordinate in thermal image (0-511)
        apply_offset: Whether to reverse the alignment offset (default True)

    Returns:
        Tuple of (visual_x, visual_y) in visual image space (0-1279, 0-1023)
    """
    # Scale to visual resolution
    visual_x = thermal_x / SCALE_X
    visual_y = thermal_y / SCALE_Y

    if apply_offset:
        # Reverse the offset
        visual_x -= settings.thermal_visual_offset_x
        visual_y -= settings.thermal_visual_offset_y

    return visual_x, visual_y


def clamp_thermal_coords(
    thermal_x: float,
    thermal_y: float,
) -> Tuple[int, int]:
    """
    Clamp thermal coordinates to valid array indices.

    Args:
        thermal_x: X coordinate (can be float)
        thermal_y: Y coordinate (can be float)

    Returns:
        Tuple of (x, y) clamped to valid thermal array bounds
    """
    x = int(max(0, min(THERMAL_WIDTH - 1, thermal_x)))
    y = int(max(0, min(THERMAL_HEIGHT - 1, thermal_y)))
    return x, y


def visual_bbox_to_thermal(
    left: float,
    top: float,
    right: float,
    bottom: float,
    apply_offset: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Convert a bounding box from visual to thermal coordinates.

    Args:
        left, top, right, bottom: Bounding box in visual coordinates
        apply_offset: Whether to apply alignment offset

    Returns:
        Tuple of (left, top, right, bottom) in thermal coordinates, clamped
    """
    tl_x, tl_y = visual_to_thermal(left, top, apply_offset)
    br_x, br_y = visual_to_thermal(right, bottom, apply_offset)

    # Clamp to valid thermal bounds
    left_t = int(max(0, min(THERMAL_WIDTH - 1, tl_x)))
    top_t = int(max(0, min(THERMAL_HEIGHT - 1, tl_y)))
    right_t = int(max(0, min(THERMAL_WIDTH, br_x)))
    bottom_t = int(max(0, min(THERMAL_HEIGHT, br_y)))

    return left_t, top_t, right_t, bottom_t


def log_alignment_config():
    """Log the current thermal-visual alignment configuration."""
    offset_x = settings.thermal_visual_offset_x
    offset_y = settings.thermal_visual_offset_y

    if offset_x != 0.0 or offset_y != 0.0:
        logger.info(
            f"Thermal-visual alignment offset: ({offset_x:.1f}, {offset_y:.1f}) visual pixels"
        )
    else:
        logger.debug("Thermal-visual alignment: no offset configured (using default 0.5x scale)")
