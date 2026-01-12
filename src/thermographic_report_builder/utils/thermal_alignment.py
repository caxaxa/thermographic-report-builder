"""Thermal-Visual coordinate alignment utilities.

The DJI M30T R-JPEG contains both visual (1280x1024) and thermal (640x512) data.
While DJI's onboard processing pre-aligns these, there can be residual offset
due to manufacturing tolerances or temperature drift. These utilities provide
a centralized way to convert between visual and thermal coordinate spaces,
applying any configured alignment corrections.

For non-M30T cameras that produce 640x512 thermal-only images, no coordinate
transformation is needed since the image is already in thermal space.
"""

from enum import Enum
from typing import Tuple, Optional

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImageFormat(Enum):
    """Image format types for coordinate transformation."""
    VISUAL_THERMAL = "visual_thermal"  # M30T: 1280x1024 visual with 640x512 thermal
    THERMAL_ONLY = "thermal_only"      # Non-M30T: 640x512 thermal only (no transform needed)


# Standard DJI M30T resolutions
VISUAL_WIDTH = 1280
VISUAL_HEIGHT = 1024
THERMAL_WIDTH = 640
THERMAL_HEIGHT = 512

# Scale factors (thermal = visual * scale)
SCALE_X = THERMAL_WIDTH / VISUAL_WIDTH  # 0.5
SCALE_Y = THERMAL_HEIGHT / VISUAL_HEIGHT  # 0.5


def _resolve_sizes(
    image_format: ImageFormat,
    visual_size: Optional[Tuple[int, int]],
    thermal_size: Optional[Tuple[int, int]],
) -> Tuple[int, int, int, int]:
    """Resolve visual/thermal dimensions with sensible defaults."""
    if image_format == ImageFormat.THERMAL_ONLY:
        if thermal_size:
            t_w, t_h = thermal_size
        elif visual_size:
            t_w, t_h = visual_size
        else:
            t_w, t_h = THERMAL_WIDTH, THERMAL_HEIGHT
        v_w, v_h = visual_size or (t_w, t_h)
        return v_w, v_h, t_w, t_h

    v_w, v_h = visual_size or (VISUAL_WIDTH, VISUAL_HEIGHT)
    t_w, t_h = thermal_size or (THERMAL_WIDTH, THERMAL_HEIGHT)
    return v_w, v_h, t_w, t_h


def visual_to_thermal(
    visual_x: float,
    visual_y: float,
    apply_offset: bool = True,
    image_format: ImageFormat = ImageFormat.VISUAL_THERMAL,
    visual_size: Optional[Tuple[int, int]] = None,
    thermal_size: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """
    Convert visual image coordinates to thermal image coordinates.

    Applies the configured alignment offset before scaling.

    Args:
        visual_x: X coordinate in visual image (0-1279 for M30T, 0-639 for thermal-only)
        visual_y: Y coordinate in visual image (0-1023 for M30T, 0-511 for thermal-only)
        apply_offset: Whether to apply the alignment offset (default True)
        image_format: The image format type (VISUAL_THERMAL for M30T, THERMAL_ONLY for non-M30T)

    Returns:
        Tuple of (thermal_x, thermal_y) in thermal image space (0-639, 0-511)
    """
    # For THERMAL_ONLY images, coordinates are already in thermal space - no transformation needed
    if image_format == ImageFormat.THERMAL_ONLY:
        return visual_x, visual_y

    if apply_offset:
        # Apply offset in visual space first
        # Positive offset = shift the thermal lookup position
        adjusted_x = visual_x + settings.thermal_visual_offset_x
        adjusted_y = visual_y + settings.thermal_visual_offset_y
    else:
        adjusted_x = visual_x
        adjusted_y = visual_y

    v_w, v_h, t_w, t_h = _resolve_sizes(image_format, visual_size, thermal_size)
    scale_x = t_w / v_w
    scale_y = t_h / v_h

    # Scale to thermal resolution
    thermal_x = adjusted_x * scale_x
    thermal_y = adjusted_y * scale_y

    return thermal_x, thermal_y


def thermal_to_visual(
    thermal_x: float,
    thermal_y: float,
    apply_offset: bool = True,
    image_format: ImageFormat = ImageFormat.VISUAL_THERMAL,
    visual_size: Optional[Tuple[int, int]] = None,
    thermal_size: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """
    Convert thermal image coordinates back to visual image coordinates.

    Args:
        thermal_x: X coordinate in thermal image (0-639)
        thermal_y: Y coordinate in thermal image (0-511)
        apply_offset: Whether to reverse the alignment offset (default True)
        image_format: The image format type (VISUAL_THERMAL for M30T, THERMAL_ONLY for non-M30T)

    Returns:
        Tuple of (visual_x, visual_y) in visual image space (0-1279 for M30T, 0-639 for thermal-only)
    """
    # For THERMAL_ONLY images, coordinates are already in thermal space - no transformation needed
    if image_format == ImageFormat.THERMAL_ONLY:
        return thermal_x, thermal_y

    v_w, v_h, t_w, t_h = _resolve_sizes(image_format, visual_size, thermal_size)
    scale_x = t_w / v_w
    scale_y = t_h / v_h

    # Scale to visual resolution
    visual_x = thermal_x / scale_x
    visual_y = thermal_y / scale_y

    if apply_offset:
        # Reverse the offset
        visual_x -= settings.thermal_visual_offset_x
        visual_y -= settings.thermal_visual_offset_y

    return visual_x, visual_y


def clamp_thermal_coords(
    thermal_x: float,
    thermal_y: float,
    thermal_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Clamp thermal coordinates to valid array indices.

    Args:
        thermal_x: X coordinate (can be float)
        thermal_y: Y coordinate (can be float)

    Returns:
        Tuple of (x, y) clamped to valid thermal array bounds
    """
    t_w, t_h = thermal_size or (THERMAL_WIDTH, THERMAL_HEIGHT)
    x = int(max(0, min(t_w - 1, thermal_x)))
    y = int(max(0, min(t_h - 1, thermal_y)))
    return x, y


def visual_bbox_to_thermal(
    left: float,
    top: float,
    right: float,
    bottom: float,
    apply_offset: bool = True,
    image_format: ImageFormat = ImageFormat.VISUAL_THERMAL,
    visual_size: Optional[Tuple[int, int]] = None,
    thermal_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    Convert a bounding box from visual to thermal coordinates.

    Args:
        left, top, right, bottom: Bounding box in visual coordinates
        apply_offset: Whether to apply alignment offset
        image_format: The image format type (VISUAL_THERMAL for M30T, THERMAL_ONLY for non-M30T)

    Returns:
        Tuple of (left, top, right, bottom) in thermal coordinates, clamped
    """
    tl_x, tl_y = visual_to_thermal(
        left, top, apply_offset, image_format, visual_size, thermal_size
    )
    br_x, br_y = visual_to_thermal(
        right, bottom, apply_offset, image_format, visual_size, thermal_size
    )

    # Clamp to valid thermal bounds
    t_w, t_h = thermal_size or (THERMAL_WIDTH, THERMAL_HEIGHT)
    left_t = int(max(0, min(t_w - 1, tl_x)))
    top_t = int(max(0, min(t_h - 1, tl_y)))
    right_t = int(max(0, min(t_w, br_x)))
    bottom_t = int(max(0, min(t_h, br_y)))

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
