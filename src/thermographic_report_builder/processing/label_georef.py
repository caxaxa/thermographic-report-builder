"""Georeference of the label pixel space, and a north-up view of it for report maps.

Inference crops the orthophoto, rotates the crop by the crop annotation's angle (PIL,
counter-clockwise about its centre, expand=True) and resamples it to 1.6 cm; panel and
defect labels live in that pixel space.

Inference writes that rotation into the TIF's transform, and then the file's own
georeference is the label georeference. TIFs written before it did keep the crop's
pre-rotation, north-up transform, which is wrong for a rotated crop. The rebuild below
puts the rotation back from that transform, the image size and the annotation's angle,
so it stays right even if the original orthophoto is regenerated after inference.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from affine import Affine

from .crop_transform import CropTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Inference leaves crops at or below this angle unrotated (crop_rotate_utils).
MIN_ROTATION_DEG = 0.1

# A transform turned by more than this carries the crop rotation itself. A north-up file
# has no rotation terms at all, and inference never rotates by MIN_ROTATION_DEG or less.
_CARRIED_ROTATION_DEG = 0.01

# Near 45 degrees the crop's size can't be read back from the rotated canvas: a rectangle
# at 45 degrees has the same bounding box whatever its aspect ratio.
_MIN_CONDITION = 0.2

# The rebuilt georeference and the back-mapping route agree within a few cm on every
# consistent project; past this the original orthophoto changed after inference.
STALE_ORTHO_TOLERANCE_M = 0.5


@dataclass(frozen=True)
class LabelGeoreference:
    """Where the label pixels really are."""

    transform: Affine  # label pixel -> orthophoto CRS
    crop_size: Tuple[float, float]  # ground width/height of the crop before rotation
    rotation_deg: float
    ortho_drift_m: Optional[float] = None  # disagreement with the back-mapping route

    @property
    def rotated(self) -> bool:
        return abs(self.rotation_deg) > MIN_ROTATION_DEG


def crop_rotation_deg(crop_annotation: Optional[Dict]) -> float:
    """The angle inference rotated the crop by, from the crop annotation's rotation line."""
    line = (crop_annotation or {}).get("rotationLine") or {}
    start, end = line.get("start") or {}, line.get("end") or {}
    if not all(axis in point for point in (start, end) for axis in ("x", "y")):
        return 0.0
    return math.degrees(math.atan2(end["y"] - start["y"], end["x"] - start["x"]))


def _carried_rotation_deg(transform: Affine) -> float:
    """The crop rotation a transform already carries; 0 for a north-up file."""
    return math.degrees(math.atan2(-transform.d, transform.a))


def _crop_size_before_rotation(
    canvas_w: float, canvas_h: float, rotation_deg: float
) -> Optional[Tuple[float, float]]:
    """Undo PIL's expand: the crop's ground size from its rotated canvas's; None near 45 degrees.

    canvas_w = crop_w * c + crop_h * s,  canvas_h = crop_w * s + crop_h * c
    """
    t = math.radians(rotation_deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    if abs(c - s) < _MIN_CONDITION:
        return None
    total = (canvas_w + canvas_h) / (c + s)
    diff = (canvas_w - canvas_h) / (c - s)
    return (total + diff) / 2, (total - diff) / 2


def _rebuild(
    inference_transform: Affine, width: int, height: int, rotation_deg: float
) -> Optional[Tuple[Affine, Tuple[float, float]]]:
    """Put back the rotation an older inference left out of its transform; None near 45 degrees."""
    # The file's transform is the crop's, scaled to the rotated canvas, so this is the
    # canvas's ground size exactly.
    px_w, px_h = inference_transform.a, -inference_transform.e
    crop_size = _crop_size_before_rotation(width * px_w, height * px_h, rotation_deg)
    if crop_size is None:
        return None
    crop_w, crop_h = crop_size

    # The file's origin is the crop's corner, and PIL keeps the crop's centre fixed.
    t = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    centre_x = inference_transform.c + crop_w / 2
    centre_y = inference_transform.f - crop_h / 2
    a, b = px_w * cos_t, -px_h * sin_t
    d, e = -px_w * sin_t, -px_h * cos_t
    cx, cy = width / 2, height / 2
    transform = Affine(a, b, centre_x - a * cx - b * cy, d, e, centre_y - d * cx - e * cy)
    return transform, (crop_w, crop_h)


def _back_mapping_route(crop_transform: CropTransform, original_transform: Affine) -> Affine:
    """The back-mapping's path (label -> original orthophoto pixel -> CRS) as one affine."""

    def ground(x: float, y: float) -> Tuple[float, float]:
        return original_transform * crop_transform.cropped_to_uncropped(x, y)

    (x0, y0), (x1, y1), (x2, y2) = ground(0, 0), ground(1, 0), ground(0, 1)
    return Affine(x1 - x0, x2 - x0, x0, y1 - y0, y2 - y0, y0)


def _route_crop_size(crop_transform: CropTransform, original_transform: Affine) -> Tuple[float, float]:
    """Ground size of the crop as the back-mapping sees it."""
    p = crop_transform.params
    return p.crop_width * abs(original_transform.a), p.crop_height * abs(original_transform.e)


def build_label_georeference(
    inference_transform: Affine,
    width: int,
    height: int,
    crop_annotation: Optional[Dict],
    crop_transform: Optional[CropTransform] = None,
    original_transform: Optional[Affine] = None,
) -> LabelGeoreference:
    """Georeference of the inference image the labels were drawn on.

    Args:
        inference_transform: Transform of odm_orthophoto_1.6cm.tif as inference wrote it
        width, height: Size of that image (the label pixel space)
        crop_annotation: crop_annotation.json, for the rotation angle when the file's
            transform doesn't carry it
        crop_transform, original_transform: The back-mapping route, when the original
            orthophoto is available; used near 45 degrees and to detect a stale orthophoto
    """
    rotation_deg = crop_rotation_deg(crop_annotation)
    route = None
    if crop_transform is not None and crop_transform.available and original_transform is not None:
        route = _back_mapping_route(crop_transform, original_transform)

    transform = inference_transform
    crop_size = (width * inference_transform.a, height * -inference_transform.e)
    carried_deg = _carried_rotation_deg(inference_transform)
    if abs(carried_deg) > _CARRIED_ROTATION_DEG:
        # The file's transform is already the label georeference. Only the north-up view
        # still needs the crop's size, read back from the rotated canvas.
        rotation_deg = carried_deg
        canvas_w = width * math.hypot(transform.a, transform.d)
        canvas_h = height * math.hypot(transform.b, transform.e)
        crop_size = _crop_size_before_rotation(canvas_w, canvas_h, rotation_deg)
        if crop_size is None and route is not None:
            crop_size = _route_crop_size(crop_transform, original_transform)
        if crop_size is None:
            # Near 45 degrees and no original orthophoto: show the whole rotated canvas.
            t = math.radians(rotation_deg)
            c, s = abs(math.cos(t)), abs(math.sin(t))
            crop_size = (canvas_w * c + canvas_h * s, canvas_w * s + canvas_h * c)
        logger.info(f"Label georeference read from the file ({rotation_deg:.2f} deg crop rotation)")
    elif abs(rotation_deg) > MIN_ROTATION_DEG:
        rebuilt = _rebuild(inference_transform, width, height, rotation_deg)
        if rebuilt is not None:
            transform, crop_size = rebuilt
            logger.info(f"Label georeference rebuilt for a {rotation_deg:.2f} deg crop rotation")
        elif route is not None:
            transform = route
            crop_size = _route_crop_size(crop_transform, original_transform)
            logger.info(
                f"Label georeference from the back-mapping route ({rotation_deg:.2f} deg is "
                "too close to 45 to rebuild)"
            )
        else:
            logger.error(
                f"Crop rotated {rotation_deg:.2f} deg, too close to 45 to rebuild, and no "
                "original orthophoto: report coordinates will be wrong"
            )

    drift = None
    if route is not None:
        probes = [(width * fx, height * fy) for fx in (0.1, 0.5, 0.9) for fy in (0.1, 0.5, 0.9)]
        drift = max(math.dist(transform * p, route * p) for p in probes)
        if drift > STALE_ORTHO_TOLERANCE_M:
            logger.warning(
                f"Original orthophoto disagrees with the inference image by {drift:.1f} m: it "
                "was probably regenerated after inference. Hotspot back-mapping will be off by "
                "about that much; re-run inference (and review the labels) for this project."
            )
    return LabelGeoreference(transform, crop_size, rotation_deg, drift)


class NorthUpView:
    """Draws label-space content north-up, at the label resolution.

    The canvas is the crop as it was before inference rotated it. Identity when the
    labels weren't rotated, so unrotated projects render exactly as before.
    """

    def __init__(self, georef: LabelGeoreference, width: int, height: int):
        self.is_identity = not georef.rotated
        if self.is_identity:
            self.width, self.height = width, height
            self._to_display = Affine.identity()
            return

        t = georef.transform
        px = math.sqrt(abs(t.determinant))
        crop_w, crop_h = georef.crop_size
        self.width, self.height = max(1, round(crop_w / px)), max(1, round(crop_h / px))
        cx, cy = t * (width / 2, height / 2)
        north_up = Affine(px, 0, cx - self.width * px / 2, 0, -px, cy + self.height * px / 2)
        self._to_display = ~north_up * t

    def point(self, x: float, y: float) -> Tuple[float, float]:
        """Label pixel -> north-up pixel."""
        return self._to_display * (x, y)

    def warp(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Turn a label-space image (downscaled by `scale`) north-up, at the same scale."""
        size = (max(1, round(self.width * scale)), max(1, round(self.height * scale)))
        m = Affine.scale(scale) * self._to_display * Affine.scale(1 / scale)
        return cv2.warpAffine(image, _cv2_matrix(m), size, flags=cv2.INTER_LINEAR)

    def window(self, image: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """North-up pixels [x0, x0+w) x [y0, y0+h), cut from a full-resolution label image."""
        m = Affine.translation(-x0, -y0) * self._to_display
        return cv2.warpAffine(image, _cv2_matrix(m), (w, h), flags=cv2.INTER_LINEAR)


def _cv2_matrix(m: Affine) -> np.ndarray:
    """Affine on pixel corners -> OpenCV's 2x3 matrix on pixel centres."""
    shift = Affine.translation(0.5, 0.5)
    c = ~shift * m * shift
    return np.array([[c.a, c.b, c.c], [c.d, c.e, c.f]], dtype=np.float64)
