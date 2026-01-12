"""Transform coordinates between cropped/rotated and uncropped orthophoto spaces.

The inference pipeline crops and rotates the orthophoto before running detection:
1. Scales crop polygon from preview dimensions to full orthophoto dimensions
2. Crops to the polygon's bounding box
3. Rotates by the angle from the rotation line
4. Resamples to 1.6cm GSD (if needed)

This module provides the inverse transformation to map defect label coordinates
(in resampled/cropped/rotated space) back to the uncropped orthophoto space
(which matches the mesh coordinate system for backtracking).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CropTransformParams:
    """Parameters for the crop/rotate transformation."""

    # Crop bounding box in full orthophoto pixels
    crop_x_min: int
    crop_y_min: int
    crop_width: int
    crop_height: int

    # Rotation angle in degrees (positive = counterclockwise)
    rotation_angle: float

    # Dimensions after cropping and rotating (at original GSD)
    cropped_rotated_width: int
    cropped_rotated_height: int

    # Scale factor from resampled (1.6cm) to original GSD
    # (resampled_size / cropped_rotated_size)
    resample_scale: float = 1.0


class CropTransform:
    """Transform coordinates between cropped/rotated and uncropped orthophoto spaces."""

    def __init__(
        self,
        crop_annotation: Optional[Dict] = None,
        crop_metadata: Optional[Dict] = None,
        ortho_width: int = 0,
        ortho_height: int = 0,
        resampled_width: int = 0,
        resampled_height: int = 0,
    ):
        """
        Initialize the crop transform.

        Args:
            crop_annotation: The crop_annotation.json data containing polygon and rotationLine
            crop_metadata: The crop_annotation_metadata.json with preview dimensions
            ortho_width: Width of the full (uncropped) orthophoto
            ortho_height: Height of the full (uncropped) orthophoto
            resampled_width: Width of the resampled (1.6cm GSD) cropped/rotated orthophoto
            resampled_height: Height of the resampled (1.6cm GSD) cropped/rotated orthophoto
        """
        self.available = False
        self.params: Optional[CropTransformParams] = None

        if not crop_annotation or not crop_metadata:
            logger.info("No crop annotation provided, transform disabled")
            return

        if ortho_width <= 0 or ortho_height <= 0:
            logger.warning("Invalid orthophoto dimensions, transform disabled")
            return

        try:
            self.params = self._compute_params(
                crop_annotation, crop_metadata, ortho_width, ortho_height,
                resampled_width, resampled_height
            )
            self.available = True
            logger.info(
                f"Crop transform initialized: crop=({self.params.crop_x_min}, {self.params.crop_y_min}) "
                f"size={self.params.crop_width}x{self.params.crop_height}, "
                f"rotation={self.params.rotation_angle:.2f}°, "
                f"cropped_rotated_size={self.params.cropped_rotated_width}x{self.params.cropped_rotated_height}, "
                f"resample_scale={self.params.resample_scale:.3f}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize crop transform: {e}")

    def _compute_params(
        self,
        crop_annotation: Dict,
        crop_metadata: Dict,
        ortho_width: int,
        ortho_height: int,
        resampled_width: int = 0,
        resampled_height: int = 0,
    ) -> CropTransformParams:
        """Compute transformation parameters from crop annotation."""
        # Get preview dimensions (where annotation was made)
        preview_width = crop_metadata.get("width", 0)
        preview_height = crop_metadata.get("height", 0)

        if preview_width <= 0 or preview_height <= 0:
            raise ValueError(f"Invalid preview dimensions: {preview_width}x{preview_height}")

        # Calculate scale factors from preview to full orthophoto
        scale_x = ortho_width / preview_width
        scale_y = ortho_height / preview_height

        # Get polygon points and scale to full orthophoto
        polygon = crop_annotation.get("polygon", [])
        if len(polygon) < 3:
            raise ValueError("Polygon must have at least 3 points")

        scaled_polygon = [
            (int(p["x"] * scale_x), int(p["y"] * scale_y)) for p in polygon
        ]

        # Compute bounding box of polygon
        xs = [p[0] for p in scaled_polygon]
        ys = [p[1] for p in scaled_polygon]
        crop_x_min = min(xs)
        crop_y_min = min(ys)
        crop_x_max = max(xs)
        crop_y_max = max(ys)
        crop_width = crop_x_max - crop_x_min
        crop_height = crop_y_max - crop_y_min

        # Compute rotation angle from rotation line
        rotation_line = crop_annotation.get("rotationLine") or {}
        start = rotation_line.get("start", {}) if rotation_line else {}
        end = rotation_line.get("end", {}) if rotation_line else {}

        if "x" in start and "y" in start and "x" in end and "y" in end:
            dx = end["x"] - start["x"]
            dy = end["y"] - start["y"]
            rotation_angle = math.degrees(math.atan2(dy, dx))
        else:
            rotation_angle = 0.0

        # Compute dimensions after rotation
        # When rotating by angle, the bounding box changes
        angle_rad = math.radians(rotation_angle)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))

        # After rotation, the new bounding box size is:
        rotated_width = int(crop_width * cos_a + crop_height * sin_a)
        rotated_height = int(crop_width * sin_a + crop_height * cos_a)

        # Compute resample scale factor
        # The resampled ortho is at 1.6cm GSD, cropped_rotated is at original GSD
        # We need to scale resampled coords down to original GSD coords
        if resampled_width > 0 and resampled_height > 0 and rotated_width > 0:
            # Use width for scale (should be same as height)
            resample_scale = resampled_width / rotated_width
        else:
            resample_scale = 1.0

        return CropTransformParams(
            crop_x_min=crop_x_min,
            crop_y_min=crop_y_min,
            crop_width=crop_width,
            crop_height=crop_height,
            rotation_angle=rotation_angle,
            cropped_rotated_width=rotated_width,
            cropped_rotated_height=rotated_height,
            resample_scale=resample_scale,
        )

    def cropped_to_uncropped(
        self, cropped_x: float, cropped_y: float
    ) -> Tuple[float, float]:
        """
        Transform coordinates from resampled/cropped/rotated space to uncropped orthophoto space.

        The input coordinates are in the resampled (1.6cm GSD) cropped/rotated space.
        The output coordinates are in the original (uncropped) orthophoto space.

        Args:
            cropped_x: X coordinate in resampled/cropped/rotated space
            cropped_y: Y coordinate in resampled/cropped/rotated space

        Returns:
            (x, y) coordinates in the uncropped orthophoto space
        """
        if not self.available or not self.params:
            # No transform available, return as-is
            return cropped_x, cropped_y

        p = self.params

        # Step 0: Scale from resampled resolution to original resolution
        # The resampled ortho is at higher resolution (1.6cm vs ~16cm)
        if p.resample_scale > 0:
            scaled_x = cropped_x / p.resample_scale
            scaled_y = cropped_y / p.resample_scale
        else:
            scaled_x = cropped_x
            scaled_y = cropped_y

        # Step 1: Un-rotate around the center of the rotated image
        # The rotation was applied around the center of the cropped image
        center_x = p.cropped_rotated_width / 2
        center_y = p.cropped_rotated_height / 2

        # Translate to origin
        tx = scaled_x - center_x
        ty = scaled_y - center_y

        # Rotate by +angle here because PIL rotates in screen coordinates (y down).
        # Using the standard rotation matrix with +angle correctly inverts the PIL rotation.
        angle_rad = math.radians(p.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        # Translate back to the center of the original (pre-rotation) cropped image
        orig_center_x = p.crop_width / 2
        orig_center_y = p.crop_height / 2

        unrotated_x = rx + orig_center_x
        unrotated_y = ry + orig_center_y

        # Step 2: Un-crop (add back the offset)
        uncropped_x = unrotated_x + p.crop_x_min
        uncropped_y = unrotated_y + p.crop_y_min

        return uncropped_x, uncropped_y

    def uncropped_to_cropped(
        self, uncropped_x: float, uncropped_y: float
    ) -> Tuple[float, float]:
        """
        Transform coordinates from uncropped orthophoto space to cropped/rotated space.

        This is the forward transform (used for visualization if needed).

        Args:
            uncropped_x: X coordinate in uncropped orthophoto space
            uncropped_y: Y coordinate in uncropped orthophoto space

        Returns:
            (x, y) coordinates in the cropped/rotated space
        """
        if not self.available or not self.params:
            return uncropped_x, uncropped_y

        p = self.params

        # Step 1: Crop (subtract offset)
        cropped_x = uncropped_x - p.crop_x_min
        cropped_y = uncropped_y - p.crop_y_min

        # Step 2: Rotate around the center of the cropped image
        orig_center_x = p.crop_width / 2
        orig_center_y = p.crop_height / 2

        # Translate to origin
        tx = cropped_x - orig_center_x
        ty = cropped_y - orig_center_y

        # Rotate by angle
        angle_rad = math.radians(p.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        # Translate to center of rotated image
        center_x = p.cropped_rotated_width / 2
        center_y = p.cropped_rotated_height / 2

        rotated_x = rx + center_x
        rotated_y = ry + center_y

        return rotated_x, rotated_y

    def transform_bbox(
        self, left: int, top: int, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """
        Transform a bounding box from cropped/rotated to uncropped space.

        Note: After inverse rotation, the bbox may not be axis-aligned.
        This returns the axis-aligned bounding box that contains the transformed corners.

        Args:
            left, top, width, height: Bounding box in cropped/rotated space

        Returns:
            (left, top, width, height) in uncropped orthophoto space
        """
        if not self.available:
            return left, top, width, height

        # Transform all four corners
        corners = [
            (left, top),
            (left + width, top),
            (left, top + height),
            (left + width, top + height),
        ]

        transformed = [self.cropped_to_uncropped(x, y) for x, y in corners]

        # Compute axis-aligned bounding box
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]

        new_left = int(min(xs))
        new_top = int(min(ys))
        new_right = int(max(xs))
        new_bottom = int(max(ys))

        return new_left, new_top, new_right - new_left, new_bottom - new_top

    def apply_crop_rotate(self, input_path: Path, output_path: Path) -> Path:
        """
        Apply crop and rotation to an orthophoto image.

        This produces the cropped/rotated orthophoto that matches the defect label
        coordinate space. Use this for visualization.

        Args:
            input_path: Path to the uncropped orthophoto (TIFF or image)
            output_path: Path to save the cropped/rotated result

        Returns:
            Path to the output file
        """
        from PIL import Image

        if not self.available or not self.params:
            # No transform, just copy the file
            import shutil
            shutil.copy(input_path, output_path)
            return output_path

        p = self.params

        # Open the image
        with Image.open(input_path) as img:
            # Step 1: Crop to bounding box
            cropped = img.crop((
                p.crop_x_min,
                p.crop_y_min,
                p.crop_x_min + p.crop_width,
                p.crop_y_min + p.crop_height,
            ))

            # Step 2: Rotate (PIL's rotate is counter-clockwise, and we want clockwise rotation)
            # The inference used img.rotate(angle, expand=True), which means:
            # - positive angle = counter-clockwise in PIL
            # - We stored the angle from atan2(dy, dx), which is mathematically standard (CCW positive)
            # - To match inference, we rotate by the stored angle
            if abs(p.rotation_angle) > 0.1:
                rotated = cropped.rotate(p.rotation_angle, expand=True, resample=Image.BILINEAR)
            else:
                rotated = cropped

            # Save result
            rotated.save(output_path)
            logger.info(
                f"Applied crop/rotate: {img.size[0]}x{img.size[1]} -> "
                f"crop({p.crop_width}x{p.crop_height}) -> rotate({p.rotation_angle:.1f}°) -> "
                f"{rotated.size[0]}x{rotated.size[1]}"
            )

        return output_path
