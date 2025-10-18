"""Crop defect regions from orthophoto for detailed views."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from ..models.defect import Panel
from ..io.image_loader import load_orthophoto, save_image
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


def crop_defect_regions(
    ortho_path: Path,
    panel_grid: Dict[Tuple[int, int], Panel],
    output_dir: Path,
    scale_factor: float = 0.5,
) -> None:
    """
    Crop regions around each defect for detailed views.

    For each panel with defects, creates a cropped image showing the panel
    and surrounding context.

    Args:
        ortho_path: Path to orthophoto
        panel_grid: Dictionary of panels with defects
        output_dir: Directory to save cropped images
        scale_factor: Downscale factor for crops
    """
    logger.info(f"Cropping defect regions with scale {scale_factor}")

    # Load orthophoto
    ortho_img, _, (img_h, img_w) = load_orthophoto(ortho_path)

    crop_count = 0
    for panel in panel_grid.values():
        if not panel.has_defects:
            continue

        # Group defects by type
        for defect_type in ["hotspots", "faultydiodes", "offlinepanels"]:
            defects = getattr(panel, defect_type, [])
            if not defects:
                continue

            # Create one crop per defect type per panel
            panel_bbox = panel.bbox
            crop_size_px = settings.default_panel_width_px * settings.crop_panel_size

            # Calculate crop bounds (centered on panel)
            center_x, center_y = panel_bbox.center
            half_size = crop_size_px // 2

            x1 = max(0, int(center_x - half_size))
            y1 = max(0, int(center_y - half_size))
            x2 = min(img_w, int(center_x + half_size))
            y2 = min(img_h, int(center_y + half_size))

            # Crop region
            cropped = ortho_img[y1:y2, x1:x2]

            # Downscale
            cropped = cv2.resize(
                cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
            )

            # Save
            filename = f"{defect_type}_({panel.panel_id})_cropped.jpg"
            output_path = output_dir / filename
            save_image(cropped, output_path, quality=settings.jpeg_quality)
            crop_count += 1

    logger.info(f"Created {crop_count} defect crop images")
