"""Crop defect regions from orthophoto for detailed views."""

import cv2
import numpy as np
import copy
from pathlib import Path
from typing import Dict, Tuple
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.graphics import renderPDF

from ..models.defect import Panel
from ..io.image_loader import load_orthophoto, save_image
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


def crop_defect_regions(
    ortho_path: Path,
    panel_grid: Dict[Tuple[int, int], Panel],
    output_dir: Path,
    layer_pdf_path: Path = None,
    scale_factor: float = 0.5,
) -> None:
    """
    Crop regions around each defect for detailed views.

    For each panel with defects, creates a cropped image showing the panel
    and surrounding context, with defect annotations drawn on the crop.
    Also creates a mini-map PDF showing where the crop came from.

    Args:
        ortho_path: Path to orthophoto
        panel_grid: Dictionary of panels with defects (all panels, not just with defects)
        output_dir: Directory to save cropped images
        layer_pdf_path: Path to the annotated layer PDF (for creating mini-maps)
        scale_factor: Downscale factor for crops
    """
    logger.info(f"Cropping defect regions with scale {scale_factor}")

    # Load orthophoto
    ortho_img, _, _, (img_h, img_w) = load_orthophoto(ortho_path)

    # Color map for defect annotations (BGR format for OpenCV)
    color_map = {
        "hotspots": (0, 0, 255),        # Red
        "faulty_diodes": (255, 0, 0),   # Blue
        "offline_panels": (0, 255, 255)  # Yellow
    }

    crop_count = 0
    for panel in panel_grid.values():
        if not panel.has_defects:
            continue

        # Group defects by type
        for defect_type in ["hotspots", "faulty_diodes", "offline_panels"]:
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

            # Create annotated copy of orthophoto
            annotated_ortho = ortho_img.copy()

            # Draw defect contours on the annotated image
            for defect in defects:
                bbox = defect.bbox
                # Create rectangle contour from bbox
                contour = np.array([
                    [int(bbox.left), int(bbox.top)],
                    [int(bbox.left + bbox.width), int(bbox.top)],
                    [int(bbox.left + bbox.width), int(bbox.top + bbox.height)],
                    [int(bbox.left), int(bbox.top + bbox.height)]
                ], dtype=np.int32)

                # Draw the defect contour
                cv2.drawContours(annotated_ortho, [contour], -1, color_map[defect_type], thickness=3)

            # Crop region from annotated image
            cropped = annotated_ortho[y1:y2, x1:x2]

            # Skip if crop is empty (can happen at image edges)
            if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                logger.warning(f"Skipping empty crop for panel {panel.panel_id}, bounds: ({x1},{y1})-({x2},{y2})")
                continue

            # Downscale
            cropped = cv2.resize(
                cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
            )

            # Save cropped annotated image
            filename = f"{defect_type}_({panel.panel_id})_cropped.jpg"
            output_path = output_dir / filename
            save_image(cropped, output_path, quality=settings.jpeg_quality)
            crop_count += 1

            # Create mini-map PDF with blue rectangle showing crop location
            _create_mini_map(
                panel_grid,
                img_w, img_h,
                x1, y1, x2, y2,
                panel.panel_id,
                defect_type,
                output_dir
            )

    logger.info(f"Created {crop_count} defect crop images")


def _create_mini_map(
    panel_grid: Dict[Tuple[int, int], Panel],
    img_w: int, img_h: int,
    x1: int, y1: int, x2: int, y2: int,
    panel_id: str,
    defect_type: str,
    output_dir: Path
) -> None:
    """
    Create a mini-map PDF showing where the crop came from.

    Recreates the layer drawing with panels and defects, then overlays
    a blue rectangle indicating the crop region.

    Args:
        panel_grid: Dictionary of all panels (for recreating layer)
        img_w: Image width
        img_h: Image height
        x1, y1, x2, y2: Crop bounds
        panel_id: Panel identifier
        defect_type: Type of defect
        output_dir: Directory to save mini-map
    """
    from reportlab.graphics.shapes import Polygon, String

    # Create a drawing matching the original image dimensions
    drawing = Drawing(img_w, img_h)

    # Add white background
    drawing.add(Rect(0, 0, img_w, img_h, fillColor=colors.white, strokeColor=None))

    # Recreate the layer by drawing panels and defects
    # (Similar to what create_layer_image does)
    for panel in panel_grid.values():
        bbox = panel.bbox
        # Transform from top-left to bottom-left origin
        x_draw = bbox.left
        y_draw = img_h - bbox.top - bbox.height
        w_draw = bbox.width
        h_draw = bbox.height

        # Create polygon points for panel rectangle
        pts = [x_draw, y_draw, x_draw + w_draw, y_draw,
               x_draw + w_draw, y_draw + h_draw, x_draw, y_draw + h_draw]

        # Fill with red if panel has defects, otherwise no fill
        if panel.has_defects:
            fill_color = colors.Color(1, 0, 0, alpha=0.3)
            stroke_color = colors.red
        else:
            fill_color = None
            stroke_color = colors.black

        drawing.add(Polygon(pts, fillColor=fill_color, strokeColor=stroke_color, strokeWidth=1))

        # Add panel label
        label_str = f"{panel.column}-{panel.row}"
        drawing.add(String(x_draw, y_draw + h_draw + 5, label_str, fontSize=20, fillColor=colors.black))

    # Calculate crop rectangle coordinates
    # Note: ReportLab uses bottom-left origin, so transform y coordinates
    rect_x = x1
    rect_y = img_h - y2  # Transform y coordinate
    rect_width = x2 - x1
    rect_height = y2 - y1

    # Add blue rectangle for crop region (thick stroke, no fill)
    drawing.add(
        Rect(rect_x, rect_y, rect_width, rect_height,
             strokeColor=colors.blue, strokeWidth=50, fillColor=None)
    )

    # Save as PDF
    filename = f"{defect_type}_({panel_id})_minimap.pdf"
    output_path = output_dir / filename
    renderPDF.drawToFile(drawing, str(output_path))
    logger.debug(f"Created mini-map: {filename}")
