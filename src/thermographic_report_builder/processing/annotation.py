"""Annotate orthophotos with defect bounding boxes."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from ..models.defect import Panel
from ..config import DEFECT_COLORS
from ..io.image_loader import load_orthophoto, save_image
from ..utils.logger import get_logger

logger = get_logger(__name__)


def annotate_orthophoto(
    ortho_path: Path,
    panel_grid: Dict[Tuple[int, int], Panel],
    output_path: Path,
    scale_factor: float = 0.25,
) -> None:
    """
    Annotate orthophoto with defect bounding boxes and downsample.

    Args:
        ortho_path: Path to original orthophoto
        panel_grid: Dictionary of panels with defects
        output_path: Path to save annotated image
        scale_factor: Downscale factor (e.g., 0.25 = 25% of original size)
    """
    logger.info(f"Annotating orthophoto with scale factor {scale_factor}")

    # Load orthophoto
    ortho_img, _, _, _ = load_orthophoto(ortho_path)

    # Draw all defects
    for panel in panel_grid.values():
        for defect in panel.all_defects():
            bbox = defect.bbox
            color = DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))

            # Draw rectangle (convert floats to ints for OpenCV)
            cv2.rectangle(
                ortho_img,
                (int(bbox.left), int(bbox.top)),
                (int(bbox.right), int(bbox.bottom)),
                color,
                thickness=3,
            )

            # Add label (convert floats to ints for OpenCV)
            label = defect.defect_type
            text_pos = (int(bbox.left), int(bbox.top) - 5)
            cv2.putText(
                ortho_img,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )

    # Downscale
    downscaled = cv2.resize(
        ortho_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )

    # Save
    save_image(downscaled, output_path, quality=90)
    logger.info(f"Saved annotated orthophoto to {output_path}")


def create_layer_image(
    panel_grid: Dict[Tuple[int, int], Panel],
    img_width: int,
    img_height: int,
    output_path: Path,
) -> None:
    """
    Create vectorized layer image showing panel grid and defects.

    This generates a PDF with panel outlines and defect markers.

    Args:
        panel_grid: Dictionary of panels
        img_width: Original image width
        img_height: Original image height
        output_path: Path to save PDF
    """
    from reportlab.graphics.shapes import Drawing, Rect, Polygon, String
    from reportlab.lib import colors
    from reportlab.graphics import renderPDF

    logger.info(f"Creating layer image: {img_width}x{img_height} px")

    # Create drawing (convert coordinates: OpenCV top-left to ReportLab bottom-left)
    drawing = Drawing(img_width, img_height)

    # Draw panels and defects
    for panel in panel_grid.values():
        bbox = panel.bbox

        # Convert y-coordinate (flip vertical axis)
        y_bottom = img_height - bbox.bottom

        # Draw panel outline (gray)
        if not panel.has_defects:
            panel_rect = Rect(bbox.left, y_bottom, bbox.width, bbox.height)
            panel_rect.fillColor = None
            panel_rect.strokeColor = colors.gray
            panel_rect.strokeWidth = 1
            drawing.add(panel_rect)
        else:
            # Panels with defects: red fill at 50% opacity
            panel_rect = Rect(bbox.left, y_bottom, bbox.width, bbox.height)
            panel_rect.fillColor = colors.Color(1, 0, 0, alpha=0.5)
            panel_rect.strokeColor = colors.red
            panel_rect.strokeWidth = 2
            drawing.add(panel_rect)

            # Add panel number label
            center_x, center_y = bbox.center
            center_y_flipped = img_height - center_y
            label = String(center_x, center_y_flipped, panel.panel_id)
            label.fontSize = 12
            label.fillColor = colors.black
            drawing.add(label)

    # Render to PDF
    renderPDF.drawToFile(drawing, str(output_path))
    logger.info(f"Saved layer image to {output_path}")
