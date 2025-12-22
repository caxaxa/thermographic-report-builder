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
    The PDF is scaled to fit within A4 page dimensions while maintaining aspect ratio.

    Args:
        panel_grid: Dictionary of panels
        img_width: Original image width
        img_height: Original image height
        output_path: Path to save PDF
    """
    from reportlab.graphics.shapes import Drawing, Rect, Polygon, String
    from reportlab.lib import colors
    from reportlab.graphics import renderPDF
    import numpy as np

    logger.info(f"Creating layer image: {img_width}x{img_height} px")

    # Log panel bbox stats to verify alignment
    if panel_grid:
        widths = [p.bbox.width for p in panel_grid.values()]
        heights = [p.bbox.height for p in panel_grid.values()]
        lefts = [p.bbox.left for p in panel_grid.values()]
        unique_lefts = len(set(round(l, 1) for l in lefts))
        logger.info(
            "create_layer_image received: %d panels, width=[min=%.1f, median=%.1f, max=%.1f], "
            "height=[min=%.1f, median=%.1f, max=%.1f], unique_lefts=%d",
            len(panel_grid),
            min(widths), float(np.median(widths)), max(widths),
            min(heights), float(np.median(heights)), max(heights),
            unique_lefts,
        )
        # Log first 5 panels' exact coordinates for verification
        sample_panels = list(panel_grid.values())[:5]
        for p in sample_panels:
            logger.info(
                "DRAW COORDS panel %s: left=%.1f, top=%.1f, width=%.1f, height=%.1f",
                p.panel_id, p.bbox.left, p.bbox.top, p.bbox.width, p.bbox.height
            )

    # Scale to fit within A4 page dimensions (in points: 1 inch = 72 points)
    # A4 is 595 x 842 points, leave margins for LaTeX embedding
    max_pdf_width = 500  # points (~7 inches)
    max_pdf_height = 700  # points (~9.7 inches)

    # Calculate scale factor to fit within bounds while maintaining aspect ratio
    scale_x = max_pdf_width / img_width
    scale_y = max_pdf_height / img_height
    scale_factor = min(scale_x, scale_y)

    pdf_width = img_width * scale_factor
    pdf_height = img_height * scale_factor

    logger.info(f"Scaling layer image by {scale_factor:.4f}: {img_width}x{img_height}px -> {pdf_width:.0f}x{pdf_height:.0f}pt")

    # Create drawing with scaled dimensions
    drawing = Drawing(pdf_width, pdf_height)

    # Scale font size based on the scale factor (keep readable but proportional)
    base_font_size = 12
    scaled_font_size = max(6, min(12, base_font_size * scale_factor * 10))  # Keep between 6-12pt

    # Draw panels and defects
    for panel in panel_grid.values():
        bbox = panel.bbox

        # Scale coordinates to PDF dimensions
        scaled_left = bbox.left * scale_factor
        scaled_width = bbox.width * scale_factor
        scaled_height = bbox.height * scale_factor

        # Convert y-coordinate (flip vertical axis) and scale
        y_bottom = (img_height - bbox.bottom) * scale_factor

        # Draw panel outline (gray)
        if not panel.has_defects:
            panel_rect = Rect(scaled_left, y_bottom, scaled_width, scaled_height)
            panel_rect.fillColor = None
            panel_rect.strokeColor = colors.gray
            panel_rect.strokeWidth = max(0.5, 1 * scale_factor)
            drawing.add(panel_rect)
        else:
            # Panels with defects: red fill at 50% opacity
            panel_rect = Rect(scaled_left, y_bottom, scaled_width, scaled_height)
            panel_rect.fillColor = colors.Color(1, 0, 0, alpha=0.5)
            panel_rect.strokeColor = colors.red
            panel_rect.strokeWidth = max(1, 2 * scale_factor)
            drawing.add(panel_rect)

            # Add panel number label
            center_x, center_y = bbox.center
            scaled_center_x = center_x * scale_factor
            scaled_center_y = (img_height - center_y) * scale_factor
            label = String(scaled_center_x, scaled_center_y, panel.panel_id)
            label.fontSize = scaled_font_size
            label.fillColor = colors.black
            drawing.add(label)

    # Render to PDF
    renderPDF.drawToFile(drawing, str(output_path))
    logger.info(f"Saved layer image to {output_path}")
