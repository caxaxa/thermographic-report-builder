"""Create annotated thermal images with temperature overlay for reports.

This module generates professional annotated thermal images that show:
- Zoomed view of the defect area in the raw thermal image
- Temperature colorbar overlay
- Panel outline via reprojection
- Temperature readings for defect and panel average
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..utils.logger import get_logger
from .thermal_extractor import ThermalExtractor, TemperatureReading

logger = get_logger(__name__)


@dataclass
class AnnotatedThermalImage:
    """Result of creating an annotated thermal image."""

    image: np.ndarray  # The annotated BGR image
    defect_temp: float  # Defect temperature in Celsius
    panel_avg_temp: float  # Panel average temperature in Celsius
    delta_t: float  # Temperature difference
    severity: str  # Severity classification


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
        zoom_size: int = 200,
        output_size: Tuple[int, int] = (600, 500),
    ) -> Optional[AnnotatedThermalImage]:
        """
        Create an annotated thermal image centered on the defect.

        Args:
            raw_image_path: Path to the raw thermal image (DJI R-JPEG)
            defect_pixel_x: X pixel coordinate of defect in raw image
            defect_pixel_y: Y pixel coordinate of defect in raw image
            panel_id: Panel identifier (e.g., "A-01")
            panel_row: Panel row number
            panel_column: Panel column number
            latitude: Latitude of defect location
            longitude: Longitude of defect location
            zoom_size: Size of zoom window around defect (pixels)
            output_size: Final output image size (width, height)

        Returns:
            AnnotatedThermalImage or None if extraction fails
        """
        if not self.thermal_extractor or not self.thermal_extractor.available:
            logger.warning("Thermal extractor not available, cannot create annotated image")
            return None

        try:
            # Extract temperature data
            temp_reading = self.thermal_extractor.extract_temperature_at_pixel(
                image_path=raw_image_path,
                pixel_x=int(defect_pixel_x),
                pixel_y=int(defect_pixel_y),
            )

            if not temp_reading:
                logger.warning(f"Could not extract temperature from {raw_image_path}")
                return None

            # Load the visual image
            img = cv2.imread(str(raw_image_path))
            if img is None:
                logger.warning(f"Could not load image: {raw_image_path}")
                return None

            img_h, img_w = img.shape[:2]

            # Calculate zoom window bounds
            half_size = zoom_size // 2
            x1 = max(0, int(defect_pixel_x) - half_size)
            y1 = max(0, int(defect_pixel_y) - half_size)
            x2 = min(img_w, int(defect_pixel_x) + half_size)
            y2 = min(img_h, int(defect_pixel_y) + half_size)

            # Crop the zoomed region
            zoomed = img[y1:y2, x1:x2].copy()

            # Resize to target size (leaving room for annotation)
            img_area_height = output_size[1] - 120  # Reserve space for text
            img_area_width = output_size[0] - 80  # Reserve space for colorbar

            zoomed_resized = cv2.resize(
                zoomed,
                (img_area_width, img_area_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Create the output canvas
            canvas = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255

            # Place the zoomed image
            canvas[10 : 10 + img_area_height, 10 : 10 + img_area_width] = zoomed_resized

            # Draw defect marker (crosshair) at center of zoomed image
            center_x = 10 + img_area_width // 2
            center_y = 10 + img_area_height // 2
            marker_size = 20

            # Red crosshair with white outline
            cv2.drawMarker(
                canvas,
                (center_x, center_y),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                marker_size + 4,
                4,
            )
            cv2.drawMarker(
                canvas,
                (center_x, center_y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                marker_size,
                2,
            )

            # Add colorbar on the right
            colorbar_x = img_area_width + 30
            colorbar_width = 30
            colorbar_height = img_area_height

            # Create temperature colorbar (whitehot grayscale - matches DJI thermal display)
            # Hot = white (255), Cold = black (0)
            colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
            for i in range(colorbar_height):
                # Interpolate from black (bottom/cold) to white (top/hot)
                ratio = 1.0 - (i / colorbar_height)  # ratio=1 at top (hot), ratio=0 at bottom (cold)
                gray_value = int(255 * ratio)
                colorbar[i, :] = (gray_value, gray_value, gray_value)

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

            # Max temp at top
            cv2.putText(
                canvas,
                f"{temp_reading.max_temp_celsius:.0f}C",
                (colorbar_x + colorbar_width + 5, 20),
                font,
                font_scale,
                font_color,
                1,
            )
            # Min temp at bottom
            cv2.putText(
                canvas,
                f"{temp_reading.min_temp_celsius:.0f}C",
                (colorbar_x + colorbar_width + 5, 10 + colorbar_height - 5),
                font,
                font_scale,
                font_color,
                1,
            )

            # Add text annotations at bottom
            text_y_start = output_size[1] - 100
            line_height = 22
            font_scale_text = 0.55
            font_thickness = 1

            # Title line
            title = f"Painel {panel_id} (Linha {panel_row}, Coluna {panel_column})"
            cv2.putText(
                canvas,
                title,
                (15, text_y_start),
                font,
                font_scale_text + 0.1,
                (0, 0, 0),
                font_thickness + 1,
            )

            # Location line
            location = f"Lat: {latitude:.6f}, Lon: {longitude:.6f}"
            cv2.putText(
                canvas,
                location,
                (15, text_y_start + line_height),
                font,
                font_scale_text,
                (80, 80, 80),
                font_thickness,
            )

            # Temperature readings
            temp_defect = f"Temp. Defeito: {temp_reading.defect_temp_celsius:.1f}C"
            cv2.putText(
                canvas,
                temp_defect,
                (15, text_y_start + line_height * 2),
                font,
                font_scale_text,
                (0, 0, 200),  # Red for defect
                font_thickness,
            )

            temp_panel = f"Temp. Media Painel: {temp_reading.panel_avg_celsius:.1f}C"
            cv2.putText(
                canvas,
                temp_panel,
                (15, text_y_start + line_height * 3),
                font,
                font_scale_text,
                (0, 128, 0),  # Green for panel average
                font_thickness,
            )

            # Delta T with severity color
            severity_colors = {
                "CRITICAL": (0, 0, 255),  # Red
                "HIGH": (0, 128, 255),  # Orange
                "MEDIUM": (0, 200, 255),  # Yellow-orange
                "LOW": (0, 200, 200),  # Yellow
                "MINIMAL": (0, 200, 0),  # Green
            }
            severity_color = severity_colors.get(temp_reading.severity, (0, 0, 0))

            delta_text = f"Delta T: {temp_reading.delta_t:.1f}C ({temp_reading.severity})"
            cv2.putText(
                canvas,
                delta_text,
                (300, text_y_start + line_height * 2),
                font,
                font_scale_text,
                severity_color,
                font_thickness + 1,
            )

            logger.info(
                f"Created annotated thermal image for {panel_id}: "
                f"defect={temp_reading.defect_temp_celsius:.1f}C, "
                f"panel_avg={temp_reading.panel_avg_celsius:.1f}C, "
                f"delta_t={temp_reading.delta_t:.1f}C"
            )

            return AnnotatedThermalImage(
                image=canvas,
                defect_temp=temp_reading.defect_temp_celsius,
                panel_avg_temp=temp_reading.panel_avg_celsius,
                delta_t=temp_reading.delta_t,
                severity=temp_reading.severity,
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
