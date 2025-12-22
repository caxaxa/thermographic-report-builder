"""Extract temperature data from DJI thermal images.

Uses thermal_parser to read radiometric data from DJI R-JPEG images
and calculate temperature differences for defect analysis.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import thermal_parser, but make it optional for backwards compatibility
# Note: thermal_parser may be importable but fail when instantiated if DJI SDK is missing
try:
    from thermal_parser import Thermal
    THERMAL_PARSER_AVAILABLE = True
except ImportError:
    Thermal = None
    THERMAL_PARSER_AVAILABLE = False
    logger.warning("thermal_parser not available - temperature extraction disabled")


@dataclass
class TemperatureReading:
    """Temperature data extracted from a thermal image at a specific location."""

    defect_temp_celsius: float  # Temperature at defect location
    panel_avg_celsius: float  # Average temperature of surrounding panel
    delta_t: float  # Temperature difference (defect - panel average)
    min_temp_celsius: float  # Minimum temperature in panel region
    max_temp_celsius: float  # Maximum temperature in panel region

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


class ThermalExtractor:
    """Extract temperature data from DJI thermal images."""

    def __init__(self):
        """Initialize thermal extractor."""
        if THERMAL_PARSER_AVAILABLE and Thermal is not None:
            try:
                self.thermal = Thermal(dtype=np.float32)
                self.available = True
                logger.info("Thermal extractor initialized")
            except SystemExit:
                # thermal_parser calls sys.exit() if DJI SDK libraries are missing
                logger.warning("thermal_parser SDK libraries not available - temperature extraction disabled")
                self.thermal = None
                self.available = False
            except Exception as e:
                # thermal_parser imported but SDK libraries not available
                logger.warning(f"thermal_parser SDK not available: {e}")
                self.thermal = None
                self.available = False
        else:
            self.thermal = None
            self.available = False

    def extract_temperature_at_pixel(
        self,
        image_path: Path,
        pixel_x: int,
        pixel_y: int,
        panel_region: Optional[Tuple[int, int, int, int]] = None,
        sample_radius: int = 50,
    ) -> Optional[TemperatureReading]:
        """
        Extract temperature at a specific pixel location.

        Args:
            image_path: Path to DJI thermal R-JPEG image
            pixel_x: X coordinate of defect in image
            pixel_y: Y coordinate of defect in image
            panel_region: Optional (x1, y1, x2, y2) bounding box for panel.
                         If None, uses a region around the defect.
            sample_radius: Radius around defect to exclude when calculating
                          panel average (to avoid including the defect itself)

        Returns:
            TemperatureReading or None if extraction fails
        """
        if not self.available:
            logger.debug("Temperature extraction not available")
            return None

        try:
            # Parse thermal image
            temp_array = self.thermal.parse(filepath_image=str(image_path))

            if temp_array is None:
                logger.warning(f"Failed to parse thermal data from {image_path}")
                return None

            height, width = temp_array.shape

            # Clamp pixel coordinates to image bounds
            pixel_x = max(0, min(int(pixel_x), width - 1))
            pixel_y = max(0, min(int(pixel_y), height - 1))

            # Get temperature at defect location
            defect_temp = float(temp_array[pixel_y, pixel_x])

            # Define panel region for average calculation
            if panel_region:
                x1, y1, x2, y2 = panel_region
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
            else:
                # Default: use a region around the defect
                margin = 100
                x1 = max(0, pixel_x - margin)
                y1 = max(0, pixel_y - margin)
                x2 = min(width, pixel_x + margin)
                y2 = min(height, pixel_y + margin)

            # Extract panel region
            panel_temp_region = temp_array[y1:y2, x1:x2].copy()

            # Create a mask to exclude the defect area from panel average
            mask = np.ones_like(panel_temp_region, dtype=bool)
            defect_x_local = pixel_x - x1
            defect_y_local = pixel_y - y1

            # Zero out the defect area in the mask
            y_start = max(0, defect_y_local - sample_radius)
            y_end = min(panel_temp_region.shape[0], defect_y_local + sample_radius)
            x_start = max(0, defect_x_local - sample_radius)
            x_end = min(panel_temp_region.shape[1], defect_x_local + sample_radius)
            mask[y_start:y_end, x_start:x_end] = False

            # Calculate panel average excluding defect
            if mask.sum() > 0:
                panel_avg = float(np.mean(panel_temp_region[mask]))
            else:
                # Fallback if mask is empty
                panel_avg = float(np.mean(panel_temp_region))

            # Calculate ΔT
            delta_t = defect_temp - panel_avg

            reading = TemperatureReading(
                defect_temp_celsius=round(defect_temp, 1),
                panel_avg_celsius=round(panel_avg, 1),
                delta_t=round(delta_t, 1),
                min_temp_celsius=round(float(panel_temp_region.min()), 1),
                max_temp_celsius=round(float(panel_temp_region.max()), 1),
            )

            logger.debug(
                f"Temperature extracted: defect={reading.defect_temp_celsius}°C, "
                f"panel_avg={reading.panel_avg_celsius}°C, ΔT={reading.delta_t}°C"
            )

            return reading

        except Exception as e:
            logger.warning(f"Failed to extract temperature from {image_path}: {e}")
            return None

    def find_local_hotspot(
        self,
        image_path: Path,
        initial_x: int,
        initial_y: int,
        search_radius: int = 100,
    ) -> Tuple[int, int, float]:
        """
        Find the local temperature maximum near a given pixel.

        The projected coordinates from the orthophoto defect bbox center often
        don't land exactly on the hottest pixel. This method searches the area
        around the initial projection to find the actual hotspot.

        Args:
            image_path: Path to DJI thermal R-JPEG image
            initial_x: Initial X coordinate (from projection)
            initial_y: Initial Y coordinate (from projection)
            search_radius: Radius to search around initial point (pixels)

        Returns:
            Tuple of (hotspot_x, hotspot_y, temperature) or (initial_x, initial_y, temp)
            if search fails
        """
        if not self.available:
            return initial_x, initial_y, 0.0

        try:
            temp_array = self.thermal.parse(filepath_image=str(image_path))
            if temp_array is None:
                return initial_x, initial_y, 0.0

            height, width = temp_array.shape

            # Define search region
            x1 = max(0, initial_x - search_radius)
            y1 = max(0, initial_y - search_radius)
            x2 = min(width, initial_x + search_radius)
            y2 = min(height, initial_y + search_radius)

            # Extract search region
            search_region = temp_array[y1:y2, x1:x2]

            # Find maximum temperature location
            local_max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
            hotspot_y = y1 + local_max_idx[0]
            hotspot_x = x1 + local_max_idx[1]
            max_temp = float(search_region[local_max_idx])

            # Get temperature at initial point for comparison
            initial_temp = float(temp_array[
                max(0, min(height - 1, initial_y)),
                max(0, min(width - 1, initial_x))
            ])

            # Only use the hotspot if it's significantly warmer than initial point
            # This avoids jumping to a different hotspot in the same image
            if max_temp > initial_temp + 2.0:  # At least 2°C warmer
                logger.info(
                    f"Hotspot search: adjusted ({initial_x}, {initial_y}) -> ({hotspot_x}, {hotspot_y}), "
                    f"temp {initial_temp:.1f}°C -> {max_temp:.1f}°C (+{max_temp - initial_temp:.1f}°C)"
                )
                return hotspot_x, hotspot_y, max_temp
            else:
                logger.debug(
                    f"Hotspot search: keeping initial ({initial_x}, {initial_y}), "
                    f"max in region only {max_temp - initial_temp:.1f}°C warmer"
                )
                return initial_x, initial_y, initial_temp

        except Exception as e:
            logger.warning(f"Hotspot search failed for {image_path}: {e}")
            return initial_x, initial_y, 0.0

    def get_full_temperature_array(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Get the full temperature array from a thermal image.

        Args:
            image_path: Path to DJI thermal R-JPEG image

        Returns:
            2D numpy array of temperatures in Celsius, or None if fails
        """
        if not self.available:
            return None

        try:
            return self.thermal.parse(filepath_image=str(image_path))
        except Exception as e:
            logger.warning(f"Failed to get temperature array from {image_path}: {e}")
            return None
