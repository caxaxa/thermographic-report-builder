"""Extract temperature data from DJI thermal images.

Uses thermal_parser to read radiometric data from DJI R-JPEG images
and calculate temperature differences for defect analysis.

Supports calibrated temperature extraction using DJI DIRP2 API with:
- Emissivity correction for solar panel surfaces
- Object distance (flight altitude) compensation
- Relative humidity correction
- Reflected apparent temperature compensation
"""

import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..utils.logger import get_logger
from ..config import settings

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


@dataclass
class ThermalCalibration:
    """Calibration parameters for thermal image processing."""

    emissivity: float = 0.95  # Solar panel surface emissivity
    object_distance: float = 25.0  # Distance from camera to target (meters)
    relative_humidity: float = 50.0  # Ambient relative humidity (%)
    reflected_temp: float = 25.0  # Reflected apparent temperature (Celsius)

    @classmethod
    def from_settings(cls) -> "ThermalCalibration":
        """Create calibration from application settings."""
        return cls(
            emissivity=settings.thermal_emissivity,
            object_distance=settings.thermal_object_distance,
            relative_humidity=settings.thermal_relative_humidity,
            reflected_temp=settings.thermal_reflected_temp,
        )


def extract_altitude_from_exif(image_path: Path) -> Optional[float]:
    """
    Extract relative altitude from DJI image EXIF/XMP metadata.

    Reads the XMP data embedded in the JPEG to find drone-dji:RelativeAltitude.

    Args:
        image_path: Path to the thermal image

    Returns:
        Altitude in meters, or None if not found
    """
    try:
        # Read file and search for DJI relative altitude in XMP
        with open(image_path, 'rb') as f:
            content = f.read()

        # Look for drone-dji:RelativeAltitude in XMP
        match = re.search(rb'drone-dji:RelativeAltitude="([^"]+)"', content)
        if match:
            altitude_str = match.group(1).decode('utf-8')
            altitude = float(altitude_str.replace('+', ''))
            logger.debug(f"Extracted altitude from XMP: {altitude}m")
            return altitude

    except Exception as e:
        logger.debug(f"Failed to extract altitude from EXIF: {e}")

    return None


class ThermalExtractor:
    """Extract temperature data from DJI thermal images with calibration support."""

    def __init__(self, calibration: Optional[ThermalCalibration] = None):
        """
        Initialize thermal extractor with optional calibration.

        Args:
            calibration: Calibration parameters. Uses settings defaults if None.
        """
        self.calibration = calibration or ThermalCalibration.from_settings()

        if THERMAL_PARSER_AVAILABLE and Thermal is not None:
            try:
                self.thermal = Thermal(dtype=np.float32)
                self.available = True
                logger.info(
                    f"Thermal extractor initialized with calibration: "
                    f"emissivity={self.calibration.emissivity}, "
                    f"distance={self.calibration.object_distance}m, "
                    f"humidity={self.calibration.relative_humidity}%, "
                    f"reflected_temp={self.calibration.reflected_temp}C"
                )
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

    def _parse_thermal_image(
        self,
        image_path: Path,
        use_calibration: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Parse thermal image with optional calibration.

        Uses parse_dirp2() with calibration parameters for accurate temperature
        readings, falling back to parse() if calibration fails.

        Args:
            image_path: Path to the thermal R-JPEG image
            use_calibration: Whether to use calibrated parsing (DIRP2)

        Returns:
            2D numpy array of temperatures in Celsius, or None if fails
        """
        if not self.available:
            return None

        try:
            if use_calibration:
                # Determine object distance
                object_distance = self.calibration.object_distance

                # Try to extract actual altitude from EXIF if auto_altitude is enabled
                if settings.thermal_auto_altitude:
                    exif_altitude = extract_altitude_from_exif(image_path)
                    if exif_altitude is not None:
                        # Clamp to valid range (1-25m)
                        object_distance = max(1.0, min(25.0, exif_altitude))
                        logger.debug(f"Using EXIF altitude: {object_distance}m")

                # Use calibrated parsing with DIRP2
                temp_array = self.thermal.parse_dirp2(
                    filepath_image=str(image_path),
                    image_height=512,
                    image_width=640,
                    object_distance=object_distance,
                    relative_humidity=self.calibration.relative_humidity,
                    emissivity=self.calibration.emissivity,
                    reflected_apparent_temperature=self.calibration.reflected_temp,
                )

                if temp_array is not None:
                    logger.debug(
                        f"Calibrated thermal parse successful: "
                        f"distance={object_distance}m, emissivity={self.calibration.emissivity}"
                    )
                    return temp_array

            # Fallback to simple parsing without calibration
            logger.debug("Using uncalibrated thermal parsing")
            return self.thermal.parse(filepath_image=str(image_path))

        except Exception as e:
            logger.warning(f"Calibrated parsing failed, trying uncalibrated: {e}")
            try:
                return self.thermal.parse(filepath_image=str(image_path))
            except Exception as e2:
                logger.warning(f"Uncalibrated parsing also failed: {e2}")
                return None

    def extract_temperature_at_pixel(
        self,
        image_path: Path,
        pixel_x: int,
        pixel_y: int,
        panel_region: Optional[Tuple[int, int, int, int]] = None,
        sample_radius: int = 50,
        image_format=None,
        visual_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[TemperatureReading]:
        """
        Extract temperature at a specific pixel location.

        Args:
            image_path: Path to DJI thermal R-JPEG image
            pixel_x: X coordinate of defect in VISUAL image (1280x1024)
            pixel_y: Y coordinate of defect in VISUAL image (1280x1024)
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
            # Parse thermal image with calibration
            temp_array = self._parse_thermal_image(image_path, use_calibration=True)

            if temp_array is None:
                logger.warning(f"Failed to parse thermal data from {image_path}")
                return None

            height, width = temp_array.shape

            # Convert from visual (1280x1024) to thermal (640x512) coordinates
            # This applies any configured alignment offset and scales by 0.5
            from ..utils.thermal_alignment import (
                visual_to_thermal,
                clamp_thermal_coords,
                ImageFormat,
            )

            if image_format is None:
                image_format = ImageFormat.VISUAL_THERMAL

            thermal_size = (width, height)
            thermal_x, thermal_y = visual_to_thermal(
                pixel_x,
                pixel_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            pixel_x, pixel_y = clamp_thermal_coords(
                thermal_x,
                thermal_y,
                thermal_size=thermal_size,
            )

            # Also scale the sample radius for the thermal resolution
            if image_format == ImageFormat.VISUAL_THERMAL and visual_size:
                scale_x = thermal_size[0] / visual_size[0]
                sample_radius = int(sample_radius * scale_x)

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

            # Calculate delta T
            delta_t = defect_temp - panel_avg

            reading = TemperatureReading(
                defect_temp_celsius=round(defect_temp, 1),
                panel_avg_celsius=round(panel_avg, 1),
                delta_t=round(delta_t, 1),
                min_temp_celsius=round(float(panel_temp_region.min()), 1),
                max_temp_celsius=round(float(panel_temp_region.max()), 1),
            )

            logger.debug(
                f"Temperature extracted (calibrated): defect={reading.defect_temp_celsius}C, "
                f"panel_avg={reading.panel_avg_celsius}C, delta_t={reading.delta_t}C"
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
        image_format=None,
        visual_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, float]:
        """
        Find the local temperature maximum near a given pixel.

        The projected coordinates from the orthophoto defect bbox center often
        don't land exactly on the hottest pixel. This method searches the area
        around the initial projection to find the actual hotspot.

        Args:
            image_path: Path to DJI thermal R-JPEG image
            initial_x: Initial X coordinate in VISUAL image (1280x1024)
            initial_y: Initial Y coordinate in VISUAL image (1280x1024)
            search_radius: Radius to search around initial point (in visual pixels)

        Returns:
            Tuple of (hotspot_x, hotspot_y, temperature) in VISUAL coordinates,
            or (initial_x, initial_y, temp) if search fails
        """
        if not self.available:
            return initial_x, initial_y, 0.0

        try:
            temp_array = self._parse_thermal_image(image_path, use_calibration=True)
            if temp_array is None:
                return initial_x, initial_y, 0.0

            height, width = temp_array.shape

            # Convert from visual (1280x1024) to thermal (640x512) coordinates
            # This applies any configured alignment offset
            from ..utils.thermal_alignment import (
                visual_to_thermal,
                thermal_to_visual,
                clamp_thermal_coords,
                ImageFormat,
            )

            if image_format is None:
                image_format = ImageFormat.VISUAL_THERMAL

            thermal_size = (width, height)
            thermal_x_f, thermal_y_f = visual_to_thermal(
                initial_x,
                initial_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            thermal_x, thermal_y = clamp_thermal_coords(
                thermal_x_f,
                thermal_y_f,
                thermal_size=thermal_size,
            )
            if image_format == ImageFormat.VISUAL_THERMAL and visual_size:
                scale_x = thermal_size[0] / visual_size[0]
                thermal_radius = int(search_radius * scale_x)
            else:
                thermal_radius = search_radius

            # Define search region in thermal coordinates
            x1 = max(0, thermal_x - thermal_radius)
            y1 = max(0, thermal_y - thermal_radius)
            x2 = min(width, thermal_x + thermal_radius)
            y2 = min(height, thermal_y + thermal_radius)

            # Extract search region
            search_region = temp_array[y1:y2, x1:x2]

            # Find maximum temperature location (in thermal coordinates)
            local_max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
            thermal_hotspot_y = y1 + local_max_idx[0]
            thermal_hotspot_x = x1 + local_max_idx[1]
            max_temp = float(search_region[local_max_idx])

            # Get temperature at initial point for comparison
            initial_temp = float(temp_array[
                max(0, min(height - 1, thermal_y)),
                max(0, min(width - 1, thermal_x))
            ])

            # Convert hotspot coordinates back to visual resolution
            visual_hotspot_x, visual_hotspot_y = thermal_to_visual(
                thermal_hotspot_x,
                thermal_hotspot_y,
                image_format=image_format,
                visual_size=visual_size,
                thermal_size=thermal_size,
            )
            visual_hotspot_x = int(visual_hotspot_x)
            visual_hotspot_y = int(visual_hotspot_y)

            # Only use the hotspot if it's significantly warmer than initial point
            # This avoids jumping to a different hotspot in the same image
            if max_temp > initial_temp + 2.0:  # At least 2C warmer
                logger.info(
                    f"Hotspot search: adjusted ({initial_x}, {initial_y}) -> ({visual_hotspot_x}, {visual_hotspot_y}), "
                    f"temp {initial_temp:.1f}C -> {max_temp:.1f}C (+{max_temp - initial_temp:.1f}C)"
                )
                return visual_hotspot_x, visual_hotspot_y, max_temp
            else:
                logger.debug(
                    f"Hotspot search: keeping initial ({initial_x}, {initial_y}), "
                    f"max in region only {max_temp - initial_temp:.1f}C warmer"
                )
                return initial_x, initial_y, initial_temp

        except Exception as e:
            logger.warning(f"Hotspot search failed for {image_path}: {e}")
            return initial_x, initial_y, 0.0

    def get_full_temperature_array(
        self,
        image_path: Path,
        use_calibration: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get the full temperature array from a thermal image.

        Args:
            image_path: Path to DJI thermal R-JPEG image
            use_calibration: Whether to use calibrated parsing

        Returns:
            2D numpy array of temperatures in Celsius, or None if fails
        """
        if not self.available:
            return None

        return self._parse_thermal_image(image_path, use_calibration=use_calibration)
