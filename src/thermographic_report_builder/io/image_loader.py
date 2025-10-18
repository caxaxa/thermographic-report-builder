"""Image loading utilities for orthophotos and raw thermal images."""

import cv2
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
from typing import Any

from ..utils.logger import get_logger
from ..utils.exceptions import ImageProcessingError

logger = get_logger(__name__)


def load_orthophoto(path: Path) -> tuple[np.ndarray, Any, tuple[int, int]]:
    """
    Load orthophoto GeoTIFF with georeferencing information.

    Args:
        path: Path to orthophoto GeoTIFF file

    Returns:
        Tuple of (image_array, rasterio_transform, (height, width))

    Raises:
        ImageProcessingError: If loading fails
    """
    logger.info(f"Loading orthophoto from {path}")

    try:
        with rasterio.open(path) as src:
            # Read all bands and convert to numpy array (C, H, W) -> (H, W, C)
            img = src.read()
            img = np.moveaxis(img, 0, -1).astype(np.uint8)

            # Convert RGB to BGR for OpenCV compatibility
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            transform = src.transform
            height, width = img.shape[:2]

            logger.info(f"Loaded orthophoto: {width}x{height} pixels, {img.shape[2]} channels")
            return img, transform, (height, width)

    except Exception as e:
        error_msg = f"Failed to load orthophoto from {path}: {e}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def load_image_bgr(path: Path) -> np.ndarray:
    """
    Load regular image file in BGR format (OpenCV compatible).

    Args:
        path: Path to image file

    Returns:
        Image as numpy array in BGR format

    Raises:
        ImageProcessingError: If loading fails
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
    except Exception as e:
        error_msg = f"Failed to load image {path}: {e}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def load_raw_image_with_exif(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load thermal image and extract EXIF metadata (including GPS).

    Args:
        path: Path to thermal JPG image

    Returns:
        Tuple of (image_array, exif_dict)

    Raises:
        ImageProcessingError: If loading fails
    """
    logger.debug(f"Loading raw image with EXIF: {path}")

    try:
        # Load image
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Extract EXIF data using PIL
        pil_img = Image.open(path)
        exif_data = pil_img._getexif() or {}

        # Extract GPS coordinates if available
        gps_info = exif_data.get(34853, {})  # GPS IFD tag

        exif_dict = {
            "path": str(path),
            "width": img.shape[1],
            "height": img.shape[0],
            "has_gps": bool(gps_info),
            "gps_data": gps_info,
        }

        # Parse GPS coordinates if available
        if gps_info:
            try:
                lat = _convert_gps_to_decimal(gps_info.get(2), gps_info.get(1))
                lon = _convert_gps_to_decimal(gps_info.get(4), gps_info.get(3))
                exif_dict["latitude"] = lat
                exif_dict["longitude"] = lon
                logger.debug(f"GPS coordinates: {lat}, {lon}")
            except Exception as e:
                logger.warning(f"Failed to parse GPS data: {e}")

        return img, exif_dict

    except Exception as e:
        error_msg = f"Failed to load raw image {path}: {e}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def _convert_gps_to_decimal(coords: tuple, ref: str) -> float:
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal.

    Args:
        coords: Tuple of (degrees, minutes, seconds)
        ref: Reference direction ('N', 'S', 'E', 'W')

    Returns:
        Decimal coordinate
    """
    degrees, minutes, seconds = coords
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

    if ref in ["S", "W"]:
        decimal = -decimal

    return decimal


def save_image(img: np.ndarray, path: Path, quality: int = 90) -> None:
    """
    Save image to file with specified quality.

    Args:
        img: Image array in BGR format
        path: Output path
        quality: JPEG quality (1-100)

    Raises:
        ImageProcessingError: If saving fails
    """
    try:
        if path.suffix.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif path.suffix.lower() == ".png":
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(path), img)

        logger.debug(f"Saved image to {path}")
    except Exception as e:
        error_msg = f"Failed to save image to {path}: {e}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e
