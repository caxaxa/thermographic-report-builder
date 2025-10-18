"""JSON file handling utilities."""

import json
from pathlib import Path
from typing import Any

from ..models.defect import DefectLabelsJSON
from ..utils.logger import get_logger
from ..utils.exceptions import InvalidInputError

logger = get_logger(__name__)


def load_defect_labels(path: Path) -> DefectLabelsJSON:
    """
    Load and parse defect_labels.json file.

    Args:
        path: Path to defect_labels.json

    Returns:
        Parsed DefectLabelsJSON object

    Raises:
        InvalidInputError: If file is invalid or missing
    """
    logger.info(f"Loading defect labels from {path}")

    try:
        labels = DefectLabelsJSON.from_json_file(str(path))
        logger.info(
            f"Loaded {len(labels.bounding_boxes)} bounding boxes "
            f"({len(labels.get_defects())} defects, {len(labels.get_panels())} panels)"
        )
        return labels
    except Exception as e:
        error_msg = f"Failed to load defect labels from {path}: {e}"
        logger.error(error_msg)
        raise InvalidInputError(error_msg) from e


def save_json(data: dict[str, Any] | list[Any], path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (dict or list)
        path: Output path
        indent: JSON indentation (default: 2)
    """
    logger.info(f"Saving JSON to {path}")

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        size_kb = path.stat().st_size / 1_000
        logger.info(f"Saved JSON: {size_kb:.1f} KB")
    except Exception as e:
        error_msg = f"Failed to save JSON to {path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    logger.debug(f"Loading JSON from {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        error_msg = f"Failed to load JSON from {path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e
