"""Configuration management for thermographic report builder."""

from .settings import settings
from .constants import (
    DEFECT_COLORS,
    DEFECT_LABELS_PT,
    DEFAULT_PANEL_WIDTH_PX,
    ORTHOPHOTO_DOWNSCALE_FACTOR,
    CROP_DOWNSCALE_FACTOR,
)

__all__ = [
    "settings",
    "DEFECT_COLORS",
    "DEFECT_LABELS_PT",
    "DEFAULT_PANEL_WIDTH_PX",
    "ORTHOPHOTO_DOWNSCALE_FACTOR",
    "CROP_DOWNSCALE_FACTOR",
]
