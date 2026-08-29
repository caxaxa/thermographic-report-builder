"""Data models for defects, panels, and bounding boxes."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ValidationError
import json
import logging


logger = logging.getLogger(__name__)


class DefectType(str, Enum):
    """Types of defects detected in thermal images."""

    HOTSPOTS = "hotspots"
    FAULTY_DIODES = "faultydiodes"
    OFFLINE_PANELS = "offlinepanels"
    SOLAR_PANELS = "solarpanels"  # For panel detection


class BoundingBox(BaseModel):
    """Bounding box in image pixel coordinates."""

    left: float = Field(ge=0, description="Left coordinate in pixels")
    top: float = Field(ge=0, description="Top coordinate in pixels")
    width: float = Field(gt=0, description="Width in pixels")
    height: float = Field(gt=0, description="Height in pixels")
    label: str = Field(description="Defect type label")

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box (x, y)."""
        return (self.left + self.width / 2, self.top + self.height / 2)

    @property
    def right(self) -> int:
        """Right edge coordinate."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Bottom edge coordinate."""
        return self.top + self.height

    @property
    def area(self) -> int:
        """Area of bounding box in pixels."""
        return self.width * self.height


class GeospatialCoordinate(BaseModel):
    """Longitude/Latitude coordinates."""

    longitude: float = Field(description="Longitude in degrees")
    latitude: float = Field(description="Latitude in degrees")

    def to_tuple(self) -> tuple[float, float]:
        """Return as (lon, lat) tuple."""
        return (self.longitude, self.latitude)


class Defect(BaseModel):
    """Single defect instance with location and type."""

    bbox: BoundingBox
    defect_center_px: tuple[float, float] = Field(
        description="Defect center in pixel coordinates (x, y)"
    )
    panel_centroid_geospatial: GeospatialCoordinate
    defect_type: str

    @property
    def is_hotspot(self) -> bool:
        return self.defect_type == DefectType.HOTSPOTS.value

    @property
    def is_faulty_diode(self) -> bool:
        return self.defect_type == DefectType.FAULTY_DIODES.value

    @property
    def is_offline_panel(self) -> bool:
        return self.defect_type == DefectType.OFFLINE_PANELS.value


class Panel(BaseModel):
    """Solar panel with potential defects."""

    column: int = Field(ge=1, description="Column number (1-indexed, left to right)")
    row: int = Field(ge=1, description="Row within column (1-indexed, top to bottom)")
    bbox: BoundingBox
    hotspots: list[Defect] = Field(default_factory=list)
    faulty_diodes: list[Defect] = Field(default_factory=list)
    offline_panels: list[Defect] = Field(default_factory=list)
    tracker: str = Field(default="A", description="Tracker section label (A, B, C, ...)")
    tracker_column: int = Field(default=1, ge=1, description="Row within tracker section (1-indexed)")
    is_horizontal: bool = Field(default=False, description="If true, use row-column format instead of column-row")
    double_suffix: str = Field(default="", description="For double trackers: 'A' for top panel, 'B' for bottom glued panel")

    @property
    def panel_id(self) -> str:
        """
        Get panel identifier based on layout configuration.

        Vertical layout (default, is_horizontal=False):
        - Column-first numbering: columns left-to-right, rows top-to-bottom within column
        - Format: 'Col-Row' (e.g., '3-45') or 'Col-RowTracker' (e.g., '3-1B')

        Horizontal layout (is_horizontal=True):
        - Row-first numbering: rows top-to-bottom, columns left-to-right within row
        - Format: 'Row-Col' (e.g., '1-3') or 'Row-ColSuffix' (e.g., '1-3A' for double tracker)

        Double tracker suffix (double_suffix):
        - 'A' = top panel of glued pair
        - 'B' = bottom panel of glued pair
        """
        if self.is_horizontal:
            # Horizontal layout: Row-Column format
            base = f"{self.row}-{self.column}"
            if self.double_suffix:
                return f"{base}{self.double_suffix}"
            elif self.tracker != "A":
                return f"{base}{self.tracker}"
            return base
        else:
            # Vertical layout (default): Column-Row format
            if self.tracker == "A":
                return f"{self.column}-{self.tracker_column}"
            else:
                return f"{self.column}-{self.tracker_column}{self.tracker}"

    @property
    def has_defects(self) -> bool:
        """Check if panel has any defects."""
        return bool(self.hotspots or self.faulty_diodes or self.offline_panels)

    @property
    def defect_count(self) -> int:
        """Total number of defects in this panel."""
        return len(self.hotspots) + len(self.faulty_diodes) + len(self.offline_panels)

    def all_defects(self) -> list[Defect]:
        """Get all defects as a single list."""
        return self.hotspots + self.faulty_diodes + self.offline_panels


class DefectLabelsJSON(BaseModel):
    """
    Root structure for defect_labels.json file.

    Expected format from masking stage:
    [
        {
            "boundingBox": {
                "boundingBoxes": [
                    {"label": "hotspots", "left": 498, "top": 10641, "width": 7, "height": 19},
                    ...
                ]
            }
        }
    ]
    """

    bounding_boxes: list[BoundingBox]

    @classmethod
    def from_json_file(cls, path: str) -> "DefectLabelsJSON":
        """Load and parse defect labels from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Handle multiple input formats:
        # 1. List format from masking stage: [{"boundingBox": {"boundingBoxes": [...]}}]
        # 2. Dict format from annotation tool: {"boundingBox": {"boundingBoxes": [...]}}
        if isinstance(data, list) and len(data) > 0:
            # List format - take first element
            boxes_data = data[0].get("boundingBox", {}).get("boundingBoxes", [])
        elif isinstance(data, dict):
            # Dict format from annotation tool (groundtruth bucket)
            boxes_data = data.get("boundingBox", {}).get("boundingBoxes", [])
        else:
            boxes_data = []

        bounding_boxes = cls._sanitize_boxes(boxes_data)
        return cls(bounding_boxes=bounding_boxes)

    @staticmethod
    def _sanitize_boxes(boxes_data: list[dict]) -> list["BoundingBox"]:
        """Build boxes, tolerating stray annotations near/off the canvas.

        Annotation tools let a box be dragged partly or fully off the image,
        which yields negative left/top. BoundingBox requires >= 0, so a single
        stray box used to abort an entire report (2026-08-29: 3 junk boxes out
        of 1,512 killed a customer report). Clamp the partially-visible ones
        into frame and drop the ones with no on-canvas area.
        """
        boxes: list[BoundingBox] = []
        clamped = dropped = invalid = 0
        for bb in boxes_data:
            data = dict(bb)
            left = data.get("left", 0)
            top = data.get("top", 0)
            width = data.get("width", 0)
            height = data.get("height", 0)

            # No visible area once projected onto the canvas -> carries nothing.
            if left + width <= 0 or top + height <= 0:
                dropped += 1
                continue

            if left < 0:
                width += left  # keep the right edge fixed
                left = 0
                clamped += 1
            if top < 0:
                height += top  # keep the bottom edge fixed
                top = 0
                clamped += 1

            data.update(left=left, top=top, width=width, height=height)
            try:
                boxes.append(BoundingBox(**data))
            except ValidationError:
                invalid += 1

        if clamped or dropped or invalid:
            logger.warning(
                "Sanitized annotation boxes: %d clamped, %d dropped (off-canvas), "
                "%d invalid, %d kept",
                clamped,
                dropped,
                invalid,
                len(boxes),
            )
        return boxes

    def get_panels(self) -> list[BoundingBox]:
        """Get all solar panel bounding boxes."""
        # Accept both "solarpanels" and "default_panel" as panel labels
        # TODO: Standardize label naming across training, inference, and report generation
        panel_labels = {DefectType.SOLAR_PANELS.value, "default_panel"}
        return [bb for bb in self.bounding_boxes if bb.label in panel_labels]

    def get_defects(self) -> list[BoundingBox]:
        """Get all defect bounding boxes (excluding panels)."""
        # Exclude both "solarpanels" and "default_panel" labels
        panel_labels = {DefectType.SOLAR_PANELS.value, "default_panel"}
        return [bb for bb in self.bounding_boxes if bb.label not in panel_labels]

    def get_by_type(self, defect_type: DefectType) -> list[BoundingBox]:
        """Get bounding boxes of a specific defect type."""
        return [bb for bb in self.bounding_boxes if bb.label == defect_type.value]
