"""Data models for defects, panels, and bounding boxes."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import json


class DefectType(str, Enum):
    """Types of defects detected in thermal images."""

    HOTSPOTS = "hotspots"
    FAULTY_DIODES = "faultydiodes"
    OFFLINE_PANELS = "offlinepanels"
    SOLAR_PANELS = "solarpanels"  # For panel detection


class BoundingBox(BaseModel):
    """Bounding box in image pixel coordinates."""

    left: int = Field(ge=0, description="Left coordinate in pixels")
    top: int = Field(ge=0, description="Top coordinate in pixels")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")
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

    column: int = Field(ge=1, description="Panel column number (1-indexed)")
    row: int = Field(ge=1, description="Panel row number (1-indexed)")
    bbox: BoundingBox
    hotspots: list[Defect] = Field(default_factory=list)
    faulty_diodes: list[Defect] = Field(default_factory=list)
    offline_panels: list[Defect] = Field(default_factory=list)

    @property
    def panel_id(self) -> str:
        """Get panel identifier as 'col-row'."""
        return f"{self.column}-{self.row}"

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

        # Parse nested structure from masking stage
        if isinstance(data, list) and len(data) > 0:
            boxes_data = data[0].get("boundingBox", {}).get("boundingBoxes", [])
        else:
            boxes_data = []

        bounding_boxes = [BoundingBox(**bb) for bb in boxes_data]
        return cls(bounding_boxes=bounding_boxes)

    def get_panels(self) -> list[BoundingBox]:
        """Get all solar panel bounding boxes."""
        return [bb for bb in self.bounding_boxes if bb.label == DefectType.SOLAR_PANELS.value]

    def get_defects(self) -> list[BoundingBox]:
        """Get all defect bounding boxes (excluding panels)."""
        return [bb for bb in self.bounding_boxes if bb.label != DefectType.SOLAR_PANELS.value]

    def get_by_type(self, defect_type: DefectType) -> list[BoundingBox]:
        """Get bounding boxes of a specific defect type."""
        return [bb for bb in self.bounding_boxes if bb.label == defect_type.value]
