"""Data models for annotation manifest and human overrides.

The annotation manifest captures all thermal annotations generated during
report building, enabling a human-in-the-loop workflow where operators can
review and override hot/cold point positions before final report generation.

Coordinate System:
- All coordinates are in THERMAL space (640x512 pixels)
- These are direct indices into the thermal temperature array
- Conversion: thermal_x = visual_x * (640/1280), thermal_y = visual_y * (512/1024)

Rotation Handling:
- Coordinates are stored in ROTATED space (as displayed in the report)
- The `flight_direction` field indicates if 180° rotation was applied:
  - 'south': Image was rotated 180° for display (drone flying south)
  - 'north' or None: No rotation applied
- For the override UI: if flight_direction is 'south', the image should be
  displayed rotated 180° to match the coordinate system of the annotations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class AnnotationPoint(BaseModel):
    """A point in thermal coordinate space with temperature."""

    x: int = Field(ge=0, lt=640, description="X coordinate in thermal space (0-639)")
    y: int = Field(ge=0, lt=512, description="Y coordinate in thermal space (0-511)")
    temp: Optional[float] = Field(default=None, description="Temperature in Celsius")


class AnnotationEntry(BaseModel):
    """A single defect annotation with hot/cold points."""

    defect_id: str = Field(description="Unique defect identifier (e.g., 'hotspots_(A-01)_defeito1')")
    panel_id: str = Field(description="Panel identifier (e.g., 'A-01')")
    defect_type: str = Field(description="Type of defect (e.g., 'hotspots', 'soiling')")
    defect_index: int = Field(description="1-based index within panel")
    raw_image_path: str = Field(description="S3 path to raw thermal image")
    raw_image_name: str = Field(description="Filename only (e.g., 'DJI_xxx.JPG')")
    annotated_image: str = Field(description="Local filename of annotated image")
    hot_point: AnnotationPoint = Field(description="Hot point in thermal coordinates")
    cold_point: AnnotationPoint = Field(description="Cold/reference point in thermal coordinates")
    delta_t: float = Field(description="Temperature difference (hot - cold) in Celsius")
    severity: str = Field(description="Severity classification (CRITICAL, ATTENTION, MONITORING)")
    flight_direction: Optional[str] = Field(
        default=None,
        description="Flight direction ('north' or 'south'). If 'south', image needs 180° rotation for display."
    )


class AnnotationManifest(BaseModel):
    """Complete manifest of all thermal annotations for a project."""

    project_id: str = Field(description="Project identifier")
    user_id: str = Field(description="User/org identifier")
    generated_at: str = Field(description="ISO timestamp when manifest was generated")
    annotations: List[AnnotationEntry] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AnnotationManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class AnnotationOverride(BaseModel):
    """Override for a single defect's hot and/or cold point.

    If hot_point or cold_point is None, the original automatic annotation is kept.
    """

    defect_id: str = Field(description="Defect identifier to override")
    hot_point: Optional[AnnotationPoint] = Field(
        default=None, description="Override hot point (null = keep automatic)"
    )
    cold_point: Optional[AnnotationPoint] = Field(
        default=None, description="Override cold point (null = keep automatic)"
    )


class OverrideManifest(BaseModel):
    """Collection of annotation overrides from human review."""

    project_id: str = Field(description="Project identifier")
    user_id: str = Field(description="User/org identifier")
    created_at: str = Field(description="ISO timestamp when overrides were created")
    overrides: List[AnnotationOverride] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save overrides to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "OverrideManifest":
        """Load overrides from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def get_override(self, defect_id: str) -> Optional[AnnotationOverride]:
        """Get override for a specific defect, or None if not overridden."""
        for override in self.overrides:
            if override.defect_id == defect_id:
                return override
        return None
