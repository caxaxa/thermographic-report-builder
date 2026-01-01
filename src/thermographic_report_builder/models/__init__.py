"""Data models for thermographic report builder."""

from .defect import (
    DefectType,
    BoundingBox,
    GeospatialCoordinate,
    Defect,
    Panel,
    DefectLabelsJSON,
)
from .report import ReportConfig, ReportMetadata, DefectMetrics
from .job import JobInput, JobOutput
from .annotation_manifest import (
    AnnotationPoint,
    AnnotationEntry,
    AnnotationManifest,
    AnnotationOverride,
    OverrideManifest,
)

__all__ = [
    "DefectType",
    "BoundingBox",
    "GeospatialCoordinate",
    "Defect",
    "Panel",
    "DefectLabelsJSON",
    "ReportConfig",
    "ReportMetadata",
    "DefectMetrics",
    "JobInput",
    "JobOutput",
    "AnnotationPoint",
    "AnnotationEntry",
    "AnnotationManifest",
    "AnnotationOverride",
    "OverrideManifest",
]
