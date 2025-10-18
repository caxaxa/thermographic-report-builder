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
]
