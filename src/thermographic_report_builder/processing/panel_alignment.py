"""Alignment helpers to keep solar panel grids tracker-straight.

This module keeps the same guarantees Detectron2 provides during inference:
- panels are resized to a consensus footprint
- columns/rows are aligned inside a tracker-oriented coordinate frame (PCA-based)
- neighbouring trackers receive smoothed offsets so large fields stay visually straight
- the final positions stay within the original orthophoto footprint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..models.defect import BoundingBox
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _PanelPlacement:
    """Panel representation in tracker space."""

    bbox: BoundingBox
    tracker_coords: np.ndarray  # [parallel, perpendicular]


class _TrackerFrame:
    """Describe a rotated coordinate frame fitted to the tracker orientation."""

    def __init__(self, origin: np.ndarray, rotation: np.ndarray):
        self.origin = origin  # centroid of all panels
        self.rotation = rotation  # columns = orthonormal axes
        self.rotation_T = rotation.T

    @property
    def axis_parallel(self) -> np.ndarray:
        return self.rotation[:, 0]

    @property
    def axis_perpendicular(self) -> np.ndarray:
        return self.rotation[:, 1]

    def to_tracker(self, point: np.ndarray) -> np.ndarray:
        """Project (x, y) from image space into tracker coordinates."""
        return (point - self.origin) @ self.rotation

    def to_image(self, tracker_point: np.ndarray) -> np.ndarray:
        """Transform a tracker coordinate back to image space."""
        return self.origin + tracker_point @ self.rotation_T

    @classmethod
    def from_boxes(cls, boxes: Sequence[BoundingBox]) -> "_TrackerFrame":
        centers = np.array([_bbox_center(box) for box in boxes], dtype=float)
        origin = centers.mean(axis=0)

        if len(centers) < 2:
            return cls(origin, np.eye(2))

        centered = centers - origin
        cov = np.cov(centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order = np.argsort(eig_vals)[::-1]
        rotation = eig_vecs[:, order]

        # Ensure right-handed frame
        if np.linalg.det(rotation) < 0:
            rotation[:, 1] *= -1

        return cls(origin, rotation)


def align_panel_grid(
    panel_boxes: list[BoundingBox],
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[BoundingBox]:
    """Return tracker-aligned copies of the incoming panel boxes."""
    if len(panel_boxes) < 3:
        return panel_boxes

    try:
        original_bounds = _compute_envelope(panel_boxes)
        aligned = [box.model_copy(deep=True) for box in panel_boxes]

        widths = [box.width for box in aligned]
        heights = [box.height for box in aligned]
        median_width = float(np.median(widths)) if widths else 0.0
        median_height = float(np.median(heights)) if heights else 0.0

        if median_width <= 0 or median_height <= 0:
            return panel_boxes

        _standardize_dimensions(aligned, median_width, median_height, image_width, image_height)

        frame = _TrackerFrame.from_boxes(aligned)
        placements = [
            _PanelPlacement(bbox=box, tracker_coords=frame.to_tracker(_bbox_center(box)))
            for box in aligned
        ]

        panel_extent_parallel = _projected_extent(median_width, median_height, frame.axis_parallel)
        panel_extent_perp = _projected_extent(median_width, median_height, frame.axis_perpendicular)

        columns = _cluster_columns(placements, tolerance=panel_extent_perp * 0.55)
        if not columns:
            return aligned

        _synchronize_columns(columns, panel_extent_perp)
        row_tracks = _compute_row_tracks(columns, panel_extent_parallel)
        if row_tracks:
            row_tracks = _smooth_sequence(row_tracks)
            _apply_row_tracks(columns, row_tracks)

        _resolve_tracker_overlaps(columns, panel_extent_parallel)
        _update_boxes_from_tracker(columns, frame, image_width, image_height)
        _apply_guardrails(aligned, original_bounds)
        _log_alignment_shift(panel_boxes, aligned)
        return aligned
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Panel alignment failed, using raw detections: %s", exc)
        return panel_boxes


def _standardize_dimensions(
    panels: list[BoundingBox],
    median_width: float,
    median_height: float,
    image_width: int | None,
    image_height: int | None,
) -> None:
    for panel in panels:
        cx, cy = _bbox_center(panel)
        new_left = cx - median_width / 2
        new_top = cy - median_height / 2

        if image_width is not None:
            new_left = max(0.0, min(new_left, image_width - median_width))
        if image_height is not None:
            new_top = max(0.0, min(new_top, image_height - median_height))

        panel.left = new_left
        panel.top = new_top
        panel.width = median_width
        panel.height = median_height


def _cluster_columns(placements: list[_PanelPlacement], tolerance: float) -> list[list[_PanelPlacement]]:
    sorted_panels = sorted(placements, key=lambda p: p.tracker_coords[1])

    columns: list[list[_PanelPlacement]] = []
    current: list[_PanelPlacement] = []

    for panel in sorted_panels:
        if not current:
            current.append(panel)
            continue

        center = float(np.median([p.tracker_coords[1] for p in current]))
        if abs(panel.tracker_coords[1] - center) <= tolerance:
            current.append(panel)
        else:
            columns.append(sorted(current, key=lambda p: p.tracker_coords[0]))
            current = [panel]

    if current:
        columns.append(sorted(current, key=lambda p: p.tracker_coords[0]))

    return columns


def _synchronize_columns(columns: list[list[_PanelPlacement]], panel_extent_perp: float) -> None:
    medians = [float(np.median([p.tracker_coords[1] for p in column])) for column in columns]
    smoothed = _smooth_sequence(medians)

    for column, target in zip(columns, smoothed):
        for panel in column:
            panel.tracker_coords[1] = target

    if len(smoothed) >= 2:
        gaps = np.diff(smoothed)
        median_gap = np.median(gaps) if len(gaps) else 0.0
        logger.info(
            "Column alignment: %d trackers, median spacing %.1f px (projected width %.1f)",
            len(columns),
            median_gap,
            panel_extent_perp,
        )


def _compute_row_tracks(columns: list[list[_PanelPlacement]], panel_extent_parallel: float) -> list[float]:
    if not columns:
        return []

    max_rows = max(len(column) for column in columns)
    if max_rows == 0:
        return []

    first_row_samples = [column[0].tracker_coords[0] for column in columns if column]
    anchor = float(np.median(first_row_samples)) if first_row_samples else 0.0
    row_tracks = [anchor]

    for row_idx in range(1, max_rows):
        gaps = []
        for column in columns:
            if len(column) > row_idx:
                prev = column[row_idx - 1].tracker_coords[0]
                curr = column[row_idx].tracker_coords[0]
                gaps.append(curr - prev)
        median_gap = float(np.median(gaps)) if gaps else panel_extent_parallel
        if median_gap < 0:
            median_gap = panel_extent_parallel
        row_tracks.append(row_tracks[row_idx - 1] + median_gap)

    return row_tracks


def _apply_row_tracks(columns: list[list[_PanelPlacement]], row_tracks: list[float]) -> None:
    for column in columns:
        for row_idx, panel in enumerate(column):
            if row_idx < len(row_tracks):
                panel.tracker_coords[0] = row_tracks[row_idx]


def _resolve_tracker_overlaps(columns: list[list[_PanelPlacement]], panel_extent_parallel: float) -> None:
    minimum_gap = panel_extent_parallel * 0.1
    for column in columns:
        previous = None
        for panel in column:
            if previous is not None and panel.tracker_coords[0] <= previous + minimum_gap:
                panel.tracker_coords[0] = previous + minimum_gap
            previous = panel.tracker_coords[0]


def _update_boxes_from_tracker(
    columns: list[list[_PanelPlacement]],
    frame: _TrackerFrame,
    image_width: int | None,
    image_height: int | None,
) -> None:
    for panel in _flatten(columns):
        center = frame.to_image(panel.tracker_coords)
        left = center[0] - panel.bbox.width / 2
        top = center[1] - panel.bbox.height / 2

        if image_width is not None:
            left = max(0.0, min(left, image_width - panel.bbox.width))
        if image_height is not None:
            top = max(0.0, min(top, image_height - panel.bbox.height))

        panel.bbox.left = left
        panel.bbox.top = top


def _apply_guardrails(boxes: Sequence[BoundingBox], bounds: tuple[float, float, float, float]) -> None:
    min_left, min_top, max_right, max_bottom = bounds

    for box in boxes:
        box.left = max(min_left, min(box.left, max_right - box.width))
        box.top = max(min_top, min(box.top, max_bottom - box.height))


def _compute_envelope(boxes: Sequence[BoundingBox]) -> tuple[float, float, float, float]:
    min_left = min(box.left for box in boxes)
    min_top = min(box.top for box in boxes)
    max_right = max(box.left + box.width for box in boxes)
    max_bottom = max(box.top + box.height for box in boxes)
    return min_left, min_top, max_right, max_bottom


def _smooth_sequence(values: Sequence[float], kernel: Sequence[float] | None = None) -> list[float]:
    if len(values) < 3:
        return list(values)

    kernel = np.array(kernel if kernel is not None else [0.25, 0.5, 0.25])
    padded = np.pad(values, (1, 1), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.tolist()


def _projected_extent(width: float, height: float, axis: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    return abs(axis[0]) * width + abs(axis[1]) * height


def _bbox_center(box: BoundingBox) -> np.ndarray:
    return np.array([box.left + box.width / 2, box.top + box.height / 2], dtype=float)


def _flatten(columns: list[list[_PanelPlacement]]) -> list[_PanelPlacement]:
    return [panel for column in columns for panel in column]


def _log_alignment_shift(original: Sequence[BoundingBox], aligned: Sequence[BoundingBox]) -> None:
    if len(original) != len(aligned):
        return

    before = np.array([_bbox_center(box) for box in original], dtype=float)
    after = np.array([_bbox_center(box) for box in aligned], dtype=float)

    shifts = np.linalg.norm(after - before, axis=1)
    if shifts.size == 0:
        return

    logger.info(
        "Panel alignment adjustment stats → mean: %.1f px, median: %.1f px, max: %.1f px",
        float(np.mean(shifts)),
        float(np.median(shifts)),
        float(np.max(shifts)),
    )
