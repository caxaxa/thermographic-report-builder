"""Utilities to convert Ground Truth detection review output into defect_labels.json.

This mirrors the post-processing pipeline used in the Detectron2 inference job:
1. normalize panel bounding boxes to a common footprint
2. deduplicate overlaps with IoU thresholding
3. align panels into tracker rows/columns to guarantee straight ladders
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Use report builder's logger instead of structlog
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Label normalization maps Ground Truth labels back to inference classes
LABEL_NORMALIZATION = {
    "default_panel": "default_panel",
    "solarpanels": "default_panel",
    "solar_panel": "default_panel",
    "panel": "default_panel",
    "hotspot": "hotspots",
    "hotspots": "hotspots",
    "faultydiodes": "faultydiodes",
    "faulty_diode": "faultydiodes",
    "offlinepanels": "offlinepanels",
    "offline_panel": "offlinepanels",
}

# Report builder expects solarpanels instead of default_panel
REPORT_LABEL_MAPPING = {
    "default_panel": "solarpanels",
    "hotspots": "hotspots",
    "faultydiodes": "faultydiodes",
    "offlinepanels": "offlinepanels",
}


def convert_annotations_to_defect_labels(
    annotations: List[Dict[str, Any]],
    preview_metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert Ground Truth detection review output to defect_labels.json payload.

    Args:
        annotations: Parsed manifest entries from GroundTruthManager.get_annotation_results
        preview_metadata: Metadata captured when the preview JPEG was generated

    Returns:
        Tuple of (defect_labels_payload, statistics dict)
    """
    detections = _annotations_to_detections(annotations, preview_metadata)

    if not detections:
        raise ValueError("No bounding boxes found in annotation results")

    processed = standardize_and_deduplicate(detections, iou_threshold=0.2)
    bounding_boxes = _detections_to_report_format(processed)
    class_counts = Counter(box["label"] for box in bounding_boxes)

    payload = [
        {
            "boundingBox": {
                "boundingBoxes": bounding_boxes,
            }
        }
    ]

    stats = {
        "total_boxes": len(bounding_boxes),
        "class_counts": dict(class_counts),
        "total_panels": class_counts.get("solarpanels", 0),
    }

    logger.info(
        "Converted Ground Truth annotations",
        total_boxes=stats["total_boxes"],
        class_counts=stats["class_counts"],
    )

    return payload, stats


def _extract_annotation_payload(annotation: Dict[str, Any]) -> Dict[str, Any] | None:
    """Find the annotation payload that contains bounding boxes."""
    for key, value in annotation.items():
        if not isinstance(value, dict):
            continue
        if not key.endswith("detection-review") or key.endswith("detection-review-metadata"):
            continue
        if "boundingBoxes" in value:
            return value
    return None


def _annotations_to_detections(
    annotations: Iterable[Dict[str, Any]],
    preview_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Scale Ground Truth preview coordinates back to orthophoto coordinates."""
    default_preview_width = preview_metadata.get("preview_width")
    default_preview_height = preview_metadata.get("preview_height")
    original_width = preview_metadata.get("original_width") or default_preview_width
    original_height = preview_metadata.get("original_height") or default_preview_height

    detections: List[Dict[str, Any]] = []

    for annotation in annotations:
        payload = _extract_annotation_payload(annotation)
        if not payload:
            continue

        image_props = payload.get("inputImageProperties", {}) or {}
        preview_width = image_props.get("width") or default_preview_width
        preview_height = image_props.get("height") or default_preview_height

        if not preview_width or not preview_height:
            logger.warning("Missing preview dimensions in annotation entry, skipping")
            continue

        scale_x = (original_width or preview_width) / preview_width
        scale_y = (original_height or preview_height) / preview_height

        for bbox in payload.get("boundingBoxes", []):
            cls = _normalize_label(bbox.get("label", "default_panel"))
            if cls is None:
                continue

            left = float(bbox.get("left", 0)) * scale_x
            top = float(bbox.get("top", 0)) * scale_y
            width = float(bbox.get("width", 0)) * scale_x
            height = float(bbox.get("height", 0)) * scale_y
            if width <= 0 or height <= 0:
                continue

            detection = {
                "class": cls,
                "confidence": float(bbox.get("confidence", 0.99)),
                "bbox": [
                    round(left, 2),
                    round(top, 2),
                    round(left + width, 2),
                    round(top + height, 2),
                ],
            }
            detections.append(detection)

    return detections


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    canonical = LABEL_NORMALIZATION.get(label.lower())
    if canonical is None:
        logger.warning("Skipping unknown label from Ground Truth", label=label)
    return canonical


def _detections_to_report_format(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bounding_boxes: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)
        label = REPORT_LABEL_MAPPING.get(det["class"], det["class"])

        bounding_boxes.append(
            {
                "left": round(x1, 2),
                "top": round(y1, 2),
                "width": round(width, 2),
                "height": round(height, 2),
                "label": label,
            }
        )

    return bounding_boxes


# ===== Inference post-processing (mirrors solar-detection-model/scripts/inference_tiled.py) =====


def calculate_median_panel_size(detections: List[Dict[str, Any]]) -> Tuple[float | None, float | None]:
    panel_detections = [d for d in detections if d["class"] == "default_panel"]

    if not panel_detections:
        return None, None

    widths = []
    heights = []

    for det in panel_detections:
        bbox = det["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        widths.append(width)
        heights.append(height)

    median_width = float(np.median(widths))
    median_height = float(np.median(heights))

    return median_width, median_height


def standardize_panel_size(
    detection: Dict[str, Any],
    median_width: float,
    median_height: float,
) -> Dict[str, Any]:
    bbox = detection["bbox"]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    new_bbox = [
        cx - median_width / 2,
        cy - median_height / 2,
        cx + median_width / 2,
        cy + median_height / 2,
    ]
    detection["bbox"] = [round(v, 2) for v in new_bbox]
    return detection


def standardize_and_deduplicate(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    panel_count = sum(1 for d in detections if d["class"] == "default_panel")
    defect_count = len(detections) - panel_count
    logger.info(f"standardize_and_deduplicate: input {panel_count} panels, {defect_count} defects")

    median_width, median_height = calculate_median_panel_size(detections)
    logger.info(f"Median panel size: {median_width:.1f} x {median_height:.1f} px")

    standardized = []
    for det in detections:
        if det["class"] == "default_panel" and median_width and median_height:
            det = standardize_panel_size(det, median_width, median_height)
        standardized.append(det)
    logger.info(f"After standardization: {len(standardized)} detections")

    deduplicated = merge_duplicate_detections(standardized, iou_threshold=iou_threshold)
    dedup_panels = sum(1 for d in deduplicated if d["class"] == "default_panel")
    logger.info(f"After deduplication (IoU>{iou_threshold}): {dedup_panels} panels")

    if median_width and median_height:
        deduplicated = align_panels_to_grid(deduplicated, median_width, median_height)
        aligned_panels = sum(1 for d in deduplicated if d["class"] == "default_panel")
        logger.info(f"After grid alignment: {aligned_panels} panels")

    return deduplicated


def align_panels_to_grid(
    detections: List[Dict[str, Any]],
    panel_width: float,
    panel_height: float,
) -> List[Dict[str, Any]]:
    panels = [d for d in detections if d["class"] == "default_panel"]
    other_detections = [d for d in detections if d["class"] != "default_panel"]

    if len(panels) < 3:
        logger.info("align_panels_to_grid: skipped (less than 3 panels)")
        return detections

    orientation = "HORIZONTAL" if panel_width > panel_height else "VERTICAL"
    logger.info(f"Panel orientation: {orientation} (w={panel_width:.1f}, h={panel_height:.1f})")

    if orientation == "HORIZONTAL":
        columns = cluster_by_x_coordinate(panels, tolerance=panel_width * 0.5)
        logger.info(f"Clustered into {len(columns)} vertical ladders (columns)")
        aligned_panels = align_vertical_ladders(panels, panel_width, panel_height)
    else:
        rows = cluster_by_y_coordinate(panels, tolerance=panel_height * 0.5)
        logger.info(f"Clustered into {len(rows)} horizontal ladders (rows)")
        aligned_panels = align_horizontal_ladders(panels, panel_width, panel_height)

    aligned_panels = eliminate_overlaps_and_renormalize(aligned_panels, orientation)
    logger.info(f"After overlap elimination: {len(aligned_panels)} panels")

    return aligned_panels + other_detections


def eliminate_overlaps_and_renormalize(
    panels: List[Dict[str, Any]],
    orientation: str,
) -> List[Dict[str, Any]]:
    if not panels:
        return panels

    if orientation == "HORIZONTAL":
        panel_width = panels[0]["bbox"][2] - panels[0]["bbox"][0]
        columns = cluster_by_x_coordinate(panels, tolerance=panel_width * 0.5)

        non_overlapping_panels: List[Dict[str, Any]] = []
        for col in columns:
            col.sort(key=lambda p: p["bbox"][1])
            for i in range(1, len(col)):
                prev = col[i - 1]
                curr = col[i]
                prev_bottom = prev["bbox"][3]
                curr_top = curr["bbox"][1]
                if curr_top < prev_bottom:
                    overlap = prev_bottom - curr_top
                    half_overlap = overlap / 2.0
                    prev["bbox"][3] = prev_bottom - half_overlap
                    curr["bbox"][1] = curr_top + half_overlap
            non_overlapping_panels.extend(col)
        return non_overlapping_panels

    panel_height = panels[0]["bbox"][3] - panels[0]["bbox"][1]
    rows = cluster_by_y_coordinate(panels, tolerance=panel_height * 0.5)

    non_overlapping_panels = []
    for row in rows:
        row.sort(key=lambda p: p["bbox"][0])
        for i in range(1, len(row)):
            prev = row[i - 1]
            curr = row[i]
            prev_right = prev["bbox"][2]
            curr_left = curr["bbox"][0]
            if curr_left < prev_right:
                overlap = prev_right - curr_left
                half_overlap = overlap / 2.0
                prev["bbox"][2] = prev_right - half_overlap
                curr["bbox"][0] = curr_left + half_overlap
        non_overlapping_panels.extend(row)

    return non_overlapping_panels


def align_vertical_ladders(
    panels: List[Dict[str, Any]],
    panel_width: float,
    panel_height: float,
) -> List[Dict[str, Any]]:
    columns = cluster_by_x_coordinate(panels, tolerance=panel_width * 0.5)

    aligned_panels = []
    for column in columns:
        centers_x = [(p["bbox"][0] + p["bbox"][2]) / 2 for p in column]
        median_x = np.median(centers_x)
        x1_col = round(median_x - panel_width / 2, 2)
        x2_col = round(median_x + panel_width / 2, 2)
        for panel in column:
            panel["bbox"][0] = x1_col
            panel["bbox"][2] = x2_col
        column.sort(key=lambda p: p["bbox"][1])
        if len(column) > 1:
            gaps = [column[i + 1]["bbox"][1] - column[i]["bbox"][3] for i in range(len(column) - 1)]
            median_gap = np.median(gaps) if gaps else 0
            for i in range(1, len(column)):
                prev = column[i - 1]
                curr = column[i]
                expected_y1 = prev["bbox"][3] + median_gap
                if curr["bbox"][1] < prev["bbox"][3]:
                    curr["bbox"][1] = expected_y1
                    curr["bbox"][3] = expected_y1 + panel_height
                elif abs(curr["bbox"][1] - expected_y1) <= panel_height * 0.3:
                    curr["bbox"][1] = expected_y1
                    curr["bbox"][3] = expected_y1 + panel_height
        aligned_panels.extend(column)

    return aligned_panels


def align_horizontal_ladders(
    panels: List[Dict[str, Any]],
    panel_width: float,
    panel_height: float,
) -> List[Dict[str, Any]]:
    rows = cluster_by_y_coordinate(panels, tolerance=panel_height * 0.5)

    aligned_panels = []
    for row in rows:
        centers_y = [(p["bbox"][1] + p["bbox"][3]) / 2 for p in row]
        median_y = np.median(centers_y)
        y1_row = round(median_y - panel_height / 2, 2)
        y2_row = round(median_y + panel_height / 2, 2)
        for panel in row:
            panel["bbox"][1] = y1_row
            panel["bbox"][3] = y2_row
        row.sort(key=lambda p: p["bbox"][0])
        if len(row) > 1:
            gaps = [row[i + 1]["bbox"][0] - row[i]["bbox"][2] for i in range(len(row) - 1)]
            median_gap = np.median(gaps) if gaps else 0
            for i in range(1, len(row)):
                prev = row[i - 1]
                curr = row[i]
                expected_x1 = prev["bbox"][2] + median_gap
                if curr["bbox"][0] < prev["bbox"][2]:
                    curr["bbox"][0] = expected_x1
                    curr["bbox"][2] = expected_x1 + panel_width
                elif abs(curr["bbox"][0] - expected_x1) <= panel_width * 0.3:
                    curr["bbox"][0] = expected_x1
                    curr["bbox"][2] = expected_x1 + panel_width
        aligned_panels.extend(row)

    return aligned_panels


def cluster_by_x_coordinate(panels: List[Dict[str, Any]], tolerance: float) -> List[List[Dict[str, Any]]]:
    sorted_panels = sorted(panels, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)
    columns: List[List[Dict[str, Any]]] = []
    current_column: List[Dict[str, Any]] = []

    for panel in sorted_panels:
        center_x = (panel["bbox"][0] + panel["bbox"][2]) / 2
        if not current_column:
            current_column.append(panel)
            continue
        column_center = np.median([(p["bbox"][0] + p["bbox"][2]) / 2 for p in current_column])
        if abs(center_x - column_center) < tolerance:
            current_column.append(panel)
        else:
            columns.append(current_column)
            current_column = [panel]

    if current_column:
        columns.append(current_column)

    return columns


def cluster_by_y_coordinate(panels: List[Dict[str, Any]], tolerance: float) -> List[List[Dict[str, Any]]]:
    sorted_panels = sorted(panels, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2)
    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = []

    for panel in sorted_panels:
        center_y = (panel["bbox"][1] + panel["bbox"][3]) / 2
        if not current_row:
            current_row.append(panel)
            continue
        row_center = np.median([(p["bbox"][1] + p["bbox"][3]) / 2 for p in current_row])
        if abs(center_y - row_center) < tolerance:
            current_row.append(panel)
        else:
            rows.append(current_row)
            current_row = [panel]

    if current_row:
        rows.append(current_row)

    return rows


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_duplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(det["class"], []).append(det)

    merged_results: List[Dict[str, Any]] = []

    for cls, class_detections in grouped.items():
        class_detections = sorted(class_detections, key=lambda x: x.get("confidence", 0), reverse=True)

        while class_detections:
            current = class_detections.pop(0)
            overlapping = [current]
            remaining: List[Dict[str, Any]] = []

            for det in class_detections:
                if calculate_iou(current["bbox"], det["bbox"]) >= iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)

            if len(overlapping) > 1:
                centers_x = []
                centers_y = []
                confidences = []
                for det in overlapping:
                    bbox = det["bbox"]
                    centers_x.append((bbox[0] + bbox[2]) / 2)
                    centers_y.append((bbox[1] + bbox[3]) / 2)
                    confidences.append(det.get("confidence", 0))

                avg_cx = float(np.mean(centers_x))
                avg_cy = float(np.mean(centers_y))
                max_conf = max(confidences) if confidences else 0
                width = current["bbox"][2] - current["bbox"][0]
                height = current["bbox"][3] - current["bbox"][1]

                merged_results.append(
                    {
                        "class": cls,
                        "confidence": round(max_conf, 4),
                        "bbox": [
                            round(avg_cx - width / 2, 2),
                            round(avg_cy - height / 2, 2),
                            round(avg_cx + width / 2, 2),
                            round(avg_cy + height / 2, 2),
                        ],
                    }
                )
            else:
                merged_results.append(current)

            class_detections = remaining

    return merged_results
