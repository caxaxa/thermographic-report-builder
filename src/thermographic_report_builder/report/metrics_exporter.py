"""Export defect metrics to JSON and CSV formats."""

import csv
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from ..models.defect import Panel
from ..models.report import DefectMetrics
from ..models.annotation_manifest import AnnotationManifest
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Defect class display names (Portuguese)
DEFECT_DISPLAY_NAMES = {
    "hotspots": "Ponto Quente (Hot Spot)",
    "faultydiodes": "Diodo de Bypass Queimado",
    "offlinepanels": "Painel Desligado",
}


def calculate_metrics(panel_grid: Dict[Tuple[int, int], Panel]) -> DefectMetrics:
    """
    Calculate defect metrics from panel grid.

    Args:
        panel_grid: Dictionary of panels

    Returns:
        DefectMetrics object
    """
    total_panels = len(panel_grid)
    panels_with_defects = sum(1 for p in panel_grid.values() if p.has_defects)

    hotspots_count = sum(len(p.hotspots) for p in panel_grid.values())
    faulty_diodes_count = sum(len(p.faulty_diodes) for p in panel_grid.values())
    offline_panels_count = sum(len(p.offline_panels) for p in panel_grid.values())
    total_defects = hotspots_count + faulty_diodes_count + offline_panels_count

    return DefectMetrics(
        total_panels=total_panels,
        panels_with_defects=panels_with_defects,
        total_defects=total_defects,
        hotspots_count=hotspots_count,
        faulty_diodes_count=faulty_diodes_count,
        offline_panels_count=offline_panels_count,
    )


def export_metrics_json(
    panel_grid: Dict[Tuple[int, int], Panel],
    output_path: Path,
    include_details: bool = True,
    ortho_width: Optional[int] = None,
    ortho_height: Optional[int] = None,
    ortho_scale_factor: float = 0.25,
) -> Path:
    """
    Export metrics to JSON file.

    Args:
        panel_grid: Dictionary of panels
        output_path: Path to save JSON file
        include_details: If True, include per-panel details
        ortho_width: Original orthophoto width (before scaling)
        ortho_height: Original orthophoto height (before scaling)
        ortho_scale_factor: Scale factor applied to create ortho.png

    Returns:
        Path to saved file
    """
    logger.info(f"Exporting metrics to JSON: {output_path}")

    metrics = calculate_metrics(panel_grid)

    data = {
        "export_date": datetime.utcnow().isoformat(),
        "summary": metrics.to_dict(),
    }

    # Add ortho.png dimensions for interactive defect map
    if ortho_width and ortho_height:
        data["ortho_dimensions"] = {
            "original_width": ortho_width,
            "original_height": ortho_height,
            "scale_factor": ortho_scale_factor,
            "scaled_width": int(ortho_width * ortho_scale_factor),
            "scaled_height": int(ortho_height * ortho_scale_factor),
        }

    if include_details:
        # Add per-panel breakdown
        panels_list = []
        for (col, row), panel in sorted(panel_grid.items()):
            if panel.has_defects:
                panels_list.append(
                    {
                        "panel_id": panel.panel_id,
                        "column": col,
                        "row": row,
                        "defect_count": panel.defect_count,
                        "hotspots": len(panel.hotspots),
                        "faulty_diodes": len(panel.faulty_diodes),
                        "offline_panels": len(panel.offline_panels),
                    }
                )
        data["panels_with_defects"] = panels_list

        # Add per-defect details for interactive map
        defects_list = []
        defect_counter = 0
        for (col, row), panel in sorted(panel_grid.items()):
            for defect_type_attr, defect_class in [
                ("hotspots", "hotspots"),
                ("faulty_diodes", "faultydiodes"),
                ("offline_panels", "offlinepanels"),
            ]:
                defects = getattr(panel, defect_type_attr, [])
                for defect in defects:
                    defect_counter += 1
                    defect_entry = {
                        "defect_id": f"defect_{defect_counter}",
                        "panel_id": panel.panel_id,
                        "column": col,
                        "row": row,
                        "defect_class": defect_class,
                        "display_class": DEFECT_DISPLAY_NAMES.get(defect_class, defect_class),
                        "latitude": defect.panel_centroid_geospatial.latitude,
                        "longitude": defect.panel_centroid_geospatial.longitude,
                        # Bounding box in original coordinates
                        "bbox_original": {
                            "left": round(defect.bbox.left, 1),
                            "top": round(defect.bbox.top, 1),
                            "width": round(defect.bbox.width, 1),
                            "height": round(defect.bbox.height, 1),
                        },
                    }
                    # Add scaled bbox for ortho.png if dimensions provided
                    if ortho_width and ortho_height:
                        defect_entry["bbox_scaled"] = {
                            "left": round(defect.bbox.left * ortho_scale_factor, 1),
                            "top": round(defect.bbox.top * ortho_scale_factor, 1),
                            "width": round(defect.bbox.width * ortho_scale_factor, 1),
                            "height": round(defect.bbox.height * ortho_scale_factor, 1),
                        }
                    defects_list.append(defect_entry)

        data["defects"] = defects_list
        logger.info(f"Added {len(defects_list)} defects to metrics JSON")

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported metrics JSON: {output_path.stat().st_size / 1_000:.1f} KB")
    return output_path


def export_metrics_csv(
    panel_grid: Dict[Tuple[int, int], Panel],
    output_path: Path,
) -> Path:
    """
    Export per-panel metrics to CSV file.

    Args:
        panel_grid: Dictionary of panels
        output_path: Path to save CSV file

    Returns:
        Path to saved file
    """
    logger.info(f"Exporting metrics to CSV: {output_path}")

    # CSV columns
    fieldnames = [
        "panel_id",
        "column",
        "row",
        "has_defects",
        "total_defects",
        "hotspots",
        "faulty_diodes",
        "offline_panels",
        "longitude",
        "latitude",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (col, row), panel in sorted(panel_grid.items()):
            # Get geospatial coordinates from first defect (if any)
            lon, lat = None, None
            if panel.has_defects:
                first_defect = panel.all_defects()[0]
                lon = first_defect.panel_centroid_geospatial.longitude
                lat = first_defect.panel_centroid_geospatial.latitude

            writer.writerow(
                {
                    "panel_id": panel.panel_id,
                    "column": col,
                    "row": row,
                    "has_defects": panel.has_defects,
                    "total_defects": panel.defect_count,
                    "hotspots": len(panel.hotspots),
                    "faulty_diodes": len(panel.faulty_diodes),
                    "offline_panels": len(panel.offline_panels),
                    "longitude": lon if lon else "",
                    "latitude": lat if lat else "",
                }
            )

    logger.info(f"Exported metrics CSV: {output_path.stat().st_size / 1_000:.1f} KB")
    return output_path


def export_panel_grid_json(
    panel_grid: Dict[Tuple[int, int], Panel],
    output_path: Path,
    ortho_width: int,
    ortho_height: int,
    ortho_scale_factor: float = 0.25,
    images_dir: Optional[Path] = None,
    annotation_manifest: Optional[AnnotationManifest] = None,
) -> Path:
    """
    Export complete panel grid to JSON for interactive defect map.

    This file contains all panels with their bounding boxes and defects,
    using the report's panel_id format (e.g., "3-45", "1-2B") so the
    interactive map matches exactly what appears in the PDF report.

    For hotspots with thermal analysis, the annotated image (with temperature
    markers) is used for the hover display if available.

    Args:
        panel_grid: Dictionary of panels from DefectMapper
        output_path: Path to save JSON file
        ortho_width: Original orthophoto width in pixels
        ortho_height: Original orthophoto height in pixels
        ortho_scale_factor: Scale factor used to create ortho.png
        images_dir: Directory containing report images (for checking annotated versions)
        annotation_manifest: Optional annotation manifest with temperature data

    Returns:
        Path to saved file
    """
    logger.info(f"Exporting panel grid to JSON: {output_path}")

    # Build index of temperature data from annotation manifest
    # Key: (panel_id, defect_type, defect_index) -> annotation entry
    temp_index: Dict[str, dict] = {}
    if annotation_manifest:
        for entry in annotation_manifest.annotations:
            # Key by defect_id for exact lookup
            temp_index[entry.defect_id] = {
                "hot_temp": entry.hot_point.temp,
                "cold_temp": entry.cold_point.temp,
                "delta_t": entry.delta_t,
                "severity": entry.severity,
            }
        logger.info(f"Built temperature index with {len(temp_index)} entries")

    # Build list of panels with defects (for the interactive map)
    panels_with_defects = []

    for (col, row), panel in sorted(panel_grid.items()):
        if not panel.has_defects:
            continue

        # Get geospatial coords from first defect
        first_defect = panel.all_defects()[0]
        lat = first_defect.panel_centroid_geospatial.latitude
        lon = first_defect.panel_centroid_geospatial.longitude

        # Build defects list for this panel
        defects = []
        for defect_type_attr, defect_class in [
            ("hotspots", "hotspots"),
            ("faulty_diodes", "faultydiodes"),
            ("offline_panels", "offlinepanels"),
        ]:
            defect_list = getattr(panel, defect_type_attr, [])
            for defect_idx, defect in enumerate(defect_list, 1):
                # Build defect_id to look up temperature data
                # Format: hotspots_(panel_id)_defeito{index} or hotspots_(panel_id) for single
                if len(defect_list) > 1:
                    defect_id = f"{defect_class}_({panel.panel_id})_defeito{defect_idx}"
                else:
                    defect_id = f"{defect_class}_({panel.panel_id})_defeito1"

                defect_entry = {
                    "defect_class": defect_class,
                    "display_class": DEFECT_DISPLAY_NAMES.get(defect_class, defect_class),
                    "bbox": {
                        "left": round(defect.bbox.left, 1),
                        "top": round(defect.bbox.top, 1),
                        "width": round(defect.bbox.width, 1),
                        "height": round(defect.bbox.height, 1),
                    },
                    "bbox_scaled": {
                        "left": round(defect.bbox.left * ortho_scale_factor, 1),
                        "top": round(defect.bbox.top * ortho_scale_factor, 1),
                        "width": round(defect.bbox.width * ortho_scale_factor, 1),
                        "height": round(defect.bbox.height * ortho_scale_factor, 1),
                    },
                }

                # Add temperature data if available from annotation manifest
                if defect_id in temp_index:
                    temp_data = temp_index[defect_id]
                    defect_entry["hot_temp"] = temp_data["hot_temp"]
                    defect_entry["cold_temp"] = temp_data["cold_temp"]
                    defect_entry["delta_t"] = round(temp_data["delta_t"], 1)
                    defect_entry["severity"] = temp_data["severity"]

                defects.append(defect_entry)

        # Build thermal image filenames for each defect type present in this panel
        # For hotspots: prefer zoombox image (raw with crop indicator) for hover display
        # Fallback chain: _zoombox.jpg -> _annotated.jpg -> regular .jpg
        thermal_images = {}
        if panel.hotspots:
            # Check for zoombox version first (raw image with zoom area marked)
            zoombox_filename = f"hotspots_({panel.panel_id})_zoombox.jpg"
            annotated_filename = f"hotspots_({panel.panel_id})_annotated.jpg"
            regular_filename = f"hotspots_({panel.panel_id}).jpg"
            if images_dir and (images_dir / zoombox_filename).exists():
                thermal_images["hotspots"] = zoombox_filename
                logger.debug(f"Using zoombox thermal image for {panel.panel_id}")
            elif images_dir and (images_dir / annotated_filename).exists():
                thermal_images["hotspots"] = annotated_filename
                logger.debug(f"Using annotated thermal image for {panel.panel_id}")
            else:
                thermal_images["hotspots"] = regular_filename
        if panel.faulty_diodes:
            thermal_images["faultydiodes"] = f"faultydiodes_({panel.panel_id}).jpg"
        if panel.offline_panels:
            thermal_images["offlinepanels"] = f"offlinepanels_({panel.panel_id}).jpg"

        panels_with_defects.append({
            "panel_id": panel.panel_id,
            "column": panel.column,
            "row": panel.row,
            "tracker": panel.tracker,
            "tracker_row": panel.tracker_column,
            "latitude": lat,
            "longitude": lon,
            "panel_bbox": {
                "left": round(panel.bbox.left, 1),
                "top": round(panel.bbox.top, 1),
                "width": round(panel.bbox.width, 1),
                "height": round(panel.bbox.height, 1),
            },
            "panel_bbox_scaled": {
                "left": round(panel.bbox.left * ortho_scale_factor, 1),
                "top": round(panel.bbox.top * ortho_scale_factor, 1),
                "width": round(panel.bbox.width * ortho_scale_factor, 1),
                "height": round(panel.bbox.height * ortho_scale_factor, 1),
            },
            "defect_count": panel.defect_count,
            "hotspots_count": len(panel.hotspots),
            "faulty_diodes_count": len(panel.faulty_diodes),
            "offline_panels_count": len(panel.offline_panels),
            "thermal_images": thermal_images,
            "defects": defects,
        })

    data = {
        "export_date": datetime.utcnow().isoformat(),
        "ortho_dimensions": {
            "original_width": ortho_width,
            "original_height": ortho_height,
            "scale_factor": ortho_scale_factor,
            "scaled_width": int(ortho_width * ortho_scale_factor),
            "scaled_height": int(ortho_height * ortho_scale_factor),
        },
        "summary": {
            "total_panels": len(panel_grid),
            "panels_with_defects": len(panels_with_defects),
            "total_defects": sum(p["defect_count"] for p in panels_with_defects),
        },
        "panels": panels_with_defects,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported panel grid JSON with {len(panels_with_defects)} panels: {output_path.stat().st_size / 1_000:.1f} KB")
    return output_path
