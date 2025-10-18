"""Export defect metrics to JSON and CSV formats."""

import csv
import json
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

from ..models.defect import Panel
from ..models.report import DefectMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


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
) -> Path:
    """
    Export metrics to JSON file.

    Args:
        panel_grid: Dictionary of panels
        output_path: Path to save JSON file
        include_details: If True, include per-panel details

    Returns:
        Path to saved file
    """
    logger.info(f"Exporting metrics to JSON: {output_path}")

    metrics = calculate_metrics(panel_grid)

    data = {
        "export_date": datetime.utcnow().isoformat(),
        "summary": metrics.to_dict(),
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
