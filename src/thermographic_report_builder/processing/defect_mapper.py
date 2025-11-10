"""Map defects to solar panels and create panel grid."""

from typing import Dict, Tuple
import numpy as np

from ..models.defect import Panel, Defect, BoundingBox, GeospatialCoordinate
from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter

logger = get_logger(__name__)


class DefectMapper:
    """Maps defects to solar panels and creates structured panel grid."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        geo_converter: PixelToLatLonConverter,
    ):
        """
        Initialize defect mapper.

        Args:
            image_width: Orthophoto width in pixels
            image_height: Orthophoto height in pixels
            geo_converter: Helper to convert pixels into geodetic coordinates
        """
        self.img_width = image_width
        self.img_height = image_height
        self.geo_converter = geo_converter

    def map_defects_to_panels(
        self, panel_boxes: list[BoundingBox], defect_boxes: list[BoundingBox]
    ) -> Dict[Tuple[int, int], Panel]:
        """
        Map defects to nearest panels and create panel grid.

        Args:
            panel_boxes: List of solar panel bounding boxes
            defect_boxes: List of defect bounding boxes

        Returns:
            Dictionary mapping (col, row) -> Panel
        """
        logger.info(f"Mapping {len(defect_boxes)} defects to {len(panel_boxes)} panels")

        # 1. Group panels into rows
        panels_sorted = sorted(panel_boxes, key=lambda b: (b.top, b.left))
        rows = self._group_into_rows(panels_sorted)

        # 2. Create panel grid with (col, row) coordinates
        panel_grid: Dict[Tuple[int, int], Panel] = {}
        for row_idx, row_panels in enumerate(rows, start=1):
            for col_idx, panel_bbox in enumerate(row_panels, start=1):
                panel_grid[(col_idx, row_idx)] = Panel(
                    column=col_idx, row=row_idx, bbox=panel_bbox
                )

        max_cols = max((len(r) for r in rows), default=0)
        logger.info(f"Created panel grid: {len(rows)} rows, max {max_cols} columns")

        # 3. Assign each defect to nearest panel
        defects_assigned = 0
        for defect_bbox in defect_boxes:
            nearest_panel = self._find_nearest_panel(defect_bbox, panel_grid)
            if nearest_panel:
                defect = self._create_defect(defect_bbox, nearest_panel)
                self._add_defect_to_panel(nearest_panel, defect)
                defects_assigned += 1

        logger.info(f"Assigned {defects_assigned} defects to panels")
        return panel_grid

    def _group_into_rows(self, panels: list[BoundingBox]) -> list[list[BoundingBox]]:
        """
        Group panels into rows based on vertical proximity.

        Args:
            panels: Sorted list of panel bounding boxes

        Returns:
            List of rows, each containing panel bounding boxes
        """
        if not panels:
            return []

        rows = []
        current_row = [panels[0]]
        ref_y = panels[0].top

        for panel in panels[1:]:
            # Same row if top values are within half panel height
            if abs(panel.top - ref_y) <= current_row[0].height * 0.5:
                current_row.append(panel)
            else:
                # Start new row
                rows.append(sorted(current_row, key=lambda p: p.left))
                current_row = [panel]
                ref_y = panel.top

        if current_row:
            rows.append(sorted(current_row, key=lambda p: p.left))

        return rows

    def _find_nearest_panel(
        self, defect: BoundingBox, panels: Dict[Tuple[int, int], Panel]
    ) -> Panel | None:
        """
        Find panel with center closest to defect center.

        Args:
            defect: Defect bounding box
            panels: Dictionary of panels

        Returns:
            Nearest panel or None
        """
        defect_center = defect.center
        min_dist = float("inf")
        nearest = None

        for panel in panels.values():
            panel_center = panel.bbox.center
            dist = np.hypot(defect_center[0] - panel_center[0], defect_center[1] - panel_center[1])

            if dist < min_dist:
                min_dist = dist
                nearest = panel

        return nearest

    def _create_defect(self, bbox: BoundingBox, panel: Panel) -> Defect:
        """
        Create Defect object with geospatial coordinates.

        Args:
            bbox: Defect bounding box
            panel: Parent panel

        Returns:
            Defect instance
        """
        # Get panel center in pixel coordinates
        panel_center_px = panel.bbox.center

        # Convert to geospatial coordinates (always WGS84)
        lon, lat = self.geo_converter.pixel_to_lonlat(panel_center_px)

        return Defect(
            bbox=bbox,
            defect_center_px=bbox.center,
            panel_centroid_geospatial=GeospatialCoordinate(longitude=lon, latitude=lat),
            defect_type=bbox.label,
        )

    def _add_defect_to_panel(self, panel: Panel, defect: Defect) -> None:
        """
        Add defect to appropriate panel list based on type.

        Args:
            panel: Target panel
            defect: Defect to add
        """
        if defect.is_hotspot:
            panel.hotspots.append(defect)
        elif defect.is_faulty_diode:
            panel.faulty_diodes.append(defect)
        elif defect.is_offline_panel:
            panel.offline_panels.append(defect)
