"""Map defects to solar panels and create panel grid."""

from typing import Dict, List, Tuple
import numpy as np

from ..models.defect import Panel, Defect, BoundingBox, GeospatialCoordinate
from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter

logger = get_logger(__name__)

# Tracker separation threshold: if vertical gap between panels in a column is >= this many panel heights,
# start a new tracker section (e.g., 1, 2, 3 -> 1B, 2B, 3B)
TRACKER_GAP_THRESHOLD = 3.0

# Column grouping tolerance: panels within this many panel widths horizontally are in the same column
COLUMN_TOLERANCE = 1.0


# Row grouping tolerance for horizontal layout: panels within this many panel heights vertically are in same row
ROW_TOLERANCE = 1.0

# Double tracker detection: if vertical gap between panels is less than this fraction of panel height,
# they are considered "glued" (A=top, B=bottom of same tracker)
DOUBLE_TRACKER_GAP_THRESHOLD = 0.5


class DefectMapper:
    """Maps defects to solar panels and creates structured panel grid."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        geo_converter: PixelToLatLonConverter,
        is_horizontal: bool = False,
        is_double_tracker: bool = False,
    ):
        """
        Initialize defect mapper.

        Args:
            image_width: Orthophoto width in pixels
            image_height: Orthophoto height in pixels
            geo_converter: Helper to convert pixels into geodetic coordinates
            is_horizontal: If True, use row-first numbering (Row-Column format)
            is_double_tracker: If True, detect glued panel pairs (A=top, B=bottom)
        """
        self.img_width = image_width
        self.img_height = image_height
        self.geo_converter = geo_converter
        self.is_horizontal = is_horizontal
        self.is_double_tracker = is_double_tracker
        logger.info(f"DefectMapper: is_horizontal={is_horizontal}, is_double_tracker={is_double_tracker}")

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

        # Log input panel statistics for debugging
        if panel_boxes:
            widths = [b.width for b in panel_boxes]
            heights = [b.height for b in panel_boxes]
            lefts = [b.left for b in panel_boxes]
            unique_lefts = len(set(round(l, 1) for l in lefts))
            logger.info(
                "Panel stats: count=%d, width=[min=%.1f, median=%.1f, max=%.1f], "
                "height=[min=%.1f, median=%.1f, max=%.1f], unique_lefts=%d",
                len(panel_boxes),
                min(widths), float(np.median(widths)), max(widths),
                min(heights), float(np.median(heights)), max(heights),
                unique_lefts,
            )

        # Use panels directly - alignment is now done interactively in annotation tool
        logger.info("Using pre-aligned panels from annotation tool (no batch alignment)")
        panels_to_use = panel_boxes

        # Get median panel dimensions for grouping tolerances
        median_width = float(np.median([p.width for p in panels_to_use])) if panels_to_use else 100.0
        median_height = float(np.median([p.height for p in panels_to_use])) if panels_to_use else 100.0

        # Choose layout strategy based on configuration
        if self.is_horizontal:
            # Horizontal layout: group by rows first (top to bottom), then columns (left to right)
            logger.info("Using HORIZONTAL layout (row-first numbering)")
            rows = self._group_into_rows(panels_to_use, median_height)
            logger.info(f"Grouped panels into {len(rows)} rows")
            panel_grid = self._build_panel_grid_row_first(rows, median_width)
            max_cols = max((len(r) for r in rows), default=0)
            logger.info(f"Created panel grid: {len(rows)} rows, max {max_cols} columns per row")
        else:
            # Vertical layout (default): group by columns first (left to right), then rows (top to bottom)
            logger.info("Using VERTICAL layout (column-first numbering)")
            columns = self._group_into_columns(panels_to_use, median_width)
            logger.info(f"Grouped panels into {len(columns)} columns")
            panel_grid = self._build_panel_grid_column_first(columns, median_height)
            max_rows = max((len(c) for c in columns), default=0)
            logger.info(f"Created panel grid: {len(columns)} columns, max {max_rows} rows per column")

        # Verify panel grid contains aligned panels
        if panel_grid:
            grid_widths = [p.bbox.width for p in panel_grid.values()]
            grid_heights = [p.bbox.height for p in panel_grid.values()]
            grid_lefts = [p.bbox.left for p in panel_grid.values()]
            unique_lefts = len(set(round(l, 1) for l in grid_lefts))
            logger.info(
                "PANEL_GRID bbox stats: count=%d, width=[min=%.1f, median=%.1f, max=%.1f], "
                "height=[min=%.1f, median=%.1f, max=%.1f], unique_lefts=%d",
                len(panel_grid),
                min(grid_widths), float(np.median(grid_widths)), max(grid_widths),
                min(grid_heights), float(np.median(grid_heights)), max(grid_heights),
                unique_lefts,
            )
            # Log first 5 grid panels' exact coords
            for i, (key, p) in enumerate(list(panel_grid.items())[:5]):
                logger.info(
                    "PANEL_GRID[%s] left=%.1f, top=%.1f, width=%.1f, height=%.1f",
                    key, p.bbox.left, p.bbox.top, p.bbox.width, p.bbox.height
                )

        # 3. Assign defects to panels
        # Note: Micro-alignment is DISABLED because panels are now pre-aligned
        # interactively in the annotation tool. The old _reinforce_alignment_with_defects
        # was shifting panels and breaking the user's careful alignment.
        assignments = self._assign_defects_to_panels(panel_grid, defect_boxes, reset=True)
        logger.info(f"Assigned {assignments} defects to panels (using pre-aligned coordinates)")

        panels_with_defects = sum(1 for panel in panel_grid.values() if panel.has_defects)
        total_defects = sum(panel.defect_count for panel in panel_grid.values())
        logger.info(
            "Final panel stats: %d panels, %d with defects, %d total defects",
            len(panel_grid),
            panels_with_defects,
            total_defects,
        )

        return panel_grid

    def _group_into_columns(
        self, panels: list[BoundingBox], median_width: float
    ) -> List[List[BoundingBox]]:
        """
        Group panels into columns based on horizontal proximity.

        Panels within COLUMN_TOLERANCE * median_width horizontally are in the same column.

        Args:
            panels: List of panel bounding boxes
            median_width: Median panel width for tolerance calculation

        Returns:
            List of columns, each containing panel bounding boxes sorted by Y (top to bottom)
        """
        if not panels:
            return []

        # Sort by left position (X coordinate)
        sorted_by_x = sorted(panels, key=lambda p: p.left)

        tolerance_x = median_width * COLUMN_TOLERANCE
        columns: List[List[BoundingBox]] = []
        current_col = [sorted_by_x[0]]
        ref_x = sorted_by_x[0].left

        for panel in sorted_by_x[1:]:
            # Same column if X positions are within tolerance
            if abs(panel.left - ref_x) <= tolerance_x:
                current_col.append(panel)
            else:
                # Start new column - sort current by Y before adding
                columns.append(sorted(current_col, key=lambda p: p.top))
                current_col = [panel]
                ref_x = panel.left

        if current_col:
            columns.append(sorted(current_col, key=lambda p: p.top))

        logger.info(
            "Grouped %d panels into %d columns (tolerance=%.1f px)",
            len(panels), len(columns), tolerance_x
        )
        for i, col in enumerate(columns[:5]):
            x_vals = [p.left for p in col]
            logger.debug(
                "Column %d: %d panels, X range=[%.0f-%.0f]",
                i + 1, len(col), min(x_vals), max(x_vals)
            )

        return columns

    def _build_panel_grid_column_first(
        self,
        columns: List[List[BoundingBox]],
        median_height: float,
    ) -> Dict[Tuple[int, int], Panel]:
        """
        Create panel grid with column-first numbering.

        For each column (left to right):
        - Number panels 1, 2, 3... from top to bottom
        - If vertical gap >= TRACKER_GAP_THRESHOLD * median_height, start new tracker (B, C, ...)

        Panel ID format:
        - Tracker A: "col-row" e.g., "3-45"
        - Tracker B+: "colB-row" e.g., "1B-12"

        Args:
            columns: List of columns, each containing panel bounding boxes sorted by Y
            median_height: Median panel height for tracker gap detection

        Returns:
            Dictionary mapping (column, row) -> Panel
        """
        panel_grid: Dict[Tuple[int, int], Panel] = {}

        for col_idx, col_panels in enumerate(columns, start=1):
            if not col_panels:
                continue

            # Panels should already be sorted by Y (top to bottom)
            tracker_label = "A"
            row_in_tracker = 0
            prev_bottom = None

            for panel_bbox in col_panels:
                # Check for tracker gap (vertical separation)
                if prev_bottom is not None:
                    gap = panel_bbox.top - prev_bottom
                    gap_panels = gap / median_height

                    if gap_panels >= TRACKER_GAP_THRESHOLD:
                        # Start new tracker section
                        tracker_label = chr(ord(tracker_label) + 1)
                        row_in_tracker = 0
                        logger.debug(
                            "Column %d: Tracker break at gap %.1f panels, starting tracker %s",
                            col_idx, gap_panels, tracker_label
                        )

                row_in_tracker += 1
                prev_bottom = panel_bbox.top + panel_bbox.height

                # Sequential row within this column
                seq_row = len([k for k in panel_grid.keys() if k[0] == col_idx]) + 1

                panel_grid[(col_idx, seq_row)] = Panel(
                    column=col_idx,
                    row=seq_row,
                    bbox=panel_bbox,
                    tracker=tracker_label,
                    tracker_column=row_in_tracker,  # Actually row within tracker for column-first
                )

        # Log tracker summary
        trackers_found = set()
        for panel in panel_grid.values():
            trackers_found.add(panel.tracker)
        if len(trackers_found) > 1:
            logger.info("Found %d tracker sections: %s", len(trackers_found), sorted(trackers_found))

        return panel_grid

    def _group_into_rows(
        self, panels: list[BoundingBox], median_height: float
    ) -> List[List[BoundingBox]]:
        """
        Group panels into rows based on vertical proximity (for horizontal layout).

        Panels within ROW_TOLERANCE * median_height vertically are in the same row.

        Args:
            panels: List of panel bounding boxes
            median_height: Median panel height for tolerance calculation

        Returns:
            List of rows, each containing panel bounding boxes sorted by X (left to right)
        """
        if not panels:
            return []

        # Sort by top position (Y coordinate)
        sorted_by_y = sorted(panels, key=lambda p: p.top)

        tolerance_y = median_height * ROW_TOLERANCE
        rows: List[List[BoundingBox]] = []
        current_row = [sorted_by_y[0]]
        ref_y = sorted_by_y[0].top

        for panel in sorted_by_y[1:]:
            # Same row if Y positions are within tolerance
            if abs(panel.top - ref_y) <= tolerance_y:
                current_row.append(panel)
            else:
                # Start new row - sort current by X before adding
                rows.append(sorted(current_row, key=lambda p: p.left))
                current_row = [panel]
                ref_y = panel.top

        if current_row:
            rows.append(sorted(current_row, key=lambda p: p.left))

        logger.info(
            "Grouped %d panels into %d rows (tolerance=%.1f px)",
            len(panels), len(rows), tolerance_y
        )
        for i, row in enumerate(rows[:5]):
            y_vals = [p.top for p in row]
            logger.debug(
                "Row %d: %d panels, Y range=[%.0f-%.0f]",
                i + 1, len(row), min(y_vals), max(y_vals)
            )

        return rows

    def _build_panel_grid_row_first(
        self,
        rows: List[List[BoundingBox]],
        median_width: float,
    ) -> Dict[Tuple[int, int], Panel]:
        """
        Create panel grid with row-first numbering (for horizontal layout).

        For each row (top to bottom):
        - Number panels 1, 2, 3... from left to right

        For double tracker:
        - Detect "glued" panel pairs (small vertical gap)
        - Top panel gets suffix 'A', bottom panel gets suffix 'B'

        Panel ID format:
        - Single tracker: "row-col" e.g., "1-3"
        - Double tracker: "row-colA" / "row-colB" e.g., "1-3A", "1-3B"

        Args:
            rows: List of rows, each containing panel bounding boxes sorted by X
            median_width: Median panel width for double tracker detection

        Returns:
            Dictionary mapping (row, column) -> Panel
        """
        panel_grid: Dict[Tuple[int, int], Panel] = {}

        if self.is_double_tracker:
            # For double tracker, we need to pair panels and assign A/B suffixes
            panel_grid = self._build_double_tracker_grid(rows, median_width)
        else:
            # Simple row-first numbering
            for row_idx, row_panels in enumerate(rows, start=1):
                if not row_panels:
                    continue

                for col_idx, panel_bbox in enumerate(row_panels, start=1):
                    panel_grid[(row_idx, col_idx)] = Panel(
                        column=col_idx,
                        row=row_idx,
                        bbox=panel_bbox,
                        tracker="A",
                        tracker_column=col_idx,
                        is_horizontal=True,
                    )

        return panel_grid

    def _build_double_tracker_grid(
        self,
        rows: List[List[BoundingBox]],
        median_width: float,
    ) -> Dict[Tuple[int, int], Panel]:
        """
        Build panel grid for double tracker horizontal layout.

        For double trackers with horizontal layout:
        - Rows come in pairs (top row A, bottom row B - the "glued" panels)
        - Each pair forms one logical tracker row
        - Panel ID format: Row-ColumnA / Row-ColumnB

        Example: If we have 4 physical rows of panels (rows 1-2 are glued, rows 3-4 are glued):
        - Physical row 1, col 3 -> 1-3A
        - Physical row 2, col 3 -> 1-3B (glued below)
        - Physical row 3, col 3 -> 2-3A
        - Physical row 4, col 3 -> 2-3B (glued below)

        Args:
            rows: List of rows, each containing panel bounding boxes sorted by X
            median_width: Median panel width for column matching

        Returns:
            Dictionary mapping (row, column) -> Panel
        """
        panel_grid: Dict[Tuple[int, int], Panel] = {}

        # Process rows in pairs (A row, B row)
        tracker_row = 0
        row_idx = 0

        while row_idx < len(rows):
            row_a = rows[row_idx]

            # Check if there's a B row (glued row below)
            row_b = None
            if row_idx + 1 < len(rows):
                # Check if next row is "glued" (small vertical gap)
                if row_a and rows[row_idx + 1]:
                    first_a = row_a[0]
                    first_b = rows[row_idx + 1][0]
                    gap = first_b.top - (first_a.top + first_a.height)
                    median_height = (first_a.height + first_b.height) / 2

                    if gap < median_height * DOUBLE_TRACKER_GAP_THRESHOLD:
                        row_b = rows[row_idx + 1]
                        row_idx += 1  # Skip the B row in main loop

            tracker_row += 1

            # Process A row
            for col_idx, panel_bbox in enumerate(row_a, start=1):
                panel_grid[(tracker_row, col_idx * 2 - 1)] = Panel(
                    column=col_idx,
                    row=tracker_row,
                    bbox=panel_bbox,
                    tracker="A",
                    tracker_column=col_idx,
                    is_horizontal=True,
                    double_suffix="A",
                )

            # Process B row if exists
            if row_b:
                for col_idx, panel_bbox in enumerate(row_b, start=1):
                    panel_grid[(tracker_row, col_idx * 2)] = Panel(
                        column=col_idx,
                        row=tracker_row,
                        bbox=panel_bbox,
                        tracker="A",
                        tracker_column=col_idx,
                        is_horizontal=True,
                        double_suffix="B",
                    )

            row_idx += 1

        logger.info(f"Double tracker grid: {len(panel_grid)} panels in {tracker_row} tracker rows")
        return panel_grid

    def _assign_defects_to_panels(
        self,
        panel_grid: Dict[Tuple[int, int], Panel],
        defect_boxes: list[BoundingBox],
        reset: bool = True,
    ) -> int:
        """Assign every defect bounding box to the nearest panel."""
        if reset:
            for panel in panel_grid.values():
                panel.hotspots.clear()
                panel.faulty_diodes.clear()
                panel.offline_panels.clear()

        assignments = 0
        for defect_bbox in defect_boxes:
            nearest_panel = self._find_nearest_panel(defect_bbox, panel_grid)
            if nearest_panel:
                defect = self._create_defect(defect_bbox, nearest_panel)
                self._add_defect_to_panel(nearest_panel, defect)
                assignments += 1

        return assignments

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

    def _reinforce_alignment_with_defects(self, panel_grid: Dict[Tuple[int, int], Panel]) -> None:
        """Use defect centroids to micro-adjust entire rows and columns."""
        panel_widths = [panel.bbox.width for panel in panel_grid.values()]
        panel_heights = [panel.bbox.height for panel in panel_grid.values()]
        if not panel_widths or not panel_heights:
            return

        median_width = float(np.median(panel_widths))
        median_height = float(np.median(panel_heights))

        column_offsets: Dict[int, list[float]] = {}
        row_offsets: Dict[int, list[float]] = {}

        for panel in panel_grid.values():
            defects = panel.all_defects()
            if not defects:
                continue

            centers = np.array([defect.defect_center_px for defect in defects], dtype=float)
            mean_x, mean_y = centers.mean(axis=0)
            panel_center = panel.bbox.center

            delta_x = mean_x - panel_center[0]
            delta_y = mean_y - panel_center[1]

            column_offsets.setdefault(panel.column, []).append(delta_x)
            row_offsets.setdefault(panel.row, []).append(delta_y)

        if not column_offsets and not row_offsets:
            return

        max_dx = median_width * 0.2
        max_dy = median_height * 0.2

        column_shifts = {
            col: float(np.clip(np.median(shifts), -max_dx, max_dx))
            for col, shifts in column_offsets.items()
        }
        row_shifts = {
            row: float(np.clip(np.median(shifts), -max_dy, max_dy))
            for row, shifts in row_offsets.items()
        }

        logger.info(
            "Applying defect-informed offsets to %d columns and %d rows",
            len(column_shifts),
            len(row_shifts),
        )

        for panel in panel_grid.values():
            dx = column_shifts.get(panel.column, 0.0)
            dy = row_shifts.get(panel.row, 0.0)

            if dx == 0 and dy == 0:
                continue

            new_left = panel.bbox.left + dx
            new_top = panel.bbox.top + dy

            new_left = max(0.0, min(new_left, self.img_width - panel.bbox.width))
            new_top = max(0.0, min(new_top, self.img_height - panel.bbox.height))

            panel.bbox.left = new_left
            panel.bbox.top = new_top
