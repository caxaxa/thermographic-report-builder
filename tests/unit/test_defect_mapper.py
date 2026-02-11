"""Tests for DefectMapper — panel grid construction and defect assignment.

Tests use the public map_defects_to_panels() API with synthetic bounding boxes
to verify column/row grouping, tracker detection, and nearest-panel assignment.
"""
import pytest
from unittest.mock import MagicMock
from thermographic_report_builder.models.defect import BoundingBox


@pytest.fixture
def mock_geo_converter():
    """Mock PixelToLatLonConverter that returns fixed coordinates."""
    converter = MagicMock()
    converter.pixel_to_lonlat.return_value = (-47.0, -15.0)
    return converter


@pytest.fixture
def mapper_vertical(mock_geo_converter):
    """DefectMapper configured for vertical (column-first) layout."""
    from thermographic_report_builder.processing.defect_mapper import DefectMapper

    return DefectMapper(
        image_width=1000,
        image_height=2000,
        geo_converter=mock_geo_converter,
        is_horizontal=False,
        is_double_tracker=False,
    )


@pytest.fixture
def mapper_horizontal(mock_geo_converter):
    """DefectMapper configured for horizontal (row-first) layout."""
    from thermographic_report_builder.processing.defect_mapper import DefectMapper

    return DefectMapper(
        image_width=2000,
        image_height=1000,
        geo_converter=mock_geo_converter,
        is_horizontal=True,
        is_double_tracker=False,
    )


class TestMapDefectsVertical:
    def test_single_column_panels(self, mapper_vertical):
        """Panels aligned vertically should form a single column."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=103, top=200, width=50, height=80, label="solarpanels"),
            BoundingBox(left=101, top=300, width=50, height=80, label="solarpanels"),
        ]
        grid = mapper_vertical.map_defects_to_panels(panels, [])
        assert len(grid) == 3
        # All panels should be in column 1
        columns = set(panel.column for panel in grid.values())
        assert len(columns) == 1

    def test_two_columns(self, mapper_vertical):
        """Panels with wide X gap should form two columns."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=100, top=200, width=50, height=80, label="solarpanels"),
            BoundingBox(left=300, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=300, top=200, width=50, height=80, label="solarpanels"),
        ]
        grid = mapper_vertical.map_defects_to_panels(panels, [])
        assert len(grid) == 4
        columns = set(panel.column for panel in grid.values())
        assert len(columns) == 2

    def test_defect_assigned_to_nearest_panel(self, mapper_vertical):
        """A defect should be mapped to the closest panel."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=100, top=400, width=50, height=80, label="solarpanels"),
        ]
        defects = [
            BoundingBox(left=110, top=120, width=10, height=15, label="hotspots"),
        ]
        grid = mapper_vertical.map_defects_to_panels(panels, defects)

        panels_with_defects = [p for p in grid.values() if p.has_defects]
        assert len(panels_with_defects) == 1
        assert panels_with_defects[0].defect_count == 1

    def test_no_defects_all_panels_clean(self, mapper_vertical):
        """With no defects, all panels should be defect-free."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=100, top=300, width=50, height=80, label="solarpanels"),
        ]
        grid = mapper_vertical.map_defects_to_panels(panels, [])

        for panel in grid.values():
            assert panel.has_defects is False

    def test_multiple_defect_types(self, mapper_vertical):
        """Different defect types should be routed to correct panel lists."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
        ]
        defects = [
            BoundingBox(left=110, top=120, width=10, height=15, label="hotspots"),
            BoundingBox(left=115, top=125, width=8, height=12, label="faultydiodes"),
        ]
        grid = mapper_vertical.map_defects_to_panels(panels, defects)

        panel = list(grid.values())[0]
        assert len(panel.hotspots) == 1
        assert len(panel.faulty_diodes) == 1
        assert panel.defect_count == 2


class TestMapDefectsHorizontal:
    def test_single_row_panels(self, mapper_horizontal):
        """Panels aligned horizontally should form a single row."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=200, top=103, width=50, height=80, label="solarpanels"),
            BoundingBox(left=300, top=101, width=50, height=80, label="solarpanels"),
        ]
        grid = mapper_horizontal.map_defects_to_panels(panels, [])
        assert len(grid) == 3
        # All panels should be in row 1
        rows = set(panel.row for panel in grid.values())
        assert len(rows) == 1

    def test_two_rows(self, mapper_horizontal):
        """Panels with wide Y gap should form two rows."""
        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=200, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=100, top=400, width=50, height=80, label="solarpanels"),
            BoundingBox(left=200, top=400, width=50, height=80, label="solarpanels"),
        ]
        grid = mapper_horizontal.map_defects_to_panels(panels, [])
        assert len(grid) == 4
        rows = set(panel.row for panel in grid.values())
        assert len(rows) == 2


class TestEmptyInputs:
    def test_empty_panels_list(self, mapper_vertical):
        """With no panels, should handle gracefully."""
        grid = mapper_vertical.map_defects_to_panels([], [])
        assert len(grid) == 0

    def test_defects_with_no_panels(self, mapper_vertical):
        """Defects without panels should not crash."""
        defects = [
            BoundingBox(left=110, top=120, width=10, height=15, label="hotspots"),
        ]
        grid = mapper_vertical.map_defects_to_panels([], defects)
        assert len(grid) == 0
