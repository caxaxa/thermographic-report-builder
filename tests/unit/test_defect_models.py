"""Tests for defect.py Pydantic models — BoundingBox, Panel, DefectType.

Bugs in these models propagate to every report.
"""
import pytest
from pydantic import ValidationError

from thermographic_report_builder.models.defect import (
    BoundingBox,
    Defect,
    DefectType,
    GeospatialCoordinate,
    Panel,
)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_valid_box(self):
        bb = BoundingBox(left=10, top=20, width=50, height=80, label="hotspots")
        assert bb.left == 10
        assert bb.top == 20
        assert bb.width == 50
        assert bb.height == 80

    def test_center_calculation(self):
        bb = BoundingBox(left=100, top=200, width=50, height=80, label="solarpanels")
        cx, cy = bb.center
        assert cx == pytest.approx(125.0)
        assert cy == pytest.approx(240.0)

    def test_right_edge(self):
        bb = BoundingBox(left=100, top=0, width=50, height=10, label="hotspots")
        assert bb.right == 150

    def test_bottom_edge(self):
        bb = BoundingBox(left=0, top=100, width=10, height=80, label="hotspots")
        assert bb.bottom == 180

    def test_area(self):
        bb = BoundingBox(left=0, top=0, width=50, height=80, label="hotspots")
        assert bb.area == 4000

    def test_zero_width_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(left=0, top=0, width=0, height=10, label="hotspots")

    def test_negative_left_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(left=-1, top=0, width=10, height=10, label="hotspots")

    def test_negative_top_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(left=0, top=-1, width=10, height=10, label="hotspots")


# ---------------------------------------------------------------------------
# DefectType enum
# ---------------------------------------------------------------------------

class TestDefectType:
    def test_hotspots_value(self):
        assert DefectType.HOTSPOTS.value == "hotspots"

    def test_faulty_diodes_value(self):
        assert DefectType.FAULTY_DIODES.value == "faultydiodes"

    def test_offline_panels_value(self):
        assert DefectType.OFFLINE_PANELS.value == "offlinepanels"

    def test_solar_panels_value(self):
        assert DefectType.SOLAR_PANELS.value == "solarpanels"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class TestPanel:
    def _make_panel(self, **kwargs):
        defaults = {
            "column": 3,
            "row": 1,
            "bbox": BoundingBox(left=100, top=200, width=50, height=80, label="solarpanels"),
        }
        defaults.update(kwargs)
        return Panel(**defaults)

    def test_panel_id_vertical_default(self):
        """Vertical layout: 'column-tracker_column' format."""
        panel = self._make_panel(column=3, tracker_column=45)
        assert panel.panel_id == "3-45"

    def test_panel_id_vertical_with_tracker(self):
        """With non-default tracker: 'column-tracker_column+tracker' format."""
        panel = self._make_panel(column=3, tracker_column=1, tracker="B")
        assert panel.panel_id == "3-1B"

    def test_panel_id_horizontal(self):
        """Horizontal layout: 'row-column' format."""
        panel = self._make_panel(row=1, column=3, is_horizontal=True)
        assert panel.panel_id == "1-3"

    def test_panel_id_horizontal_with_double_suffix(self):
        """Horizontal layout with double tracker suffix."""
        panel = self._make_panel(row=1, column=3, is_horizontal=True, double_suffix="B")
        assert panel.panel_id == "1-3B"

    def test_panel_id_horizontal_with_non_default_tracker(self):
        panel = self._make_panel(row=2, column=5, is_horizontal=True, tracker="C")
        assert panel.panel_id == "2-5C"

    def test_has_defects_false_when_empty(self):
        panel = self._make_panel()
        assert panel.has_defects is False

    def test_has_defects_true_with_hotspot(self):
        defect = Defect(
            bbox=BoundingBox(left=110, top=220, width=10, height=15, label="hotspots"),
            defect_center_px=(115, 227.5),
            panel_centroid_geospatial=GeospatialCoordinate(longitude=-47.0, latitude=-15.0),
            defect_type="hotspots",
        )
        panel = self._make_panel(hotspots=[defect])
        assert panel.has_defects is True
        assert panel.defect_count == 1

    def test_defect_count_sums_all_types(self):
        hotspot = Defect(
            bbox=BoundingBox(left=110, top=220, width=10, height=15, label="hotspots"),
            defect_center_px=(115, 227.5),
            panel_centroid_geospatial=GeospatialCoordinate(longitude=-47.0, latitude=-15.0),
            defect_type="hotspots",
        )
        diode = Defect(
            bbox=BoundingBox(left=120, top=230, width=8, height=12, label="faultydiodes"),
            defect_center_px=(124, 236),
            panel_centroid_geospatial=GeospatialCoordinate(longitude=-47.0, latitude=-15.0),
            defect_type="faultydiodes",
        )
        panel = self._make_panel(hotspots=[hotspot], faulty_diodes=[diode])
        assert panel.defect_count == 2
        assert len(panel.all_defects()) == 2

    def test_column_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._make_panel(column=0)


# ---------------------------------------------------------------------------
# Defect
# ---------------------------------------------------------------------------

class TestDefect:
    def _make_defect(self, defect_type="hotspots"):
        return Defect(
            bbox=BoundingBox(left=110, top=220, width=10, height=15, label=defect_type),
            defect_center_px=(115, 227.5),
            panel_centroid_geospatial=GeospatialCoordinate(longitude=-47.0, latitude=-15.0),
            defect_type=defect_type,
        )

    def test_is_hotspot(self):
        d = self._make_defect("hotspots")
        assert d.is_hotspot is True
        assert d.is_faulty_diode is False

    def test_is_faulty_diode(self):
        d = self._make_defect("faultydiodes")
        assert d.is_faulty_diode is True
        assert d.is_hotspot is False

    def test_is_offline_panel(self):
        d = self._make_defect("offlinepanels")
        assert d.is_offline_panel is True


class TestBoxSanitization:
    """Stray annotations must not abort a whole report (2026-08-29 incident:
    3 off-canvas boxes out of 1,512 failed a customer's report)."""

    def _payload(self, boxes):
        return {"boundingBox": {"boundingBoxes": boxes}}

    def _load(self, tmp_path, boxes):
        import json as _json
        from thermographic_report_builder.models.defect import DefectLabelsJSON
        p = tmp_path / "labels.json"
        p.write_text(_json.dumps(self._payload(boxes)))
        return DefectLabelsJSON.from_json_file(str(p))

    def test_drops_fully_offcanvas_box(self, tmp_path):
        boxes = [
            {"left": 10, "top": 10, "width": 50, "height": 50, "label": "hotspots"},
            {"left": -2440, "top": -2455, "width": 156, "height": 73, "label": "offlinepanels"},
        ]
        labels = self._load(tmp_path, boxes)
        assert len(labels.bounding_boxes) == 1
        assert labels.bounding_boxes[0].label == "hotspots"

    def test_clamps_partially_visible_box(self, tmp_path):
        # Left edge off-canvas by 20px: clamp to 0 and keep the right edge at 30.
        boxes = [{"left": -20, "top": -5, "width": 50, "height": 40, "label": "hotspots"}]
        labels = self._load(tmp_path, boxes)
        bb = labels.bounding_boxes[0]
        assert (bb.left, bb.width) == (0, 30)
        assert (bb.top, bb.height) == (0, 35)
        assert bb.right == 30

    def test_clean_boxes_untouched(self, tmp_path):
        boxes = [{"left": 5, "top": 6, "width": 7, "height": 8, "label": "default_panel"}]
        bb = self._load(tmp_path, boxes).bounding_boxes[0]
        assert (bb.left, bb.top, bb.width, bb.height) == (5, 6, 7, 8)


class TestThermalSuffixes:
    """Radiometric frames must be discovered regardless of drone vendor
    convention — XT2 ships _R.JPG, M30T ships _T.JPG (2026-08-29: an XT2
    flight produced zero temperatures because only _t.jpg was matched)."""

    def test_suffix_set_covers_known_conventions(self):
        from thermographic_report_builder.io.s3_client import THERMAL_SUFFIXES
        for name in ("DJI_0001_T.JPG", "DJI_0001_R.JPG", "dji_0002_t.jpeg", "x_r.jpeg"):
            assert name.lower().endswith(THERMAL_SUFFIXES), name

    def test_visual_frames_excluded(self):
        from thermographic_report_builder.io.s3_client import THERMAL_SUFFIXES
        for name in ("DJI_0001.jpg", "DJI_0001_W.JPG", "IMG_1234.JPG"):
            assert not name.lower().endswith(THERMAL_SUFFIXES), name
