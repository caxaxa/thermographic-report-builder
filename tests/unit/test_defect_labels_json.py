"""Tests for DefectLabelsJSON parsing — the bridge between inference output and reports.

Covers both input formats (list vs dict), panel/defect filtering, and edge cases.
"""
import json
import pytest

from thermographic_report_builder.models.defect import DefectLabelsJSON, DefectType


# ---------------------------------------------------------------------------
# Helpers — write temp JSON files
# ---------------------------------------------------------------------------

def _write_json(tmp_path, data, filename="defect_labels.json"):
    p = tmp_path / filename
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# List format (masking stage output)
# ---------------------------------------------------------------------------

class TestListFormat:
    def test_parse_list_with_boxes(self, tmp_path):
        data = [
            {
                "boundingBox": {
                    "boundingBoxes": [
                        {"label": "hotspots", "left": 498, "top": 10641, "width": 7, "height": 19},
                        {"label": "solarpanels", "left": 100, "top": 200, "width": 50, "height": 80},
                    ]
                }
            }
        ]
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 2

    def test_parse_list_empty_boxes(self, tmp_path):
        data = [{"boundingBox": {"boundingBoxes": []}}]
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 0

    def test_parse_empty_list(self, tmp_path):
        path = _write_json(tmp_path, [])
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 0

    def test_parse_list_missing_bounding_box_key(self, tmp_path):
        data = [{"other": "data"}]
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 0

    def test_parse_list_uses_first_element_only(self, tmp_path):
        """Only the first element of the list is processed."""
        data = [
            {"boundingBox": {"boundingBoxes": [
                {"label": "hotspots", "left": 10, "top": 20, "width": 5, "height": 5},
            ]}},
            {"boundingBox": {"boundingBoxes": [
                {"label": "faultydiodes", "left": 30, "top": 40, "width": 5, "height": 5},
            ]}},
        ]
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 1
        assert result.bounding_boxes[0].label == "hotspots"


# ---------------------------------------------------------------------------
# Dict format (annotation tool / groundtruth bucket)
# ---------------------------------------------------------------------------

class TestDictFormat:
    def test_parse_dict_with_boxes(self, tmp_path):
        data = {
            "boundingBox": {
                "boundingBoxes": [
                    {"label": "faultydiodes", "left": 120, "top": 300, "width": 8, "height": 12},
                ]
            }
        }
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 1
        assert result.bounding_boxes[0].label == "faultydiodes"

    def test_parse_dict_empty_boxes(self, tmp_path):
        data = {"boundingBox": {"boundingBoxes": []}}
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 0

    def test_parse_dict_missing_bounding_box_key(self, tmp_path):
        data = {"metadata": "something"}
        path = _write_json(tmp_path, data)
        result = DefectLabelsJSON.from_json_file(path)
        assert len(result.bounding_boxes) == 0


# ---------------------------------------------------------------------------
# get_panels / get_defects filtering
# ---------------------------------------------------------------------------

class TestPanelDefectFiltering:
    @pytest.fixture
    def mixed_labels(self, tmp_path):
        """JSON with panels and defects of various types."""
        data = [
            {
                "boundingBox": {
                    "boundingBoxes": [
                        {"label": "solarpanels", "left": 100, "top": 200, "width": 50, "height": 80},
                        {"label": "default_panel", "left": 160, "top": 200, "width": 50, "height": 80},
                        {"label": "hotspots", "left": 110, "top": 220, "width": 10, "height": 15},
                        {"label": "faultydiodes", "left": 170, "top": 230, "width": 8, "height": 12},
                        {"label": "offlinepanels", "left": 100, "top": 290, "width": 50, "height": 80},
                    ]
                }
            }
        ]
        path = _write_json(tmp_path, data)
        return DefectLabelsJSON.from_json_file(path)

    def test_get_panels_includes_solarpanels(self, mixed_labels):
        panels = mixed_labels.get_panels()
        labels = [p.label for p in panels]
        assert "solarpanels" in labels

    def test_get_panels_includes_default_panel(self, mixed_labels):
        panels = mixed_labels.get_panels()
        labels = [p.label for p in panels]
        assert "default_panel" in labels

    def test_get_panels_count(self, mixed_labels):
        panels = mixed_labels.get_panels()
        assert len(panels) == 2

    def test_get_defects_excludes_panels(self, mixed_labels):
        defects = mixed_labels.get_defects()
        labels = [d.label for d in defects]
        assert "solarpanels" not in labels
        assert "default_panel" not in labels

    def test_get_defects_count(self, mixed_labels):
        defects = mixed_labels.get_defects()
        assert len(defects) == 3

    def test_get_defects_includes_all_defect_types(self, mixed_labels):
        defects = mixed_labels.get_defects()
        labels = {d.label for d in defects}
        assert labels == {"hotspots", "faultydiodes", "offlinepanels"}

    def test_panels_plus_defects_equals_total(self, mixed_labels):
        total = len(mixed_labels.bounding_boxes)
        panels = len(mixed_labels.get_panels())
        defects = len(mixed_labels.get_defects())
        assert panels + defects == total


# ---------------------------------------------------------------------------
# get_by_type
# ---------------------------------------------------------------------------

class TestGetByType:
    @pytest.fixture
    def multi_type_labels(self, tmp_path):
        data = [
            {
                "boundingBox": {
                    "boundingBoxes": [
                        {"label": "hotspots", "left": 10, "top": 20, "width": 5, "height": 5},
                        {"label": "hotspots", "left": 30, "top": 40, "width": 5, "height": 5},
                        {"label": "faultydiodes", "left": 50, "top": 60, "width": 5, "height": 5},
                        {"label": "solarpanels", "left": 100, "top": 200, "width": 50, "height": 80},
                    ]
                }
            }
        ]
        path = _write_json(tmp_path, data)
        return DefectLabelsJSON.from_json_file(path)

    def test_get_hotspots(self, multi_type_labels):
        hotspots = multi_type_labels.get_by_type(DefectType.HOTSPOTS)
        assert len(hotspots) == 2
        assert all(bb.label == "hotspots" for bb in hotspots)

    def test_get_faulty_diodes(self, multi_type_labels):
        diodes = multi_type_labels.get_by_type(DefectType.FAULTY_DIODES)
        assert len(diodes) == 1

    def test_get_offline_panels_returns_empty(self, multi_type_labels):
        offline = multi_type_labels.get_by_type(DefectType.OFFLINE_PANELS)
        assert len(offline) == 0

    def test_get_solar_panels(self, multi_type_labels):
        panels = multi_type_labels.get_by_type(DefectType.SOLAR_PANELS)
        assert len(panels) == 1
