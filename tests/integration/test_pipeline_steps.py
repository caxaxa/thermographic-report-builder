"""Integration tests — verify pipeline step composition and data flow.

Tests run the DefectMapper → metrics_exporter → AnnotationManifest
chain with synthetic data to verify the full pipeline logic works
end-to-end without real S3 or image files.
"""
import json
import csv
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def mock_geo_converter():
    """Mock PixelToLatLonConverter that returns fixed coordinates."""
    converter = MagicMock()
    converter.pixel_to_lonlat.return_value = (-47.0, -15.0)
    return converter


def _make_mapper(geo_converter):
    """Create a DefectMapper with test-friendly defaults."""
    from thermographic_report_builder.processing.defect_mapper import DefectMapper

    return DefectMapper(
        image_width=1000,
        image_height=2000,
        geo_converter=geo_converter,
    )


class TestDefectMapperToMetrics:
    """DefectMapper output feeds directly into metrics_exporter."""

    def test_panel_grid_produces_valid_metrics_json(
        self, sample_panel_boxes, sample_defect_boxes, tmp_path, mock_geo_converter
    ):
        from thermographic_report_builder.report.metrics_exporter import export_metrics_json

        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)

        output_path = tmp_path / "metrics.json"
        result = export_metrics_json(panel_grid, output_path)

        assert result == output_path
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert "summary" in data
        assert data["summary"]["total_panels"] == len(sample_panel_boxes)

    def test_panel_grid_produces_valid_metrics_csv(
        self, sample_panel_boxes, sample_defect_boxes, tmp_path, mock_geo_converter
    ):
        from thermographic_report_builder.report.metrics_exporter import export_metrics_csv

        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)

        output_path = tmp_path / "metrics.csv"
        result = export_metrics_csv(panel_grid, output_path)

        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(sample_panel_boxes)
        assert "panel_id" in rows[0]

    def test_metrics_defect_counts_match_mapper_output(
        self, sample_panel_boxes, sample_defect_boxes, mock_geo_converter
    ):
        from thermographic_report_builder.report.metrics_exporter import calculate_metrics

        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)

        metrics = calculate_metrics(panel_grid)
        assert metrics.total_panels == len(sample_panel_boxes)
        assert metrics.total_defects == len(sample_defect_boxes)
        assert metrics.hotspots_count + metrics.faulty_diodes_count + metrics.offline_panels_count == len(
            sample_defect_boxes
        )

    def test_panel_grid_json_export(
        self, sample_panel_boxes, sample_defect_boxes, tmp_path, mock_geo_converter
    ):
        from thermographic_report_builder.report.metrics_exporter import export_panel_grid_json

        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)

        output_path = tmp_path / "panel_grid.json"
        result = export_panel_grid_json(
            panel_grid, output_path, ortho_width=8000, ortho_height=6000
        )

        assert result == output_path
        data = json.loads(output_path.read_text())
        assert "panels" in data
        # Only panels with defects are included in panel_grid JSON
        assert len(data["panels"]) == len(sample_defect_boxes)


class TestAnnotationManifest:
    """AnnotationManifest serialization roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        from thermographic_report_builder.models.annotation_manifest import (
            AnnotationManifest,
            AnnotationEntry,
            AnnotationPoint,
        )

        manifest = AnnotationManifest(
            project_id="test-project-001",
            user_id="test-user-001",
            generated_at="2024-06-15T12:00:00Z",
            annotations=[
                AnnotationEntry(
                    defect_id="hotspots_(1-1)_defeito1",
                    panel_id="1-1",
                    defect_type="hotspots",
                    defect_index=1,
                    raw_image_path="s3://bucket/raw/DJI_001.JPG",
                    raw_image_name="DJI_001.JPG",
                    annotated_image="hotspots_1-1_defeito1.jpg",
                    hot_point=AnnotationPoint(x=320, y=256, temp=85.3),
                    cold_point=AnnotationPoint(x=100, y=100, temp=45.1),
                    delta_t=40.2,
                    severity="CRITICAL",
                ),
            ],
        )

        path = tmp_path / "manifest.json"
        manifest.save(path)
        assert path.exists()

        loaded = AnnotationManifest.load(path)
        assert loaded.project_id == "test-project-001"
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].delta_t == 40.2
        assert loaded.annotations[0].severity == "CRITICAL"
        assert loaded.annotations[0].hot_point.temp == 85.3

    def test_manifest_with_multiple_annotations(self, tmp_path):
        from thermographic_report_builder.models.annotation_manifest import (
            AnnotationManifest,
            AnnotationEntry,
            AnnotationPoint,
        )

        entries = [
            AnnotationEntry(
                defect_id=f"hotspots_(1-{i})_defeito1",
                panel_id=f"1-{i}",
                defect_type="hotspots",
                defect_index=1,
                raw_image_path=f"s3://bucket/raw/DJI_{i:03d}.JPG",
                raw_image_name=f"DJI_{i:03d}.JPG",
                annotated_image=f"hotspots_1-{i}_defeito1.jpg",
                hot_point=AnnotationPoint(x=320, y=256, temp=70.0 + i),
                cold_point=AnnotationPoint(x=100, y=100, temp=40.0),
                delta_t=30.0 + i,
                severity="CRITICAL" if i > 3 else "ATTENTION",
            )
            for i in range(1, 6)
        ]

        manifest = AnnotationManifest(
            project_id="test-project-001",
            user_id="test-user-001",
            generated_at="2024-06-15T12:00:00Z",
            annotations=entries,
        )

        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = AnnotationManifest.load(path)
        assert len(loaded.annotations) == 5
        severities = [a.severity for a in loaded.annotations]
        assert severities.count("CRITICAL") == 2
        assert severities.count("ATTENTION") == 3


class TestOverrideManifest:
    """OverrideManifest save/load and lookup."""

    def test_save_and_load_roundtrip(self, tmp_path):
        from thermographic_report_builder.models.annotation_manifest import (
            OverrideManifest,
            AnnotationOverride,
            AnnotationPoint,
        )

        override = OverrideManifest(
            project_id="test-project-001",
            user_id="test-user-001",
            created_at="2024-06-16T10:00:00Z",
            overrides=[
                AnnotationOverride(
                    defect_id="hotspots_(1-1)_defeito1",
                    hot_point=AnnotationPoint(x=350, y=270, temp=90.0),
                    cold_point=None,
                ),
            ],
        )

        path = tmp_path / "overrides.json"
        override.save(path)
        loaded = OverrideManifest.load(path)
        assert len(loaded.overrides) == 1
        assert loaded.overrides[0].hot_point.temp == 90.0
        assert loaded.overrides[0].cold_point is None

    def test_get_override_finds_matching(self, tmp_path):
        from thermographic_report_builder.models.annotation_manifest import (
            OverrideManifest,
            AnnotationOverride,
            AnnotationPoint,
        )

        manifest = OverrideManifest(
            project_id="test-project-001",
            user_id="test-user-001",
            created_at="2024-06-16T10:00:00Z",
            overrides=[
                AnnotationOverride(
                    defect_id="hotspots_(1-1)_defeito1",
                    hot_point=AnnotationPoint(x=350, y=270, temp=90.0),
                ),
                AnnotationOverride(
                    defect_id="hotspots_(1-2)_defeito1",
                    hot_point=AnnotationPoint(x=200, y=150, temp=75.0),
                ),
            ],
        )

        result = manifest.get_override("hotspots_(1-2)_defeito1")
        assert result is not None
        assert result.hot_point.temp == 75.0

    def test_get_override_returns_none_when_not_found(self):
        from thermographic_report_builder.models.annotation_manifest import (
            OverrideManifest,
        )

        manifest = OverrideManifest(
            project_id="test-project-001",
            user_id="test-user-001",
            created_at="2024-06-16T10:00:00Z",
            overrides=[],
        )

        assert manifest.get_override("nonexistent") is None


class TestEndToEndDataFlow:
    """Verify data flows correctly through multiple pipeline steps."""

    def test_mapper_to_metrics_to_json_roundtrip(
        self, sample_panel_boxes, sample_defect_boxes, tmp_path, mock_geo_converter
    ):
        """Full chain: raw bboxes → DefectMapper → metrics → JSON → verify."""
        from thermographic_report_builder.report.metrics_exporter import (
            calculate_metrics,
            export_metrics_json,
        )

        # Step 1: Map defects to panels
        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)
        assert len(panel_grid) == len(sample_panel_boxes)

        # Step 2: Calculate metrics
        metrics = calculate_metrics(panel_grid)
        assert metrics.total_panels == 3
        assert metrics.total_defects == 2

        # Step 3: Export to JSON
        json_path = tmp_path / "full_metrics.json"
        export_metrics_json(panel_grid, json_path, include_details=True)

        # Step 4: Verify JSON structure
        data = json.loads(json_path.read_text())
        assert data["summary"]["total_panels"] == 3
        assert data["summary"]["total_defects"] == 2
        assert data["summary"]["panels_with_defects"] <= data["summary"]["total_panels"]

    def test_panels_without_defects_are_tracked(self, tmp_path, mock_geo_converter):
        """Panels without defects should still appear in the grid and metrics."""
        from thermographic_report_builder.models.defect import BoundingBox
        from thermographic_report_builder.report.metrics_exporter import calculate_metrics

        panels = [
            BoundingBox(left=100, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=200, top=100, width=50, height=80, label="solarpanels"),
            BoundingBox(left=300, top=100, width=50, height=80, label="solarpanels"),
        ]
        # Only one defect — assigned to nearest panel
        defects = [
            BoundingBox(left=105, top=110, width=10, height=10, label="hotspots"),
        ]

        mapper = _make_mapper(mock_geo_converter)
        panel_grid = mapper.map_defects_to_panels(panels, defects)
        metrics = calculate_metrics(panel_grid)

        assert metrics.total_panels == 3
        assert metrics.panels_with_defects == 1
        assert metrics.hotspots_count == 1
