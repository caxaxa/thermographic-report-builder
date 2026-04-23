"""End-to-end test for the report builder main pipeline.

Tests the 11-step pipeline with mocked S3 I/O, mocked pdflatex,
and mocked GPSMatcher. Exercises the real DefectMapper, metrics
exporters, and report config assembly.

This is the most critical integration test — it verifies that the
pipeline orchestrator (main.py) correctly chains all processing
steps and produces the expected output files.
"""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from thermographic_report_builder.models.defect import BoundingBox


@pytest.fixture
def work_dir(tmp_path):
    """Create temporary work directory structure matching pipeline expectations."""
    work = tmp_path / "work"
    work.mkdir()
    images = work / "report_images"
    images.mkdir()
    raw = work / "raw_images"
    raw.mkdir()
    return work


@pytest.fixture
def mock_orthophoto(work_dir):
    """Create a minimal GeoTIFF-like file for the pipeline to load."""
    ortho_path = work_dir / "odm_orthophoto.tif"
    # Create a minimal 100x100 pixel image as numpy array
    ortho_path.write_bytes(b"\x00" * 1024)  # Placeholder - will be mocked during load
    return ortho_path


@pytest.fixture
def mock_defect_labels(work_dir):
    """Create a synthetic defect_labels.json matching inference output format."""
    labels = {
        "predictions": [
            {
                "class": "default_panel",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
            },
            {
                "class": "default_panel",
                "confidence": 0.93,
                "bbox": [100, 250, 200, 350],
            },
            {
                "class": "default_panel",
                "confidence": 0.91,
                "bbox": [100, 400, 200, 500],
            },
            {
                "class": "hotspots",
                "confidence": 0.88,
                "bbox": [120, 120, 180, 180],
            },
            {
                "class": "faultydiodes",
                "confidence": 0.82,
                "bbox": [110, 260, 190, 340],
            },
        ]
    }
    labels_path = work_dir / "defect_labels.json"
    labels_path.write_text(json.dumps(labels))
    return labels_path


@pytest.fixture
def mock_s3_client():
    """Create a fully mocked S3Client that returns test data."""
    s3 = MagicMock()

    # download_orthophoto_resampled returns (path, version)
    s3.download_orthophoto_resampled.side_effect = lambda path, **kw: (path, "1.6cm")
    s3.download_orthophoto.side_effect = lambda path: path
    s3.download_defect_labels.side_effect = lambda path: path
    s3.download_crop_annotation.return_value = None
    s3.download_crop_annotation_metadata.return_value = None
    s3.download_reconstruction_json.return_value = None
    s3.download_tracks_csv.return_value = None
    s3.download_dsm.return_value = None
    s3.download_textured_mesh.return_value = None
    s3.download_labeling_vec.return_value = None
    s3.download_reconstruction_nvm.return_value = None
    s3.download_source_map.return_value = None
    s3.download_odm_stats.return_value = None
    s3.download_annotation_overrides.return_value = False
    s3.download_tex_bundle.return_value = False

    # Upload methods return S3 URIs
    s3.upload_report.side_effect = lambda path, name: f"s3://test-reports/test-user/test-project/{name}"
    s3.upload_tex_bundle.return_value = "s3://test-reports/test-user/test-project/tex_bundle.tar.gz"
    s3.upload_file.side_effect = lambda path, key, **kw: f"s3://test-reports/{key}"

    return s3


@pytest.fixture
def mock_settings(work_dir):
    """Patch settings to use test work directory and minimal config."""
    settings_patches = {
        "work_dir": work_dir,
        "project_id": "test-project-001",
        "user_id": "test-user-001",
        "area_name": "Test Solar Farm",
        "client_name": "Test Client",
        "engineer_name": "Test Engineer",
        "location": "Test Location",
        "report_language": "pt-BR",
        "orthophoto_downscale_factor": 1.0,
        "crop_downscale_factor": 1.0,
        "crea_number": "12345678",
        "address": "Test Address, Brazil",
        "compile_only": False,
        "generate_preview_pdf": False,
        "enable_mesh_fallback": False,
        "log_level": "WARNING",
        "log_json": False,
    }
    return settings_patches


class TestMainPipelineE2E:
    """Test the main() pipeline function end-to-end."""

    @staticmethod
    def _ensure_main_imported():
        """Pre-import main module so patch() can resolve it.

        The module-level Settings() requires env vars that are only set
        by the autouse monkeypatch fixture, so we must import lazily.
        """
        import thermographic_report_builder.main  # noqa: F401

    def test_pipeline_runs_to_completion(
        self, work_dir, mock_defect_labels, mock_s3_client, mock_settings
    ):
        """Verify the full pipeline runs without error and produces expected outputs."""
        self._ensure_main_imported()
        images_dir = work_dir / "report_images"

        # Mock load_orthophoto to return synthetic raster data
        mock_transform = MagicMock()
        mock_transform.__mul__ = MagicMock(return_value=mock_transform)
        mock_crs = MagicMock()
        mock_crs.to_epsg.return_value = 4326

        # Mock the geo converter
        mock_geo_converter = MagicMock()
        mock_geo_converter.pixel_to_lonlat.return_value = (-47.0, -15.0)

        with (
            patch("thermographic_report_builder.main.S3Client", return_value=mock_s3_client),
            patch("thermographic_report_builder.main.load_orthophoto") as mock_load_ortho,
            patch("thermographic_report_builder.main.PixelToLatLonConverter", return_value=mock_geo_converter),
            patch("thermographic_report_builder.main.reverse_geocode", return_value="Test Location, Brazil"),
            patch("thermographic_report_builder.main.get_orthophoto_center", return_value=(-47.0, -15.0)),
            patch("thermographic_report_builder.main.annotate_orthophoto") as mock_annotate,
            patch("thermographic_report_builder.main.create_layer_image") as mock_layer,
            patch("thermographic_report_builder.main.create_dxf_layers") as mock_dxf,
            patch("thermographic_report_builder.main.crop_defect_regions") as mock_crop,
            patch("thermographic_report_builder.main.GPSMatcher") as MockGPSMatcher,
            patch("thermographic_report_builder.main.ReportBuilder") as MockReportBuilder,
            patch("thermographic_report_builder.main.settings") as mock_settings_obj,
            patch("thermographic_report_builder.main.setup_logging"),
            patch("thermographic_report_builder.main.log_alignment_config"),
            patch("thermographic_report_builder.main.CropTransform", return_value=MagicMock(is_horizontal=False, is_double_tracker=False)),
            patch("subprocess.run") as mock_subprocess,
        ):
            # Configure settings mock
            for k, v in mock_settings.items():
                setattr(mock_settings_obj, k, v)

            # Configure load_orthophoto to return (raster_data, transform, crs, (h, w))
            mock_raster = MagicMock()
            mock_load_ortho.return_value = (mock_raster, mock_transform, mock_crs, (1000, 2000))

            # Configure annotate_orthophoto to create a placeholder file
            def fake_annotate(ortho_path, panel_grid, output_path, **kw):
                output_path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
                return output_path
            mock_annotate.side_effect = fake_annotate

            # Configure create_layer_image
            def fake_layer(panel_grid, output_path, **kw):
                output_path.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)
                return output_path
            mock_layer.side_effect = fake_layer

            # Configure create_dxf_layers to write a placeholder
            mock_dxf.return_value = None

            # Configure crop_defect_regions to do nothing (no actual images to crop)
            mock_crop.return_value = {}

            # Configure GPSMatcher to return minimal matches
            mock_gps_instance = MockGPSMatcher.return_value
            mock_gps_instance.match_images_to_panels.return_value = (0, {})
            mock_gps_instance.export_annotation_manifest.return_value = MagicMock()

            # Configure ReportBuilder to write a placeholder .tex file
            mock_builder_instance = MockReportBuilder.return_value
            def fake_generate_tex(output_path):
                output_path.write_text(r"\documentclass{article}\begin{document}Test\end{document}")
                return output_path
            mock_builder_instance.generate_tex.side_effect = fake_generate_tex

            # Configure subprocess (pdflatex) to succeed and create PDF
            def fake_pdflatex(args, **kw):
                # Create the output PDF
                cwd = Path(kw.get("cwd", "."))
                for arg in args:
                    if arg.endswith(".tex"):
                        pdf_name = Path(arg).stem + ".pdf"
                        (cwd / pdf_name).write_bytes(b"%PDF-1.4\nFake PDF content")
                result = MagicMock()
                result.returncode = 0
                return result
            mock_subprocess.side_effect = fake_pdflatex

            # Import and run the pipeline
            from thermographic_report_builder.main import main
            exit_code = main()

        assert exit_code == 0

        # Verify S3 uploads happened
        assert mock_s3_client.upload_report.call_count >= 1
        assert mock_s3_client.upload_tex_bundle.call_count >= 1

    def test_pipeline_exports_metrics_files(
        self, work_dir, mock_defect_labels, mock_s3_client, mock_settings, sample_panel_boxes, sample_defect_boxes
    ):
        """Verify that metrics JSON, CSV, and panel_grid JSON are produced."""
        from thermographic_report_builder.report.metrics_exporter import (
            export_metrics_json,
            export_metrics_csv,
            export_panel_grid_json,
        )
        from thermographic_report_builder.processing.defect_mapper import DefectMapper

        # Create a real mapper with mock geo converter
        mock_geo = MagicMock()
        mock_geo.pixel_to_lonlat.return_value = (-47.0, -15.0)
        mapper = DefectMapper(
            image_width=1000, image_height=2000, geo_converter=mock_geo
        )
        panel_grid = mapper.map_defects_to_panels(sample_panel_boxes, sample_defect_boxes)

        # Export all metrics
        metrics_json_path = work_dir / "metrics.json"
        metrics_csv_path = work_dir / "metrics.csv"
        panel_grid_path = work_dir / "panel_grid.json"

        export_metrics_json(panel_grid, metrics_json_path)
        export_metrics_csv(panel_grid, metrics_csv_path)
        export_panel_grid_json(panel_grid, panel_grid_path, ortho_width=1000, ortho_height=2000)

        # Verify all files exist and are valid
        assert metrics_json_path.exists()
        assert metrics_csv_path.exists()
        assert panel_grid_path.exists()

        # Verify JSON content
        data = json.loads(metrics_json_path.read_text())
        assert "summary" in data
        assert data["summary"]["total_panels"] == len(sample_panel_boxes)
        assert data["summary"]["total_defects"] == len(sample_defect_boxes)

        # Verify CSV has header + rows
        csv_lines = metrics_csv_path.read_text().strip().split("\n")
        assert len(csv_lines) >= 2  # header + at least 1 data row

        # Verify panel_grid JSON
        grid_data = json.loads(panel_grid_path.read_text())
        assert isinstance(grid_data, (list, dict))

    def test_compile_only_mode(self, work_dir, mock_s3_client):
        """Verify compile-only mode downloads tex bundle and compiles PDF."""
        self._ensure_main_imported()
        # Create a fake tex file
        tex_path = work_dir / "report.tex"
        tex_path.write_text(r"\documentclass{article}\begin{document}Test\end{document}")

        # Mock S3 download_tex_bundle to return True (success)
        mock_s3_client.download_tex_bundle.return_value = True

        with (
            patch("thermographic_report_builder.main.S3Client", return_value=mock_s3_client),
            patch("thermographic_report_builder.main.settings") as mock_settings_obj,
            patch("thermographic_report_builder.main.setup_logging"),
            patch("thermographic_report_builder.main.log_alignment_config"),
            patch("subprocess.run") as mock_subprocess,
        ):
            mock_settings_obj.work_dir = work_dir
            mock_settings_obj.compile_only = True
            mock_settings_obj.log_level = "WARNING"
            mock_settings_obj.log_json = False
            mock_settings_obj.project_id = "test-project"
            mock_settings_obj.user_id = "test-user"

            # Subprocess creates the PDF
            def fake_pdflatex(args, **kw):
                cwd = Path(kw.get("cwd", "."))
                (cwd / "report.pdf").write_bytes(b"%PDF-1.4\nFake")
                result = MagicMock()
                result.returncode = 0
                return result
            mock_subprocess.side_effect = fake_pdflatex

            from thermographic_report_builder.main import main
            exit_code = main()

        assert exit_code == 0
        assert mock_s3_client.download_tex_bundle.called
        assert mock_s3_client.upload_report.call_count >= 1

    def test_compile_only_fails_without_bundle(self, work_dir, mock_s3_client):
        """Verify compile-only mode returns 1 when tex bundle download fails."""
        self._ensure_main_imported()
        mock_s3_client.download_tex_bundle.return_value = False

        with (
            patch("thermographic_report_builder.main.S3Client", return_value=mock_s3_client),
            patch("thermographic_report_builder.main.settings") as mock_settings_obj,
            patch("thermographic_report_builder.main.setup_logging"),
            patch("thermographic_report_builder.main.log_alignment_config"),
        ):
            mock_settings_obj.work_dir = work_dir
            mock_settings_obj.compile_only = True
            mock_settings_obj.log_level = "WARNING"
            mock_settings_obj.log_json = False
            mock_settings_obj.project_id = "test-project"
            mock_settings_obj.user_id = "test-user"

            from thermographic_report_builder.main import main
            exit_code = main()

        assert exit_code == 1
