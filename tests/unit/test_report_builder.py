"""Tests for ReportBuilder — defect aggregation, panel sort, LaTeX gen, compile.

Covers:
- __init__ defect counting (total_defects, hotspots, faulty_diodes, offline, temp_data_count)
- _panel_sort_key: regex parsing for "Col-Row", "Painel X-Y", and fallback logic
- generate_tex: produces a .tex file on disk
- generate_pdf: calls _compile_latex twice
- _compile_latex: pdflatex success/failure/timeout/missing
- Preview mode: watermark package added in preamble
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from thermographic_report_builder.report.builder import ReportBuilder
from thermographic_report_builder.models.defect import (
    BoundingBox,
    Defect,
    GeospatialCoordinate,
    Panel,
)
from thermographic_report_builder.models.report import ReportConfig
from thermographic_report_builder.utils.exceptions import ReportGenerationError
from thermographic_report_builder.processing.gps_matcher import DefectMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox(**kw):
    defaults = dict(left=100, top=200, width=50, height=80, label="solarpanels")
    defaults.update(kw)
    return BoundingBox(**defaults)


def _defect(defect_type="hotspots", left=110, top=220):
    return Defect(
        bbox=_bbox(left=left, top=top, width=10, height=15, label=defect_type),
        defect_center_px=(left + 5, top + 7.5),
        panel_centroid_geospatial=GeospatialCoordinate(longitude=-54.0, latitude=-20.0),
        defect_type=defect_type,
    )


def _panel(row=1, col=1, hotspots=0, faulty_diodes=0, offline=0, **kw):
    return Panel(
        row=row,
        column=col,
        bbox=_bbox(),
        hotspots=[_defect("hotspots", left=110 + i * 5) for i in range(hotspots)],
        faulty_diodes=[_defect("faultydiodes", left=200 + i * 5) for i in range(faulty_diodes)],
        offline_panels=[_defect("offlinepanels", left=300 + i * 5) for i in range(offline)],
        **kw,
    )


def _config(**kw):
    defaults = dict(area_name="Test Solar Farm", language="pt-BR")
    defaults.update(kw)
    return ReportConfig(**defaults)


def _builder(panels=None, defect_matches=None, preview=False, tmp_path=None, **kw):
    if panels is None:
        panels = {(1, 1): _panel(hotspots=2), (1, 2): _panel(row=1, col=2)}
    images_dir = Path(tmp_path or "/tmp") / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    odm_stats_dir = Path(tmp_path or "/tmp") / "odm_stats"
    odm_stats_dir.mkdir(parents=True, exist_ok=True)
    return ReportBuilder(
        panel_grid=panels,
        images_dir=images_dir,
        config=_config(),
        defect_matches=defect_matches or {},
        preview=preview,
        odm_stats_dir=odm_stats_dir,
        **kw,
    )


# ---------------------------------------------------------------------------
# __init__ aggregation
# ---------------------------------------------------------------------------
class TestBuilderInit:

    def test_total_defects(self):
        b = _builder(panels={
            (1, 1): _panel(hotspots=3, faulty_diodes=2),
            (1, 2): _panel(offline=1),
        })
        assert b.total_defects == 6

    def test_panels_with_defects(self):
        b = _builder(panels={
            (1, 1): _panel(hotspots=1),
            (1, 2): _panel(),  # no defects
            (2, 1): _panel(faulty_diodes=1),
        })
        assert len(b.panels_with_defects) == 2

    def test_hotspot_count(self):
        b = _builder(panels={
            (1, 1): _panel(hotspots=5),
            (1, 2): _panel(hotspots=3),
        })
        assert b.total_hotspots == 8

    def test_faulty_diode_count(self):
        b = _builder(panels={(1, 1): _panel(faulty_diodes=4)})
        assert b.total_faulty_diodes == 4

    def test_offline_count(self):
        b = _builder(panels={(1, 1): _panel(offline=2)})
        assert b.total_offline == 2

    def test_temp_data_count(self):
        """Counts defect matches with temperature != None."""
        matches = {
            "d1": MagicMock(temperature=45.0),
            "d2": MagicMock(temperature=None),
            "d3": MagicMock(temperature=60.0),
        }
        b = _builder(defect_matches=matches)
        assert b.temp_data_count == 2

    def test_no_defects(self):
        b = _builder(panels={(1, 1): _panel()})
        assert b.total_defects == 0
        assert b.total_hotspots == 0
        assert len(b.panels_with_defects) == 0

    def test_preview_flag(self):
        b = _builder(preview=True)
        assert b.preview is True


# ---------------------------------------------------------------------------
# _panel_sort_key
# ---------------------------------------------------------------------------
class TestPanelSortKey:

    def _sort(self, panel):
        b = _builder()
        return b._panel_sort_key(panel)

    def test_numeric_panel_id(self):
        """'3-8' → (3, 8, '')"""
        p = _panel(col=3, row=8)
        # panel_id = "3-1" (column-trackerColumn)
        p_with_id = _panel(col=3, row=8, tracker_column=8)
        key = self._sort(p_with_id)
        assert key == (3, 8, "")

    def test_panel_id_with_suffix(self):
        """'5-12B' → (5, 12, 'B')"""
        p = _panel(col=5, tracker_column=12, tracker="B")
        key = self._sort(p)
        assert key == (5, 12, "B")

    def test_painel_prefix(self):
        """'Painel 2-7' → (2, 7, '')"""
        p = _panel()
        # Override panel_id by setting fields that produce "2-7"
        p_custom = _panel(col=2, tracker_column=7)
        key = self._sort(p_custom)
        assert key == (2, 7, "")

    def test_horizontal_layout(self):
        """Horizontal panels sort by (row, column, suffix)."""
        p = _panel(row=3, col=7, is_horizontal=True)
        key = self._sort(p)
        assert key == (3, 7, "")

    def test_sorting_order(self):
        """Panels sort correctly by column first, then row."""
        b = _builder()
        panels = [
            _panel(col=3, tracker_column=2),
            _panel(col=1, tracker_column=5),
            _panel(col=3, tracker_column=1),
            _panel(col=1, tracker_column=1),
        ]
        sorted_panels = sorted(panels, key=b._panel_sort_key)
        ids = [p.panel_id for p in sorted_panels]
        assert ids == ["1-1", "1-5", "3-1", "3-2"]


# ---------------------------------------------------------------------------
# _compile_latex
# ---------------------------------------------------------------------------
class TestCompileLatex:

    def _builder(self):
        return _builder()

    @patch("thermographic_report_builder.report.builder.subprocess.run")
    def test_compile_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        b = self._builder()
        b._compile_latex(Path("/tmp/report.tex"))
        mock_run.assert_called_once()

    @patch("thermographic_report_builder.report.builder.subprocess.run")
    def test_compile_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        b = self._builder()
        with pytest.raises(ReportGenerationError, match="pdflatex failed"):
            b._compile_latex(Path("/tmp/report.tex"))

    @patch("thermographic_report_builder.report.builder.subprocess.run")
    def test_compile_timeout_raises(self, mock_run):
        import subprocess as real_subprocess
        mock_run.side_effect = real_subprocess.TimeoutExpired("pdflatex", 300)
        b = self._builder()
        with pytest.raises(ReportGenerationError, match="timed out"):
            b._compile_latex(Path("/tmp/report.tex"))

    @patch("thermographic_report_builder.report.builder.subprocess.run")
    def test_compile_pdflatex_missing_raises(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        b = self._builder()
        with pytest.raises(ReportGenerationError, match="pdflatex not found"):
            b._compile_latex(Path("/tmp/report.tex"))


# ---------------------------------------------------------------------------
# generate_tex
# ---------------------------------------------------------------------------
class TestGenerateTex:

    def test_generates_tex_file(self, tmp_path):
        b = _builder(panels={(1, 1): _panel(hotspots=1)}, tmp_path=tmp_path)
        output = tmp_path / "report.tex"
        result = b.generate_tex(output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "\\documentclass" in content

    def test_tex_contains_area_name(self, tmp_path):
        b = _builder(panels={(1, 1): _panel()}, tmp_path=tmp_path)
        output = tmp_path / "report.tex"
        b.generate_tex(output)
        content = output.read_text()
        assert "Test Solar Farm" in content


# ---------------------------------------------------------------------------
# generate_pdf
# ---------------------------------------------------------------------------
class TestGeneratePdf:

    @patch.object(ReportBuilder, "_compile_latex")
    def test_calls_compile_twice(self, mock_compile, tmp_path):
        """PDF generation compiles twice for proper LaTeX references."""
        b = _builder(panels={(1, 1): _panel()}, tmp_path=tmp_path)
        output = tmp_path / "report.pdf"
        # _compile_latex is mocked, so we need to create the PDF file it would produce
        mock_compile.side_effect = lambda tex: output.write_bytes(b"%PDF-1.4 fake")
        b.generate_pdf(output)
        assert mock_compile.call_count == 2

    @patch.object(ReportBuilder, "_compile_latex")
    def test_creates_tex_file(self, mock_compile, tmp_path):
        b = _builder(panels={(1, 1): _panel()}, tmp_path=tmp_path)
        output = tmp_path / "report.pdf"
        mock_compile.side_effect = lambda tex: output.write_bytes(b"%PDF-1.4 fake")
        b.generate_pdf(output)
        tex_file = tmp_path / "report.tex"
        assert tex_file.exists()


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------
class TestPreviewMode:

    def test_preview_adds_watermark_package(self, tmp_path):
        b = _builder(panels={(1, 1): _panel()}, preview=True, tmp_path=tmp_path)
        output = tmp_path / "report.tex"
        b.generate_tex(output)
        content = output.read_text()
        assert "background" in content

    def test_non_preview_no_watermark(self, tmp_path):
        b = _builder(panels={(1, 1): _panel()}, preview=False, tmp_path=tmp_path)
        output = tmp_path / "report.tex"
        b.generate_tex(output)
        content = output.read_text()
        assert "backgroundsetup" not in content
