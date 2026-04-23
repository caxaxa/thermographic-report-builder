"""Tests for thermal_annotator.py — LUT interpolation, colorbar, severity, palette detection.

Covers:
- _build_lut_256: shape, dtype, known endpoints, alias resolution, unknown fallback
- create_colorbar: shape, BGR ordering, top-hot/bottom-cold, width uniformity
- HotColdPoints.severity: boundary-value classification for ΔT thresholds
- detect_thermal_palette: subprocess success/failure/timeout
- ThermalAnnotator.find_hot_cold_points: synthetic temp array with mocked extractor
- ThermalAnnotator.save_annotated_image: cv2.imwrite success/failure
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from thermographic_report_builder.processing.thermal_annotator import (
    _build_lut_256,
    _DJI_PALETTE_CONTROL_POINTS,
    _PALETTE_ALIASES,
    create_colorbar,
    detect_thermal_palette,
    HotColdPoints,
    AnnotatedThermalImage,
    ThermalAnnotator,
)


# ---------------------------------------------------------------------------
# _build_lut_256
# ---------------------------------------------------------------------------
class TestBuildLut256:

    def test_returns_256x3_shape(self):
        lut = _build_lut_256("whitehot")
        assert lut.shape == (256, 3)

    def test_returns_uint8_dtype(self):
        lut = _build_lut_256("ironred")
        assert lut.dtype == np.uint8

    def test_whitehot_endpoints(self):
        lut = _build_lut_256("whitehot")
        np.testing.assert_array_equal(lut[0], [0, 0, 0])
        np.testing.assert_array_equal(lut[255], [255, 255, 255])

    def test_blackhot_endpoints(self):
        lut = _build_lut_256("blackhot")
        np.testing.assert_array_equal(lut[0], [255, 255, 255])
        np.testing.assert_array_equal(lut[255], [0, 0, 0])

    def test_alias_resolution(self):
        """'iron' should resolve to 'ironred'."""
        lut_alias = _build_lut_256("iron")
        lut_canonical = _build_lut_256("ironred")
        np.testing.assert_array_equal(lut_alias, lut_canonical)

    def test_all_aliases_resolve(self):
        for alias, canonical in _PALETTE_ALIASES.items():
            lut = _build_lut_256(alias)
            assert lut.shape == (256, 3), f"Alias '{alias}' -> '{canonical}' failed"

    def test_unknown_palette_falls_back_to_whitehot(self):
        lut_unknown = _build_lut_256("nonexistent_palette")
        lut_whitehot = _build_lut_256("whitehot")
        np.testing.assert_array_equal(lut_unknown, lut_whitehot)

    def test_all_known_palettes_produce_valid_luts(self):
        for name in _DJI_PALETTE_CONTROL_POINTS:
            lut = _build_lut_256(name)
            assert lut.shape == (256, 3), f"Palette '{name}' failed"
            assert lut.dtype == np.uint8
            # Values should be in [0, 255]
            assert lut.min() >= 0
            assert lut.max() <= 255

    def test_control_points_are_exact(self):
        """LUT values at control-point indices should match exactly."""
        for palette_name, points in _DJI_PALETTE_CONTROL_POINTS.items():
            lut = _build_lut_256(palette_name)
            for idx, r, g, b in points:
                np.testing.assert_array_equal(
                    lut[idx], [r, g, b],
                    err_msg=f"Palette '{palette_name}' control point {idx} mismatch",
                )


# ---------------------------------------------------------------------------
# create_colorbar
# ---------------------------------------------------------------------------
class TestCreateColorbar:

    def test_correct_shape(self):
        cb = create_colorbar(height=200, width=30, palette="whitehot")
        assert cb.shape == (200, 30, 3)

    def test_bgr_ordering(self):
        """Colorbar returns BGR (for OpenCV), not RGB."""
        # Whitehot LUT: index 255 = [255,255,255] RGB → BGR same
        # Use fulgurite where index 0 = [0,0,0] (same in both), index 255 = [255,255,255]
        # Use ironred where index 0 = [0,0,32] RGB → [32,0,0] BGR
        cb = create_colorbar(height=256, width=1, palette="ironred")
        # Bottom row = cold (LUT index 0). IronRed index 0 = (0, 0, 32) RGB → (32, 0, 0) BGR
        np.testing.assert_array_equal(cb[-1, 0], [32, 0, 0])

    def test_top_is_hot_bottom_is_cold(self):
        """Top of colorbar = hot (LUT index 255), bottom = cold (LUT index 0)."""
        cb = create_colorbar(height=256, width=1, palette="whitehot")
        # Whitehot: top = index 255 = (255,255,255) → BGR (255,255,255)
        np.testing.assert_array_equal(cb[0, 0], [255, 255, 255])
        # Bottom = index 0 = (0,0,0) → BGR (0,0,0)
        np.testing.assert_array_equal(cb[-1, 0], [0, 0, 0])

    def test_uniform_across_width(self):
        """All columns in the same row should be identical."""
        cb = create_colorbar(height=100, width=50, palette="rainbow1")
        for row in range(0, 100, 20):
            first_col = cb[row, 0]
            for col in range(1, 50):
                np.testing.assert_array_equal(cb[row, col], first_col)


# ---------------------------------------------------------------------------
# HotColdPoints.severity
# ---------------------------------------------------------------------------
class TestHotColdPointsSeverity:

    def _make(self, delta_t):
        return HotColdPoints(
            hot_x=0, hot_y=0, hot_temp=50.0,
            cold_x=10, cold_y=10, cold_temp=50.0 - delta_t,
            delta_t=delta_t,
        )

    def test_critical(self):
        assert self._make(30.0).severity == "CRITICAL"
        assert self._make(50.0).severity == "CRITICAL"

    def test_high(self):
        assert self._make(20.0).severity == "HIGH"
        assert self._make(29.9).severity == "HIGH"

    def test_medium(self):
        assert self._make(10.0).severity == "MEDIUM"
        assert self._make(19.9).severity == "MEDIUM"

    def test_low(self):
        assert self._make(5.0).severity == "LOW"
        assert self._make(9.9).severity == "LOW"

    def test_minimal(self):
        assert self._make(4.9).severity == "MINIMAL"
        assert self._make(0.0).severity == "MINIMAL"

    def test_boundary_exactly_30(self):
        assert self._make(30.0).severity == "CRITICAL"

    def test_boundary_just_below_30(self):
        assert self._make(29.99).severity == "HIGH"


# ---------------------------------------------------------------------------
# detect_thermal_palette
# ---------------------------------------------------------------------------
class TestDetectThermalPalette:

    @patch("thermographic_report_builder.processing.thermal_annotator.subprocess")
    def test_returns_palette_from_exiftool(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="IronRed\n"
        )
        result = detect_thermal_palette(Path("/fake/image.jpg"))
        assert result == "ironred"

    @patch("thermographic_report_builder.processing.thermal_annotator.subprocess")
    def test_returns_whitehot_on_failure(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(
            returncode=1, stdout=""
        )
        result = detect_thermal_palette(Path("/fake/image.jpg"))
        assert result == "whitehot"

    @patch("thermographic_report_builder.processing.thermal_annotator.subprocess")
    def test_returns_whitehot_on_empty_stdout(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="  \n"
        )
        result = detect_thermal_palette(Path("/fake/image.jpg"))
        assert result == "whitehot"

    @patch("thermographic_report_builder.processing.thermal_annotator.subprocess")
    def test_returns_whitehot_on_exception(self, mock_subprocess):
        mock_subprocess.run.side_effect = Exception("timeout")
        result = detect_thermal_palette(Path("/fake/image.jpg"))
        assert result == "whitehot"


# ---------------------------------------------------------------------------
# ThermalAnnotator.find_hot_cold_points (mocked extractor)
# ---------------------------------------------------------------------------
class TestFindHotColdPoints:

    def _make_extractor(self, temp_array):
        """Create a mock ThermalExtractor returning the given temperature array."""
        extractor = MagicMock()
        extractor.available = True
        extractor.get_full_temperature_array.return_value = temp_array
        return extractor

    def test_returns_none_when_extractor_unavailable(self):
        extractor = MagicMock()
        extractor.available = False
        annotator = ThermalAnnotator(thermal_extractor=extractor)
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=100, initial_y=100,
        )
        assert result is None

    def test_returns_none_when_temp_array_is_none(self):
        extractor = MagicMock()
        extractor.available = True
        extractor.get_full_temperature_array.return_value = None
        annotator = ThermalAnnotator(thermal_extractor=extractor)
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=100, initial_y=100,
        )
        assert result is None

    def test_finds_hottest_and_coldest_in_panel(self):
        """Synthetic 640x512 temp array with known hot/cold spots inside panel bbox."""
        temp_array = np.full((512, 640), 30.0, dtype=np.float64)
        # Place a hot spot at (320, 256) — center of the image
        temp_array[250:260, 315:325] = 80.0
        # Place a cold spot at (300, 280)
        temp_array[275:285, 295:305] = 15.0

        extractor = self._make_extractor(temp_array)
        annotator = ThermalAnnotator(thermal_extractor=extractor)

        # Use THERMAL_ONLY format (no visual-to-thermal conversion)
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=320, initial_y=256,
            panel_bbox_visual=(280, 240, 360, 300),
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
        )

        assert result is not None
        assert result.hot_temp == 80.0
        assert result.cold_temp == 15.0
        assert result.delta_t == 65.0
        assert result.severity == "CRITICAL"

    def test_south_flight_rotates_array(self):
        """South-facing flight should rotate the thermal array 180 degrees."""
        temp_array = np.full((512, 640), 30.0, dtype=np.float64)
        # Place hot spot at top-left (which becomes bottom-right after 180° rotation)
        temp_array[5:15, 5:15] = 90.0

        extractor = self._make_extractor(temp_array)
        annotator = ThermalAnnotator(thermal_extractor=extractor)

        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        # After rotation, hot spot moves to (625-635, 497-507)
        # Search near the rotated position
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=630, initial_y=502,
            panel_bbox_visual=(600, 480, 640, 512),
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
            flight_direction="south",
        )

        assert result is not None
        assert result.hot_temp == 90.0


# ---------------------------------------------------------------------------
# ThermalAnnotator.save_annotated_image
# ---------------------------------------------------------------------------
class TestSaveAnnotatedImage:

    def test_save_success(self):
        annotator = ThermalAnnotator(thermal_extractor=None)
        canvas = np.zeros((500, 600, 3), dtype=np.uint8)
        annotated = AnnotatedThermalImage(
            image=canvas, defect_temp=60.0, panel_avg_temp=35.0,
            delta_t=25.0, severity="HIGH",
        )

        with patch("thermographic_report_builder.processing.thermal_annotator.cv2") as mock_cv2:
            mock_cv2.IMWRITE_JPEG_QUALITY = 1
            result = annotator.save_annotated_image(annotated, Path("/tmp/test.jpg"))

        assert result is True
        mock_cv2.imwrite.assert_called_once()

    def test_save_failure(self):
        annotator = ThermalAnnotator(thermal_extractor=None)
        canvas = np.zeros((500, 600, 3), dtype=np.uint8)
        annotated = AnnotatedThermalImage(
            image=canvas, defect_temp=60.0, panel_avg_temp=35.0,
            delta_t=25.0, severity="HIGH",
        )

        with patch("thermographic_report_builder.processing.thermal_annotator.cv2") as mock_cv2:
            mock_cv2.IMWRITE_JPEG_QUALITY = 1
            mock_cv2.imwrite.side_effect = Exception("disk full")
            result = annotator.save_annotated_image(annotated, Path("/tmp/test.jpg"))

        assert result is False


# ---------------------------------------------------------------------------
# AnnotatedThermalImage dataclass
# ---------------------------------------------------------------------------
class TestAnnotatedThermalImage:

    def test_fields_set_correctly(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ati = AnnotatedThermalImage(
            image=img, defect_temp=55.0, panel_avg_temp=30.0,
            delta_t=25.0, severity="HIGH",
        )
        assert ati.defect_temp == 55.0
        assert ati.panel_avg_temp == 30.0
        assert ati.delta_t == 25.0
        assert ati.severity == "HIGH"
        assert ati.hot_cold is None

    def test_hot_cold_attached(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        hc = HotColdPoints(
            hot_x=10, hot_y=20, hot_temp=60.0,
            cold_x=50, cold_y=60, cold_temp=30.0,
            delta_t=30.0,
        )
        ati = AnnotatedThermalImage(
            image=img, defect_temp=60.0, panel_avg_temp=30.0,
            delta_t=30.0, severity="CRITICAL", hot_cold=hc,
        )
        assert ati.hot_cold is not None
        assert ati.hot_cold.severity == "CRITICAL"
