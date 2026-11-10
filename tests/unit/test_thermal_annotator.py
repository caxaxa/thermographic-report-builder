"""Tests for thermal_annotator.py — LUT interpolation, colorbar, severity, palette detection.

Covers:
- _build_lut_256: shape, dtype, known endpoints, alias resolution, unknown fallback
- create_colorbar: shape, BGR ordering, top-hot/bottom-cold, width uniformity
- HotColdPoints.severity: boundary-value classification for ΔT thresholds
- detect_thermal_palette: subprocess success/failure/timeout
- ThermalAnnotator.find_hot_cold_points: synthetic temp array with mocked extractor
- find_module_reference_point / _shorth_midpoint: module-surface reference next to a hot point
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
    find_module_reference_point,
    _shorth_midpoint,
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

    def test_finds_hot_point_and_module_reference(self):
        """Hot spot on a uniform module: the reference is the module surface, not the coldest pixel."""
        temp_array = np.full((512, 640), 30.0, dtype=np.float64)
        # Place a hot spot at (320, 256) — center of the image
        temp_array[250:260, 315:325] = 80.0
        # Cold patch inside the panel box (shadow, gap): must not become the reference
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
        assert result.cold_temp == 30.0
        assert result.delta_t == 50.0
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

    def _edge_cell_scene(self):
        """Module (280..360, 240..300) with a hot cell on its right edge and a warm interior patch."""
        temp_array = np.full((512, 640), 30.0, dtype=np.float64)
        temp_array[240:300, 280:360] = 40.0
        temp_array[266:272, 352:358] = 50.0  # warm module edge around the hot cell
        temp_array[268:270, 356:358] = 85.0  # hot cell, inside the 10 % inset margin
        temp_array[262:266, 328:332] = 70.0  # warm interior patch 25 px away
        return temp_array

    def test_panel_search_drops_edge_hot_cell(self):
        """Default (imprecise point): the inset keeps the hot search off the module edge."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(self._edge_cell_scene()))
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=354, initial_y=268,
            panel_bbox_visual=(280, 240, 360, 300),
            hot_search_radius=30,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
        )
        assert result is not None
        assert result.hot_temp == 70.0

    def test_precise_point_finds_edge_hot_cell(self):
        """Source-map points: hottest pixel near the label, even on the module edge."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(self._edge_cell_scene()))
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=354, initial_y=268,
            panel_bbox_visual=(280, 240, 360, 300),
            hot_search_radius=6,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
            hot_within_panel=False,
        )
        assert result is not None
        assert (result.hot_x, result.hot_y, result.hot_temp) == (356, 268, 85.0)
        # The reference is the module surface, not the warm edge or the background
        assert result.cold_temp == 40.0

    def test_precise_point_ignores_offset_panel_box(self):
        """A panel box projected off the module must not pull the hot point to its edge."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        temp_array = np.full((512, 640), 40.0, dtype=np.float64)
        temp_array[200:202, 200:202] = 80.0  # labelled hot cell
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(temp_array))
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=201, initial_y=201,
            panel_bbox_visual=(210, 180, 250, 230),  # box 9 px right of the cell
            hot_search_radius=6,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
            hot_within_panel=False,
        )
        assert result is not None
        assert result.hot_temp == 80.0

    def test_precise_point_respects_prior_hot_points(self):
        """A second defect on the same image must not reuse the first one's hot pixel."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        temp_array = np.full((512, 640), 40.0, dtype=np.float64)
        temp_array[200, 200] = 90.0  # first defect's hot cell
        temp_array[200, 208] = 70.0  # second defect's cell
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(temp_array))
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=205, initial_y=200,
            panel_bbox_visual=(150, 150, 260, 260),
            hot_search_radius=6,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
            excluded_hot_points=[(200, 200)],
            min_hot_distance=4,
            hot_within_panel=False,
        )
        assert result is not None
        assert (result.hot_x, result.hot_temp) == (208, 70.0)

    def test_reference_without_panel_box(self):
        """No panel box: the reference comes from the surface around the hot point."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        temp_array = np.full((512, 640), 45.0, dtype=np.float64)
        temp_array[255:258, 319:322] = 70.0
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(temp_array))
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=320, initial_y=256,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
        )
        assert result is not None
        assert (result.hot_temp, result.cold_temp, result.delta_t) == (70.0, 45.0, 25.0)

    def test_reference_reads_like_an_override_click(self):
        """cold_temp is the 5x5 minimum at the marker: what the override path reads there."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat, visual_to_thermal
        temp_array = _table_scene(noise=0.3)
        annotator = ThermalAnnotator(thermal_extractor=self._make_extractor(temp_array))
        # 1280x1024 visual frame over the 640x512 thermal array, as for R-JPEGs
        result = annotator.find_hot_cold_points(
            image_path=Path("/fake.jpg"),
            initial_x=2 * 330, initial_y=2 * 251,
            panel_bbox_visual=(2 * 314, 2 * 230, 2 * 344, 2 * 274),
            hot_search_radius=12,
            image_format=ImageFormat.VISUAL_THERMAL,
            visual_size=(1280, 1024),
            hot_within_panel=False,
        )
        assert result is not None
        assert result.hot_temp == 75.0
        tx, ty = visual_to_thermal(
            result.cold_x, result.cold_y,
            image_format=ImageFormat.VISUAL_THERMAL, visual_size=(1280, 1024), thermal_size=(640, 512),
        )
        tx, ty = int(tx), int(ty)
        assert 314 <= tx < 344 and 230 <= ty < 274
        assert result.cold_temp == round(float(temp_array[ty - 2:ty + 3, tx - 2:tx + 3].min()), 1)
        assert abs(result.cold_temp - 45.0) <= 0.5

    def test_south_flight_reference_in_rotated_frame(self):
        """South flights: same reference as the north-up scene, in rotated coordinates."""
        from thermographic_report_builder.utils.thermal_alignment import ImageFormat
        scene = _table_scene(noise=0.3)
        kwargs = dict(
            image_path=Path("/fake.jpg"),
            initial_x=330, initial_y=251,
            panel_bbox_visual=(314, 230, 344, 274),
            hot_search_radius=6,
            image_format=ImageFormat.THERMAL_ONLY,
            visual_size=(640, 512),
            hot_within_panel=False,
        )
        north = ThermalAnnotator(thermal_extractor=self._make_extractor(scene)).find_hot_cold_points(**kwargs)
        south = ThermalAnnotator(
            thermal_extractor=self._make_extractor(np.rot90(scene, 2))
        ).find_hot_cold_points(flight_direction="south", **kwargs)
        assert north is not None
        assert south == north


def _table_scene(noise: float = 0.0) -> np.ndarray:
    """Three modules (45 °C) side by side with 4 px gaps (32 °C) on hot textured ground.

    Modules span y 230..274 and x 280..310, 314..344, 348..378; the middle one has a
    75 °C cell at x 329..332, y 250..253. Temperatures are quantised to 0.1 °C like R-JPEG data.
    """
    rng = np.random.default_rng(7)
    temp_array = 60.0 + rng.uniform(-4.0, 4.0, size=(512, 640))
    temp_array[230:274, 280:378] = 32.0
    for left in (280, 314, 348):
        temp_array[230:274, left:left + 30] = 45.0 + rng.uniform(-noise, noise, size=(44, 30))
    temp_array[250:253, 329:332] = 75.0
    return np.round(temp_array, 1)


def _box_mask(left, top, right, bottom, shape=(512, 640)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


# ---------------------------------------------------------------------------
# find_module_reference_point / _shorth_midpoint
# ---------------------------------------------------------------------------
class TestFindModuleReferencePoint:

    def test_shorth_midpoint_is_the_densest_level(self):
        """Half of the values are spread out (gaps, ground, hot cells): the tight cluster wins."""
        values = np.concatenate([
            np.linspace(20.0, 40.0, 55),  # gaps and shadow, majority of the pixels
            np.full(40, 45.0),            # module surface
            np.linspace(50.0, 75.0, 5),   # hot cells
        ])
        assert np.median(values) < 40.0
        assert _shorth_midpoint(values) == 45.0

    def test_shorth_midpoint_of_few_values_is_the_median(self):
        assert _shorth_midpoint(np.array([40.0, 50.0, 42.0])) == 42.0

    def test_box_covering_gaps_and_ground(self):
        """A box larger than the module: gaps and ground inside it do not move the reference."""
        temp_array = _table_scene(noise=0.3)
        x, y, temp = find_module_reference_point(temp_array, _box_mask(310, 222, 348, 282), 330, 251)
        assert 314 <= x < 344 and 230 <= y < 274
        assert abs(temp - 45.0) <= 0.5

    def test_box_projected_off_the_module(self):
        """A box 14 px off (half on the neighbouring module) still gives the hot module's surface."""
        temp_array = _table_scene(noise=0.3)
        temp_array[230:274, 348:378] += 6.0  # neighbour runs warmer
        x, y, temp = find_module_reference_point(temp_array, _box_mask(328, 230, 358, 274), 330, 251)
        assert 314 <= x < 344 and 230 <= y < 274
        assert abs(temp - 45.0) <= 0.5

    def test_marker_is_clear_of_the_hot_point(self):
        """The 5x5 window at the marker must not include the hot pixel."""
        temp_array = np.full((512, 640), 45.0)
        temp_array[251, 330] = 90.0
        x, y, temp = find_module_reference_point(temp_array, _box_mask(314, 230, 344, 274), 330, 251)
        assert np.hypot(x - 330, y - 251) >= 4
        assert temp == 45.0

    def test_marker_leaves_room_for_the_hot_marker(self):
        """Report markers are 51 px rings: the reference sits 16 px away when the module has room."""
        x, y, temp = find_module_reference_point(_table_scene(), _box_mask(314, 230, 344, 274), 330, 251)
        assert 314 <= x < 344 and 230 <= y < 274
        assert 16 <= np.hypot(x - 330, y - 251) < 17
        assert temp == 45.0

    def test_marker_prefers_the_hot_module_over_a_nearer_neighbour(self):
        """Hot cell on the module edge: the marker stays on its module, not across the gap."""
        temp_array = _table_scene()
        temp_array[250:253, 329:332] = 45.0
        yy, xx = np.ogrid[:512, :640]
        halo = (np.hypot(xx - 341.5, yy - 251.5) <= 9) & (xx >= 314) & (xx < 344)
        temp_array[halo] = 46.0            # warm halo around the cell, on its own module
        temp_array[250:254, 340:344] = 75.0  # hot cell at the module's right edge
        x, y, temp = find_module_reference_point(temp_array, _box_mask(314, 230, 378, 274), 341, 251)
        assert 314 <= x < 344 and 230 <= y < 274
        assert temp == 45.0

    def test_small_module(self):
        """Modules of a few pixels (high flights) still give their own surface temperature."""
        temp_array = np.full((512, 640), 35.0)
        temp_array[100:108, 200:212] = 50.0
        temp_array[103, 205] = 70.0
        x, y, temp = find_module_reference_point(temp_array, _box_mask(200, 100, 212, 108), 205, 103)
        assert (x, y) == (209, 105)  # flat patch farthest from the hot point, still on the module
        assert temp == 50.0

    def test_invalid_pixels_are_ignored(self):
        temp_array = np.full((512, 640), 45.0)
        temp_array[240:246, 320:326] = np.nan
        temp_array[251, 330] = 80.0
        x, y, temp = find_module_reference_point(temp_array, _box_mask(314, 230, 344, 274), 330, 251)
        assert temp == 45.0
        assert not np.isnan(temp_array[y - 2:y + 3, x - 2:x + 3]).any()

    def test_returns_none_without_valid_temperatures(self):
        temp_array = np.full((512, 640), np.nan)
        assert find_module_reference_point(temp_array, _box_mask(314, 230, 344, 274), 330, 251) is None

    def test_hot_point_on_the_frame_edge(self):
        """Windows are clipped at the frame border, as in the override path."""
        temp_array = np.full((512, 640), 45.0)
        temp_array[0:2, 0:2] = 80.0
        x, y, temp = find_module_reference_point(temp_array, _box_mask(0, 0, 30, 44), 0, 0)
        assert temp == 45.0


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
