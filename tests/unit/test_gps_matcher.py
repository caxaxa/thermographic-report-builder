"""Tests for GPSMatcher — image-to-defect matching logic.

Tests the pure helper functions (closest image search, candidate finding,
flight direction detection, track coordinate conversion, filename normalization)
without requiring S3 or real image data.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stub_gps_matcher(**overrides):
    """Create a GPSMatcher-like object with manually set attributes.

    We don't call __init__ because it requires many heavy dependencies.
    Instead we create a bare instance and set only the attributes tested.
    """
    from thermographic_report_builder.processing.gps_matcher import GPSMatcher

    # Bypass __init__
    obj = object.__new__(GPSMatcher)
    obj.image_cache = {}
    obj.orientation_map = {}
    obj.annotation_entries = []
    obj.camera_projector = MagicMock()
    obj.camera_projector.available = False
    obj.camera_projector.shots = {}
    obj._track_coord_mode = None
    obj._track_calibration_offsets = {}
    for key, val in overrides.items():
        setattr(obj, key, val)
    return obj


@pytest.fixture
def matcher_with_images():
    """GPSMatcher stub with 5 synthetic images at known GPS positions."""
    images = {
        "DJI_001.JPG": {"filename": "DJI_001.JPG", "latitude": -15.0000, "longitude": -47.0000, "path": "/tmp/DJI_001.JPG"},
        "DJI_002.JPG": {"filename": "DJI_002.JPG", "latitude": -15.0001, "longitude": -47.0000, "path": "/tmp/DJI_002.JPG"},
        "DJI_003.JPG": {"filename": "DJI_003.JPG", "latitude": -15.0002, "longitude": -47.0000, "path": "/tmp/DJI_003.JPG"},
        "DJI_004.JPG": {"filename": "DJI_004.JPG", "latitude": -15.0003, "longitude": -47.0000, "path": "/tmp/DJI_004.JPG"},
        "DJI_005.JPG": {"filename": "DJI_005.JPG", "latitude": -15.0004, "longitude": -47.0000, "path": "/tmp/DJI_005.JPG"},
    }
    return _stub_gps_matcher(image_cache=images)


# ---------------------------------------------------------------------------
# _find_closest_image
# ---------------------------------------------------------------------------

class TestFindClosestImage:
    def test_finds_exact_match(self, matcher_with_images):
        result = matcher_with_images._find_closest_image(-15.0002, -47.0000)
        assert result is not None
        assert result["filename"] == "DJI_003.JPG"

    def test_finds_nearest_when_no_exact(self, matcher_with_images):
        # Between DJI_001 and DJI_002 — closer to DJI_001
        result = matcher_with_images._find_closest_image(-15.00003, -47.0000)
        assert result is not None
        assert result["filename"] == "DJI_001.JPG"

    def test_empty_cache_returns_none(self):
        matcher = _stub_gps_matcher(image_cache={})
        result = matcher._find_closest_image(-15.0, -47.0)
        assert result is None


# ---------------------------------------------------------------------------
# _find_candidate_images
# ---------------------------------------------------------------------------

class TestFindCandidateImages:
    def test_returns_images_within_distance(self, matcher_with_images):
        # Distance between consecutive images is ~0.0001 degrees
        candidates = matcher_with_images._find_candidate_images(
            -15.0001, -47.0000, max_distance_deg=0.00015
        )
        # Should find DJI_001, DJI_002 (exact), and possibly DJI_003
        assert len(candidates) >= 2
        # First should be closest
        assert candidates[0]["filename"] == "DJI_002.JPG"

    def test_respects_max_candidates(self, matcher_with_images):
        candidates = matcher_with_images._find_candidate_images(
            -15.0002, -47.0000, max_distance_deg=0.001, max_candidates=2
        )
        assert len(candidates) <= 2

    def test_sorted_by_distance(self, matcher_with_images):
        candidates = matcher_with_images._find_candidate_images(
            -15.0002, -47.0000, max_distance_deg=0.001
        )
        # Verify sorted by ascending distance
        for i in range(len(candidates) - 1):
            dist_a = np.hypot(
                -15.0002 - candidates[i]["latitude"],
                -47.0000 - candidates[i]["longitude"],
            )
            dist_b = np.hypot(
                -15.0002 - candidates[i + 1]["latitude"],
                -47.0000 - candidates[i + 1]["longitude"],
            )
            assert dist_a <= dist_b

    def test_empty_when_all_too_far(self, matcher_with_images):
        candidates = matcher_with_images._find_candidate_images(
            0.0, 0.0, max_distance_deg=0.0001
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# _compute_flight_directions
# ---------------------------------------------------------------------------

class TestComputeFlightDirections:
    def test_southward_flight(self):
        """Images with decreasing latitude → flying south."""
        images = {
            "DJI_001.JPG": {"latitude": -15.0000, "longitude": -47.0},
            "DJI_002.JPG": {"latitude": -15.0001, "longitude": -47.0},
            "DJI_003.JPG": {"latitude": -15.0002, "longitude": -47.0},
        }
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()

        assert matcher.orientation_map["DJI_001.JPG"] == "south"
        assert matcher.orientation_map["DJI_002.JPG"] == "south"

    def test_northward_flight(self):
        """Images with increasing latitude → flying north."""
        images = {
            "DJI_001.JPG": {"latitude": -15.0002, "longitude": -47.0},
            "DJI_002.JPG": {"latitude": -15.0001, "longitude": -47.0},
            "DJI_003.JPG": {"latitude": -15.0000, "longitude": -47.0},
        }
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()

        assert matcher.orientation_map["DJI_001.JPG"] == "north"
        assert matcher.orientation_map["DJI_002.JPG"] == "north"

    def test_last_image_inherits_direction(self):
        images = {
            "DJI_001.JPG": {"latitude": -15.0000, "longitude": -47.0},
            "DJI_002.JPG": {"latitude": -15.0001, "longitude": -47.0},
        }
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()

        # Last image should inherit direction from previous
        assert matcher.orientation_map["DJI_002.JPG"] == matcher.orientation_map["DJI_001.JPG"]

    def test_empty_cache_does_nothing(self):
        matcher = _stub_gps_matcher(image_cache={})
        matcher._compute_flight_directions()
        assert matcher.orientation_map == {}

    def test_single_image_handled(self):
        images = {"DJI_001.JPG": {"latitude": -15.0, "longitude": -47.0}}
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()
        # Not enough images to determine direction
        assert len(matcher.orientation_map) == 0

    def test_mixed_flight_pattern(self):
        """Alternating lat changes → mixed directions."""
        images = {
            "DJI_001.JPG": {"latitude": -15.0000, "longitude": -47.0},
            "DJI_002.JPG": {"latitude": -15.0001, "longitude": -47.0},  # south
            "DJI_003.JPG": {"latitude": -15.0000, "longitude": -47.0},  # north
            "DJI_004.JPG": {"latitude": -15.0001, "longitude": -47.0},  # south
        }
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()

        assert matcher.orientation_map["DJI_001.JPG"] == "south"
        assert matcher.orientation_map["DJI_002.JPG"] == "north"
        assert matcher.orientation_map["DJI_003.JPG"] == "south"

    def test_skips_images_without_lat(self):
        images = {
            "DJI_001.JPG": {"latitude": -15.0, "longitude": -47.0},
            "DJI_002.JPG": {"latitude": None, "longitude": None},
            "DJI_003.JPG": {"latitude": -15.0001, "longitude": -47.0},
        }
        matcher = _stub_gps_matcher(image_cache=images)
        matcher._compute_flight_directions()

        # Only 2 valid images used
        assert "DJI_002.JPG" not in matcher.orientation_map
        assert "DJI_001.JPG" in matcher.orientation_map


# ---------------------------------------------------------------------------
# _tracks_xy_to_pixels
# ---------------------------------------------------------------------------

class TestTracksXyToPixels:
    """Test coordinate mode conversions for tracks.csv data."""

    def test_pixel_mode_passthrough(self):
        matcher = _stub_gps_matcher()
        px, py = matcher._tracks_xy_to_pixels(320, 256, 640, 512, "pixel")
        assert px == 320
        assert py == 256

    def test_norm01_mode(self):
        matcher = _stub_gps_matcher()
        px, py = matcher._tracks_xy_to_pixels(0.5, 0.5, 640, 512, "norm01")
        assert px == 320.0
        assert py == 256.0

    def test_norm01_corners(self):
        matcher = _stub_gps_matcher()
        px, py = matcher._tracks_xy_to_pixels(0.0, 0.0, 640, 512, "norm01")
        assert px == 0.0
        assert py == 0.0

        px, py = matcher._tracks_xy_to_pixels(1.0, 1.0, 640, 512, "norm01")
        assert px == 640.0
        assert py == 512.0

    def test_centered_mode(self):
        matcher = _stub_gps_matcher()
        # (0, 0) in centered mode → center of image
        px, py = matcher._tracks_xy_to_pixels(0.0, 0.0, 640, 512, "centered")
        assert px == 320.0
        assert py == 256.0

    def test_centered_max_mode(self):
        matcher = _stub_gps_matcher()
        # (0, 0) in centered_max → center of image
        px, py = matcher._tracks_xy_to_pixels(0.0, 0.0, 640, 512, "centered_max")
        assert px == 320.0
        assert py == 256.0

    def test_unknown_mode_passthrough(self):
        matcher = _stub_gps_matcher()
        px, py = matcher._tracks_xy_to_pixels(123, 456, 640, 512, "unknown")
        assert px == 123
        assert py == 456


# ---------------------------------------------------------------------------
# _normalize_track_image_name
# ---------------------------------------------------------------------------

class TestNormalizeTrackImageName:
    def test_exact_match(self):
        mock_projector = MagicMock()
        mock_projector.shots = {"DJI_001.JPG": {}}
        matcher = _stub_gps_matcher(camera_projector=mock_projector)

        result = matcher._normalize_track_image_name("DJI_001.JPG")
        assert result == "DJI_001.JPG"

    def test_strips_path_prefix(self):
        mock_projector = MagicMock()
        mock_projector.shots = {"DJI_001.JPG": {}}
        matcher = _stub_gps_matcher(camera_projector=mock_projector)

        result = matcher._normalize_track_image_name("images/DJI_001.JPG")
        assert result == "DJI_001.JPG"

    def test_adds_t_suffix(self):
        """DJI M30T images have _T suffix in reconstruction but not in tracks."""
        mock_projector = MagicMock()
        mock_projector.shots = {"DJI_001_T.JPG": {}}
        matcher = _stub_gps_matcher(camera_projector=mock_projector)

        result = matcher._normalize_track_image_name("DJI_001.JPG")
        assert result == "DJI_001_T.JPG"

    def test_removes_t_suffix(self):
        """Reverse case: tracks have _T but reconstruction does not."""
        mock_projector = MagicMock()
        mock_projector.shots = {"DJI_001.JPG": {}}
        matcher = _stub_gps_matcher(camera_projector=mock_projector)

        result = matcher._normalize_track_image_name("DJI_001_T.JPG")
        assert result == "DJI_001.JPG"

    def test_returns_none_for_no_match(self):
        mock_projector = MagicMock()
        mock_projector.shots = {"DJI_001.JPG": {}}
        matcher = _stub_gps_matcher(camera_projector=mock_projector)

        result = matcher._normalize_track_image_name("UNKNOWN.JPG")
        assert result is None

    def test_empty_string_returns_none(self):
        matcher = _stub_gps_matcher()
        result = matcher._normalize_track_image_name("")
        assert result is None


# ---------------------------------------------------------------------------
# _find_annotation_entry
# ---------------------------------------------------------------------------

class TestFindAnnotationEntry:
    def test_finds_existing(self):
        from thermographic_report_builder.models.annotation_manifest import (
            AnnotationEntry,
            AnnotationPoint,
        )

        entry = AnnotationEntry(
            defect_id="hotspots_(1-1)_defeito1",
            panel_id="1-1",
            defect_type="hotspots",
            defect_index=1,
            raw_image_path="s3://bucket/DJI_001.JPG",
            raw_image_name="DJI_001.JPG",
            annotated_image="hotspots_1-1_defeito1.jpg",
            hot_point=AnnotationPoint(x=100, y=200, temp=85.0),
            cold_point=AnnotationPoint(x=50, y=50, temp=40.0),
            delta_t=45.0,
            severity="CRITICAL",
        )
        matcher = _stub_gps_matcher(annotation_entries=[entry])
        result = matcher._find_annotation_entry("hotspots_(1-1)_defeito1")
        assert result is not None
        assert result.delta_t == 45.0

    def test_returns_none_when_not_found(self):
        matcher = _stub_gps_matcher(annotation_entries=[])
        result = matcher._find_annotation_entry("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# DefectMatch dataclass
# ---------------------------------------------------------------------------

class TestDefectMatch:
    def test_creation(self):
        from thermographic_report_builder.processing.gps_matcher import DefectMatch

        match = DefectMatch(
            image_name="DJI_001.JPG",
            image_path=Path("/tmp/DJI_001.JPG"),
            method="gps",
        )
        assert match.image_name == "DJI_001.JPG"
        assert match.method == "gps"
        assert match.pixel_x is None
        assert match.temperature is None

    def test_with_pixel_coordinates(self):
        from thermographic_report_builder.processing.gps_matcher import DefectMatch

        match = DefectMatch(
            image_name="DJI_001.JPG",
            image_path=Path("/tmp/DJI_001.JPG"),
            method="reprojection",
            pixel_x=320.5,
            pixel_y=256.3,
        )
        assert match.pixel_x == 320.5
        assert match.pixel_y == 256.3


# ---------------------------------------------------------------------------
# _index_raw_images (two-phase: sync detection + parallel downloads)
# ---------------------------------------------------------------------------

class TestIndexRawImages:
    """Tests for the two-phase (sync first image + parallel rest) indexing."""

    @staticmethod
    def _make_matcher(keys, fail_names=frozenset()):
        s3 = MagicMock()
        s3.list_raw_images.return_value = keys

        def fake_download(s3_key, local_path):
            if Path(s3_key).name in fail_names:
                raise RuntimeError("simulated download failure")
            return local_path

        s3.download_raw_image.side_effect = fake_download
        return _stub_gps_matcher(
            s3_client=s3,
            _detected_image_width=None,
            _detected_image_height=None,
            _thermal_only_images=False,
            _use_gps_fallback=False,
        )

    @staticmethod
    def _exif_for(name, idx):
        return {
            "has_gps": True,
            "latitude": -15.0 - idx * 0.0001,
            "longitude": -47.0,
            "width": 640,
            "height": 512,
            "filename": name,
        }

    def test_indexes_all_images_and_detects_dimensions(self, tmp_path):
        keys = [f"user/projects/proj/images/DJI_{i:03d}_T.JPG" for i in range(12)]
        matcher = self._make_matcher(keys)

        def fake_exif(path):
            return None, self._exif_for(path.name, int(path.name[4:7]))

        with patch(
            "thermographic_report_builder.processing.gps_matcher.load_raw_image_with_exif",
            side_effect=fake_exif,
        ):
            matcher._index_raw_images(tmp_path)

        assert len(matcher.image_cache) == 12
        # First-image detection ran (sync phase)
        assert matcher._detected_image_width == 640
        assert matcher._detected_image_height == 512
        assert matcher._thermal_only_images is True

    def test_single_bad_image_does_not_poison_run(self, tmp_path):
        keys = [f"user/projects/proj/images/DJI_{i:03d}_T.JPG" for i in range(10)]
        matcher = self._make_matcher(keys, fail_names={"DJI_005_T.JPG"})

        def fake_exif(path):
            return None, self._exif_for(path.name, int(path.name[4:7]))

        with patch(
            "thermographic_report_builder.processing.gps_matcher.load_raw_image_with_exif",
            side_effect=fake_exif,
        ):
            matcher._index_raw_images(tmp_path)

        assert len(matcher.image_cache) == 9
        assert "DJI_005_T.JPG" not in matcher.image_cache

    def test_detection_survives_first_image_failure(self, tmp_path):
        """If image 1 fails, the sync phase continues until detection runs."""
        keys = [f"user/projects/proj/images/DJI_{i:03d}_T.JPG" for i in range(5)]
        matcher = self._make_matcher(keys, fail_names={"DJI_000_T.JPG"})

        def fake_exif(path):
            return None, self._exif_for(path.name, int(path.name[4:7]))

        with patch(
            "thermographic_report_builder.processing.gps_matcher.load_raw_image_with_exif",
            side_effect=fake_exif,
        ):
            matcher._index_raw_images(tmp_path)

        assert matcher._detected_image_width == 640
        assert len(matcher.image_cache) == 4
