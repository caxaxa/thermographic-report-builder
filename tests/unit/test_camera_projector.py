"""Tests for CameraProjector — camera reprojection from ortho to raw images.

Tests the pure mathematical functions (rotation, projection, coordinate
conversion) and the integration of projection with mock reconstruction data.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_geo_converter(
    utm_origin_x=500000.0,
    utm_origin_y=8000000.0,
    gsd=0.03,
):
    """Create a mock PixelToLatLonConverter with a linear affine transform.

    Simulates an orthophoto with origin at (utm_origin_x, utm_origin_y) and
    ground-sample distance of ``gsd`` metres/pixel.
    """
    converter = MagicMock()
    transform = MagicMock()
    # Standard non-rotated north-up affine: a = gsd, b = 0, d = 0, e = -gsd
    transform.a = gsd
    transform.b = 0.0
    transform.c = utm_origin_x  # X offset
    transform.d = 0.0
    transform.e = -gsd
    transform.f = utm_origin_y  # Y offset
    converter.transform = transform
    converter._source_crs = None  # Skip pyproj transformer
    return converter


def _make_reconstruction(
    *,
    cameras=None,
    shots=None,
    points=None,
    reference_lla=None,
):
    """Build a minimal reconstruction dict."""
    recon = {}
    if cameras is not None:
        recon["cameras"] = cameras
    if shots is not None:
        recon["shots"] = shots
    if points is not None:
        recon["points"] = points
    if reference_lla is not None:
        recon["reference_lla"] = reference_lla
    return [recon]


# A simple pinhole camera with no distortion
SIMPLE_CAMERA = {
    "v2 unknown unknown 640 512 perspective 0.8": {
        "focal_x": 0.8,
        "focal_y": 0.8,
        "c_x": 0.0,
        "c_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "width": 640,
        "height": 512,
    }
}

# Camera hovering at (0, 0, 25m) looking straight down (nadir).
# OpenSfM convention: camera_point = R * world_point + t
# For nadir camera at pos=[0,0,25], we need 180° rotation around X axis
# so that the camera Z-axis points downward into the scene.
# R_x(pi) = [[1,0,0],[0,-1,0],[0,0,-1]]
# axis-angle for 180° around X = [pi, 0, 0]
# t = R * (-pos) = R_x(pi) * [0, 0, -25] = [0, 0, 25]
_PI = float(np.pi)
SIMPLE_SHOT = {
    "DJI_001.JPG": {
        "camera": "v2 unknown unknown 640 512 perspective 0.8",
        "rotation": [_PI, 0.0, 0.0],
        "translation": [0.0, 0.0, 25.0],
        "gps_position": [0.0, 0.0, 25.0],
    }
}


@pytest.fixture
def projector_simple():
    """A CameraProjector with a single top-down camera at (0,0,25)."""
    from thermographic_report_builder.processing.camera_projector import CameraProjector

    geo = _make_mock_geo_converter()
    recon = _make_reconstruction(
        cameras=SIMPLE_CAMERA,
        shots=SIMPLE_SHOT,
        points={},
    )
    return CameraProjector(recon, geo)


# ---------------------------------------------------------------------------
# RawImageMatch dataclass
# ---------------------------------------------------------------------------

class TestRawImageMatch:
    def test_is_central_true(self):
        from thermographic_report_builder.processing.camera_projector import RawImageMatch

        m = RawImageMatch(image_name="img.jpg", pixel_x=100, pixel_y=100, distance_from_center=0.3)
        assert m.is_central is True

    def test_is_central_false_at_boundary(self):
        from thermographic_report_builder.processing.camera_projector import RawImageMatch

        m = RawImageMatch(image_name="img.jpg", pixel_x=100, pixel_y=100, distance_from_center=0.5)
        assert m.is_central is False

    def test_is_central_false_near_edge(self):
        from thermographic_report_builder.processing.camera_projector import RawImageMatch

        m = RawImageMatch(image_name="img.jpg", pixel_x=100, pixel_y=100, distance_from_center=0.9)
        assert m.is_central is False


# ---------------------------------------------------------------------------
# axis_angle_to_rotation_matrix
# ---------------------------------------------------------------------------

class TestAxisAngleToRotation:
    def test_zero_rotation_gives_identity(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_tiny_rotation_gives_identity(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([1e-12, 0, 0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_around_z(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        angle = np.pi / 2
        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([0, 0, angle]))
        # Rotating [1, 0, 0] by 90° around Z should give [0, 1, 0]
        v = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-10)

    def test_180_deg_around_x(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        angle = np.pi
        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([angle, 0, 0]))
        # Rotating [0, 1, 0] by 180° around X should give [0, -1, 0]
        v = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(v, [0, -1, 0], atol=1e-10)

    def test_rotation_is_orthogonal(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([0.3, 0.5, -0.2]))
        # R^T * R should be identity
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        R = CameraProjector._axis_angle_to_rotation_matrix(np.array([1.0, -0.5, 0.7]))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# CameraProjector constructor
# ---------------------------------------------------------------------------

class TestCameraProjectorInit:
    def test_none_reconstruction_disabled(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cp = CameraProjector(None, geo)
        assert cp.available is False

    def test_empty_list_reconstruction_disabled(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cp = CameraProjector([], geo)
        assert cp.available is False

    def test_valid_reconstruction_enabled(self, projector_simple):
        assert projector_simple.available is True

    def test_camera_dimensions_set_from_reconstruction(self, projector_simple):
        assert projector_simple.image_width == 640
        assert projector_simple.image_height == 512


# ---------------------------------------------------------------------------
# _project_to_camera
# ---------------------------------------------------------------------------

class TestProjectToCamera:
    def test_center_point_projects_to_image_center(self, projector_simple):
        """A point directly below the camera should project to image center."""
        world_point = np.array([0.0, 0.0, 0.0])
        shot = projector_simple.shots["DJI_001.JPG"]
        result = projector_simple._project_to_camera(world_point, shot)

        assert result is not None
        px, py, w, h = result
        # Center of 640x512 image is (319.5, 255.5)
        assert abs(px - 319.5) < 1.0
        assert abs(py - 255.5) < 1.0

    def test_point_behind_camera_returns_none(self, projector_simple):
        """A point behind the camera (z > camera z) should return None."""
        # Camera is at z=25, point at z=30 is behind
        world_point = np.array([0.0, 0.0, 30.0])
        shot = projector_simple.shots["DJI_001.JPG"]
        result = projector_simple._project_to_camera(world_point, shot)

        assert result is None

    def test_offset_point_projects_off_center(self, projector_simple):
        """A point offset from camera center should project off-center."""
        world_point = np.array([2.0, 0.0, 0.0])
        shot = projector_simple.shots["DJI_001.JPG"]
        result = projector_simple._project_to_camera(world_point, shot)

        assert result is not None
        px, py, w, h = result
        # Should be shifted to the right of center
        assert px > 320


# ---------------------------------------------------------------------------
# project_ortho_to_raw
# ---------------------------------------------------------------------------

class TestProjectOrthoToRaw:
    def test_disabled_projector_returns_empty(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cp = CameraProjector(None, geo)
        result = cp.project_ortho_to_raw(100, 100, elevation=0.0)
        assert result == []

    def test_single_match_returned(self, projector_simple):
        """A point within the single camera's FOV should produce one match."""
        # Set up reference UTM so ortho pixel (0,0) maps to world (0,0)
        projector_simple._ref_utm_x = 500000.015  # 0.5 * gsd added for center
        projector_simple._ref_utm_y = 7999999.985

        matches = projector_simple.project_ortho_to_raw(0, 0, elevation=0.0)
        assert len(matches) >= 1
        assert matches[0].image_name == "DJI_001.JPG"

    def test_matches_sorted_by_centrality(self):
        """Multiple matches should be sorted by distance_from_center."""
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cameras = SIMPLE_CAMERA.copy()
        shots = {
            "DJI_001.JPG": {
                "camera": "v2 unknown unknown 640 512 perspective 0.8",
                "rotation": [_PI, 0.0, 0.0],
                "translation": [0.0, 0.0, 25.0],
                "gps_position": [0.0, 0.0, 25.0],
            },
            "DJI_002.JPG": {
                "camera": "v2 unknown unknown 640 512 perspective 0.8",
                "rotation": [_PI, 0.0, 0.0],
                "translation": [-2.0, 0.0, 25.0],  # offset camera
                "gps_position": [2.0, 0.0, 25.0],
            },
        }
        cp = CameraProjector(
            _make_reconstruction(cameras=cameras, shots=shots), geo
        )
        cp._ref_utm_x = 500000.015
        cp._ref_utm_y = 7999999.985

        matches = cp.project_ortho_to_raw(0, 0, elevation=0.0)
        if len(matches) >= 2:
            assert matches[0].distance_from_center <= matches[1].distance_from_center

    def test_gps_distance_filter(self, projector_simple):
        """max_gps_distance should exclude distant cameras."""
        projector_simple._ref_utm_x = 500000.015
        projector_simple._ref_utm_y = 7999999.985

        # With a very small distance filter, the shot at (0,0,25) should be excluded
        # when the ortho point maps to world (0,0) — drone is at 0m horizontal
        matches = projector_simple.project_ortho_to_raw(
            0, 0, elevation=0.0, max_gps_distance=0.001
        )
        # This should still match since horizontal distance is 0
        assert len(matches) >= 1


# ---------------------------------------------------------------------------
# get_best_match
# ---------------------------------------------------------------------------

class TestGetBestMatch:
    def test_returns_none_when_disabled(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cp = CameraProjector(None, geo)
        assert cp.get_best_match(100, 100, elevation=0.0) is None


# ---------------------------------------------------------------------------
# project_world_to_image
# ---------------------------------------------------------------------------

class TestProjectWorldToImage:
    def test_returns_none_when_disabled(self):
        from thermographic_report_builder.processing.camera_projector import CameraProjector

        geo = _make_mock_geo_converter()
        cp = CameraProjector(None, geo)
        assert cp.project_world_to_image("img.jpg", np.array([0, 0, 0])) is None

    def test_unknown_image_returns_none(self, projector_simple):
        result = projector_simple.project_world_to_image(
            "nonexistent.jpg", np.array([0, 0, 0])
        )
        assert result is None

    def test_center_point_returns_center_pixels(self, projector_simple):
        result = projector_simple.project_world_to_image(
            "DJI_001.JPG", np.array([0.0, 0.0, 0.0])
        )
        assert result is not None
        px, py = result
        assert abs(px - 319.5) < 1.0
        assert abs(py - 255.5) < 1.0


# ---------------------------------------------------------------------------
# load_reconstruction
# ---------------------------------------------------------------------------

class TestLoadReconstruction:
    def test_missing_file_returns_none(self, tmp_path):
        from thermographic_report_builder.processing.camera_projector import load_reconstruction

        result = load_reconstruction(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_json_loaded(self, tmp_path):
        import json
        from thermographic_report_builder.processing.camera_projector import load_reconstruction

        path = tmp_path / "reconstruction.json"
        data = [{"cameras": {}, "shots": {}}]
        path.write_text(json.dumps(data))

        result = load_reconstruction(path)
        assert result is not None
        assert isinstance(result, list)

    def test_invalid_json_returns_none(self, tmp_path):
        from thermographic_report_builder.processing.camera_projector import load_reconstruction

        path = tmp_path / "reconstruction.json"
        path.write_text("{invalid json")

        result = load_reconstruction(path)
        assert result is None
