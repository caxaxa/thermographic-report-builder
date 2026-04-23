import math
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
MODULE_PATH = SRC_ROOT / "thermographic_report_builder" / "processing" / "crop_transform.py"


@pytest.fixture()
def crop_transform_module():
    """Import crop_transform.py in isolation without polluting sys.modules.

    crop_transform.py has a relative import (..utils.logger), so we need
    parent packages in sys.modules. We save/restore module state to avoid
    leaving broken stubs that block later imports of the full processing package.
    """
    saved = {}
    stub_keys = [
        "thermographic_report_builder",
        "thermographic_report_builder.processing",
        "thermographic_report_builder.utils",
        "thermographic_report_builder.utils.logger",
    ]
    for key in stub_keys:
        if key in sys.modules:
            saved[key] = sys.modules[key]

    # Create minimal parent stubs for relative import resolution
    pkg = types.ModuleType("thermographic_report_builder")
    pkg.__path__ = [str(SRC_ROOT / "thermographic_report_builder")]
    sys.modules.setdefault("thermographic_report_builder", pkg)

    proc_pkg = types.ModuleType("thermographic_report_builder.processing")
    proc_pkg.__path__ = [str(SRC_ROOT / "thermographic_report_builder" / "processing")]
    sys.modules.setdefault("thermographic_report_builder.processing", proc_pkg)

    # Also need utils.logger for the relative import in crop_transform.py
    utils_pkg = types.ModuleType("thermographic_report_builder.utils")
    utils_pkg.__path__ = [str(SRC_ROOT / "thermographic_report_builder" / "utils")]
    sys.modules.setdefault("thermographic_report_builder.utils", utils_pkg)

    logger_spec = importlib.util.spec_from_file_location(
        "thermographic_report_builder.utils.logger",
        SRC_ROOT / "thermographic_report_builder" / "utils" / "logger.py",
    )
    logger_mod = importlib.util.module_from_spec(logger_spec)
    sys.modules.setdefault("thermographic_report_builder.utils.logger", logger_mod)
    logger_spec.loader.exec_module(logger_mod)

    spec = importlib.util.spec_from_file_location(
        "thermographic_report_builder.processing.crop_transform", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    yield mod

    # Restore original sys.modules state — remove stubs we added
    for key in stub_keys:
        if key in saved:
            sys.modules[key] = saved[key]
        else:
            sys.modules.pop(key, None)


def _rotate_point_screen(x, y, width, height, angle_deg):
    """Rotate a point using the same screen-coordinate math as PIL (y down)."""
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rot_w = int(width * abs(cos_a) + height * abs(sin_a))
    rot_h = int(width * abs(sin_a) + height * abs(cos_a))

    cx, cy = width / 2.0, height / 2.0
    rcx, rcy = rot_w / 2.0, rot_h / 2.0

    tx, ty = x - cx, y - cy
    rx = tx * cos_a + ty * sin_a
    ry = -tx * sin_a + ty * cos_a

    return rx + rcx, ry + rcy, rot_w, rot_h


def _make_transform(CropTransform, params):
    transform = CropTransform()
    transform.params = params
    transform.available = True
    return transform


def test_cropped_to_uncropped_no_rotation(crop_transform_module):
    CropTransform = crop_transform_module.CropTransform
    CropTransformParams = crop_transform_module.CropTransformParams

    params = CropTransformParams(
        crop_x_min=100,
        crop_y_min=200,
        crop_width=1000,
        crop_height=800,
        rotation_angle=0.0,
        cropped_rotated_width=1000,
        cropped_rotated_height=800,
        resample_scale=2.0,
    )
    transform = _make_transform(CropTransform, params)

    uncropped_x, uncropped_y = transform.cropped_to_uncropped(200.0, 400.0)

    # Scale down (resample_scale=2.0) and add crop offsets.
    assert abs(uncropped_x - 200.0) < 1e-6
    assert abs(uncropped_y - 400.0) < 1e-6


def test_cropped_to_uncropped_roundtrip_rotation(crop_transform_module):
    CropTransform = crop_transform_module.CropTransform
    CropTransformParams = crop_transform_module.CropTransformParams

    width, height = 400, 300
    angle = 27.0
    x, y = 123.4, 210.7

    rotated_x, rotated_y, rot_w, rot_h = _rotate_point_screen(
        x, y, width, height, angle
    )

    params = CropTransformParams(
        crop_x_min=0,
        crop_y_min=0,
        crop_width=width,
        crop_height=height,
        rotation_angle=angle,
        cropped_rotated_width=rot_w,
        cropped_rotated_height=rot_h,
        resample_scale=1.0,
    )
    transform = _make_transform(CropTransform, params)

    uncropped_x, uncropped_y = transform.cropped_to_uncropped(rotated_x, rotated_y)

    assert abs(uncropped_x - x) < 1e-6
    assert abs(uncropped_y - y) < 1e-6
