import math
import importlib.util
import sys
import types
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
MODULE_PATH = SRC_ROOT / "thermographic_report_builder" / "processing" / "crop_transform.py"

sys.path.insert(0, str(SRC_ROOT))

package = types.ModuleType("thermographic_report_builder")
package.__path__ = [str(SRC_ROOT / "thermographic_report_builder")]
sys.modules.setdefault("thermographic_report_builder", package)

processing_pkg = types.ModuleType("thermographic_report_builder.processing")
processing_pkg.__path__ = [str(SRC_ROOT / "thermographic_report_builder" / "processing")]
sys.modules.setdefault("thermographic_report_builder.processing", processing_pkg)

spec = importlib.util.spec_from_file_location(
    "thermographic_report_builder.processing.crop_transform", MODULE_PATH
)
crop_transform = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(crop_transform)

CropTransform = crop_transform.CropTransform
CropTransformParams = crop_transform.CropTransformParams


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


def _make_transform(params):
    transform = CropTransform()
    transform.params = params
    transform.available = True
    return transform


def test_cropped_to_uncropped_no_rotation():
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
    transform = _make_transform(params)

    uncropped_x, uncropped_y = transform.cropped_to_uncropped(200.0, 400.0)

    # Scale down (resample_scale=2.0) and add crop offsets.
    assert abs(uncropped_x - 200.0) < 1e-6
    assert abs(uncropped_y - 400.0) < 1e-6


def test_cropped_to_uncropped_roundtrip_rotation():
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
    transform = _make_transform(params)

    uncropped_x, uncropped_y = transform.cropped_to_uncropped(rotated_x, rotated_y)

    assert abs(uncropped_x - x) < 1e-6
    assert abs(uncropped_y - y) < 1e-6
