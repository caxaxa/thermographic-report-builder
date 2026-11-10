"""Label-space georeference of a rotated crop.

The fixture replays what inference does (crop, PIL rotate with expand, resample to
1.6 cm) on an image with markers at known ground positions. Inference used to scale the
transform it wrote but never rotate it, which is wrong for a rotated crop; the tests check
the rebuilt georeference puts every marker back where it is, and that a file whose
transform does carry the rotation is used as it is.
"""

import math

import cv2
import ezdxf
import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image

from thermographic_report_builder.models.defect import BoundingBox, Defect, GeospatialCoordinate, Panel
from thermographic_report_builder.processing.crop_transform import CropTransform
from thermographic_report_builder.processing.label_georef import (
    NorthUpView,
    build_label_georeference,
    crop_rotation_deg,
)

ORIGINAL = Affine(0.05, 0, 500000.0, 0, -0.05, 7500000.0)  # 5 cm orthophoto, 480 x 360 px
ORIGINAL_SIZE = (480, 360)
CROP = (40, 30, 440, 330)  # x0, y0, x1, y1 in original pixels
MARKERS = [(x, y) for x in range(70, 420, 70) for y in range(60, 310, 60)]
SCALE = 0.05 / 0.016  # original px -> 1.6 cm px
# PIL rounds the expanded canvas up by as much as 2 px, which the rebuild can't see: a
# few cm of error, where inference's own transform is off by metres.
GROUND_TOLERANCE_M = 0.1


def _rotation_line(deg):
    t = math.radians(deg)
    return {"start": {"x": 0, "y": 0}, "end": {"x": 100 * math.cos(t), "y": 100 * math.sin(t)}}


def _crop_annotation(deg):
    x0, y0, x1, y1 = CROP
    polygon = [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]
    return {"polygon": polygon, "rotationLine": _rotation_line(deg) if deg else None}


def _run_inference(deg, rotation_in_transform=False):
    """Crop, rotate and resample a marker image the way inference does.

    The transform is the one inference used to write, without the rotation, unless
    rotation_in_transform asks for the one it writes now.
    """
    x0, y0, x1, y1 = CROP
    img = np.full((y1 - y0, x1 - x0), 100, np.uint8)
    for mx, my in MARKERS:
        img[my - y0 - 1 : my - y0 + 2, mx - x0 - 1 : mx - x0 + 2] = 255
    rotated = Image.fromarray(img)
    transform = ORIGINAL * Affine.translation(x0, y0)
    if abs(deg) > 0.1:
        rotated = rotated.rotate(deg, expand=True, resample=Image.BILINEAR)
        if rotation_in_transform:
            transform = (
                transform
                * Affine.translation(img.shape[1] / 2, img.shape[0] / 2)
                * Affine.rotation(deg)
                * Affine.translation(-rotated.width / 2, -rotated.height / 2)
            )
    w16, h16 = int(rotated.width * SCALE), int(rotated.height * SCALE)
    label = np.array(rotated.resize((w16, h16), Image.BILINEAR))
    transform = transform * Affine.scale(rotated.width / w16, rotated.height / h16)
    return label, transform


def _back_mapping(deg, label):
    return CropTransform(
        crop_annotation=_crop_annotation(deg),
        crop_metadata={"width": ORIGINAL_SIZE[0], "height": ORIGINAL_SIZE[1]},
        ortho_width=ORIGINAL_SIZE[0],
        ortho_height=ORIGINAL_SIZE[1],
        resampled_width=label.shape[1],
        resampled_height=label.shape[0],
    )


def _markers(image, threshold=180):
    """Marker centroids in pixel-corner coordinates."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats((image > threshold).astype(np.uint8))
    return [(cx + 0.5, cy + 0.5) for (cx, cy), area in zip(centroids[1:], stats[1:, 4]) if area >= 4]


def _truth():
    return [ORIGINAL * (mx + 0.5, my + 0.5) for mx, my in MARKERS]


def _nearest(point, candidates):
    return min(math.dist(point, c) for c in candidates)


def _layout_error(points, expected):
    """Common offset between two marker sets, and the worst residual once it's removed."""
    pairs = [(p, min(expected, key=lambda e: math.dist(p, e))) for p in points]
    dx = float(np.mean([p[0] - e[0] for p, e in pairs]))
    dy = float(np.mean([p[1] - e[1] for p, e in pairs]))
    residual = max(math.dist((p[0] - dx, p[1] - dy), e) for p, e in pairs)
    return math.hypot(dx, dy), residual


@pytest.mark.parametrize("deg", [20.85, -16.02, 30, 60, 89, -75, 40, 45])
def test_markers_land_on_their_ground_position(deg):
    label, transform = _run_inference(deg)
    georef = build_label_georeference(
        transform, label.shape[1], label.shape[0], _crop_annotation(deg),
        crop_transform=_back_mapping(deg, label), original_transform=ORIGINAL,
    )
    found = _markers(label)
    assert len(found) == len(MARKERS)
    for p in found:
        assert _nearest(georef.transform * p, _truth()) < GROUND_TOLERANCE_M
    # Inference's own transform is metres off
    assert max(_nearest(transform * p, _truth()) for p in found) > 1.0
    assert georef.ortho_drift_m < 0.5


@pytest.mark.parametrize("deg", [20.85, -16.02, 30, 60, 89, 90, 180, -75, 40, 45])
def test_file_that_carries_the_rotation_is_used_as_it_is(deg):
    label, transform = _run_inference(deg, rotation_in_transform=True)
    # No annotation: the file alone is enough, whatever was redrawn since inference
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], None)
    assert georef.transform == transform
    assert georef.rotated
    assert (georef.rotation_deg - deg + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
    found = _markers(label)
    assert len(found) == len(MARKERS)
    for p in found:
        assert _nearest(transform * p, _truth()) < 0.03


@pytest.mark.parametrize("deg", [20.85, -16.02])
def test_stamping_the_rebuilt_transform_on_an_old_file_changes_nothing(deg):
    """Files from before inference wrote the rotation were re-stamped with the rebuilt one."""
    label, transform = _run_inference(deg)
    width, height = label.shape[1], label.shape[0]
    route = dict(crop_transform=_back_mapping(deg, label), original_transform=ORIGINAL)
    before = build_label_georeference(transform, width, height, _crop_annotation(deg), **route)
    after = build_label_georeference(before.transform, width, height, _crop_annotation(deg), **route)
    assert after.transform == before.transform
    assert after.ortho_drift_m == before.ortho_drift_m
    assert after.rotation_deg == pytest.approx(before.rotation_deg, abs=1e-9)
    assert after.crop_size == pytest.approx(before.crop_size, abs=1e-9)
    old_view, new_view = NorthUpView(before, width, height), NorthUpView(after, width, height)
    assert (new_view.width, new_view.height) == (old_view.width, old_view.height)
    assert new_view.point(123.0, 456.0) == old_view.point(123.0, 456.0)


def test_near_45_degrees_a_file_that_carries_the_rotation_still_gets_a_north_up_view():
    label, transform = _run_inference(45, rotation_in_transform=True)
    width, height = label.shape[1], label.shape[0]
    x0, y0, x1, y1 = CROP

    # With the original orthophoto, the crop's size comes from the back-mapping route
    georef = build_label_georeference(
        transform, width, height, _crop_annotation(45),
        crop_transform=_back_mapping(45, label), original_transform=ORIGINAL,
    )
    assert georef.transform == transform
    assert georef.crop_size == pytest.approx(((x1 - x0) * 0.05, (y1 - y0) * 0.05), abs=0.2)
    assert georef.ortho_drift_m < 0.5

    # Without it, the view holds the whole rotated image
    georef = build_label_georeference(transform, width, height, None)
    assert georef.transform == transform
    view = NorthUpView(georef, width, height)
    for corner in ((0, 0), (width, 0), (width, height), (0, height)):
        x, y = view.point(*corner)
        assert -1 <= x <= view.width + 1 and -1 <= y <= view.height + 1
    assert len(_markers(view.warp(label))) == len(MARKERS)


def test_rebuild_does_not_need_the_original_orthophoto():
    label, transform = _run_inference(20.85)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(20.85))
    assert georef.ortho_drift_m is None
    for p in _markers(label):
        assert _nearest(georef.transform * p, _truth()) < GROUND_TOLERANCE_M


def test_near_45_degrees_without_the_original_keeps_the_file_transform():
    label, transform = _run_inference(45)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(45))
    assert georef.transform == transform


def test_unrotated_crop_keeps_the_file_transform():
    label, transform = _run_inference(0)
    georef = build_label_georeference(
        transform, label.shape[1], label.shape[0], _crop_annotation(0),
        crop_transform=_back_mapping(0, label), original_transform=ORIGINAL,
    )
    assert georef.transform == transform
    assert not georef.rotated
    view = NorthUpView(georef, label.shape[1], label.shape[0])
    assert view.is_identity
    assert view.point(10.5, 20.25) == (10.5, 20.25)


def test_rotation_angle_comes_from_the_rotation_line():
    assert crop_rotation_deg(None) == 0.0
    assert crop_rotation_deg({"rotationLine": None}) == 0.0
    assert crop_rotation_deg({"rotationLine": _rotation_line(20.85)}) == pytest.approx(20.85)


def test_regenerated_orthophoto_is_flagged():
    label, transform = _run_inference(20.85)
    shifted = Affine.translation(3.0, 0) * ORIGINAL
    georef = build_label_georeference(
        transform, label.shape[1], label.shape[0], _crop_annotation(20.85),
        crop_transform=_back_mapping(20.85, label), original_transform=shifted,
    )
    assert georef.ortho_drift_m == pytest.approx(3.0, abs=0.1)
    for p in _markers(label):
        assert _nearest(georef.transform * p, _truth()) < GROUND_TOLERANCE_M


@pytest.mark.parametrize("rotation_in_transform", [False, True])
@pytest.mark.parametrize("deg", [20.85, -16.02, 60])
def test_north_up_view_restores_the_crop_layout(deg, rotation_in_transform):
    label, transform = _run_inference(deg, rotation_in_transform)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(deg))
    view = NorthUpView(georef, label.shape[1], label.shape[0])
    x0, y0, x1, y1 = CROP
    assert not view.is_identity
    # Canvas = the crop before rotation, give or take PIL's rounding (in 1.6 cm px)
    assert view.width == pytest.approx((x1 - x0) * SCALE, abs=10)
    assert view.height == pytest.approx((y1 - y0) * SCALE, abs=10)

    # North-up: the markers get their crop layout back, rotation and scale exact
    expected = [((mx + 0.5 - x0) * SCALE, (my + 0.5 - y0) * SCALE) for mx, my in MARKERS]
    offset, residual = _layout_error([view.point(*p) for p in _markers(label)], expected)
    assert offset < 8 and residual < 1.5

    warped = view.warp(label)
    assert warped.shape == (view.height, view.width)
    found = _markers(warped)
    assert len(found) == len(MARKERS)
    offset, residual = _layout_error(found, expected)
    assert offset < 8 and residual < 1.5

    assert view.warp(label, 0.25).shape == (round(view.height * 0.25), round(view.width * 0.25))
    window = view.window(label, 100, 80, 200, 150)
    assert np.abs(window.astype(int) - warped[80:230, 100:300].astype(int)).max() <= 1


def _panel_with_hotspot(bbox):
    defect = Defect(
        bbox=BoundingBox(left=bbox.left + 10, top=bbox.top + 10, width=10, height=10, label="hotspots"),
        defect_center_px=(bbox.left + 15, bbox.top + 15),
        panel_centroid_geospatial=GeospatialCoordinate(longitude=0, latitude=0),
        defect_type="hotspots",
    )
    return Panel(column=1, row=1, bbox=bbox, hotspots=[defect])


def test_dxf_draws_rotated_panels_where_they_are(tmp_path):
    from thermographic_report_builder.processing.dxf_generator import create_dxf_layers
    from thermographic_report_builder.utils.geospatial import PixelToLatLonConverter

    deg = 20.85
    label, transform = _run_inference(deg)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(deg))
    bbox = BoundingBox(left=500, top=400, width=60, height=100, label="solarpanels")
    converter = PixelToLatLonConverter(georef.transform, "EPSG:32722")
    create_dxf_layers({(1, 1): _panel_with_hotspot(bbox)}, converter, tmp_path / "layers.dxf", "T")

    doc = ezdxf.readfile(tmp_path / "layers.dxf")
    outline = next(e for e in doc.modelspace() if e.dxftype() == "LWPOLYLINE" and "Panels" in e.dxf.layer)
    points = [tuple(p[:2]) for p in outline.get_points()][:4]
    corners = [(500, 400), (560, 400), (560, 500), (500, 500)]
    for p, c in zip(points, corners):
        assert math.dist(p, georef.transform * c) < 1e-6
    (ax, ay), (bx, by) = points[0], points[1]
    assert math.degrees(math.atan2(by - ay, bx - ax)) == pytest.approx(-deg, abs=0.01)


def test_report_maps_are_drawn_north_up(tmp_path):
    from thermographic_report_builder.processing.annotation import annotate_orthophoto, create_layer_image
    from thermographic_report_builder.processing.cropper import crop_defect_regions

    deg = 20.85
    label, transform = _run_inference(deg)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(deg))
    view = NorthUpView(georef, label.shape[1], label.shape[0])
    ortho = tmp_path / "ortho.tif"
    profile = dict(driver="GTiff", width=label.shape[1], height=label.shape[0], count=3,
                   dtype="uint8", crs="EPSG:32722", transform=transform)
    with rasterio.open(ortho, "w", **profile) as dst:
        dst.write(np.stack([label] * 3))

    # Panel centred on the middle marker: its north-up crop must have that marker in the middle
    mx, my = min(_markers(label), key=lambda p: math.dist(p, (label.shape[1] / 2, label.shape[0] / 2)))
    bbox = BoundingBox(left=mx - 30, top=my - 50, width=60, height=100, label="solarpanels")
    grid = {(1, 1): _panel_with_hotspot(bbox)}

    annotate_orthophoto(ortho, grid, tmp_path / "ortho.png", 0.25, view, tmp_path / "ortho_north_up.png")
    assert (tmp_path / "ortho.png").exists() and (tmp_path / "ortho_north_up.png").exists()
    north_up = cv2.imread(str(tmp_path / "ortho_north_up.png"))
    assert north_up.shape[:2] == (round(view.height * 0.25), round(view.width * 0.25))

    create_layer_image(grid, label.shape[1], label.shape[0], tmp_path / "layer_img.pdf", view=view)
    assert (tmp_path / "layer_img.pdf").stat().st_size > 0

    crop_defect_regions(ortho, grid, tmp_path, scale_factor=1.0, view=view)
    crop = cv2.imread(str(tmp_path / "hotspots_(1-1)_cropped.jpg"), cv2.IMREAD_GRAYSCALE)
    assert (tmp_path / "hotspots_(1-1)_minimap.pdf").exists()
    centre = (crop.shape[1] / 2, crop.shape[0] / 2)
    assert _nearest(centre, _markers(crop)) < 3


def test_unrotated_report_maps_skip_the_north_up_copy(tmp_path):
    from thermographic_report_builder.processing.annotation import annotate_orthophoto

    label, transform = _run_inference(0)
    georef = build_label_georeference(transform, label.shape[1], label.shape[0], _crop_annotation(0))
    view = NorthUpView(georef, label.shape[1], label.shape[0])
    ortho = tmp_path / "ortho.tif"
    profile = dict(driver="GTiff", width=label.shape[1], height=label.shape[0], count=3,
                   dtype="uint8", crs="EPSG:32722", transform=transform)
    with rasterio.open(ortho, "w", **profile) as dst:
        dst.write(np.stack([label] * 3))
    annotate_orthophoto(ortho, {}, tmp_path / "ortho.png", 0.25, view, tmp_path / "ortho_north_up.png")
    assert (tmp_path / "ortho.png").exists()
    assert not (tmp_path / "ortho_north_up.png").exists()
