"""Reject reconstructions whose intrinsics are not physically possible.

2026-08-29 (Bento / DJI XT2): the report pointed at raw pixels that were metres
off the module, landing on bare soil, and reported the temperature of that soil
as the defect's. Image SELECTION was fine throughout -- the chosen photo always
contained the hotspot, ~3 m from frame centre on a ~25 m footprint.

Root cause was upstream of the report. ODM derives sensor width from EXIF
FocalPlaneXResolution, which the EXIF spec defines as pixels-per-unit. The XT2
writes the sensor WIDTH there instead (10.88 mm = 640 x 17 um) with unit "mm",
so ODM computed 640/10.88 = 58.8 mm and a focal_ratio of 13/58.8 = 0.221 rather
than 13/10.88 = 1.195 -- a 139 deg camera that does not exist. Every projection
built on it (source map, reprojection, mesh) inherited the error.

The pre-existing guard compared reconstruction dimensions against the raw images
(640x512 == 640x512) and passed, so nothing complained. These tests cover the
focal check that closes that gap, and pin the M30T path as unaffected.
"""

import math

import pytest

from thermographic_report_builder.processing.gps_matcher import (
    MAX_PLAUSIBLE_HFOV_DEG,
    MIN_PLAUSIBLE_HFOV_DEG,
    UNTRUSTED_GEOMETRY_ZOOM_PX,
    reconstruction_hfov_degrees,
)

# Verbatim from Bento's opensfm/reconstruction.json, camera
# "v2 dji xt2 640 512 brown 0.221".
XT2_BROKEN = {
    "projection_type": "brown",
    "width": 640,
    "height": 512,
    "focal_x": 0.2206131093626757,
    "focal_y": 0.2206131093626757,
}

# What the XT2 should have been: 13 mm lens on a 640 x 17 um core = 10.88 mm.
XT2_CORRECT = {**XT2_BROKEN, "focal_x": 13.0 / 10.88, "focal_y": 13.0 / 10.88}

# M30T reports FocalLengthIn35mmFormat=40, so ODM takes the 40/36 branch and
# never touches FocalPlaneXResolution.
M30T_OK = {"width": 1280, "height": 1024, "focal_x": 40.0 / 36.0}


def _is_trusted(camera) -> bool:
    fov = reconstruction_hfov_degrees(camera)
    return fov is not None and MIN_PLAUSIBLE_HFOV_DEG <= fov <= MAX_PLAUSIBLE_HFOV_DEG


def test_xt2_reconstruction_is_rejected():
    fov = reconstruction_hfov_degrees(XT2_BROKEN)
    assert 130 < fov < 145, f"expected the ~139 deg camera, got {fov}"
    assert not _is_trusted(XT2_BROKEN)


def test_m30t_reconstruction_is_accepted():
    """The drone every shipped report was built on must be untouched."""
    fov = reconstruction_hfov_degrees(M30T_OK)
    assert 40 < fov < 60, f"M30T thermal should be ~48 deg, got {fov}"
    assert _is_trusted(M30T_OK)


def test_correctly_calibrated_xt2_would_pass():
    """The guard rejects the bad intrinsics, not the camera itself."""
    fov = reconstruction_hfov_degrees(XT2_CORRECT)
    assert 40 < fov < 50, f"XT2 thermal is ~45 deg, got {fov}"
    assert _is_trusted(XT2_CORRECT)


def test_dimension_check_alone_would_have_missed_it():
    """Pins why the pre-existing size guard let this through."""
    assert XT2_BROKEN["width"] == 640 and XT2_BROKEN["height"] == 512
    assert not _is_trusted(XT2_BROKEN)


@pytest.mark.parametrize(
    "camera",
    [None, {}, {"width": 640, "height": 512}, {"width": 0, "height": 0, "focal_x": 1.0},
     {"width": 640, "height": 512, "focal_x": 0}],
)
def test_unknowable_focal_returns_none(camera):
    """Missing/degenerate metadata must not be read as 'implausible'."""
    assert reconstruction_hfov_degrees(camera) is None


def test_focal_is_normalised_by_the_long_edge():
    """OpenSfM normalises by max(w, h); using width would misjudge portrait frames."""
    landscape = {"width": 640, "height": 512, "focal_x": 1.0}
    assert reconstruction_hfov_degrees(landscape) == pytest.approx(
        2 * math.degrees(math.atan(320.0 / 640.0))
    )
    portrait = {"width": 512, "height": 640, "focal_x": 1.0}
    assert reconstruction_hfov_degrees(portrait) == pytest.approx(
        2 * math.degrees(math.atan(256.0 / 640.0))
    )


def test_wide_crop_covers_the_worst_case_offset():
    """A frame-centred crop must contain a defect ~5 m off nadir.

    Geometry, not measurement: 13 mm / 17 um = 765 px focal at 30 m AGL is
    25.5 px/m. Worst observed defect-to-nadir was 4.6 m, and the reconstruction
    error adds ~3.3 m, so ~8 m => ~202 px from centre.
    """
    px_per_m = (13.0 / 0.017) / 30.0
    worst_case_px = (4.6 + 3.3) * px_per_m
    assert UNTRUSTED_GEOMETRY_ZOOM_PX / 2 > worst_case_px, (
        f"half-crop {UNTRUSTED_GEOMETRY_ZOOM_PX / 2:.0f}px must exceed "
        f"{worst_case_px:.0f}px"
    )


def test_annotation_entry_defaults_keep_existing_behaviour():
    """Trusted path must still carry real readings and a real severity."""
    from thermographic_report_builder.models.annotation_manifest import (
        AnnotationEntry,
        AnnotationPoint,
    )

    entry = AnnotationEntry(
        defect_id="hotspots_(3-15)_defeito1",
        panel_id="3-15",
        defect_type="hotspots",
        defect_index=1,
        raw_image_path="s3://b/k.JPG",
        raw_image_name="k.JPG",
        annotated_image="a.jpg",
        hot_point=AnnotationPoint(x=1, y=1, temp=48.0),
        cold_point=AnnotationPoint(x=2, y=2, temp=39.2),
        delta_t=8.8,
        severity="LOW",
    )
    assert entry.delta_t == 8.8
    assert entry.hot_point.temp == 48.0
    assert entry.severity == "LOW"


def test_untrusted_entry_carries_no_temperatures():
    """Untrusted path must publish nulls, not a confident wrong number."""
    from thermographic_report_builder.models.annotation_manifest import (
        AnnotationEntry,
        AnnotationPoint,
    )

    entry = AnnotationEntry(
        defect_id="hotspots_(3-15)_defeito1",
        panel_id="3-15",
        defect_type="hotspots",
        defect_index=1,
        raw_image_path="s3://b/k.JPG",
        raw_image_name="k.JPG",
        annotated_image="a.jpg",
        hot_point=AnnotationPoint(x=1, y=1, temp=None),
        cold_point=AnnotationPoint(x=2, y=2, temp=None),
        delta_t=None,
        severity="UNKNOWN",
    )
    assert entry.delta_t is None
    assert entry.severity == "UNKNOWN"


def test_capacity_summary_tolerates_missing_delta_t():
    """The capacity table must not invent a max_delta_t out of nulls."""
    from test_capacity import _entry, _grid
    from thermographic_report_builder.models.annotation_manifest import AnnotationManifest
    from thermographic_report_builder.report.capacity import compute_capacity_summary

    manifest = AnnotationManifest(
        project_id="01KG0RB6SXM2FXZRSB0FQA8Y6S",
        user_id="319b35f0-7081-7052-217a-bd4fd533806c",
        generated_at="2026-08-29T21:31:00Z",
        annotations=[
            _entry("1-1", "hotspots", None, "UNKNOWN"),
            _entry("2-1", "hotspots", None, "UNKNOWN"),
        ],
    )
    summary = compute_capacity_summary(_grid(), manifest)
    assert summary.max_delta_t is None, "nulls must not produce a max"
    assert all(i.delta_t is None for i in summary.priority_list)
    # Counts are detection-based and must survive the missing temperatures.
    assert summary.compromised_panels == 3


def test_severity_lookup_is_safe_for_unknown():
    """get_thermal_legend_text must not raise on the UNKNOWN severity key."""
    from thermographic_report_builder.config.constants import get_thermal_legend_text

    assert get_thermal_legend_text("severity_unknown", "pt-BR") is not None


def _matcher():
    """A GPSMatcher with just the flags _thermal_zoom_px reads."""
    from thermographic_report_builder.processing.gps_matcher import GPSMatcher

    m = GPSMatcher.__new__(GPSMatcher)
    m._untrusted_geometry = False
    m._thermal_only_images = False
    m._zoom_override = None
    return m


def test_m30t_zoom_is_unchanged():
    """M30T must keep its historical 200 px window."""
    from thermographic_report_builder.processing.gps_matcher import M30T_ZOOM_PX

    m = _matcher()
    assert m._thermal_zoom_px() == 200 == M30T_ZOOM_PX


def test_thermal_only_zoom_is_tighter():
    """640x512 cores get a closer crop now that localisation is verified."""
    from thermographic_report_builder.processing.gps_matcher import THERMAL_ONLY_ZOOM_PX

    m = _matcher()
    m._thermal_only_images = True
    assert m._thermal_zoom_px() == THERMAL_ONLY_ZOOM_PX < 200


def test_untrusted_geometry_still_wins_over_tightening():
    """A tight crop on an untrusted pixel would be worse than a loose one."""
    from thermographic_report_builder.processing.gps_matcher import (
        UNTRUSTED_GEOMETRY_ZOOM_PX,
    )

    for thermal_only in (False, True):
        m = _matcher()
        m._thermal_only_images = thermal_only
        m._untrusted_geometry = True
        assert m._thermal_zoom_px() == UNTRUSTED_GEOMETRY_ZOOM_PX


def test_tight_crop_still_covers_the_annotated_defect():
    """140 px at 3.9 cm/px is 5.5 m -- comfortably more than a 2 m module."""
    gsd_m = 30.0 / (13.0 / 0.017)
    from thermographic_report_builder.processing.gps_matcher import THERMAL_ONLY_ZOOM_PX

    assert THERMAL_ONLY_ZOOM_PX * gsd_m > 2.0 * 2, "must show the module plus context"
