"""Tests for the affected-capacity summary (physical quantities, no currency).

Pins the 2026-08-28 design decision: the report states modules/kWp/severity —
traceable facts — and never prints financial figures (PPA terms, DC/AC
oversizing and module technology make aerial € estimates indefensible).
"""

from thermographic_report_builder.models.annotation_manifest import (
    AnnotationManifest,
    AnnotationEntry,
    AnnotationPoint,
)
from thermographic_report_builder.models.defect import (
    BoundingBox,
    Defect,
    GeospatialCoordinate,
    Panel,
)
from thermographic_report_builder.report.capacity import (
    DEFAULT_MODULE_WP,
    PRIORITY_LIST_LIMIT,
    compute_capacity_summary,
)


def _defect(dtype: str) -> Defect:
    return Defect(
        bbox=BoundingBox(left=10, top=10, width=20, height=30, label=dtype),
        defect_center_px=(20.0, 25.0),
        panel_centroid_geospatial=GeospatialCoordinate(longitude=-55.0, latitude=-20.0),
        defect_type=dtype,
    )


def _panel(col, row, hotspots=0, diodes=0, offline=0) -> Panel:
    return Panel(
        column=col,
        row=row,
        bbox=BoundingBox(left=col * 10, top=row * 10, width=9, height=9, label="solarpanels"),
        hotspots=[_defect("hotspots") for _ in range(hotspots)],
        faulty_diodes=[_defect("faulty_diodes") for _ in range(diodes)],
        offline_panels=[_defect("offline_panels") for _ in range(offline)],
    )


def _entry(panel_id, dtype, delta_t, severity) -> AnnotationEntry:
    return AnnotationEntry(
        defect_id=f"{dtype}_({panel_id})_1",
        panel_id=panel_id,
        defect_type=dtype,
        defect_index=1,
        raw_image_path="s3://x/raw.jpg",
        raw_image_name="raw.jpg",
        annotated_image="ann.jpg",
        hot_point=AnnotationPoint(x=1, y=1, temp=60.0),
        cold_point=AnnotationPoint(x=2, y=2, temp=30.0),
        delta_t=delta_t,
        severity=severity,
    )


def _grid():
    return {
        "1-1": _panel(1, 1, hotspots=2),
        "1-2": _panel(1, 2, diodes=1),
        "2-1": _panel(2, 1, offline=1),
        "2-2": _panel(2, 2),  # healthy
        "2-3": _panel(2, 3),  # healthy
    }


class TestCapacitySummary:
    def test_counts_and_kwp(self):
        s = compute_capacity_summary(_grid(), None, module_wp=500)
        assert s.total_panels == 5
        assert s.compromised_panels == 3
        assert s.compromised_pct == 60.0
        assert s.total_kwp == 2.5
        assert s.affected_kwp == 1.5  # 3 panels x 500 Wp
        by_type = {c.defect_type: c for c in s.by_class}
        assert by_type["hotspots"].panels_affected == 1
        assert by_type["faultydiodes"].kwp_affected == 0.5
        assert by_type["offlinepanels"].panels_affected == 1

    def test_default_wp(self):
        s = compute_capacity_summary(_grid())
        assert s.module_wp == DEFAULT_MODULE_WP

    def test_priority_from_manifest_orders_by_severity_then_dt(self):
        manifest = AnnotationManifest(
            project_id="p1",
            user_id="u1",
            generated_at="2026-08-28T00:00:00Z",
            annotations=[
                _entry("1-1", "hotspots", 12.0, "MONITORING"),
                _entry("1-2", "hotspots", 55.0, "CRITICAL"),
                _entry("2-1", "hotspots", 30.0, "ATTENTION"),
                _entry("1-1", "hotspots", 80.0, "CRITICAL"),
            ]
        )
        s = compute_capacity_summary(_grid(), manifest)
        ids = [(i.severity, i.delta_t) for i in s.priority_list]
        assert ids == [
            ("CRITICAL", 80.0),
            ("CRITICAL", 55.0),
            ("ATTENTION", 30.0),
            ("MONITORING", 12.0),
        ]
        assert s.severity_counts == {"CRITICAL": 2, "ATTENTION": 1, "MONITORING": 1}
        assert s.max_delta_t == 80.0

    def test_priority_capped(self):
        manifest = AnnotationManifest(
            project_id="p1",
            user_id="u1",
            generated_at="2026-08-28T00:00:00Z",
            annotations=[
                _entry(f"1-{i}", "hotspots", float(i), "MONITORING") for i in range(40)
            ]
        )
        s = compute_capacity_summary(_grid(), manifest)
        assert len(s.priority_list) == PRIORITY_LIST_LIMIT

    def test_fallback_without_manifest(self):
        s = compute_capacity_summary(_grid(), None)
        assert s.severity_counts == {}
        assert len(s.priority_list) == 3  # the compromised panels
        assert s.priority_list[0].panel_id == "1-1"  # most defects first
        assert s.priority_list[0].delta_t is None

    def test_empty_grid(self):
        s = compute_capacity_summary({}, None)
        assert s.total_panels == 0
        assert s.compromised_pct == 0.0
        assert s.priority_list == []
