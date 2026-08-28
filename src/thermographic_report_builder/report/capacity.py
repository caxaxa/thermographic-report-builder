"""Affected-capacity summary — physical quantities only, no financial modeling.

Deliberate design (2026-08-28): the report states verifiable physics — module
counts, affected DC capacity, ΔT severity — and explicitly does NOT convert to
currency. Revenue impact depends on PPA terms, DC/AC oversizing (clipping can
absorb DC losses entirely), module technology and degradation state, none of
which are observable from the air. A wrong € figure would undermine the
report's evidence-grade positioning; affected kWp is the technology- and
tariff-neutral quantity every engineer, CFO and loss adjuster accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models.annotation_manifest import AnnotationManifest
from ..models.defect import Panel

# Printed as an explicit assumption; the only parameter in the section.
DEFAULT_MODULE_WP = 550

# Manifest severity taxonomy, most urgent first.
SEVERITY_ORDER = {"CRITICAL": 0, "ATTENTION": 1, "MONITORING": 2}

# Cap for the priority worklist table (full detail lives in the per-defect
# sections that follow).
PRIORITY_LIST_LIMIT = 20


@dataclass
class ClassCapacity:
    """Affected panels and DC capacity for one defect class."""

    defect_type: str
    panels_affected: int
    kwp_affected: float


@dataclass
class PriorityItem:
    """One row of the severity-ranked maintenance worklist."""

    panel_id: str
    defect_type: str
    delta_t: Optional[float]
    severity: str


@dataclass
class CapacitySummary:
    """Everything the capacity section prints."""

    module_wp: int
    total_panels: int
    compromised_panels: int
    compromised_pct: float
    total_kwp: float
    affected_kwp: float
    by_class: List[ClassCapacity] = field(default_factory=list)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    max_delta_t: Optional[float] = None
    priority_list: List[PriorityItem] = field(default_factory=list)


def compute_capacity_summary(
    panel_grid: Dict[str, Panel],
    annotation_manifest: Optional[AnnotationManifest] = None,
    module_wp: int = DEFAULT_MODULE_WP,
) -> CapacitySummary:
    """Derive the affected-capacity summary from detections (+ manifest ΔT)."""
    panels = list(panel_grid.values())
    total = len(panels)
    compromised = [p for p in panels if p.has_defects]

    by_class = []
    for attr, dtype in (
        ("hotspots", "hotspots"),
        ("faulty_diodes", "faultydiodes"),
        ("offline_panels", "offlinepanels"),
    ):
        affected = sum(1 for p in panels if getattr(p, attr))
        if affected:
            by_class.append(
                ClassCapacity(
                    defect_type=dtype,
                    panels_affected=affected,
                    kwp_affected=round(affected * module_wp / 1000.0, 1),
                )
            )

    severity_counts: Dict[str, int] = {}
    max_dt: Optional[float] = None
    priority: List[PriorityItem] = []
    if annotation_manifest and annotation_manifest.annotations:
        for a in annotation_manifest.annotations:
            sev = (a.severity or "MONITORING").upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if a.delta_t is not None:
                max_dt = a.delta_t if max_dt is None else max(max_dt, a.delta_t)
            priority.append(
                PriorityItem(
                    panel_id=a.panel_id,
                    defect_type=a.defect_type,
                    delta_t=a.delta_t,
                    severity=sev,
                )
            )
        priority.sort(
            key=lambda i: (
                SEVERITY_ORDER.get(i.severity, 99),
                -(i.delta_t if i.delta_t is not None else 0.0),
            )
        )
        priority = priority[:PRIORITY_LIST_LIMIT]
    else:
        # No manifest (older projects): fall back to compromised panels so the
        # worklist still exists, ordered by defect count.
        for p in sorted(compromised, key=lambda p: -p.defect_count)[
            :PRIORITY_LIST_LIMIT
        ]:
            dtype = (
                "offlinepanels"
                if p.offline_panels
                else "faultydiodes" if p.faulty_diodes else "hotspots"
            )
            priority.append(
                PriorityItem(
                    panel_id=p.panel_id,
                    defect_type=dtype,
                    delta_t=None,
                    severity="—",
                )
            )

    return CapacitySummary(
        module_wp=module_wp,
        total_panels=total,
        compromised_panels=len(compromised),
        compromised_pct=round(100.0 * len(compromised) / total, 1) if total else 0.0,
        total_kwp=round(total * module_wp / 1000.0, 1),
        affected_kwp=round(len(compromised) * module_wp / 1000.0, 1),
        by_class=by_class,
        severity_counts=severity_counts,
        max_delta_t=max_dt,
        priority_list=priority,
    )
