"""Generate DXF layers from panel grid with georeferenced coordinates.

Creates a DXF file with layers showing:
- All panels with outlines
- Panels with defects (colored/filled)
- Defect markers with labels
- Panel ID annotations
"""

import ezdxf
from pathlib import Path
from typing import Dict, Tuple
from ezdxf.math import Vec2

from ..models.defect import Panel
from ..utils.logger import get_logger
from ..utils import PixelToLatLonConverter

logger = get_logger(__name__)

# DXF color codes (AutoCAD ACI colors)
COLOR_WHITE = 7
COLOR_RED = 1      # Hotspots
COLOR_BLUE = 5     # Faulty diodes
COLOR_YELLOW = 2   # Offline panels
COLOR_GREEN = 3    # Normal panels
COLOR_GRAY = 8


def create_dxf_layers(
    panel_grid: Dict[Tuple[int, int], Panel],
    geo_converter: PixelToLatLonConverter,
    output_path: Path,
    area_name: str = "Solar Farm",
) -> Path:
    """
    Create a DXF file with georeferenced panel and defect layers.

    Args:
        panel_grid: Dictionary of panels with defects
        geo_converter: Converter for pixel to geospatial coordinates
        output_path: Path to save DXF file
        area_name: Name of the solar farm/area for layer naming

    Returns:
        Path to created DXF file
    """
    logger.info(f"Creating DXF layers for {len(panel_grid)} panels")

    # Create new DXF document (AutoCAD 2018 format for compatibility)
    doc = ezdxf.new(dxfversion="R2018")
    msp = doc.modelspace()

    # Create layers with appropriate colors
    _create_layers(doc, area_name)

    # Statistics
    panels_drawn = 0
    defects_drawn = 0

    # Draw all panels and defects
    for panel in panel_grid.values():
        _draw_panel(msp, panel, geo_converter, area_name)
        panels_drawn += 1

        if panel.has_defects:
            defects_drawn += _draw_defects(msp, panel, geo_converter, area_name)

    # Save DXF file
    doc.saveas(str(output_path))

    logger.info(
        f"Created DXF file: {output_path.name} "
        f"({panels_drawn} panels, {defects_drawn} defects)"
    )

    return output_path


def _create_layers(doc: ezdxf.document.Drawing, area_name: str) -> None:
    """Create DXF layers with appropriate colors and properties."""
    layers = [
        (f"GRETA - {area_name} - All Panels", COLOR_GRAY),
        (f"GRETA - {area_name} - Affected Panels", COLOR_RED),
        (f"GRETA - {area_name} - Hotspots", COLOR_RED),
        (f"GRETA - {area_name} - Faulty Diodes", COLOR_BLUE),
        (f"GRETA - {area_name} - Offline Panels", COLOR_YELLOW),
        (f"GRETA - {area_name} - Panel Labels", COLOR_WHITE),
        (f"GRETA - {area_name} - Defect Labels", COLOR_RED),
    ]

    for layer_name, color in layers:
        doc.layers.add(layer_name, color=color)


def _pixel_to_geo(
    x: float, y: float, geo_converter: PixelToLatLonConverter
) -> Tuple[float, float]:
    """Convert pixel coordinates to geospatial coordinates for DXF.

    Note: DXF typically uses projected coordinates (meters) not lat/lon.
    We use the rasterio transform to get the projected coordinates directly.
    """
    # Use the transform to convert pixel to projected coordinates
    # transform * (col, row) = (x, y) in projected CRS
    transform = geo_converter.transform
    geo_x = transform.c + x * transform.a + y * transform.b
    geo_y = transform.f + x * transform.d + y * transform.e
    return (geo_x, geo_y)


def _draw_panel(
    msp,
    panel: Panel,
    geo_converter: PixelToLatLonConverter,
    area_name: str,
) -> None:
    """Draw a panel outline and label on the DXF."""
    bbox = panel.bbox

    # Get corner coordinates in pixels
    corners_px = [
        (bbox.left, bbox.top),
        (bbox.right, bbox.top),
        (bbox.right, bbox.bottom),
        (bbox.left, bbox.bottom),
    ]

    # Convert to geospatial coordinates
    corners_geo = [_pixel_to_geo(x, y, geo_converter) for x, y in corners_px]

    # Determine layer and color based on defect status
    if panel.has_defects:
        layer_name = f"GRETA - {area_name} - Affected Panels"
        color = COLOR_RED
    else:
        layer_name = f"GRETA - {area_name} - All Panels"
        color = COLOR_GRAY

    # Draw panel outline as closed polyline
    msp.add_lwpolyline(
        corners_geo + [corners_geo[0]],  # Close the polygon
        dxfattribs={"layer": layer_name, "color": color},
    )

    # Add panel label at center
    center_x, center_y = bbox.center
    label_pos = _pixel_to_geo(center_x, center_y, geo_converter)

    # Calculate text height based on panel size (roughly 10% of panel width)
    geo_width = abs(corners_geo[1][0] - corners_geo[0][0])
    text_height = max(0.5, geo_width * 0.1)  # Minimum 0.5 units

    msp.add_text(
        panel.panel_id,
        dxfattribs={
            "layer": f"GRETA - {area_name} - Panel Labels",
            "insert": label_pos,
            "height": text_height,
            "color": COLOR_WHITE if panel.has_defects else COLOR_GRAY,
        },
    )


def _draw_defects(
    msp,
    panel: Panel,
    geo_converter: PixelToLatLonConverter,
    area_name: str,
) -> int:
    """Draw defect markers for a panel. Returns count of defects drawn."""
    defects_drawn = 0

    defect_config = [
        ("hotspots", f"GRETA - {area_name} - Hotspots", COLOR_RED),
        ("faulty_diodes", f"GRETA - {area_name} - Faulty Diodes", COLOR_BLUE),
        ("offline_panels", f"GRETA - {area_name} - Offline Panels", COLOR_YELLOW),
    ]

    for defect_attr, layer_name, color in defect_config:
        defects = getattr(panel, defect_attr, [])

        for defect in defects:
            bbox = defect.bbox

            # Get defect bounding box corners
            corners_px = [
                (bbox.left, bbox.top),
                (bbox.right, bbox.top),
                (bbox.right, bbox.bottom),
                (bbox.left, bbox.bottom),
            ]

            corners_geo = [_pixel_to_geo(x, y, geo_converter) for x, y in corners_px]

            # Draw defect outline
            msp.add_lwpolyline(
                corners_geo + [corners_geo[0]],
                dxfattribs={"layer": layer_name, "color": color},
            )

            # Add filled hatch for visibility
            hatch = msp.add_hatch(dxfattribs={"layer": layer_name, "color": color})
            hatch.paths.add_polyline_path(
                [Vec2(p) for p in corners_geo],
                is_closed=True,
            )
            hatch.set_solid_fill(color=color)

            defects_drawn += 1

    return defects_drawn


def create_dxf_from_orthophoto(
    ortho_path: Path,
    panel_grid: Dict[Tuple[int, int], Panel],
    output_path: Path,
    area_name: str = "Solar Farm",
) -> Path:
    """
    Convenience function to create DXF directly from orthophoto file.

    Args:
        ortho_path: Path to georeferenced orthophoto (GeoTIFF)
        panel_grid: Dictionary of panels with defects
        output_path: Path to save DXF file
        area_name: Name of the solar farm/area

    Returns:
        Path to created DXF file
    """
    import rasterio

    with rasterio.open(ortho_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs

    # Create a simple geo converter
    geo_converter = PixelToLatLonConverter(transform, crs)

    return create_dxf_layers(
        panel_grid=panel_grid,
        geo_converter=geo_converter,
        output_path=output_path,
        area_name=area_name,
    )
