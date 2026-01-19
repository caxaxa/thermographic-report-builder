"""Flight visualization module for generating flight and orthophoto appendix.

This module replaces the ODM HTML report (skipped via --skip-report flag) with custom
visualizations built from the raw artifacts that ARE uploaded:
- reconstruction.json (camera poses, orientations, GPS coordinates)
- Orthophoto GeoTIFF (bounds, GSD, CRS metadata)
- DSM/DTM GeoTIFFs (elevation data)
- tracks.csv (feature correspondences, optional)

The module generates:
- Flight path maps overlaid on orthophoto
- Coverage/overlap heatmaps
- Altitude, speed, and heading time-series charts
- GSD heatmaps
- Summary tables (flight stats, coverage stats, reconstruction health)

CRITICAL: Coordinate System Handling
    OpenSfM uses local ENU (East-North-Up) coordinates, NOT WGS84 lat/lon.
    All coordinate conversions from ENU to WGS84 are handled in parsers.py.
    Never use reconstruction.json coordinates directly without conversion!

CRITICAL: GPS Prediction Error Metric
    The GPS error (distance between raw GPS and bundle-adjusted position) is
    a key quality metric that indicates reconstruction accuracy.
    - Mean GPS Error: Average error across all images
    - Max GPS Error: Worst-case error in dataset
    See docs/FLIGHT_APPENDIX.md for interpretation guidelines.

Documentation:
    Complete technical documentation available at:
    docs/FLIGHT_APPENDIX.md

Example Usage:
    from thermographic_report_builder.flight_viz import generate_flight_appendix

    viz_paths, stats = generate_flight_appendix(
        work_dir=Path("/tmp/work"),
        output_dir=Path("/tmp/output")
    )

    # GPS error metrics (IMPORTANT)
    print(f"Mean GPS Error: {stats['mean_gps_error_m']} m")
    print(f"Max GPS Error: {stats['max_gps_error_m']} m")

    # Other statistics
    print(f"Total Distance: {stats['total_distance_km']} km")
    print(f"Coverage Area: {stats['coverage_area_ha']} ha")
"""

from .models import FlightData, FlightMetrics, CoverageMetrics
from .parsers import ReconstructionParser, GeoTIFFMetadataParser
from .metrics import FlightMetricsCalculator, CoverageMetricsCalculator
from .visualizations import FlightPathVisualizer, CoverageHeatmapVisualizer, ChartsVisualizer
from .integration import FlightVisualizationGenerator, generate_flight_appendix

__all__ = [
    # Models
    "FlightData",
    "FlightMetrics",
    "CoverageMetrics",
    # Parsers
    "ReconstructionParser",
    "GeoTIFFMetadataParser",
    # Metrics
    "FlightMetricsCalculator",
    "CoverageMetricsCalculator",
    # Visualizations
    "FlightPathVisualizer",
    "CoverageHeatmapVisualizer",
    "ChartsVisualizer",
    # Integration
    "FlightVisualizationGenerator",
    "generate_flight_appendix",
]
