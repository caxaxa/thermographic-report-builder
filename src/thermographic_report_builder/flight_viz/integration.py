"""Integration module for flight visualizations with report builder."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metrics import FlightMetricsCalculator
from .models import FlightData, FlightMetrics
from .parsers import GeoTIFFMetadataParser, ReconstructionParser
from .visualizations import ChartsVisualizer, FlightPathVisualizer

logger = logging.getLogger(__name__)


class FlightVisualizationGenerator:
    """High-level interface for generating flight visualizations."""

    def __init__(self, work_dir: Path):
        """Initialize visualization generator.

        Args:
            work_dir: Working directory containing ODM artifacts
        """
        self.work_dir = Path(work_dir)

        # Initialize components
        self.reconstruction_parser = ReconstructionParser()
        self.geotiff_parser = GeoTIFFMetadataParser()
        self.metrics_calculator = FlightMetricsCalculator()
        self.flight_viz = FlightPathVisualizer()
        self.charts_viz = ChartsVisualizer()

    def generate_all(
        self,
        output_dir: Path,
        reconstruction_path: Optional[Path] = None,
        orthophoto_path: Optional[Path] = None,
        project_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Path], FlightMetrics]:
        """Generate all flight visualizations.

        Args:
            output_dir: Directory to save generated visualizations
            reconstruction_path: Path to reconstruction.json (if not in work_dir/opensfm/)
            orthophoto_path: Path to odm_orthophoto.tif (if not in work_dir/)
            project_name: Optional name of the project/solar farm for display in titles

        Returns:
            Tuple of:
                - Dictionary mapping visualization names to file paths
                - FlightMetrics object with computed statistics

        Raises:
            FileNotFoundError: If required artifacts are missing
            ValueError: If artifacts are malformed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Locate artifacts
        if reconstruction_path is None:
            reconstruction_path = self.work_dir / "opensfm" / "reconstruction.json"

        if orthophoto_path is None:
            ortho_candidates = [
                self.work_dir / "odm_orthophoto_original.tif",
                self.work_dir / "odm_orthophoto.tif",
            ]
            orthophoto_path = next(
                (path for path in ortho_candidates if path.exists()),
                ortho_candidates[-1],
            )

        # Parse flight data
        logger.info(f"Parsing reconstruction data from {reconstruction_path}")
        flight_data = self.reconstruction_parser.parse(reconstruction_path)

        # Calculate metrics
        logger.info("Calculating flight metrics")
        flight_metrics = self.metrics_calculator.calculate(flight_data)

        # Generate visualizations
        visualization_paths = {}

        try:
            # 1. Interactive flight path map
            logger.info("Generating interactive flight path map")
            map_path = output_dir / "flight_path_map.html"
            self.flight_viz.generate_map(
                flight_data,
                orthophoto_path=orthophoto_path if orthophoto_path.exists() else None,
                output_path=map_path,
            )
            visualization_paths["flight_path_map"] = map_path

        except Exception as e:
            logger.error(f"Failed to generate interactive map: {e}")

        try:
            # 2. Static flight path map (with orthophoto background)
            logger.info("Generating static flight path map with orthophoto overlay")
            static_map_path = output_dir / "flight_path_static.png"

            # Determine if orthophoto is available for background
            ortho_for_background = None
            if orthophoto_path is not None and orthophoto_path.exists():
                ortho_for_background = orthophoto_path
                logger.info(f"Orthophoto found at {orthophoto_path}, will overlay on static map")
            else:
                logger.info("No orthophoto available for static map background")

            self.flight_viz.generate_static_map(
                flight_data,
                output_path=static_map_path,
                orthophoto_path=ortho_for_background,
                project_name=project_name,
            )
            visualization_paths["flight_path_static"] = static_map_path

        except Exception as e:
            logger.error(f"Failed to generate static map: {e}")

        try:
            # 3. Altitude chart
            logger.info("Generating altitude chart")
            altitude_chart_path = output_dir / "altitude_profile.png"
            self.charts_viz.generate_altitude_chart(
                flight_data,
                output_path=altitude_chart_path,
            )
            visualization_paths["altitude_chart"] = altitude_chart_path

        except Exception as e:
            logger.error(f"Failed to generate altitude chart: {e}")

        try:
            # 4. Speed chart (if timestamps available)
            logger.info("Generating speed chart")
            speed_chart_path = output_dir / "speed_profile.png"
            result = self.charts_viz.generate_speed_chart(
                flight_data,
                output_path=speed_chart_path,
            )
            if result:
                visualization_paths["speed_chart"] = speed_chart_path

        except Exception as e:
            logger.error(f"Failed to generate speed chart: {e}")

        try:
            # 5. Combined dashboard
            logger.info("Generating combined dashboard")
            dashboard_path = output_dir / "flight_dashboard.png"
            self.charts_viz.generate_combined_dashboard(
                flight_data,
                flight_metrics,
                output_path=dashboard_path,
            )
            visualization_paths["dashboard"] = dashboard_path

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")

        logger.info(f"Generated {len(visualization_paths)} visualizations")

        return visualization_paths, flight_metrics


def generate_flight_appendix(
    work_dir: Path,
    output_dir: Path,
    reconstruction_path: Optional[Path] = None,
    orthophoto_path: Optional[Path] = None,
    project_name: Optional[str] = None,
) -> Tuple[Dict[str, Path], Dict[str, any]]:
    """Generate flight visualization appendix for report.

    This is the main entry point called by the report builder.

    Args:
        work_dir: Working directory containing ODM artifacts
        output_dir: Directory to save generated visualizations
        reconstruction_path: Optional path to reconstruction.json
        orthophoto_path: Optional path to odm_orthophoto.tif
        project_name: Optional name of the project/solar farm for display in titles

    Returns:
        Tuple of:
            - Dictionary mapping visualization names to file paths
            - Dictionary with formatted statistics for tables
    """
    generator = FlightVisualizationGenerator(work_dir)

    # Generate all visualizations
    viz_paths, metrics = generator.generate_all(
        output_dir,
        reconstruction_path=reconstruction_path,
        orthophoto_path=orthophoto_path,
        project_name=project_name,
    )

    # Format statistics for report tables
    stats_dict = {
        "total_images": metrics.total_images,
        "total_distance_m": f"{metrics.total_distance:.0f}",
        "total_distance_km": f"{metrics.total_distance / 1000:.2f}",
        "flight_duration_min": f"{metrics.flight_duration / 60:.1f}" if metrics.flight_duration else "N/A",
        "min_altitude_m": f"{metrics.min_altitude:.1f}",
        "max_altitude_m": f"{metrics.max_altitude:.1f}",
        "mean_altitude_m": f"{metrics.mean_altitude:.1f}",
        "median_altitude_m": f"{metrics.median_altitude:.1f}",
        "coverage_area_ha": f"{metrics.coverage_area / 10000:.2f}",
        "coverage_area_m2": f"{metrics.coverage_area:.0f}",
        "bbox_min_lat": f"{metrics.bbox_min_lat:.6f}",
        "bbox_max_lat": f"{metrics.bbox_max_lat:.6f}",
        "bbox_min_lon": f"{metrics.bbox_min_lon:.6f}",
        "bbox_max_lon": f"{metrics.bbox_max_lon:.6f}",
    }

    # Add optional speed stats
    if metrics.mean_speed:
        stats_dict.update({
            "min_speed_ms": f"{metrics.min_speed:.1f}",
            "max_speed_ms": f"{metrics.max_speed:.1f}",
            "mean_speed_ms": f"{metrics.mean_speed:.1f}",
            "mean_speed_kmh": f"{metrics.mean_speed * 3.6:.1f}",
        })

    # Add optional GSD stats
    if metrics.mean_gsd:
        stats_dict.update({
            "min_gsd_cm": f"{metrics.min_gsd:.2f}",
            "max_gsd_cm": f"{metrics.max_gsd:.2f}",
            "mean_gsd_cm": f"{metrics.mean_gsd:.2f}",
        })

    # Add GPS error stats (IMPORTANT - GPS prediction accuracy)
    # Componentwise errors (X/Y/Z in ENU coordinate system)
    if metrics.mean_gps_error_x is not None:
        stats_dict.update({
            "mean_gps_error_x_m": f"{metrics.mean_gps_error_x:.4f}",
            "mean_gps_error_y_m": f"{metrics.mean_gps_error_y:.4f}",
            "mean_gps_error_z_m": f"{metrics.mean_gps_error_z:.4f}",
            "std_gps_error_x_m": f"{metrics.std_gps_error_x:.4f}",
            "std_gps_error_y_m": f"{metrics.std_gps_error_y:.4f}",
            "std_gps_error_z_m": f"{metrics.std_gps_error_z:.4f}",
        })

    # 3D Euclidean distance errors
    if metrics.mean_gps_error is not None:
        stats_dict.update({
            "mean_gps_error_m": f"{metrics.mean_gps_error:.2f}",
            "max_gps_error_m": f"{metrics.max_gps_error:.2f}",
        })

    return viz_paths, stats_dict
