"""Visualization generators for flight data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from .models import FlightData, FlightMetrics

# Use non-interactive backend for server environments
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class FlightPathVisualizer:
    """Generate flight path visualization overlaid on orthophoto."""

    def generate_map(
        self,
        flight_data: FlightData,
        orthophoto_path: Optional[Path] = None,
        output_path: Path = None,
    ) -> Path:
        """Generate interactive map with flight path.

        Args:
            flight_data: Parsed flight data
            orthophoto_path: Optional path to orthophoto for overlay
            output_path: Where to save the HTML map

        Returns:
            Path to generated HTML map file
        """
        shots = flight_data.shots

        if not shots:
            raise ValueError("No shots available for visualization")

        # Calculate center point
        center_lat = np.mean([s.latitude for s in shots])
        center_lon = np.mean([s.longitude for s in shots])

        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=17,
            tiles='OpenStreetMap',
        )

        # Add satellite imagery layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        # Sort shots by timestamp if available
        if shots[0].timestamp:
            sorted_shots = sorted(shots, key=lambda s: s.timestamp)
        else:
            sorted_shots = shots

        # Create flight path coordinates
        path_coords = [[s.latitude, s.longitude] for s in sorted_shots]

        # Add flight path as polyline
        folium.PolyLine(
            path_coords,
            color='red',
            weight=2,
            opacity=0.8,
            popup='Flight Path'
        ).add_to(m)

        # Add markers for camera positions
        for i, shot in enumerate(sorted_shots):
            # Color code: start (green), end (red), middle (blue)
            if i == 0:
                color = 'green'
                icon = 'play'
            elif i == len(sorted_shots) - 1:
                color = 'red'
                icon = 'stop'
            else:
                color = 'blue'
                icon = 'camera'

            # Create popup with shot info
            popup_html = f"""
            <b>Shot:</b> {shot.shot_id}<br>
            <b>Position:</b> {i + 1}/{len(sorted_shots)}<br>
            <b>Altitude:</b> {shot.altitude:.1f}m<br>
            """

            if shot.timestamp:
                popup_html += f"<b>Time:</b> {shot.timestamp.strftime('%H:%M:%S')}<br>"

            if shot.gsd:
                popup_html += f"<b>GSD:</b> {shot.gsd:.2f} cm/px<br>"

            folium.Marker(
                location=[shot.latitude, shot.longitude],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        if output_path is None:
            output_path = Path("flight_path_map.html")

        m.save(str(output_path))
        logger.info(f"Flight path map saved to {output_path}")

        return output_path

    def generate_static_map(
        self,
        flight_data: FlightData,
        output_path: Path,
        orthophoto_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10),
        project_name: Optional[str] = None,
    ) -> Path:
        """Generate static matplotlib map with flight path.

        Args:
            flight_data: Parsed flight data
            output_path: Where to save the PNG image
            orthophoto_path: Optional path to orthophoto for background overlay
            figsize: Figure size in inches
            project_name: Optional name of the project/solar farm for display in title

        Returns:
            Path to generated PNG file
        """
        from pyproj import Geod

        shots = flight_data.shots

        if not shots:
            raise ValueError("No shots available for visualization")

        # Sort shots by timestamp if available
        if shots[0].timestamp:
            sorted_shots = sorted(shots, key=lambda s: s.timestamp)
        else:
            sorted_shots = shots

        # Extract coordinates
        lons = [s.longitude for s in sorted_shots]
        lats = [s.latitude for s in sorted_shots]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Add orthophoto background if available
        if orthophoto_path and orthophoto_path.exists():
            try:
                self._add_orthophoto_background(ax, orthophoto_path, lons, lats)
            except Exception as e:
                logger.warning(f"Failed to add orthophoto background: {e}")

        # Plot flight path (Portuguese labels)
        ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.8, label='Trajetória de Voo', zorder=3)

        # Plot camera positions
        ax.scatter(lons, lats, c='red', s=50, alpha=0.9, zorder=5,
                  edgecolors='white', linewidths=0.5, label='Posições da Câmera')

        # Mark start and end
        ax.scatter(lons[0], lats[0], c='green', s=200, marker='^',
                  edgecolors='white', linewidths=2, zorder=10, label='Início')
        ax.scatter(lons[-1], lats[-1], c='red', s=200, marker='v',
                  edgecolors='white', linewidths=2, zorder=10, label='Fim')

        # Labels and formatting for primary axes (degrees) - Portuguese
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        # Set title with project name if provided - Portuguese
        title = f'Trajetória de Voo - {project_name}' if project_name else 'Trajetória de Voo'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=1)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Calculate meter scales for secondary axes
        # Use WGS84 ellipsoid for accurate distance calculations
        geod = Geod(ellps='WGS84')

        # Get the current axis limits (in degrees)
        lon_min, lon_max = ax.get_xlim()
        lat_min, lat_max = ax.get_ylim()

        # Calculate center latitude for accurate conversion
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        # Calculate meters per degree at this location
        # Longitude distance varies with latitude
        _, _, lon_distance = geod.inv(lon_min, center_lat, lon_max, center_lat)  # X distance in meters
        _, _, lat_distance = geod.inv(center_lon, lat_min, center_lon, lat_max)  # Y distance in meters

        # Create secondary X-axis (top) for meters - Portuguese
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xlabel('Distância Leste-Oeste (m)', fontsize=12)

        # Calculate tick positions in meters (relative to left edge = 0)
        lon_range = lon_max - lon_min
        meters_per_degree_lon = lon_distance / lon_range if lon_range > 0 else 1

        def lon_to_meters(x):
            return (x - lon_min) * meters_per_degree_lon

        def meters_to_lon(m):
            return lon_min + m / meters_per_degree_lon

        ax2.set_xlim(lon_to_meters(lon_min), lon_to_meters(lon_max))

        # Format meter labels nicely
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

        # Create secondary Y-axis (right) for meters - Portuguese
        ax3 = ax.secondary_yaxis('right')
        ax3.set_ylabel('Distância Norte-Sul (m)', fontsize=12)

        # Calculate tick positions in meters (relative to bottom edge = 0)
        lat_range = lat_max - lat_min
        meters_per_degree_lat = lat_distance / lat_range if lat_range > 0 else 1

        def lat_to_meters(y):
            return (y - lat_min) * meters_per_degree_lat

        def meters_to_lat(m):
            return lat_min + m / meters_per_degree_lat

        ax3.set_ylim(lat_to_meters(lat_min), lat_to_meters(lat_max))

        # Format meter labels nicely
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Static flight path map saved to {output_path}")

        return output_path

    def _add_orthophoto_background(
        self,
        ax: plt.Axes,
        orthophoto_path: Path,
        lons: List[float],
        lats: List[float],
    ) -> None:
        """Add faded orthophoto as background to the plot.

        Args:
            ax: Matplotlib axes to add background to
            orthophoto_path: Path to orthophoto GeoTIFF
            lons: Longitude coordinates for extent calculation
            lats: Latitude coordinates for extent calculation
        """
        from osgeo import gdal
        from pyproj import Transformer

        logger.info(f"Loading orthophoto background from {orthophoto_path}")

        # Open orthophoto
        ds = gdal.Open(str(orthophoto_path))
        if ds is None:
            raise ValueError(f"Failed to open orthophoto: {orthophoto_path}")

        # Get geotransform and projection
        gt = ds.GetGeoTransform()
        proj_wkt = ds.GetProjection()

        # Read image data (downsample for performance)
        width = ds.RasterXSize
        height = ds.RasterYSize
        num_bands = ds.RasterCount

        logger.info(f"Orthophoto dimensions: {width}x{height}, bands: {num_bands}")

        # Downsample factor (use every Nth pixel for background)
        downsample = max(1, min(width, height) // 1000)
        out_width = width // downsample
        out_height = height // downsample

        logger.info(f"Downsampling by {downsample}x to {out_width}x{out_height}")

        # Read RGB bands (assuming 3-band RGB or more)
        if num_bands >= 3:
            # Read RGB
            r = ds.GetRasterBand(1).ReadAsArray(0, 0, width, height, out_width, out_height)
            g = ds.GetRasterBand(2).ReadAsArray(0, 0, width, height, out_width, out_height)
            b = ds.GetRasterBand(3).ReadAsArray(0, 0, width, height, out_width, out_height)

            # Stack into RGB image
            img = np.dstack([r, g, b])
        else:
            # Grayscale - use first band for all channels
            band = ds.GetRasterBand(1).ReadAsArray(0, 0, width, height, out_width, out_height)
            img = np.dstack([band, band, band])

        # Get data type and normalize appropriately
        # ODM orthophotos can be 8-bit or 16-bit
        img = img.astype(np.float64)

        # Handle nodata/invalid values (often 0 or very high values)
        # Replace zeros with NaN for better visualization
        nodata_mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)

        # Get the actual data range (excluding nodata)
        valid_data = img[~nodata_mask]
        if len(valid_data) > 0:
            # Use percentile normalization for better contrast
            p2 = np.percentile(valid_data, 2)
            p98 = np.percentile(valid_data, 98)
            logger.info(f"Data range: min={img.min():.1f}, max={img.max():.1f}, p2={p2:.1f}, p98={p98:.1f}")

            # Normalize using percentile stretch
            img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
        else:
            # Fallback: simple normalization based on dtype
            band1 = ds.GetRasterBand(1)
            dtype = band1.DataType
            if dtype in [gdal.GDT_Byte]:
                img = img / 255.0
            elif dtype in [gdal.GDT_UInt16, gdal.GDT_Int16]:
                img = img / 65535.0
            else:
                # Normalize to actual range
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            logger.info(f"Using dtype-based normalization (dtype={dtype})")

        # Set nodata areas to white (will be faded anyway)
        img[nodata_mask] = 1.0

        # Close dataset
        ds = None

        # Create transformer from image CRS to WGS84
        transformer = Transformer.from_crs(
            proj_wkt,
            "EPSG:4326",  # WGS84
            always_xy=True
        )

        # Calculate image extent in original CRS
        # Top-left corner
        min_x = gt[0]
        max_y = gt[3]
        # Bottom-right corner (full image extent)
        max_x = min_x + width * gt[1]
        min_y = max_y + height * gt[5]

        # Transform corners to WGS84
        lon_min, lat_min = transformer.transform(min_x, min_y)
        lon_max, lat_max = transformer.transform(max_x, max_y)

        logger.info(f"Extent in WGS84: lon=[{lon_min:.6f}, {lon_max:.6f}], lat=[{lat_min:.6f}, {lat_max:.6f}]")

        # Display image with faded alpha
        ax.imshow(
            img,
            extent=[lon_min, lon_max, lat_min, lat_max],
            aspect='auto',
            alpha=0.5,  # Faded background
            zorder=0,   # Behind everything else
            origin='upper',  # GeoTIFFs have origin at top-left
            interpolation='bilinear'
        )

        logger.info("Successfully added orthophoto background to flight path visualization")


class ChartsVisualizer:
    """Generate time-series charts for altitude, speed, and heading."""

    def generate_altitude_chart(
        self,
        flight_data: FlightData,
        output_path: Path,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Path:
        """Generate altitude vs distance/time chart.

        Args:
            flight_data: Parsed flight data
            output_path: Where to save the PNG image
            figsize: Figure size in inches

        Returns:
            Path to generated PNG file
        """
        shots = flight_data.shots

        # Sort by timestamp if available - Portuguese labels
        if shots[0].timestamp:
            sorted_shots = sorted(shots, key=lambda s: s.timestamp)
            x_data = [(s.timestamp - sorted_shots[0].timestamp).total_seconds() / 60
                     for s in sorted_shots]
            x_label = 'Tempo (minutos)'
        else:
            sorted_shots = shots
            x_data = list(range(len(sorted_shots)))
            x_label = 'Número da Imagem'

        altitudes = [s.altitude for s in sorted_shots]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot altitude
        ax.plot(x_data, altitudes, 'b-', linewidth=2)
        ax.fill_between(x_data, altitudes, alpha=0.3)

        # Add mean altitude line
        mean_alt = np.mean(altitudes)
        ax.axhline(y=mean_alt, color='r', linestyle='--', linewidth=1.5,
                  label=f'Média: {mean_alt:.1f}m')

        # Labels and formatting - Portuguese
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.set_title('Perfil de Altitude', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Altitude chart saved to {output_path}")

        return output_path

    def generate_speed_chart(
        self,
        flight_data: FlightData,
        output_path: Path,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Optional[Path]:
        """Generate speed vs time chart.

        Args:
            flight_data: Parsed flight data
            output_path: Where to save the PNG image
            figsize: Figure size in inches

        Returns:
            Path to generated PNG file, or None if timestamps unavailable
        """
        shots = flight_data.shots

        # Filter shots with timestamps
        shots_with_time = [s for s in shots if s.timestamp is not None]

        if len(shots_with_time) < 2:
            logger.warning("Not enough timestamps available for speed chart")
            return None

        # Sort by timestamp
        sorted_shots = sorted(shots_with_time, key=lambda s: s.timestamp)

        # Calculate speeds between consecutive shots
        from pyproj import Geod
        geod = Geod(ellps="WGS84")

        times = []
        speeds = []

        for i in range(len(sorted_shots) - 1):
            shot1 = sorted_shots[i]
            shot2 = sorted_shots[i + 1]

            # Calculate distance
            _, _, distance = geod.inv(
                shot1.longitude, shot1.latitude,
                shot2.longitude, shot2.latitude
            )

            # Calculate time difference
            time_diff = (shot2.timestamp - shot1.timestamp).total_seconds()

            if time_diff > 0:
                speed = distance / time_diff  # m/s
                time_minutes = (shot1.timestamp - sorted_shots[0].timestamp).total_seconds() / 60
                times.append(time_minutes)
                speeds.append(speed)

        if not speeds:
            logger.warning("Could not calculate speeds")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot speed
        ax.plot(times, speeds, 'g-', linewidth=2)
        ax.fill_between(times, speeds, alpha=0.3)

        # Add mean speed line - Portuguese
        mean_speed = np.mean(speeds)
        ax.axhline(y=mean_speed, color='r', linestyle='--', linewidth=1.5,
                  label=f'Média: {mean_speed:.1f} m/s')

        # Labels and formatting - Portuguese
        ax.set_xlabel('Tempo (minutos)', fontsize=12)
        ax.set_ylabel('Velocidade (m/s)', fontsize=12)
        ax.set_title('Perfil de Velocidade', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Speed chart saved to {output_path}")

        return output_path

    def generate_combined_dashboard(
        self,
        flight_data: FlightData,
        flight_metrics: FlightMetrics,
        output_path: Path,
        figsize: Tuple[int, int] = (14, 8),
    ) -> Path:
        """Generate combined dashboard with multiple charts.

        NOTE: Flight Path and Flight Statistics are NOT included here as they
        are already shown in Appendix A (flight_path_static.png and tables).

        Args:
            flight_data: Parsed flight data
            flight_metrics: Computed flight metrics
            output_path: Where to save the PNG image
            figsize: Figure size in inches

        Returns:
            Path to generated PNG file
        """
        shots = flight_data.shots

        # Sort by timestamp if available
        has_timestamps = shots[0].timestamp is not None
        if has_timestamps:
            sorted_shots = sorted(shots, key=lambda s: s.timestamp)
        else:
            sorted_shots = shots

        # Create figure with subplots (2 rows: altitude profile on top, histogram + intervals on bottom)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Altitude profile (full width on top row) - Portuguese
        ax1 = fig.add_subplot(gs[0, :])
        if has_timestamps:
            x_data = [(s.timestamp - sorted_shots[0].timestamp).total_seconds() / 60
                     for s in sorted_shots]
            x_label = 'Tempo (minutos)'
        else:
            x_data = list(range(len(sorted_shots)))
            x_label = 'Número da Imagem'

        altitudes = [s.altitude for s in sorted_shots]
        ax1.plot(x_data, altitudes, 'b-', linewidth=2)
        ax1.fill_between(x_data, altitudes, alpha=0.3)
        ax1.axhline(y=flight_metrics.mean_altitude, color='r', linestyle='--',
                   label=f'Média: {flight_metrics.mean_altitude:.1f}m')
        ax1.set_xlabel(x_label, fontsize=10)
        ax1.set_ylabel('Altitude (m)', fontsize=10)
        ax1.set_title('Perfil de Altitude', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Altitude histogram (bottom left) - Portuguese
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(altitudes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=flight_metrics.mean_altitude, color='r', linestyle='--',
                   linewidth=2, label=f'Média: {flight_metrics.mean_altitude:.1f}m')
        ax2.set_xlabel('Altitude (m)', fontsize=10)
        ax2.set_ylabel('Frequência', fontsize=10)
        ax2.set_title('Distribuição de Altitude', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Image spacing (bottom right, if timestamps available) - Portuguese
        ax3 = fig.add_subplot(gs[1, 1])
        if has_timestamps and len(sorted_shots) > 1:
            intervals = []
            for i in range(len(sorted_shots) - 1):
                interval = (sorted_shots[i + 1].timestamp - sorted_shots[i].timestamp).total_seconds()
                intervals.append(interval)

            ax3.plot(range(len(intervals)), intervals, 'g-', linewidth=2)
            ax3.axhline(y=np.mean(intervals), color='r', linestyle='--',
                       label=f'Média: {np.mean(intervals):.1f}s')
            ax3.set_xlabel('Índice do Par de Imagens', fontsize=10)
            ax3.set_ylabel('Intervalo de Tempo (s)', fontsize=10)
            ax3.set_title('Intervalos de Captura', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Dados de timestamp\nnão disponíveis',
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Intervalos de Captura', fontsize=12, fontweight='bold')

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Combined dashboard saved to {output_path}")

        return output_path


class CoverageHeatmapVisualizer:
    """Generate coverage and overlap heatmaps."""

    def generate_overlap_heatmap(
        self,
        point_cloud_path: Path,
        output_path: Path,
    ) -> Optional[Path]:
        """Generate heatmap showing image overlap.

        Args:
            point_cloud_path: Path to point cloud file
            output_path: Where to save the PNG image

        Returns:
            Path to generated PNG file, or None if not implemented

        Note:
            This is a placeholder. Full implementation would:
            1. Parse point cloud file (PLY or LAZ)
            2. Extract view count for each point
            3. Rasterize to create heatmap
            4. Apply color mapping
        """
        logger.warning("Coverage heatmap generation not yet implemented")
        return None
