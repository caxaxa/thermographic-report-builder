# Flight Visualization Implementation Plan

## Executive Summary

**Goal**: Replace the "Informações do Drone e Voo" section in the thermographic report with custom-built flight visualizations generated from OpenSfM artifacts.

**Current State**: ODM runs with `--skip-report` flag, which skips HTML report generation but preserves all underlying data:
- ✅ `reconstruction.json` (camera poses, GPS, orientations)
- ✅ Orthophoto GeoTIFF (bounds, GSD, CRS metadata)
- ✅ DSM/DTM GeoTIFFs (elevation data)
- ✅ `tracks.csv` (feature correspondences)
- ✅ Point cloud with view counts

**Problem**: The report currently shows outdated "Informações do Drone e Voo" content that relied on ODM HTML report files that no longer exist.

**Solution**: Build a `flight_viz` module in thermographic-report-builder that:
1. Parses `reconstruction.json` to extract camera poses and GPS coordinates
2. Extracts metadata from orthophoto GeoTIFF (bounds, GSD, CRS)
3. Computes flight statistics (altitude, speed, distance, coverage)
4. Generates visualizations (flight path maps, charts, heatmaps)
5. Creates LaTeX tables and figures for the report appendix

---

## Architecture Overview

### Module Structure

```
thermographic-report-builder/src/thermographic_report_builder/flight_viz/
├── __init__.py                 # Module exports
├── models.py                   # Data models (CREATED)
├── parsers.py                  # reconstruction.json and GeoTIFF parsers
├── coordinate_utils.py         # OpenSfM local ↔ WGS84 conversion
├── metrics.py                  # Flight and coverage metrics calculators
├── visualizations/             # Visualization generators
│   ├── __init__.py
│   ├── flight_path.py          # Flight path map over orthophoto
│   ├── coverage_heatmap.py     # Coverage/overlap heatmap
│   ├── charts.py               # Altitude/speed/heading charts
│   └── tables.py               # LaTeX summary tables
└── integration.py              # Integration point with report builder
```

---

## Implementation Phases

### Phase 0: Artifact Inventory (COMPLETED)

**Status**: ✅ Analysis complete

**Findings**:
- **reconstruction.json** location: `s3://solar-uploads-prod/{USER_ID}/projects/{PROJECT_ID}/opensfm/reconstruction.json`
- **Orthophoto**: `s3://solar-orthos-prod/{USER_ID}/projects/{PROJECT_ID}/odm_orthophoto.tif`
- **DSM/DTM**: `s3://solar-orthos-prod/{USER_ID}/projects/{PROJECT_ID}/{dsm|dtm}.tif`
- **tracks.csv**: `s3://solar-uploads-prod/{USER_ID}/projects/{PROJECT_ID}/opensfm/tracks.csv`

**Note**: Not all projects have these files. Older projects processed before January 2026 may be missing `reconstruction.json`.

---

### Phase 1: Data Parsing & Normalization

#### 1.1: reconstruction.json Parser

**File**: `parsers.py`

**Key tasks**:
- Parse OpenSfM JSON format
- Extract camera shots with GPS coordinates (already in WGS84)
- Extract camera orientations (rotation matrices)
- Extract local coordinates for reprojection
- Parse camera calibration parameters (focal length, distortion)
- Handle missing/optional fields gracefully

**OpenSfM JSON Structure** (from ODM source code reference):
```json
{
  "cameras": {
    "v2 unknown unknown 4000 3000 perspective 0.8333": {
      "projection_type": "brown",
      "width": 4000,
      "height": 3000,
      "focal": 0.8333,
      "k1": -0.123,
      "k2": 0.045
    }
  },
  "shots": {
    "DJI_0001.JPG": {
      "camera": "v2 unknown unknown 4000 3000 perspective 0.8333",
      "rotation": [0.999, -0.001, 0.042, 0.001, 1.000, 0.002, -0.042, -0.002, 0.999],
      "translation": [12.345, -5.678, 1.234],
      "gps_position": [-23.550520, -46.633309, 750.5],
      "gps_dop": 5.0,
      "capture_time": 1.634567890
    }
  },
  "reference_lla": {
    "latitude": -23.550520,
    "longitude": -46.633309,
    "altitude": 750.0
  },
  "points": {
    "point_id_1": {
      "coordinates": [1.23, 4.56, 7.89],
      "color": [128, 128, 128]
    }
  }
}
```

**Implementation**:
```python
class ReconstructionParser:
    """Parse OpenSfM reconstruction.json format."""

    def parse(self, json_path: Path) -> FlightData:
        """Parse reconstruction.json and return FlightData."""
        with open(json_path) as f:
            data = json.load(f)

        # Extract shots
        shots = []
        for shot_id, shot_data in data.get("shots", {}).items():
            # GPS position is already in WGS84 [lat, lon, alt]
            gps_pos = shot_data["gps_position"]

            shot = CameraShot(
                shot_id=shot_id,
                latitude=gps_pos[0],
                longitude=gps_pos[1],
                altitude=gps_pos[2],
                rotation=self._parse_rotation(shot_data["rotation"]),
                translation=shot_data["translation"],
                timestamp=self._parse_timestamp(shot_data.get("capture_time")),
            )
            shots.append(shot)

        # Sort by timestamp if available
        shots.sort(key=lambda s: s.timestamp or datetime.min)

        # Extract camera calibration
        camera_id = list(data["cameras"].keys())[0]
        camera_data = data["cameras"][camera_id]

        # Extract reference LLA
        ref_lla = data.get("reference_lla", {})

        return FlightData(
            shots=shots,
            reference_lla=(
                ref_lla.get("latitude", shots[0].latitude),
                ref_lla.get("longitude", shots[0].longitude),
                ref_lla.get("altitude", shots[0].altitude),
            ),
            camera_model=camera_data["projection_type"],
            focal_length=camera_data["focal"],
            k1=camera_data.get("k1", 0.0),
            k2=camera_data.get("k2", 0.0),
        )
```

#### 1.2: GeoTIFF Metadata Parser

**File**: `parsers.py`

**Key tasks**:
- Use GDAL/rasterio to read GeoTIFF metadata
- Extract bounds (min/max lat/lon)
- Extract CRS (coordinate reference system)
- Calculate GSD (ground sample distance) from pixel size and CRS
- Extract orthophoto dimensions

**Implementation**:
```python
import rasterio

class GeoTIFFMetadataParser:
    """Extract metadata from orthophoto GeoTIFF."""

    def parse(self, geotiff_path: Path) -> dict:
        """Extract orthophoto metadata."""
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds  # BoundingBox(left, bottom, right, top)
            crs = src.crs  # CRS object
            transform = src.transform  # Affine transform
            width, height = src.width, src.height

            # Calculate GSD (meters per pixel → cm per pixel)
            gsd_x = abs(transform.a) * 100  # Convert meters to cm
            gsd_y = abs(transform.e) * 100
            gsd = (gsd_x + gsd_y) / 2

            # Convert bounds to lat/lon if needed
            if crs.is_projected:
                from pyproj import Transformer
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
            else:
                min_lon, min_lat = bounds.left, bounds.bottom
                max_lon, max_lat = bounds.right, bounds.top

            return {
                "bounds_wgs84": {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                },
                "crs": str(crs),
                "gsd_cm": gsd,
                "width_px": width,
                "height_px": height,
                "area_m2": abs(bounds.right - bounds.left) * abs(bounds.top - bounds.bottom),
            }
```

---

### Phase 2: Metrics Engine

#### 2.1: Flight Metrics Calculator

**File**: `metrics.py`

**Calculations**:
- **Distance traveled**: Haversine distance between consecutive GPS points
- **Flight duration**: Last timestamp - first timestamp
- **Altitude stats**: Min, max, mean, median from GPS altitudes
- **Speed estimation**: Distance / time between consecutive points
- **GSD estimation**: Altitude-based approximation if not in EXIF
  - Formula: `GSD (cm/px) = (Altitude * SensorHeight) / (FocalLength * ImageHeight)`
- **Coverage area**: Bounding box area from GPS bounds

**Implementation**:
```python
import math

class FlightMetricsCalculator:
    """Calculate flight statistics from camera shots."""

    def calculate(self, flight_data: FlightData) -> FlightMetrics:
        shots = flight_data.shots

        # Distance traveled
        total_distance = 0.0
        for i in range(1, len(shots)):
            dist = self._haversine_distance(
                shots[i-1].latitude, shots[i-1].longitude,
                shots[i].latitude, shots[i].longitude
            )
            total_distance += dist

        # Altitude stats
        altitudes = [s.altitude for s in shots]

        # Speed stats (if timestamps available)
        speeds = []
        if shots[0].timestamp:
            for i in range(1, len(shots)):
                dt = (shots[i].timestamp - shots[i-1].timestamp).total_seconds()
                if dt > 0:
                    dist = self._haversine_distance(...)
                    speeds.append(dist / dt)

        # Flight duration
        duration = None
        if shots[0].timestamp and shots[-1].timestamp:
            duration = (shots[-1].timestamp - shots[0].timestamp).total_seconds()

        # Bounding box
        lats = [s.latitude for s in shots]
        lons = [s.longitude for s in shots]

        return FlightMetrics(
            total_images=len(shots),
            flight_duration=duration,
            total_distance=total_distance,
            min_altitude=min(altitudes),
            max_altitude=max(altitudes),
            mean_altitude=statistics.mean(altitudes),
            median_altitude=statistics.median(altitudes),
            min_speed=min(speeds) if speeds else None,
            max_speed=max(speeds) if speeds else None,
            mean_speed=statistics.mean(speeds) if speeds else None,
            bbox_min_lat=min(lats),
            bbox_max_lat=max(lats),
            bbox_min_lon=min(lons),
            bbox_max_lon=max(lons),
        )

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance in meters."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
```

#### 2.2: Coverage Metrics Calculator

**File**: `metrics.py`

**Note**: Coverage/overlap metrics require the point cloud with view counts, which is a more complex implementation. For the initial version, we can skip this and note "Requires point cloud processing (future enhancement)".

---

### Phase 3: Visualization Generation

#### 3.1: Flight Path Map

**File**: `visualizations/flight_path.py`

**Output**: PNG image showing flight path overlaid on orthophoto with color gradient by time/altitude

**Libraries**: matplotlib, PIL/Pillow

**Implementation**:
```python
import matplotlib.pyplot as plt
from PIL import Image

class FlightPathVisualizer:
    """Generate flight path map over orthophoto."""

    def generate(
        self,
        flight_data: FlightData,
        orthophoto_path: Path,
        ortho_metadata: dict,
        output_path: Path,
        color_by: str = "altitude"  # or "time"
    ):
        """Overlay flight path on orthophoto."""
        # Load orthophoto (downscale for faster rendering)
        ortho_img = Image.open(orthophoto_path)
        ortho_img.thumbnail((2000, 2000))  # Max 2000px

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(ortho_img, extent=[
            ortho_metadata["bounds_wgs84"]["min_lon"],
            ortho_metadata["bounds_wgs84"]["max_lon"],
            ortho_metadata["bounds_wgs84"]["min_lat"],
            ortho_metadata["bounds_wgs84"]["max_lat"],
        ])

        # Extract GPS coordinates
        lats = [s.latitude for s in flight_data.shots]
        lons = [s.longitude for s in flight_data.shots]

        # Color gradient
        if color_by == "altitude":
            colors = [s.altitude for s in flight_data.shots]
            cmap = "viridis"
            cbar_label = "Altitude (m)"
        elif color_by == "time":
            colors = range(len(flight_data.shots))
            cmap = "plasma"
            cbar_label = "Image Index"

        # Plot flight path
        scatter = ax.scatter(lons, lats, c=colors, cmap=cmap, s=20, alpha=0.8, edgecolors='white', linewidths=0.5)
        ax.plot(lons, lats, 'w-', alpha=0.3, linewidth=1)  # Connect with lines

        # Add start/end markers
        ax.scatter(lons[0], lats[0], c='green', s=200, marker='o', edgecolors='white', linewidths=2, label='Start', zorder=10)
        ax.scatter(lons[-1], lats[-1], c='red', s=200, marker='X', edgecolors='white', linewidths=2, label='End', zorder=10)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=cbar_label)

        # Labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Flight Path", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

#### 3.2: Charts (Altitude, Speed, Heading)

**File**: `visualizations/charts.py`

**Outputs**:
- Altitude vs. image index (line chart)
- Speed vs. image index (line chart, if timestamps available)
- Heading vs. image index (circular chart, if orientation data available)

**Implementation**:
```python
class ChartsVisualizer:
    """Generate time-series charts for flight parameters."""

    def generate_altitude_chart(self, flight_data: FlightData, output_path: Path):
        """Line chart of altitude over flight."""
        fig, ax = plt.subplots(figsize=(10, 4))

        indices = range(len(flight_data.shots))
        altitudes = [s.altitude for s in flight_data.shots]

        ax.plot(indices, altitudes, 'b-', linewidth=2)
        ax.fill_between(indices, altitudes, alpha=0.3)

        ax.set_xlabel("Image Index")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("Flight Altitude Profile")
        ax.grid(True, alpha=0.3)

        # Add min/max/mean lines
        mean_alt = sum(altitudes) / len(altitudes)
        ax.axhline(mean_alt, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_alt:.1f}m')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def generate_speed_chart(self, flight_data: FlightData, output_path: Path):
        """Line chart of ground speed (if timestamps available)."""
        # Similar implementation, calculate speed between consecutive points
        pass
```

#### 3.3: Summary Tables (LaTeX)

**File**: `visualizations/tables.py`

**Outputs**: LaTeX table code for:
- Flight summary table (images, duration, distance, altitude range)
- Coverage summary table (area, GSD, bounds)
- Reconstruction health table (features, matches, reprojection error)

**Implementation**:
```python
class LaTeXTableGenerator:
    """Generate LaTeX tables for report appendix."""

    def generate_flight_summary_table(self, metrics: FlightMetrics) -> str:
        """Generate LaTeX code for flight summary table."""
        duration_str = f"{metrics.flight_duration/60:.1f} min" if metrics.flight_duration else "N/A"

        return r"""
\begin{table}[h!]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Parâmetro} & \textbf{Valor} \\
\midrule
Imagens Totais & """ + str(metrics.total_images) + r""" \\
Duração do Voo & """ + duration_str + r""" \\
Distância Percorrida & """ + f"{metrics.total_distance:.1f} m" + r""" \\
Altitude Mínima & """ + f"{metrics.min_altitude:.1f} m" + r""" \\
Altitude Máxima & """ + f"{metrics.max_altitude:.1f} m" + r""" \\
Altitude Média & """ + f"{metrics.mean_altitude:.1f} m" + r""" \\
\bottomrule
\end{tabular}
\caption{Resumo do Voo}
\end{table}
"""
```

---

### Phase 4: Integration with Report Builder

**File**: `integration.py`

**Key tasks**:
- Download artifacts from S3 if not already cached
- Run parsers to extract flight data
- Calculate metrics
- Generate visualizations
- Return paths to generated images and LaTeX table code
- Handle missing artifacts gracefully (skip appendix if no reconstruction.json)

**Integration point**: Modify [builder.py:817-843](builder.py:817-843) in `_add_appendix` method

**New implementation**:
```python
def _add_appendix(self, doc: pl.Document) -> None:
    """Add appendix with flight data and orthophoto information."""
    logger.info("Adding flight visualization appendix")

    doc.append(NoEscape(r"\newpage"))
    doc.append(NoEscape(r"\appendix"))

    # Appendix A: Flight and Orthophoto Data
    doc.append(NoEscape(r"\section{Informações do Voo e Ortofoto}"))

    try:
        from .flight_viz.integration import generate_flight_appendix

        # Generate flight visualizations
        flight_images, flight_tables = generate_flight_appendix(
            project_id=self.config.project_id,
            user_id=self.config.user_id,
            work_dir=self.images_dir,
        )

        # Add flight path map
        if "flight_path" in flight_images:
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(f"report_images/{flight_images['flight_path'].name}", width=NoEscape(r"0.8\textwidth"))
                fig.add_caption("Trajetória de Voo sobre Ortofoto")

        # Add altitude chart
        if "altitude_chart" in flight_images:
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(f"report_images/{flight_images['altitude_chart'].name}", width=NoEscape(r"0.8\textwidth"))
                fig.add_caption("Perfil de Altitude do Voo")

        # Add flight summary table
        if "flight_summary_table" in flight_tables:
            doc.append(NoEscape(flight_tables["flight_summary_table"]))

        # Add coverage summary table
        if "coverage_table" in flight_tables:
            doc.append(NoEscape(flight_tables["coverage_table"]))

    except Exception as e:
        logger.warning(f"Failed to generate flight visualizations: {e}")
        doc.append("Dados de voo não disponíveis para este projeto.")
```

---

### Phase 5: Testing & Validation

**Test cases**:
1. **Happy path**: Project with complete artifacts (reconstruction.json, orthophoto, DSM/DTM)
2. **Missing reconstruction.json**: Gracefully skip flight appendix, log warning
3. **Missing timestamps**: Skip speed calculations, generate altitude-only charts
4. **Coordinate conversion**: Verify WGS84 coordinates align with orthophoto bounds
5. **Large datasets**: Test with 1000+ image projects, ensure visualization doesn't hang

**Validation**:
- Visual inspection: Flight path should align with orthophoto features
- Altitude values: Should match expected drone flight altitude (20-50m typical)
- Distance calculation: Compare with manual measurement on map
- GSD estimation: Compare with orthophoto metadata GSD

---

### Phase 6: Deployment

**Steps**:
1. Build Docker image with new dependencies (matplotlib, rasterio, pyproj)
2. Update requirements.txt:
   ```
   matplotlib>=3.5.0
   rasterio>=1.3.0
   pyproj>=3.4.0
   ```
3. Push to ECR
4. Update AWS Batch job definition (if needed)
5. Test with real projects

---

### Phase 7: Documentation

**Documents to create/update**:
1. **FLIGHT_VISUALIZATION_ARCHITECTURE.md**: Technical deep dive into coordinate systems, parsing logic, visualization algorithms
2. **REPORT_GENERATION.md**: Update to document new flight appendix generation
3. **Code comments**: Inline documentation for coordinate transformations, GSD calculations
4. **User guide**: How to interpret flight visualizations in the report

---

## Current Status

**Completed**:
- ✅ Module structure created (`flight_viz/` directory)
- ✅ Data models defined (`models.py`)
- ✅ Architecture planned (this document)

**Next Steps**:
1. Implement `parsers.py` (reconstruction.json and GeoTIFF metadata extraction)
2. Implement `metrics.py` (flight and coverage metrics calculators)
3. Implement `visualizations/` (flight path map, charts, tables)
4. Implement `integration.py` (glue code for report builder)
5. Add to builder.py `_add_appendix` method
6. Test with real projects
7. Deploy

**Estimated effort**: 2-3 days of focused development

---

## Dependencies

**Python packages** (add to requirements.txt):
```txt
matplotlib>=3.5.0      # Visualization
rasterio>=1.3.0        # GeoTIFF reading
pyproj>=3.4.0          # Coordinate transformations
Pillow>=9.0.0          # Image manipulation (already present)
```

**System dependencies** (already in Docker image):
- GDAL (for rasterio)
- NumPy (already present)

---

## Fallback Strategy

**If reconstruction.json is missing**:
1. Skip flight path visualizations
2. Extract metadata from orthophoto GeoTIFF only:
   - Bounds (lat/lon)
   - Area (m²)
   - GSD (cm/px)
   - CRS
3. Generate simplified appendix with orthophoto metadata table only
4. Log warning: "Flight data not available for this project (processed before Jan 2026)"

**LaTeX fallback**:
```latex
\section{Informações da Ortofoto}

\begin{table}[h!]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Parâmetro} & \textbf{Valor} \\
\midrule
Área Coberta & 12,345 m² \\
GSD (Resolução) & 2.5 cm/px \\
Sistema de Coordenadas & EPSG:32723 (UTM 23S) \\
Limites (Lat/Lon) & -23.5505°, -46.6333° \\
\bottomrule
\end{tabular}
\caption{Metadados da Ortofoto}
\end{table}

\textit{Nota: Dados de trajetória de voo não disponíveis para este projeto.}
```

---

## Future Enhancements

1. **Interactive web map**: Export flight path as GeoJSON for web viewer
2. **3D visualization**: Show flight path in 3D with terrain elevation
3. **Coverage heatmap**: Implement PDAL-based point cloud processing for overlap visualization
4. **Comparison mode**: Compare multiple flights over same area
5. **Real-time preview**: Generate flight viz during ODM processing (streaming)
6. **Video generation**: Animate flight path over time with smooth camera transitions

---

## References

- ODM source code: `/home/ubuntu/ODM/opendm/`
  - `shots.py`: Shot parsing and GeoJSON export
  - `reconstruction.py`: Reconstruction data structures
  - `report.py`: HTML report generation (reference for visualizations)
- Plan document: `/home/ubuntu/.claude/plans/transient-twirling-newt.md` (lines 136-206)
- Thermal backtracking docs: `/home/ubuntu/thermographic-report-builder/docs/THERMAL_BACKTRACKING_ARCHITECTURE.md`

---

*Document created: January 16, 2026*
*Status: Planning complete, implementation pending*
