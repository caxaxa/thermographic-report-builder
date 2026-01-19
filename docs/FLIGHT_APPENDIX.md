# Flight Appendix Documentation

## Overview

The flight appendix is an automatically generated section in thermographic inspection reports that provides detailed statistics and visualizations about the drone flight mission. This feature analyzes OpenSfM reconstruction data to extract flight metrics, camera positions, and GPS accuracy information.

## Purpose

The flight appendix serves several critical purposes:

1. **Quality Assurance**: GPS prediction error metrics indicate the accuracy of the photogrammetric reconstruction
2. **Mission Documentation**: Complete record of flight parameters (altitude, distance, coverage area)
3. **Compliance**: Demonstrates proper mission planning and execution for regulatory requirements
4. **Analysis**: Enables identification of potential data quality issues before final delivery

## Data Sources

### Primary Data: reconstruction.json

The flight appendix primarily relies on OpenSfM's `reconstruction.json` file, which contains:

- **Camera positions**: 3D coordinates in local ENU (East-North-Up) coordinate system
- **Camera orientations**: Rotation matrices for each image
- **GPS positions**: Raw GPS coordinates for each image (in ENU)
- **Optimized positions**: Bundle-adjusted camera positions (in ENU)
- **Reference point**: Geographic reference (latitude, longitude, altitude) for coordinate conversion
- **Camera calibration**: Focal length, distortion coefficients

**Location in S3**: `s3://solar-orthos-prod/{user_id}/projects/{project_id}/opensfm/reconstruction.json`

### Secondary Data: odm_orthophoto.tif

The orthophoto GeoTIFF provides:

- **Geographic bounds**: Extent of the surveyed area
- **Coordinate reference system**: Projection information
- **Ground Sample Distance (GSD)**: Spatial resolution in meters/pixel

**Location in S3**: `s3://solar-orthos-prod/{user_id}/projects/{project_id}/odm_orthophoto.tif`

## Coordinate System Transformations

### Critical: ENU to WGS84 Conversion

OpenSfM uses a **local ENU (East-North-Up)** coordinate system for computational efficiency. All positions in `reconstruction.json` are relative to a reference point and measured in **meters**, not degrees.

**Transformation Process**:

```python
# Extract reference point from reconstruction.json
reference_lla = reconstruction["reference_lla"]  # {'latitude': X, 'longitude': Y, 'altitude': Z}

# For each camera position in ENU (meters)
easting = gps_position[0]   # East offset in meters
northing = gps_position[1]  # North offset in meters
up = gps_position[2]        # Up offset in meters

# Convert to WGS84 using tangent plane approximation
meters_per_degree_lat = 111320.0
meters_per_degree_lon = 111320.0 * cos(radians(ref_lat))

delta_lat = northing / meters_per_degree_lat
delta_lon = easting / meters_per_degree_lon

latitude = ref_lat + delta_lat
longitude = ref_lon + delta_lon
altitude = ref_alt + up
```

**Why This Matters**: Without this conversion, coordinates will appear as values like -94° to +95° (impossible latitude range) and all distance/coverage calculations will fail with NaN values.

## Metrics Calculated

### GPS Prediction Error (CRITICAL METRIC)

**Definition**: The 3D Euclidean distance between the raw GPS position and the bundle-adjusted optimized position for each camera.

**Formula**:
```python
gps_error = sqrt(
    (translation[0] - easting)² +
    (translation[1] - northing)² +
    (translation[2] - up)²
)
```

**Interpretation**:
- **< 50m**: Excellent GPS accuracy, high-quality reconstruction
- **50-100m**: Good GPS accuracy, acceptable for most applications
- **100-200m**: Moderate GPS accuracy, verify alignment with ground control
- **> 200m**: Poor GPS accuracy, may indicate GPS drift or environmental issues

**Reported Values**:
- Mean GPS Error (meters): Average error across all images
- Maximum GPS Error (meters): Worst-case error in the dataset

### Flight Duration

**Source**: EXIF timestamp data from images (`capture_time` field)

**Calculation**: Time difference between first and last image timestamps

**Units**: Minutes

**Note**: Only calculated if timestamp data is available in reconstruction.json

### Total Distance Flown

**Source**: Sequential camera positions sorted by timestamp

**Calculation**: Sum of geodesic distances between consecutive camera positions using WGS84 ellipsoid

```python
# Using pyproj Geod for accurate distance calculation
geod = Geod(ellps="WGS84")
for i in range(len(shots) - 1):
    _, _, distance = geod.inv(
        lon1, lat1,
        lon2, lat2
    )
    total_distance += distance
```

**Units**: Meters (converted to kilometers in report)

### Altitude Statistics

**Source**: Altitude component of converted WGS84 coordinates

**Metrics**:
- Minimum Altitude (m)
- Maximum Altitude (m)
- Mean Altitude (m)
- Median Altitude (m)

**Note**: Altitudes are above WGS84 ellipsoid, not above ground level

### Coverage Area

**Source**: Bounding box of all camera positions

**Calculation**: Approximate area as geodesic rectangle

```python
# Width at middle latitude
_, _, width = geod.inv(min_lon, mid_lat, max_lon, mid_lat)

# Height at middle longitude
_, _, height = geod.inv(mid_lon, min_lat, mid_lon, max_lat)

coverage_area = width * height  # square meters
```

**Units**: Square meters (converted to hectares in report)

**Note**: This is an approximate bounding box area, not actual surveyed area

### Speed Statistics (Optional)

**Source**: Distance and time between consecutive images

**Calculation**: `speed = distance / time_difference`

**Metrics**:
- Minimum Speed (m/s)
- Maximum Speed (m/s)
- Mean Speed (m/s, converted to km/h in report)

**Note**: Only calculated if timestamp data is available

### Ground Sample Distance (Optional)

**Source**: GSD values from individual camera shots (if available in reconstruction)

**Metrics**:
- Minimum GSD (cm/pixel)
- Maximum GSD (cm/pixel)
- Mean GSD (cm/pixel)

**Note**: May not be available in all reconstruction.json files

## Visualizations Generated

### 1. Interactive Flight Path Map

**File**: `flight_path_map.html`

**Technology**: Folium (Leaflet.js)

**Features**:
- Camera positions plotted as markers
- Flight path lines connecting sequential positions
- Orthophoto overlay (if available)
- Interactive pan/zoom/popup

**Use Case**: Embedded in HTML reports or standalone viewing

### 2. Static Flight Path Map

**File**: `flight_path_static.png`

**Technology**: Matplotlib

**Features**:
- 2D scatter plot of camera positions
- Flight path lines
- Latitude/longitude axes
- Grid lines and labels

**Use Case**: PDF reports, presentations

### 3. Altitude Profile Chart

**File**: `altitude_profile.png`

**Features**:
- Line chart of altitude vs. image sequence
- Mean altitude reference line
- Min/max altitude bounds

**Use Case**: Identifying altitude variations during flight

### 4. Speed Profile Chart

**File**: `speed_profile.png`

**Features**:
- Line chart of speed vs. time
- Mean speed reference line
- Only generated if timestamp data available

**Use Case**: Analyzing flight dynamics and identifying irregular movements

### 5. Combined Dashboard

**File**: `flight_dashboard.png`

**Features**:
- Multi-panel visualization combining:
  - Flight path map
  - Altitude profile
  - Speed profile (if available)
  - Key statistics summary table

**Use Case**: Single-page overview of entire flight mission

## Integration with Report Builder

### Entry Point

The report builder calls `generate_flight_appendix()` from [integration.py](../src/thermographic_report_builder/flight_viz/integration.py):

```python
from thermographic_report_builder.flight_viz.integration import generate_flight_appendix

viz_paths, flight_stats = generate_flight_appendix(
    work_dir=Path("/tmp/report_work"),
    output_dir=Path("/tmp/report_work/flight_viz"),
    reconstruction_path=None,  # Auto-detected from work_dir/opensfm/
    orthophoto_path=None       # Auto-detected from work_dir/
)
```

### Return Values

**viz_paths** (Dict[str, Path]):
```python
{
    "flight_path_map": Path("flight_path_map.html"),
    "flight_path_static": Path("flight_path_static.png"),
    "altitude_chart": Path("altitude_profile.png"),
    "speed_chart": Path("speed_profile.png"),  # Optional
    "dashboard": Path("flight_dashboard.png")
}
```

**flight_stats** (Dict[str, str]):
```python
{
    "total_images": "595",
    "total_distance_m": "1970",
    "total_distance_km": "1.97",
    "flight_duration_min": "19.8",
    "min_altitude_m": "811.3",
    "max_altitude_m": "811.4",
    "mean_altitude_m": "811.4",
    "median_altitude_m": "811.4",
    "coverage_area_ha": "1.85",
    "coverage_area_m2": "18500",
    "bbox_min_lat": "-7.358924",
    "bbox_max_lat": "-7.351276",
    "bbox_min_lon": "-40.571834",
    "bbox_max_lon": "-40.566291",
    "mean_gps_error_m": "72.45",    # IMPORTANT
    "max_gps_error_m": "191.78",     # IMPORTANT
    "mean_speed_ms": "6.1",
    "mean_speed_kmh": "22.0",
    # GSD stats (if available):
    "min_gsd_cm": "1.23",
    "max_gsd_cm": "1.45",
    "mean_gsd_cm": "1.34"
}
```

### LaTeX Report Generation

The report builder inserts flight statistics into the LaTeX document in [builder.py](../src/thermographic_report_builder/report/builder.py):

```python
with doc.create(Subsection("Informações do Voo e Ortofoto")):
    # Flight statistics table
    with doc.create(Tabular("ll")) as table:
        table.add_row(["Métrica de Voo", "Valor"])
        table.add_hline()
        table.add_row(["Total de Imagens", flight_stats["total_images"]])
        table.add_row(["Duração do Voo", f"{flight_stats['flight_duration_min']} min"])
        table.add_row(["Distância Total", f"{flight_stats['total_distance_km']} km"])
        # ... altitude, coverage, speed stats ...

        # GPS error metrics (CRITICAL)
        if "mean_gps_error_m" in flight_stats:
            table.add_hline()
            table.add_row(["Erro Médio GPS", f"{flight_stats['mean_gps_error_m']} m"])
            table.add_row(["Erro Máximo GPS", f"{flight_stats['max_gps_error_m']} m"])
```

## Error Handling

### Missing reconstruction.json

**Error**: `FileNotFoundError`

**Cause**: ODM job did not generate or upload reconstruction.json

**Solution**: Verify ODM job completed successfully and check S3 uploads

**Fallback**: Report shows "Erro ao gerar visualizações de voo."

### Malformed reconstruction.json

**Error**: `ValueError: No valid shots could be parsed`

**Cause**: reconstruction.json exists but contains no camera shot data

**Solution**: Check ODM logs for reconstruction failures

**Fallback**: Report shows error message in appendix section

### Missing Orthophoto

**Error**: None (graceful degradation)

**Behavior**: Flight path map generated without orthophoto overlay

**Impact**: Minimal - all statistics still calculated correctly

### Missing Timestamps

**Error**: None (graceful degradation)

**Behavior**:
- Flight duration shows "N/A"
- Speed chart not generated
- Distance calculated using shot order instead of chronological order

**Impact**: Limited - most metrics still available

## Troubleshooting

### Problem: "Erro ao gerar visualizações de voo" in Report

**Diagnosis Steps**:

1. Check if reconstruction.json exists in S3:
   ```bash
   aws s3 ls s3://solar-orthos-prod/{user_id}/projects/{project_id}/opensfm/reconstruction.json
   ```

2. Check CloudWatch logs for flight_viz errors:
   ```bash
   grep "flight_viz" /aws/batch/job/solar-report-prod/{job_id}/default
   ```

3. Common errors to look for:
   - `ModuleNotFoundError: No module named 'osgeo'` → GDAL not installed
   - `KeyError: 0` → reference_lla format issue
   - `ValueError: No valid shots` → Empty reconstruction

**Solutions**:
- GDAL missing: Rebuild Docker image with GDAL dependencies
- reference_lla format: Update parsers.py to handle both dict and list formats
- Empty reconstruction: Re-run ODM job with correct parameters

### Problem: NaN Values in Statistics

**Symptoms**: Distance, coverage area, or speed showing as "nan" or missing

**Cause**: Coordinate conversion not working correctly

**Diagnosis**:
1. Check if latitude values are in impossible range (-90 to +90):
   ```python
   # In reconstruction.json, if you see:
   "gps_position": [-94.2, 85.3, 123.4]  # WRONG - these are ENU, not lat/lon
   ```

2. Verify ENU to WGS84 conversion is applied:
   ```python
   # parsers.py should call:
   latitude, longitude, altitude = self._enu_to_wgs84(easting, northing, up, reference_lla)
   ```

**Solution**: Ensure parsers.py implements proper coordinate transformation

### Problem: GPS Error Not Showing

**Symptoms**: GPS error rows missing from statistics table

**Diagnosis**:
1. Check if gps_error is None in FlightMetrics:
   ```python
   print(f"GPS errors: {metrics.mean_gps_error}, {metrics.max_gps_error}")
   ```

2. Verify GPS error calculation in parsers.py:
   ```python
   gps_error = np.sqrt(
       (translation[0] - easting)**2 +
       (translation[1] - northing)**2 +
       (translation[2] - up)**2
   )
   ```

**Solution**:
- Add gps_error field to CameraShot model
- Add mean_gps_error and max_gps_error to FlightMetrics
- Update integration.py to include GPS error in stats_dict

### Problem: Flight Path Inverted or Wrong Scale

**Symptoms**:
- Flight path appears flipped
- Coordinates not in degrees (-180 to +180 longitude, -90 to +90 latitude)
- Scale shows values like -94 to +95

**Cause**: Using ENU coordinates directly instead of converting to WGS84

**Solution**: Implement ENU to WGS84 conversion in parsers.py (see Coordinate System Transformations section)

## Module Architecture

### Core Components

```
flight_viz/
├── __init__.py           # Package initialization
├── models.py             # Data structures (FlightData, FlightMetrics, CameraShot)
├── parsers.py            # reconstruction.json and GeoTIFF parsing
├── metrics.py            # Statistics calculation
├── visualizations.py     # Chart and map generation
└── integration.py        # High-level API for report builder
```

### Data Flow

```
reconstruction.json
        ↓
ReconstructionParser.parse()
        ↓
FlightData (with WGS84 coordinates)
        ↓
FlightMetricsCalculator.calculate()
        ↓
FlightMetrics (statistics)
        ↓
generate_flight_appendix()
        ↓
(viz_paths, flight_stats) → Report Builder
```

## Dependencies

### Python Packages

From [pyproject.toml](../pyproject.toml):

```toml
dependencies = [
    "GDAL==3.10.*",        # GeoTIFF parsing (pinned to Debian version)
    "folium>=0.14.0",      # Interactive maps
    "matplotlib>=3.7.0",   # Charts and plots
    "numpy>=1.24.0",       # Numerical computation
    "pyproj>=3.5.0",       # Coordinate transformations
]
```

### System Libraries

From [Dockerfile](../Dockerfile):

```dockerfile
# Builder stage
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev

# Runtime stage
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal36
```

**Important**: GDAL version must match between Python package (3.10.x) and system library (3.10.x) to avoid compatibility issues.

## Testing

### Unit Tests

Test individual components:

```python
# Test coordinate conversion
def test_enu_to_wgs84():
    parser = ReconstructionParser()
    reference_lla = (-7.355, -40.569, 800.0)
    lat, lon, alt = parser._enu_to_wgs84(100.0, 200.0, 10.0, reference_lla)
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180

# Test GPS error calculation
def test_gps_error_calculation():
    shot = CameraShot(
        gps_position_enu=[10.0, 20.0, 5.0],
        translation=[12.0, 22.0, 6.0],
        ...
    )
    expected_error = np.sqrt(2**2 + 2**2 + 1**2)
    assert abs(shot.gps_error - expected_error) < 0.01
```

### Integration Tests

Test end-to-end flow:

```python
def test_generate_flight_appendix():
    viz_paths, stats = generate_flight_appendix(
        work_dir=test_data_dir,
        output_dir=output_dir
    )

    # Verify all visualizations generated
    assert viz_paths["flight_path_map"].exists()
    assert viz_paths["altitude_chart"].exists()

    # Verify statistics calculated
    assert "mean_gps_error_m" in stats
    assert float(stats["mean_gps_error_m"]) > 0

    # Verify coordinates in valid range
    assert -90 <= float(stats["bbox_min_lat"]) <= 90
    assert -180 <= float(stats["bbox_min_lon"]) <= 180
```

### Manual Testing

Test with real data:

```bash
# 1. Download test data
aws s3 sync s3://solar-orthos-prod/{user_id}/projects/{project_id}/opensfm/ /tmp/test_opensfm/

# 2. Run flight appendix generation
python -c "
from pathlib import Path
from thermographic_report_builder.flight_viz.integration import generate_flight_appendix

viz_paths, stats = generate_flight_appendix(
    work_dir=Path('/tmp/test_opensfm'),
    output_dir=Path('/tmp/flight_output')
)

print('Statistics:', stats)
print('Visualizations:', viz_paths)
"

# 3. Verify output
ls -lh /tmp/flight_output/
cat /tmp/flight_output/stats.json
```

## Performance Considerations

### Memory Usage

- **reconstruction.json**: Can be 100MB+ for large missions (1000+ images)
- **Point cloud data**: Not loaded (would be GB-scale)
- **Orthophoto**: Loaded via GDAL (memory-mapped, minimal footprint)

**Recommendation**: Use streaming JSON parsing for very large reconstruction files

### Processing Time

Typical processing times on AWS Batch (t3.medium):

- Small mission (100 images): ~10 seconds
- Medium mission (500 images): ~30 seconds
- Large mission (1000+ images): ~60 seconds

**Bottlenecks**:
1. JSON parsing (reconstruction.json)
2. Matplotlib figure generation
3. Folium HTML generation

### Optimization Opportunities

1. **Parallel visualization generation**: Generate charts in parallel threads
2. **Lazy loading**: Only generate visualizations that will be included in report
3. **Caching**: Cache parsed reconstruction.json for multiple report types
4. **Downsampling**: For flight path with 5000+ points, downsample for visualization

## Future Enhancements

### Planned Features

1. **Orthophoto overlay on flight map**: Add faded orthophoto background to flight path visualization
2. **Camera FOV visualization**: Show camera field of view cones on 3D map
3. **Coverage heatmap**: Generate overlap/coverage heatmap from point cloud
4. **GCP integration**: Show ground control points on flight map
5. **Quality score**: Compute overall reconstruction quality score (0-100)
6. **Comparison mode**: Compare multiple flights side-by-side

### Technical Debt

1. **Cleanup English text**: Remove English labels from visualizations (Portuguese only)
2. **Remove redundant plots**: Delete small flight path plot from dashboard
3. **Coordinate system documentation**: Add detailed technical documentation on ENU/WGS84 conversion
4. **Error recovery**: More graceful handling of partially-corrupt reconstruction.json

## References

### External Documentation

- [OpenSfM Reconstruction Format](https://opensfm.org/docs/reconstruction.html)
- [GDAL Python API](https://gdal.org/python/)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [WGS84 Coordinate System](https://en.wikipedia.org/wiki/World_Geodetic_System)

### Internal Documentation

- [Report Builder Documentation](REPORT_BUILDER.md)
- [GPS Matching Documentation](GPS_MATCHING.md)
- [ODM Pipeline Documentation](../../solar-web-app/jobs/odm/README.md)

## Version History

- **v1.0.0** (January 2026): Initial implementation with coordinate conversion and GPS error metrics
  - Added ENU to WGS84 transformation
  - Implemented GPS prediction error calculation
  - Fixed NaN values in distance/coverage/speed metrics
  - Added comprehensive error handling

## Support

For issues or questions:

1. Check CloudWatch logs: `/aws/batch/job/solar-report-prod/{job_id}`
2. Review this documentation
3. Contact development team with:
   - Project ID
   - Job ID
   - Error message from logs
   - reconstruction.json sample (if possible)
