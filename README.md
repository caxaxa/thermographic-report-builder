# Thermographic Report Builder

Production-ready container for generating thermographic inspection reports for solar panel installations.

## Overview

This service processes thermal defect detection data and generates comprehensive PDF reports with:
- Annotated orthophoto maps
- Per-panel defect details with images
- Metrics and statistics (JSON/CSV)
- GPS-matched raw thermal imagery

## Architecture

```
thermographic_report_builder/
├── src/thermographic_report_builder/
│   ├── models/          # Pydantic data models
│   ├── io/              # S3 and file I/O abstraction
│   ├── processing/      # Image processing and defect mapping
│   ├── report/          # PDF generation and metrics
│   ├── config/          # Settings and constants
│   ├── utils/           # Logging and exceptions
│   └── main.py          # AWS Batch entrypoint
├── latex-compiler/      # Standalone LaTeX compiler Batch job (Docker + script)
├── tests/               # Unit and integration tests
├── LEGACY_CODE/         # Original prototype (reference only)
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Key Features

### ✅ Type-Safe with Pydantic
- Full type hints throughout
- Validated data models for defects, panels, and reports
- Configuration via environment variables

### ✅ Cloud-Native
- S3 abstraction for inputs/outputs
- Structured JSON logging for CloudWatch
- AWS Batch compatible

### ✅ Production-Ready
- Comprehensive error handling
- Custom exception hierarchy
- Retry-safe and idempotent

### ✅ Testable
- Dependency injection
- Mockable I/O layer
- Unit test ready

### ✅ Flight Data Visualization
- Automated flight appendix with mission statistics
- GPS prediction error analysis (bundle adjustment quality)
- Interactive and static flight path maps
- Altitude and speed profile charts
- See [Flight Appendix Documentation](docs/FLIGHT_APPENDIX.md)

## Input/Output

### Inputs (from S3)
- **Orthophoto**: `s3://solar-orthos-{env}/{user}/projects/{project}/odm_orthophoto.tif`
- **Defect Labels**: `s3://solar-reports-{env}/{user}/projects/{project}/defect_labels.json`
- **Raw Thermal Images**: `s3://solar-uploads-{env}/{user}/projects/{project}/images/*.JPG`
- **(Optional) Reconstruction Data**: `s3://solar-orthos-{env}/{user}/projects/{project}/opensfm/reconstruction.json` - For flight appendix
- **(Optional) ODM Stats**: `s3://solar-intermediate-{env}/{user}/projects/{project}/odm/stats/`

#### Defect Labels Schema

`src/thermographic_report_builder/models/job.py` expects the Phase 2.1 JSON produced by `scripts/convert_yolo_to_report_format.py`:

```json
[
  {
    "boundingBox": {
      "boundingBoxes": [
        {
          "left": 220.06,
          "top": 329.78,
          "width": 156.90,
          "height": 84.10,
          "label": "solarpanels"
        }
      ]
    }
  }
]
```

The report builder downloads that file from `s3://solar-reports-{env}/{user}/projects/{project}/defect_labels.json`.

### Outputs (to S3)
- **Full PDF**: `s3://solar-reports-{env}/{user}/projects/{project}/thermographic-report/report-full.pdf`
- **Low-Res PDF**: `s3://solar-reports-{env}/{user}/projects/{project}/thermographic-report/report-lowres.pdf`
- **Metrics JSON**: `s3://solar-reports-{env}/{user}/projects/{project}/thermographic-report/metrics.json`
- **Metrics CSV**: `s3://solar-reports-{env}/{user}/projects/{project}/thermographic-report/metrics.csv`

`JobOutput` (see `models/job.py`) also includes the LaTeX bundle path so upstream services can archive every artifact produced during the run.

## Environment Variables

All settings are configured via environment variables (prefixed with `SOLAR_`):

```bash
# Required
SOLAR_PROJECT_ID=01K7T3CVXV08S8Y2QSA0PCG1JF
SOLAR_USER_ID=018bb5d0-4001-700f-64dd-8db4da378347

# Optional (with defaults)
SOLAR_AWS_REGION=us-east-2
SOLAR_ORTHOS_BUCKET=solar-orthos-prod
SOLAR_UPLOADS_BUCKET=solar-uploads-prod
SOLAR_REPORTS_BUCKET=solar-reports-prod
SOLAR_AREA_NAME="Solar Farm"
SOLAR_LOG_LEVEL=INFO
```

### Runtime Configuration Reference

`src/thermographic_report_builder/config/settings.py` exposes additional knobs:

| Setting | Default | Description |
|---------|---------|-------------|
| `SOLAR_ORTHOPHOTO_DOWNSCALE_FACTOR` | `0.25` | Downscale factor applied when generating the overview PNG |
| `SOLAR_CROP_DOWNSCALE_FACTOR` | `0.5` | Downscale applied to per-defect crops |
| `SOLAR_DEFAULT_PANEL_WIDTH_PX` | `127` | Used to size crops when detections are sparse |
| `SOLAR_CROP_PANEL_SIZE` | `5` | Number of panels captured per crop tile |
| `SOLAR_REPORT_LANGUAGE` | `pt-BR` | Passed to LaTeX `babel` |
| `SOLAR_CLIENT_NAME` / `SOLAR_ENGINEER_NAME` / `SOLAR_CREA_NUMBER` | anonymized values | Strings printed on the PDF cover and appendix |
| `SOLAR_LOG_JSON` | `true` | Toggle structured JSON logging for CloudWatch |

Tune these via environment variables in AWS Batch or a local `.env` file.

## Building and Running

### Docker Image Size

**Note**: The Docker image is approximately **1.2 GB** due to LaTeX dependencies required for PDF generation:
- `texlive-latex-base` (~200 MB)
- `texlive-latex-extra` (~400 MB) - Includes geometry, fancyhdr, subfig, tikz, xcolor
- `texlive-lang-portuguese` (~50 MB)
- Python packages (~300 MB)
- Base image (~250 MB)

**Alternative approaches** to reduce image size (future optimization):
1. Pre-generate LaTeX as HTML/Markdown, then convert to PDF with lighter tools (wkhtmltopdf)
2. Use a minimal LaTeX subset (would require rewriting report templates)
3. Generate reports on a dedicated report service instead of in Batch

### Build Docker Image

```bash
docker build -t solar-thermographic-report:latest .
```

**Build time**: ~5-10 minutes (due to LaTeX installation)

### LaTeX Compiler Job

The AWS Batch job that turns the uploaded `.tex` bundle into PDFs now lives alongside this repo under `./latex-compiler`:

```bash
docker build -t solar-latex-compiler:latest ./latex-compiler
```

That image still exposes the same `compile.py` entrypoint, so AWS Batch job definitions only need a new image URI.

### Run Locally (with AWS credentials)

```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN \
  -e SOLAR_PROJECT_ID=01K7T3CVXV08S8Y2QSA0PCG1JF \
  -e SOLAR_USER_ID=018bb5d0-4001-700f-64dd-8db4da378347 \
  solar-thermographic-report:latest
```

### Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Code formatting
black src/ tests/
ruff check src/ tests/
```

## Deployment

### ECR Push

```bash
# Tag and push to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-2.amazonaws.com
docker tag solar-thermographic-report:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/solar-thermographic-report-prod:latest
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/solar-thermographic-report-prod:latest
```

### AWS Batch Configuration

```typescript
// In CDK stack
const reportBuilderJob = new batch.JobDefinition(this, 'ReportBuilderJob', {
  container: {
    image: ecs.ContainerImage.fromEcrRepository(reportBuilderRepo, 'latest'),
    vcpus: 2,
    memoryLimitMiB: 8192,
    environment: {
      SOLAR_AWS_REGION: 'us-east-2',
      SOLAR_ORTHOS_BUCKET: orthosBucket.bucketName,
      SOLAR_UPLOADS_BUCKET: uploadsBucket.bucketName,
      SOLAR_REPORTS_BUCKET: reportsBucket.bucketName,
    },
  },
  timeout: Duration.hours(2),
});
```

## Processing Pipeline

1. **Download Inputs** - Fetch orthophoto, labels, raw images, and reconstruction data from S3
2. **Parse Data** - Load GeoTIFF and defect labels JSON
3. **Map Defects** - Assign defects to panel grid using spatial algorithms
4. **Annotate Orthophoto** - Draw bounding boxes on overview image
5. **Create Layer Map** - Generate vectorized PDF with panel grid
6. **Crop Defect Regions** - Extract detailed views of each defect
7. **Match Raw Thermal Images** - Use source-map backtracking to find exact pixel coordinates in raw thermal images
8. **Annotate Thermal Images** - Draw hot/cold markers on raw thermal images with temperature readings
9. **Generate Flight Appendix** - Calculate flight metrics and create visualizations (if reconstruction.json available)
10. **Generate PDF** - Create LaTeX document and compile to PDF
11. **Export Metrics** - Calculate statistics and export to JSON/CSV
12. **Upload Results** - Push all artifacts to S3

## Raw Thermal Image Backtracking

The report builder uses a sophisticated **source-map backtracking** system to locate defects in raw thermal images. This is critical for professional reports that need to show the actual thermal anomaly on the original drone footage.

### How It Works

1. **ODM Source-Map**: During orthophoto generation, a patched ODM pipeline generates `odm_orthophoto_sources.tif` - a GeoTIFF that maps each orthophoto pixel to its source raw image and pixel coordinates (Float32 precision).

2. **Coordinate Transform Chain**:
   ```
   Defect on Cropped Orthophoto
         ↓ Inverse rotation + crop offset
   Resampled Orthophoto Pixel
         ↓ ÷ resample scale factor
   Full Orthophoto Pixel
         ↓ Source-map lookup
   Raw Image + Pixel Coordinates
   ```

3. **Camera Projection**: The source-map uses camera projection from `reconstruction.nvm` to compute sub-pixel accurate coordinates through the textured 3D mesh.

### Key Files

| File | Purpose |
|------|---------|
| `processing/source_map_backtracker.py` | Loads source-map GeoTIFF, performs coordinate lookups |
| `processing/crop_transform.py` | Reverses crop/rotation transforms |
| `processing/camera_projector.py` | Projects panel corners to raw image for bounding box |
| `processing/gps_matcher.py` | Orchestrates matching, downloads raw images |
| `processing/thermal_annotator.py` | Draws hot/cold markers, extracts temperatures |

### Fallback Chain

If source-map backtracking fails, the system falls back to:
1. **Reconstruction.json camera reprojection** - Uses OpenSfM camera poses
2. **GPS proximity matching** - Finds closest image by GPS distance

See `docs/THERMAL_BACKTRACKING_ARCHITECTURE.md` for complete technical details.

## Panel Numbering System

The report builder uses a **column-first numbering system** that groups panels spatially:

### Grouping Logic

1. **Columns**: Panels within 1 panel width horizontally are grouped into the same column
   - Columns are numbered 1, 2, 3... from **left to right**

2. **Rows within columns**: Panels are numbered top to bottom within each column
   - Rows are numbered 1, 2, 3... from **top to bottom**

3. **Tracker breaks**: If the vertical gap between panels exceeds 3 panel heights, a new tracker section begins
   - Tracker A is the default (no suffix)
   - Tracker B, C, etc. restart row numbering with a suffix

### Panel ID Format

| Scenario | Format | Example | Meaning |
|----------|--------|---------|---------|
| Tracker A (default) | `Col-Row` | `3-45` | Column 3, Row 45 |
| Tracker B | `Col-RowB` | `3-1B` | Column 3, Row 1 in Tracker B |
| Tracker C | `Col-RowC` | `5-12C` | Column 5, Row 12 in Tracker C |

### Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `COLUMN_TOLERANCE` | `1.0` | Panels within this many panel widths are in the same column |
| `TRACKER_GAP_THRESHOLD` | `3.0` | Vertical gap (in panel heights) that triggers a new tracker |

## Image Orientation (North-Up)

Raw thermal images from the drone are automatically rotated to ensure **north is at the top** of the image.

### How It Works

1. Images are sorted by filename (which reflects capture order)
2. For each consecutive pair of images, GPS coordinates are compared:
   - If latitude **decreases** → drone is flying **south** → rotate image 180°
   - If latitude **increases** → drone is flying **north** → no rotation needed

3. The rotation is applied before saving images to the report

### Why This Matters

Drone thermal cameras typically capture images with the drone's heading at the top. When flying south, "south" appears at the top of the frame. Rotating 180° ensures consistent north-up orientation across all images in the report.

## Defect Ordering in Reports

Defects are ordered consistently throughout the report:

1. **By defect class** (in order):
   - Pontos Quentes (Hot Spots)
   - Diodos de Bypass Queimados (Faulty Diodes)
   - Painéis Desligados (Offline Panels)

2. **By column** (left to right)

3. **By row** (top to bottom within each column)

This ordering applies to both the summary table and the detailed sections.

## Troubleshooting

| Symptom | Likely Cause | Recommended Action |
|---------|--------------|--------------------|
| `defect_labels.json` not found | Detection pipeline failed or S3 prefix typo | Verify Detectron inference uploaded labels to `solar-reports-{env}/{user}/projects/{project}/defect_labels.json`; rerun inference if missing |
| No GPS matches | Raw thermal images missing EXIF or not uploaded | Confirm `s3://solar-uploads-{env}/{user}/projects/{project}/images/` contains original frames with GPS metadata |
| LaTeX build failure | Non-ASCII characters or missing packages | Inspect CloudWatch logs, ensure env vars are UTF-8, rebuild image only after confirming required `texlive-*` packages |
| Blank overview/crops | Orthophoto download failed or path mismatch | Confirm IAM role permissions and bucket names (`SOLAR_ORTHOS_BUCKET`, `SOLAR_UPLOADS_BUCKET`); enable `SOLAR_LOG_LEVEL=DEBUG` for verbose traces |
| Wrong panel numbering | Panel annotations not aligned or tolerances off | Check that panels are properly aligned in annotation tool; adjust `COLUMN_TOLERANCE` or `TRACKER_GAP_THRESHOLD` if needed |
| Images not rotated correctly | Flight path not purely north/south | The rotation algorithm assumes linear north/south flight paths; complex flight patterns may need manual review |
| All images rotated same direction | Single-direction flight | This is expected behavior for a single-pass flight; the algorithm correctly detects consistent flight direction |
| Hot marker on vegetation instead of panel | Panel bbox includes surrounding vegetation | Known limitation - see Hot Point Detection section below |
| Blue cross (cold point) missing | Cold point coordinates invalid or off-image | Check logs for cold point calculation errors |
| Source-map backtracking fails | Source-map not generated or corrupted | Re-run ODM processing with patched image to regenerate `odm_orthophoto_sources.tif` |
| Defect coordinates outside panel bbox | Coordinate transform chain mismatch | Enable DEBUG logging to trace coordinate transformations |
| "Erro ao gerar visualizações de voo" in report | reconstruction.json missing or malformed | Verify ODM job completed successfully and uploaded reconstruction.json to S3; see [Flight Appendix Troubleshooting](docs/FLIGHT_APPENDIX.md#troubleshooting) |
| NaN values in flight statistics | Coordinate conversion issue (ENU vs WGS84) | Check that latitude values are -90 to +90 (not -94 to +95); ensure ENU to WGS84 conversion is applied |
| GPS error metric missing | GPS error calculation not implemented | Verify FlightMetrics includes mean_gps_error and max_gps_error fields |

## Hot Point Detection Algorithm

The thermal annotator identifies the hottest and coldest points within each panel for temperature delta calculations.

### Algorithm Overview

1. **Panel Bbox Projection**: Panel corners from the orthophoto are projected onto the raw thermal image using camera parameters from `reconstruction.json`
2. **Visual-to-Thermal Coordinate Mapping**: Coordinates are converted from visual (1280x1024) to thermal (640x512) space with alignment offset
3. **Hot Point Search**: Find the hottest pixel within a search radius around the defect center, constrained to the panel bbox
4. **Cold Point Search**: Find the coldest pixel within the panel interior (excluding edges to avoid shadow artifacts)
5. **Temperature Extraction**: Use DJI thermal SDK to read actual temperature values from the R-JPEG thermal data

### Known Limitations

**Panel Bbox vs Actual Panel**: The projected panel bounding box may include surrounding vegetation or other hot objects. The algorithm searches for the hottest pixel within this bbox, which may land on vegetation rather than the actual panel hotspot.

**Attempted Fixes That Didn't Work**:
- Temperature-based vegetation filtering (vegetation and panel hotspots have similar temperatures)
- Visual brightness thresholding (whitehot thermal images show vegetation as bright, panels as dark)
- Fixed brightness threshold filtering (threshold of 100 excluded too many valid panel pixels)

**Current Behavior**: The algorithm finds the hottest point within the panel bbox. If vegetation is included in the bbox and is thermally hotter than the panel defect, the marker may land on vegetation.

**Future Improvement Ideas**:
1. Use semantic segmentation to identify actual panel pixels
2. Use panel mask from detection model instead of projected bbox
3. Multi-scale hotspot detection with local maxima filtering

### Flight Direction Handling

For south-facing flights, both the thermal array and visual image are rotated 180° to maintain consistent coordinate systems. This is critical for accurate hot/cold point detection.

## Monitoring

Structured JSON logs are sent to CloudWatch with these key events:

```json
{
  "timestamp": "2025-10-18T20:00:00",
  "level": "INFO",
  "name": "thermographic_report_builder.main",
  "message": "STEP 3: Mapping defects to panel grid",
  "project_id": "01K7T3CVXV08S8Y2QSA0PCG1JF"
}
```

## Legacy Code

The original prototype is preserved in `LEGACY_CODE/`. Production uses the modern code in `src/`, but we intentionally keep the legacy assets because the PDF layout quality is still superior for a few customer farms. Feel free to mine it for template ideas while we continue closing the gap.

- Type safety with Pydantic
- Cloud-native S3 integration
- Proper error handling
- Structured logging
- Testable architecture

## License

Copyright © 2025 Aisol. All rights reserved.
