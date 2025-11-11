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

## Input/Output

### Inputs (from S3)
- **Orthophoto**: `s3://solar-orthos-{env}/{user}/projects/{project}/odm_orthophoto.tif`
- **Defect Labels**: `s3://solar-reports-{env}/{user}/projects/{project}/defect_labels.json`
- **Raw Thermal Images**: `s3://solar-uploads-{env}/{user}/projects/{project}/images/*.JPG`
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

1. **Download Inputs** - Fetch orthophoto, labels, and raw images from S3
2. **Parse Data** - Load GeoTIFF and defect labels JSON
3. **Map Defects** - Assign defects to panel grid using spatial algorithms
4. **Annotate Orthophoto** - Draw bounding boxes on overview image
5. **Create Layer Map** - Generate vectorized PDF with panel grid
6. **Crop Defect Regions** - Extract detailed views of each defect
7. **Match GPS Images** - Find closest raw thermal image for each defect
8. **Generate PDF** - Create LaTeX document and compile to PDF
9. **Export Metrics** - Calculate statistics and export to JSON/CSV
10. **Upload Results** - Push all artifacts to S3

## Troubleshooting

| Symptom | Likely Cause | Recommended Action |
|---------|--------------|--------------------|
| `defect_labels.json` not found | Detection pipeline failed or S3 prefix typo | Verify Detectron inference uploaded labels to `solar-reports-{env}/{user}/projects/{project}/defect_labels.json`; rerun inference if missing |
| No GPS matches | Raw thermal images missing EXIF or not uploaded | Confirm `s3://solar-uploads-{env}/{user}/projects/{project}/images/` contains original frames with GPS metadata |
| LaTeX build failure | Non-ASCII characters or missing packages | Inspect CloudWatch logs, ensure env vars are UTF-8, rebuild image only after confirming required `texlive-*` packages |
| Blank overview/crops | Orthophoto download failed or path mismatch | Confirm IAM role permissions and bucket names (`SOLAR_ORTHOS_BUCKET`, `SOLAR_UPLOADS_BUCKET`); enable `SOLAR_LOG_LEVEL=DEBUG` for verbose traces |

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
