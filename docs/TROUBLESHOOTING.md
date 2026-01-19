# Troubleshooting Guide

This guide covers common issues, debugging techniques, and lessons learned from production deployments.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Debugging Techniques](#debugging-techniques)
4. [CloudWatch Log Analysis](#cloudwatch-log-analysis)
5. [Image Inspection](#image-inspection)
6. [Coordinate System Reference](#coordinate-system-reference)

---

## Quick Diagnostics

### Check Job Status
```bash
aws batch describe-jobs --jobs <JOB_ID> --region us-east-2 \
  --query 'jobs[0].{status:status,reason:statusReason}'
```

### Get Job Logs
```bash
# Get log stream name
aws batch describe-jobs --jobs <JOB_ID> --region us-east-2 \
  --query 'jobs[0].container.logStreamName' --output text

# Tail logs
aws logs tail /aws/batch/solar-report-prod \
  --log-stream-names <LOG_STREAM> \
  --region us-east-2 --follow
```

### Search for Specific Panels
```bash
aws logs filter-log-events \
  --log-group-name /aws/batch/solar-report-prod \
  --log-stream-names <LOG_STREAM> \
  --region us-east-2 \
  --filter-pattern "6-3"
```

---

## Common Issues

### 1. Job Failed to Start: "no space left on device"

**Symptom:**
```
CannotPullContainerError: write /var/lib/docker/tmp/GetImageBlob...: no space left on device
```

**Cause:** ECS container instance has insufficient disk space to pull new Docker image layers.

**Solution:**
- Wait for instance rotation (new instances have clean disk)
- Or manually terminate the problematic instance in ECS console
- Or increase EBS volume size in Batch compute environment

### 2. Source-Map Backtracking Returns None

**Symptom:** Logs show "Source-map backtracking failed" or "No valid source pixels found"

**Causes:**
1. `odm_orthophoto_sources.tif` not generated (ODM not patched)
2. Coordinates fall outside orthophoto bounds
3. Source-map has NaN values at lookup location

**Diagnosis:**
```bash
# Check if source-map exists
aws s3 ls s3://solar-orthos-prod/<USER>/<PROJECT>/odm_orthophoto/odm_orthophoto_sources.tif

# Check source-map bands
gdalinfo odm_orthophoto_sources.tif
# Should show 3 bands: View ID (UInt16), Raw X (Float32), Raw Y (Float32)
```

**Solution:** Re-run ODM processing with patched Docker image to regenerate source-map.

### 3. Hot Marker on Vegetation Instead of Panel

**Symptom:** Red circle (hot point) appears on trees/grass instead of the solar panel.

**Root Cause:** The projected panel bounding box includes surrounding vegetation. The algorithm finds the hottest pixel in this bbox, which may be vegetation.

**Why Filtering Doesn't Work:**
- Temperature filtering: Vegetation and panel defects have similar temperatures
- Brightness filtering: Some panel hotspots have similar brightness to vegetation edge pixels
- Coordinate issue: Defect center often falls outside the panel bbox in thermal space

**Current Workaround:** Accept that some edge panels may have misplaced markers. The temperature readings are still from the correct general area.

**Future Fix:** Use semantic segmentation masks instead of bounding boxes.

### 4. Blue Cross (Cold Point) Missing

**Symptom:** Annotated thermal image shows red hot marker but no blue cold marker.

**Causes:**
1. Cold point coordinates are outside image bounds
2. Panel bbox is too small after erosion (cold search excludes edges)
3. Cold point drawing failed silently

**Diagnosis:** Check logs for cold point coordinates:
```
Hot/cold detection: HOT=(x, y) ...°C, COLD=(x, y) ...°C
```

If COLD coordinates are negative or > image size, there's a coordinate calculation issue.

### 5. Wrong Image Selected for Panel

**Symptom:** Annotated thermal image shows a different area than expected.

**Causes:**
1. Source-map lookup returned wrong view ID
2. Multiple images cover the area, algorithm picked suboptimal one
3. Filename resolution failed (view ID → filename mapping)

**Diagnosis:**
```bash
# Check which image was selected
grep "Creating annotated thermal image" <LOGS> | grep "<PANEL_ID>"
```

### 6. All Images Rotated Incorrectly

**Symptom:** All thermal images appear upside-down or north is not at top.

**Cause:** Flight direction detection failed. The algorithm compares GPS coordinates between consecutive images to determine if drone flew north or south.

**Diagnosis:**
```bash
grep "flight_direction" <LOGS>
```

**Solution:** If flight direction is consistently wrong, check:
- GPS metadata quality in raw images
- Flight pattern (complex patterns may confuse the algorithm)

---

## Debugging Techniques

### Enable Debug Logging

Set environment variable:
```bash
SOLAR_LOG_LEVEL=DEBUG
```

This enables detailed logs for:
- Coordinate transformations at each step
- Source-map lookup results
- Temperature extraction details
- Panel bbox calculations

### Inspect Intermediate Files

Download the annotation manifest to see all coordinate data:
```bash
aws s3 cp s3://solar-reports-prod/<USER>/<PROJECT>/thermographic-report/annotation_manifest.json .
```

The manifest contains:
- Panel ID mappings
- Source images used
- Coordinate transformations applied
- Temperature readings

### Local Testing

Run the report builder locally with a specific project:
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e SOLAR_PROJECT_ID=<PROJECT> \
  -e SOLAR_USER_ID=<USER> \
  -e SOLAR_LOG_LEVEL=DEBUG \
  -v /tmp/output:/tmp/report_work \
  solar-report-prod:latest
```

Then inspect `/tmp/output/` for intermediate images.

### Analyze Specific Panel

To debug a specific panel, search logs:
```bash
# Find all log entries for panel 6-3
aws logs filter-log-events \
  --log-group-name /aws/batch/solar-report-prod \
  --log-stream-names <LOG_STREAM> \
  --filter-pattern "6-3" \
  --query 'events[*].message'
```

Key log entries to look for:
- `Projecting X-Y defect`: Shows coordinate transformation
- `Refined hotspot`: Shows hot point search result
- `Temperature at X-Y`: Shows extracted temperatures
- `Created annotated thermal image`: Shows final result

---

## CloudWatch Log Analysis

### Key Log Patterns

| Pattern | Meaning |
|---------|---------|
| `Source-map: (x, y) →` | Source-map lookup result |
| `Refined hotspot for X-Y defect` | Hot point refinement |
| `Temperature at X-Y defect: defect=...°C` | Temperature extraction |
| `Created annotated thermal image for X-Y` | Annotation complete |
| `flight_direction` | Detected drone flight direction |

### Filter Examples

```bash
# Find all temperature readings
aws logs filter-log-events ... --filter-pattern "Temperature at"

# Find all backtracking results
aws logs filter-log-events ... --filter-pattern "Source-map"

# Find all errors
aws logs filter-log-events ... --filter-pattern "ERROR"

# Find coordinate transformations
aws logs filter-log-events ... --filter-pattern "Projecting"
```

---

## Image Inspection

### Download Annotated Images
```bash
aws s3 sync s3://solar-reports-prod/<USER>/<PROJECT>/tex_bundle/report_images/ ./images/
```

### What to Look For

1. **Hot marker position**: Should be on the hottest visible defect
2. **Cold marker position**: Should be in the panel interior (not on edges)
3. **Image rotation**: North should be at top
4. **Panel bbox**: Should encompass the panel without too much surrounding area

### Brightness Analysis

For debugging hot point placement, extract brightness at marker location:
```python
import cv2
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Check brightness at hot marker position
brightness = img[hot_y, hot_x]
# Panels: 10-80, Vegetation: 120-255
print(f"Hot point brightness: {brightness}")
```

---

## Coordinate System Reference

### Image Spaces

| Space | Dimensions | Origin | Notes |
|-------|------------|--------|-------|
| Raw Thermal | 640 x 512 | Top-left | DJI thermal sensor |
| Raw Visual | 1280 x 1024 | Top-left | Visual component of R-JPEG |
| Full Orthophoto | ~20000 x 20000 | Top-left | Georeferenced, native resolution |
| Resampled Orthophoto | Variable | Top-left | 1.6cm/px web display |
| Cropped Orthophoto | Variable | Top-left | User-cropped and rotated |

### Transform Chain

```
Cropped Orthophoto (dx, dy)
    │
    ├─ Inverse rotation (around crop center)
    │
    ├─ Add crop offset (crop_x, crop_y)
    │
    ▼
Resampled Orthophoto (rx, ry)
    │
    ├─ Divide by resample_scale (~3.124)
    │
    ▼
Full Orthophoto (ox, oy)
    │
    ├─ Source-map lookup
    │
    ▼
Raw Image Pixel (px, py) + Image Name
```

### Thermal Alignment

Visual-to-thermal coordinate conversion includes alignment offset:
```python
thermal_x = (visual_x / 2) + alignment_offset_x  # Default: -6
thermal_y = (visual_y / 2) + alignment_offset_y  # Default: -4
```

This accounts for the physical offset between thermal and visual sensors on DJI M3T.

---

## Deployment Checklist

Before deploying a new version:

1. [ ] Build Docker image locally and test with a known project
2. [ ] Check that all coordinate systems are correctly handled
3. [ ] Verify hot/cold points land on panels (not vegetation) for sample projects
4. [ ] Push to ECR with correct tag
5. [ ] Test with AWS Batch job before marking as production

### ECR Push Commands
```bash
# Login
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 002938753233.dkr.ecr.us-east-2.amazonaws.com

# Tag and push
docker tag thermographic-report-builder:latest \
  002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-report-prod:latest
docker push 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-report-prod:latest
```
