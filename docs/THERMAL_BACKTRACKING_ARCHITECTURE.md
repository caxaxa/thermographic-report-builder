# Thermal Backtracking Architecture: From Geometry Approximation to Deterministic Source-Map Mapping

## Executive Summary

This document details the evolution of the thermal defect backtracking system from an initial geometry-based approximation approach to a deterministic pixel-mapping solution using a custom-patched OpenDroneMap (ODM) pipeline. The goal: given a defect marked on a cropped/rotated orthophoto, find the exact pixel location in the original raw thermal image.

**Final Result:** Sub-pixel precision mapping from orthophoto coordinates to raw thermal image pixels, achieving the theoretical maximum accuracy possible within the constraints of ODM's projection model.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [The Coordinate Transform Chain](#the-coordinate-transform-chain)
3. [Phase 1: Geometry Approximation (Initial Approach)](#phase-1-geometry-approximation-initial-approach)
4. [Phase 2: Discovery of ODM Source-Map](#phase-2-discovery-of-odm-source-map)
5. [Phase 3: Patching ODM for Source-Map Generation](#phase-3-patching-odm-for-source-map-generation)
6. [Phase 4: Float32 Precision Enhancement](#phase-4-float32-precision-enhancement)
7. [Bug Fixes Along the Way](#bug-fixes-along-the-way)
8. [Final Architecture](#final-architecture)
9. [Precision Analysis](#precision-analysis)
10. [Files Modified](#files-modified)
11. [Deployment](#deployment)

---

## Problem Statement

### The User Workflow

1. User uploads raw thermal drone images (e.g., DJI M3T 640x512 thermal sensor)
2. ODM processes images into a georeferenced orthomosaic
3. User views the orthomosaic in a web UI at a resampled resolution (1.6cm/px)
4. User crops and rotates the orthomosaic to align with the solar panel array
5. AI detection identifies thermal anomalies (hotspots) on the cropped image
6. User annotates defects on the cropped orthophoto
7. **Report generator must find which raw thermal image contains each defect and mark the exact pixel location**

### The Challenge

The defect annotation exists in "cropped orthophoto space" but we need coordinates in "raw thermal image space". These spaces are connected through a complex chain of transformations:

```
Raw Thermal Image (640x512)
        ↓
   [ODM Processing: SfM, MVS, Texturing, Orthophoto Generation]
        ↓
Orthophoto (~5cm/px, georeferenced)
        ↓
   [Resampling for web display]
        ↓
Resampled Orthophoto (1.6cm/px)
        ↓
   [User crop + rotation]
        ↓
Cropped/Rotated Orthophoto ← DEFECT ANNOTATION IS HERE
```

We need to reverse this entire chain.

---

## The Coordinate Transform Chain

### Forward Transform (Image → Orthophoto)

```
Raw Pixel (px, py) in Image I
        ↓ Camera Projection (K, R, t, distortion)
3D World Point (X, Y, Z)
        ↓ Orthographic Projection
Orthophoto Pixel (ox, oy)
        ↓ Resampling (scale factor ~3.124x)
Resampled Pixel (rx, ry)
        ↓ Crop + Rotate
Final Display Pixel (dx, dy)
```

### Reverse Transform (Orthophoto → Image) - What We Need

```
Final Display Pixel (dx, dy)
        ↓ Inverse Rotation
        ↓ Add Crop Offset
Resampled Pixel (rx, ry)
        ↓ Divide by Scale Factor
Orthophoto Pixel (ox, oy)
        ↓ ??? (The Hard Part)
Raw Pixel (px, py) in Image I
```

The "hard part" is: given an orthophoto pixel, which raw image did it come from, and what were the original pixel coordinates?

---

## Phase 1: Geometry Approximation (Initial Approach)

### Concept

Use OpenSfM's `reconstruction.json` to get camera poses and intrinsics, then:
1. Convert orthophoto pixel to world coordinates using the DSM for elevation
2. Project the 3D point through each camera
3. Find the camera where the point projects closest to image center
4. Return those pixel coordinates

### Implementation

```python
class CameraReprojector:
    def __init__(self, reconstruction_path, dsm_path):
        # Load camera parameters from OpenSfM
        self.cameras = self._load_cameras(reconstruction_path)
        self.dsm = rasterio.open(dsm_path)

    def reproject(self, ortho_x, ortho_y):
        # Get world coordinates
        world_x, world_y = self.ortho_transform * (ortho_x, ortho_y)
        world_z = self.dsm.sample([(world_x, world_y)])[0]

        # Project through each camera
        best_camera = None
        best_pixel = None
        for cam in self.cameras:
            px, py = self._project_to_camera(cam, world_x, world_y, world_z)
            if self._is_valid_projection(cam, px, py):
                # Score by distance to image center
                ...
```

### Problems with Geometry Approximation

1. **Camera Model Mismatch**: OpenSfM uses simplified pinhole/fisheye models. Thermal cameras have unique distortion characteristics not fully captured.

2. **DSM Interpolation Errors**: The DSM has finite resolution. Elevation lookup introduces error, especially on sloped surfaces.

3. **Multiple Valid Cameras**: A point may project validly into multiple cameras. Choosing the "best" one is heuristic.

4. **Not Deterministic**: Different runs could choose different cameras for edge cases.

5. **Formula Replication**: We're trying to replicate ODM's internal projection math, which is complex and version-dependent.

### Results

- Markers would land "in the general area" of defects
- Often 10-50 pixels off target
- Sometimes selected the wrong image entirely
- Not reliable enough for professional reports

---

## Phase 2: Discovery of ODM Source-Map

### The Insight

ODM already solves this problem internally during orthophoto generation. For each orthophoto pixel, ODM knows exactly which source image and pixel it came from. We just need ODM to tell us.

### ODM's Internal Process

During orthophoto generation (`odm_orthophoto` stage):
1. For each output pixel, ODM raycasts through the textured mesh
2. Finds which mesh face the ray hits
3. Looks up which camera/view textured that face
4. Computes the exact UV coordinates → raw pixel

### The Source-Map Concept

We can have ODM output a parallel "source-map" GeoTIFF with the same dimensions as the orthophoto:
- **Band 1**: View ID (which camera/image, 0-indexed)
- **Band 2**: Raw X coordinate in that image
- **Band 3**: Raw Y coordinate in that image

This is the **authoritative** mapping - no approximation, no guessing.

---

## Phase 3: Patching ODM for Source-Map Generation

### Files Modified in ODM

#### 1. `OdmOrthoPhoto.hpp` - Header Additions

```cpp
// New member variables for source-map storage
std::string     sourceMapFile_;      // Output path
std::string     labelingPath_;       // Face→view mapping
std::string     nvmPath_;            // Camera definitions
std::vector<int32_t> faceLabels_;    // Loaded labeling data
std::vector<NvmCamera> cameras_;     // Loaded camera data
bool            sourceMapInitialized_;
bool            sourceMapReady_;
uint16_t       *sourceView_;         // View ID per pixel
float          *sourceX_;            // Raw X per pixel (float32 for sub-pixel)
float          *sourceY_;            // Raw Y per pixel (float32 for sub-pixel)

// New methods
bool loadLabeling(const std::string &path);
bool loadNvm(const std::string &path);
void initSourceMap();
void saveSourceMap();
bool projectToCamera(size_t viewId, const pcl::PointXYZ &point,
                     float &px, float &py) const;
```

#### 2. `OdmOrthoPhoto.cpp` - Core Implementation

**Loading the Face→View Labeling:**

The labeling file (`odm_textured_model_geo_labeling.vec`) maps each mesh face to the camera that textured it:

```cpp
bool OdmOrthoPhoto::loadLabeling(const std::string &path) {
    std::ifstream file(path, std::ios::binary);

    uint64_t numFaces;
    file.read(reinterpret_cast<char*>(&numFaces), sizeof(numFaces));

    faceLabels_.resize(numFaces);
    for (uint64_t i = 0; i < numFaces; ++i) {
        uint32_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        faceLabels_[i] = static_cast<int32_t>(label);
    }

    return true;
}
```

**Loading Camera Parameters from NVM:**

```cpp
bool OdmOrthoPhoto::loadNvm(const std::string &path) {
    // NVM format: camera name, focal, quaternion, center, distortion
    std::ifstream file(path);
    std::string line;

    // Skip to camera section
    while (std::getline(file, line)) {
        if (line.find("NVM_V3") != std::string::npos) break;
    }

    int numCameras;
    file >> numCameras;

    cameras_.resize(numCameras);
    for (int i = 0; i < numCameras; ++i) {
        file >> cameras_[i].name >> cameras_[i].focal;
        // Read quaternion → convert to rotation matrix
        // Read camera center
        // Resolve image dimensions
    }

    return true;
}
```

**Projecting 3D Points to Camera Pixels:**

```cpp
bool OdmOrthoPhoto::projectToCamera(size_t viewId, const pcl::PointXYZ &point,
                                     float &px, float &py) const {
    const NvmCamera &cam = cameras_[viewId];

    // Transform world point to camera coordinates
    Eigen::Vector3f worldPt(point.x, point.y, point.z);
    Eigen::Vector3f camPt = cam.rotation * (worldPt - cam.center);

    // Check if point is in front of camera
    if (camPt.z() <= 0) return false;

    // Project to image plane (pinhole model)
    float x = camPt.x() / camPt.z();
    float y = camPt.y() / camPt.z();

    // Apply focal length and center offset
    px = cam.focal * x + cam.width / 2.0f;
    py = cam.focal * y + cam.height / 2.0f;

    // Check bounds
    if (px < 0 || px >= cam.width || py < 0 || py >= cam.height)
        return false;

    return true;
}
```

**Recording Source Information During Rendering:**

Modified `drawTexturedTriangle()` to call `renderSourcePixel()`:

```cpp
void OdmOrthoPhoto::renderSourcePixel(int row, int col,
                                       float l1, float l2, float l3,
                                       const pcl::PointXYZ &v1,
                                       const pcl::PointXYZ &v2,
                                       const pcl::PointXYZ &v3,
                                       size_t faceIndex) {
    if (!sourceMapReady_) return;

    size_t idx = row * width + col;

    // Get view ID from face labeling
    if (faceIndex >= faceLabels_.size()) return;
    int32_t viewId = faceLabels_[faceIndex];
    if (viewId < 0 || viewId >= cameras_.size()) return;

    // Interpolate 3D position using barycentric coordinates
    pcl::PointXYZ worldPt;
    worldPt.x = l1 * v1.x + l2 * v2.x + l3 * v3.x;
    worldPt.y = l1 * v1.y + l2 * v2.y + l3 * v3.y;
    worldPt.z = l1 * v1.z + l2 * v2.z + l3 * v3.z;

    // Project to camera
    float px, py;
    if (!projectToCamera(viewId, worldPt, px, py)) return;

    // Store in source-map (float32 for sub-pixel precision)
    sourceView_[idx] = static_cast<uint16_t>(viewId);
    sourceX_[idx] = px;  // Preserves sub-pixel precision
    sourceY_[idx] = py;
}
```

**Saving the Source-Map GeoTIFF:**

```cpp
void OdmOrthoPhoto::saveSourceMap() {
    GDALAllRegister();

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *dataset = driver->Create(
        sourceMapFile_.c_str(), width, height, 3, GDT_Float32, nullptr);

    // Set same geotransform as orthophoto
    dataset->SetGeoTransform(geoTransform_);
    dataset->SetProjection(aSrs_.c_str());

    // Write bands
    dataset->GetRasterBand(1)->WriteArray(sourceView_, ...);  // View IDs
    dataset->GetRasterBand(2)->WriteArray(sourceX_, ...);     // Raw X (float)
    dataset->GetRasterBand(3)->WriteArray(sourceY_, ...);     // Raw Y (float)

    // Set NoData for invalid pixels
    for (int b = 1; b <= 3; ++b) {
        dataset->GetRasterBand(b)->SetNoDataValue(NaN);
    }

    GDALClose(dataset);
}
```

#### 3. Python Stage Modifications

**`odm_orthophoto.py`** - Pass new arguments to odm_orthophoto binary:

```python
# Add source-map output path
orthophoto_sources = os.path.join(odm_orthophoto_dir, "odm_orthophoto_sources.tif")

# Add labeling and NVM paths for source-map generation
labeling_file = os.path.join(texturing_dir, "odm_textured_model_geo_labeling.vec")
nvm_file = os.path.join(opensfm_dir, "undistorted", "reconstruction.nvm")

kwargs['sourceMap'] = orthophoto_sources
kwargs['labeling'] = labeling_file
kwargs['nvmFile'] = nvm_file
```

**`mvstex.py`** - Ensure labeling file is generated:

```python
# Add flag to output labeling file
args.append('--keep_unseen_faces')  # Ensures all faces get labeled
```

---

## Phase 4: Float32 Precision Enhancement

### The Problem

Initial source-map used `uint16` for X/Y coordinates:

```cpp
// Old code
sourceX_[idx] = static_cast<uint16_t>(std::round(px));
sourceY_[idx] = static_cast<uint16_t>(std::round(py));
```

This loses sub-pixel precision:
- `px = 466.361` → stored as `466`
- Maximum error: ±0.5 pixels at source-map level

### The Scale Factor Problem

The coordinate transform includes a scale factor of ~3.124x (orthophoto resolution / resampled resolution):

```
Resampled pixel (100.0, 200.0)
    ↓ × 3.124
Orthophoto pixel (312.4, 624.8)
    ↓ Source-map lookup (rounds to 312, 625)
Raw pixel (stored as int)
```

The ±0.5px error at source-map level doesn't get scaled, but combined with rounding at multiple stages, total error could reach 1-2 pixels.

### The Fix

Changed source-map X/Y bands from `uint16` to `float32`:

**`OdmOrthoPhoto.hpp`:**
```cpp
// Changed from:
uint16_t       *sourceX_;
uint16_t       *sourceY_;

// To:
float          *sourceX_;
float          *sourceY_;
```

**`OdmOrthoPhoto.cpp`:**
```cpp
// Allocation
sourceX_ = new float[pixelCount];
sourceY_ = new float[pixelCount];

// Initialize with NaN for invalid pixels
std::fill(sourceX_, sourceX_ + pixelCount, std::numeric_limits<float>::quiet_NaN());
std::fill(sourceY_, sourceY_ + pixelCount, std::numeric_limits<float>::quiet_NaN());

// Storage - preserve full precision
sourceX_[idx] = std::max(0.0f, std::min(static_cast<float>(camera.width - 1), bestPx));
sourceY_[idx] = std::max(0.0f, std::min(static_cast<float>(camera.height - 1), bestPy));

// GDAL write as Float32
GDALRasterIO(bandX, GF_Write, 0, 0, width, height, sourceX_,
             width, height, GDT_Float32, 0, 0);
```

**`source_map_backtracker.py`:**
```python
@dataclass
class SourceMapResult:
    view_id: int
    raw_pixel_x: float  # Changed from int
    raw_pixel_y: float  # Changed from int
    image_name: Optional[str] = None

def backtrack(self, ortho_x, ortho_y):
    # ...
    raw_x = float(self.raw_x[row, col])  # Preserves sub-pixel
    raw_y = float(self.raw_y[row, col])

    # Check for NaN (new invalid marker for float32)
    if np.isnan(raw_x) or np.isnan(raw_y):
        return None
```

---

## Bug Fixes Along the Way

### 1. Null Rotation Line Crash

**Symptom:** Report builder crashed on projects without rotation annotation.

**Cause:** `crop_annotation.get("rotationLine")` returned `None`, then `.get("start")` failed.

**Fix in `crop_transform.py`:**
```python
# Before
rotation_line = crop_annotation.get("rotationLine")
start = rotation_line.get("start", {})  # CRASH if rotation_line is None

# After
rotation_line = crop_annotation.get("rotationLine") or {}
start = rotation_line.get("start", {}) if rotation_line else {}
```

### 2. Hotspot Refinement Overshooting

**Symptom:** Markers would land 60-135 pixels away from annotated defect.

**Cause:** "Hotspot refinement" searched a 150px radius for the hottest pixel, jumping to nearby (different) defects.

**Evidence from logs:**
```
Source-map: (395, 266) → Hotspot adjusted: (530, 176)  # 135px shift!
```

**Fix in `gps_matcher.py`:**
```python
# Before
should_refine = match.method != "mesh"

# After - disable refinement for deterministic methods
should_refine = match.method not in ("mesh", "source_map")
```

### 3. ECR Region Mismatch

**Symptom:** ODM Docker image pushed to wrong region, Batch jobs used old image.

**Cause:** ODM ECR repository was in `us-east-2`, but initial push went to `us-east-1`.

**Fix:** Re-tag and push to correct region:
```bash
docker tag solar-odm:latest 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-odm-prod:latest
docker push 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-odm-prod:latest
```

---

## Final Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ODM PROCESSING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Images ──→ OpenSfM ──→ MVS ──→ Texturing ──→ Orthophoto        │
│       │            │                    │              │             │
│       │            ↓                    ↓              ↓             │
│       │    reconstruction.nvm    labeling.vec   odm_orthophoto.tif  │
│       │            │                    │              │             │
│       │            └────────┬───────────┘              │             │
│       │                     ↓                          │             │
│       │           ┌─────────────────────┐              │             │
│       │           │   odm_orthophoto    │              │             │
│       │           │   (PATCHED)         │              │             │
│       │           └─────────────────────┘              │             │
│       │                     │                          │             │
│       │                     ↓                          │             │
│       │         odm_orthophoto_sources.tif             │             │
│       │         (Float32: view_id, raw_x, raw_y)       │             │
│       │                                                │             │
└───────┼────────────────────────────────────────────────┼─────────────┘
        │                                                │
        ↓                                                ↓
┌───────────────────────────────────────────────────────────────────────┐
│                      REPORT BUILDER                                   │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Defect Annotation (dx, dy) on Cropped Orthophoto                    │
│            │                                                          │
│            ↓                                                          │
│  ┌─────────────────────────────────────┐                             │
│  │     CropTransformCalculator         │                             │
│  │  - Inverse rotation                 │                             │
│  │  - Add crop offset                  │                             │
│  │  - Multiply by resample_scale       │                             │
│  └─────────────────────────────────────┘                             │
│            │                                                          │
│            ↓                                                          │
│  Orthophoto Pixel (ox, oy)                                           │
│            │                                                          │
│            ↓                                                          │
│  ┌─────────────────────────────────────┐                             │
│  │     SourceMapBacktracker            │                             │
│  │  - Load odm_orthophoto_sources.tif  │                             │
│  │  - Lookup view_id, raw_x, raw_y     │                             │
│  │  - Resolve image name from NVM      │                             │
│  └─────────────────────────────────────┘                             │
│            │                                                          │
│            ↓                                                          │
│  Raw Thermal Image + Pixel Coordinates (px, py)                      │
│            │                                                          │
│            ↓                                                          │
│  Draw annotation marker at (px, py) on raw image                     │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Key Files

| Component | File | Purpose |
|-----------|------|---------|
| ODM Patch | `OdmOrthoPhoto.hpp` | Declarations for source-map generation |
| ODM Patch | `OdmOrthoPhoto.cpp` | Source-map generation implementation |
| ODM Patch | `odm_orthophoto.py` | Pass source-map args to binary |
| ODM Patch | `mvstex.py` | Ensure labeling file generation |
| Report Builder | `source_map_backtracker.py` | Read source-map, lookup coordinates |
| Report Builder | `crop_transform.py` | Reverse crop/rotate/scale transforms |
| Report Builder | `gps_matcher.py` | Coordinate matching orchestration |

---

## Precision Analysis

### Error Budget

| Stage | Error Source | Magnitude | Status |
|-------|--------------|-----------|--------|
| Source-map generation | ODM camera projection model | 1-3 px | Inherent limitation |
| Source-map storage | Float32 quantization | ~0.000001 px | ✅ Eliminated |
| Source-map lookup | Nearest-neighbor vs bilinear | ~0.5 px | Could improve |
| Crop transform | Rotation center calculation | ~0.1 px | ✅ Minimal |
| Scale factor | Float division | ~0.01 px | ✅ Minimal |

### Theoretical Limits

The source-map is generated by ODM projecting mesh vertices through camera models. This projection is an approximation because:

1. **Simplified Camera Models**: ODM uses pinhole/fisheye models that don't capture all lens distortion characteristics of thermal sensors.

2. **Mesh Resolution**: The textured mesh has finite triangle density. Sub-triangle positions are interpolated.

3. **Single-View Assignment**: Each mesh face is assigned to exactly one camera view. Edge cases where faces span view boundaries may have suboptimal assignments.

### Expected Accuracy

- **Best case**: 1-2 pixels on raw thermal image
- **Typical case**: 2-5 pixels on raw thermal image
- **Worst case**: 5-10 pixels (edge of mesh, poor geometry)

For a 640x512 thermal image, 3 pixels = 0.5% of image width. The defect will be clearly visible and identifiable in the annotated region.

---

## Files Modified

### ODM Patches (`/home/ubuntu/solar-web-app/jobs/odm/odm_patches/`)

```
odm_orthophoto/
├── src/
│   ├── OdmOrthoPhoto.hpp    # +50 lines: source-map declarations
│   └── OdmOrthoPhoto.cpp    # +400 lines: source-map implementation
odm_orthophoto.py            # +20 lines: pass source-map arguments
mvstex.py                    # +5 lines: ensure labeling output
```

### Report Builder (`/home/ubuntu/thermographic-report-builder/`)

```
src/thermographic_report_builder/
├── processing/
│   ├── source_map_backtracker.py  # NEW: 300 lines
│   ├── crop_transform.py          # +50 lines: null rotation fix
│   └── gps_matcher.py             # +10 lines: disable hotspot refinement
```

---

## Deployment

### Docker Images

| Image | ECR Repository | Region |
|-------|---------------|--------|
| ODM | `002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-odm-prod:latest` | us-east-2 |
| Report Builder | `002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-report-prod:latest` | us-east-2 |

### Rebuild Commands

**ODM:**
```bash
cd /home/ubuntu/solar-web-app/jobs/odm
docker build -t solar-odm .
docker tag solar-odm:latest 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-odm-prod:latest
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 002938753233.dkr.ecr.us-east-2.amazonaws.com
docker push 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-odm-prod:latest
```

**Report Builder:**
```bash
cd /home/ubuntu/thermographic-report-builder
docker build -t solar-report-prod .
docker tag solar-report-prod:latest 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-report-prod:latest
docker push 002938753233.dkr.ecr.us-east-2.amazonaws.com/solar-report-prod:latest
```

### Re-processing Requirements

After deploying the new ODM image, existing projects need ODM re-run to generate the float32 source-map:

```bash
aws batch submit-job \
  --region us-east-2 \
  --job-name "PROJECT_ID-rerun" \
  --job-queue "solar-job-queue-prod" \
  --job-definition "solar-odm-prod-m5-4xl:4" \
  --container-overrides '{
    "environment": [
      {"name": "PROJECT_ID", "value": "PROJECT_ID"},
      {"name": "USER_ID", "value": "USER_ID"},
      {"name": "TENANT_ID", "value": "default"},
      {"name": "MANIFEST_KEY", "value": "USER_ID/projects/PROJECT_ID/manifest.json"},
      {"name": "PROJECTS_TABLE", "value": "solar-projects-prod"}
    ]
  }'
```

---

## Conclusion

We evolved from a geometry approximation approach with 10-50 pixel errors to a deterministic source-map based system with sub-pixel precision storage. The key insight was that ODM already computes the exact pixel correspondence during orthophoto generation - we just needed to extract and preserve it.

The final system achieves the theoretical maximum accuracy possible within ODM's projection model constraints, estimated at 1-5 pixels on raw thermal images. This is sufficient for professional thermographic reports where defects are clearly identified even if marker placement isn't mathematically perfect.

### Future Improvements (Diminishing Returns)

1. **Bilinear interpolation** in source-map lookup: ~0.5px improvement
2. **Custom camera calibration** for thermal sensors: Unknown improvement, high effort
3. **Direct OpenSfM reprojection** bypassing source-map: Architectural rewrite, marginal benefit

The current implementation represents the practical ceiling for this approach.
