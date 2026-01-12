# Plan: Deterministic Camera Selection for Pinpoint Reprojection

## Goal
Make camera selection 100% deterministic and geometrically optimal by implementing a proper selection hierarchy that mirrors ODM's internal logic.

## Current State
- `camera_projector.py`: Uses `distance_from_center` as sole criterion (wrong priority)
- `gps_matcher.py`: Adds temperature scoring for hotspots (non-deterministic)
- No stable tie-breaker (dict ordering is arbitrary)
- No incidence angle calculation
- No occlusion check

## Target State
For each ortho pixel, deterministically select the ONE best raw image using:
1. **Visibility filter** (hard requirements)
2. **Incidence angle** (primary criterion - most perpendicular view)
3. **Distance to camera** (secondary - closer = better resolution)
4. **Distance from image center** (tertiary - less lens distortion)
5. **Lexicographic filename** (stable tie-breaker)

---

## Implementation Steps

### Step 1: Add Surface Normal Calculation
**File:** `camera_projector.py`

Add method to compute surface normal at a world point using DSM gradient:

```python
def _compute_surface_normal(self, world_x: float, world_y: float) -> np.ndarray:
    """
    Compute surface normal at world point using DSM gradient.

    For flat solar panels, this will be approximately [0, 0, 1] (pointing up).
    The gradient gives us the local slope which affects viewing angle.

    Returns:
        Unit normal vector [nx, ny, nz]
    """
    if self._dsm_data is None:
        # No DSM: assume flat ground, normal pointing up
        return np.array([0.0, 0.0, 1.0])

    # Sample DSM at point and neighbors to compute gradient
    # Use small delta (1 DSM pixel) for finite difference
    delta = abs(self._dsm_transform.a)  # DSM pixel size in meters

    z_center = self.get_elevation_at_world_point(world_x, world_y)
    z_east = self.get_elevation_at_world_point(world_x + delta, world_y)
    z_north = self.get_elevation_at_world_point(world_x, world_y + delta)

    # Gradient: dz/dx (east), dz/dy (north)
    dzdx = (z_east - z_center) / delta
    dzdy = (z_north - z_center) / delta

    # Normal from gradient: n = [-dzdx, -dzdy, 1] normalized
    normal = np.array([-dzdx, -dzdy, 1.0])
    normal = normal / np.linalg.norm(normal)

    return normal
```

### Step 2: Add Incidence Angle Calculation
**File:** `camera_projector.py`

Add method to compute viewing angle between camera ray and surface normal:

```python
def _compute_incidence_angle(
    self,
    world_point: np.ndarray,
    shot: dict,
    surface_normal: np.ndarray
) -> float:
    """
    Compute incidence angle between camera viewing ray and surface normal.

    Args:
        world_point: 3D point in world coordinates [x, y, z]
        shot: Camera shot data
        surface_normal: Unit normal at the surface point

    Returns:
        Angle in radians (0 = perpendicular view, pi/2 = grazing)
    """
    # Get camera position in world coordinates
    # OpenSfM: translation = R @ (-camera_position)
    # So: camera_position = -R^T @ translation
    rotation = np.array(shot.get("rotation", [0, 0, 0]))
    translation = np.array(shot.get("translation", [0, 0, 0]))
    R = self._axis_angle_to_rotation_matrix(rotation)
    camera_position = -R.T @ translation

    # Viewing ray: from camera to world point (normalized)
    view_ray = world_point - camera_position
    view_ray = view_ray / np.linalg.norm(view_ray)

    # Incidence angle: angle between -view_ray and surface_normal
    # We use -view_ray because we want the ray pointing AT the surface
    cos_angle = np.dot(-view_ray, surface_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety

    return np.arccos(cos_angle)
```

### Step 3: Add Distance to Camera Calculation
**File:** `camera_projector.py`

Add method to compute 3D distance from camera to world point:

```python
def _compute_camera_distance(self, world_point: np.ndarray, shot: dict) -> float:
    """
    Compute 3D distance from camera to world point.

    Args:
        world_point: 3D point in world coordinates
        shot: Camera shot data

    Returns:
        Distance in meters
    """
    rotation = np.array(shot.get("rotation", [0, 0, 0]))
    translation = np.array(shot.get("translation", [0, 0, 0]))
    R = self._axis_angle_to_rotation_matrix(rotation)
    camera_position = -R.T @ translation

    return float(np.linalg.norm(world_point - camera_position))
```

### Step 4: Update RawImageMatch Dataclass
**File:** `camera_projector.py`

Extend to include all selection criteria:

```python
@dataclass
class RawImageMatch:
    """A match between an orthophoto location and a raw image pixel."""

    image_name: str
    pixel_x: float
    pixel_y: float
    distance_from_center: float  # 0.0 = center, 1.0 = corner
    incidence_angle: float = 0.0  # radians, 0 = perpendicular (best)
    camera_distance: float = 0.0  # meters

    @property
    def is_central(self) -> bool:
        """True if the point is in the central 50% of the image."""
        return self.distance_from_center < 0.5

    def selection_key(self) -> tuple:
        """
        Return sort key for deterministic selection.

        Priority (ascending = better):
        1. Incidence angle (lower = more perpendicular = better)
        2. Camera distance (lower = closer = better)
        3. Distance from center (lower = more central = better)
        4. Image name (lexicographic for stable tie-breaking)
        """
        return (
            self.incidence_angle,
            self.camera_distance,
            self.distance_from_center,
            self.image_name,
        )
```

### Step 5: Update project_ortho_to_raw Method
**File:** `camera_projector.py`

Compute all criteria and sort properly:

```python
def project_ortho_to_raw(
    self,
    ortho_x: float,
    ortho_y: float,
    elevation: Optional[float] = None,
    max_gps_distance: Optional[float] = None,
) -> List[RawImageMatch]:
    """
    Project an orthophoto pixel to raw image pixels.

    Returns matches sorted by deterministic selection criteria:
    1. Incidence angle (most perpendicular first)
    2. Camera distance (closest first)
    3. Distance from center (most central first)
    4. Image name (lexicographic tie-breaker)
    """
    if not self.available:
        return []

    # Get precise elevation from DSM if not provided
    if elevation is None:
        elevation = self.get_elevation_at_ortho_pixel(ortho_x, ortho_y)

    # Convert orthophoto pixel to world coordinates (local ENU)
    world_x, world_y = self._ortho_to_world(ortho_x, ortho_y)
    world_point = np.array([world_x, world_y, elevation])

    # Compute surface normal for incidence angle calculation
    surface_normal = self._compute_surface_normal(world_x, world_y)

    matches = []

    for image_name, shot in self.shots.items():
        try:
            # 1. Visibility check: project to camera
            pixel_coords = self._project_to_camera(world_point, shot)

            if pixel_coords is None:
                continue  # Behind camera

            px, py = pixel_coords

            # 2. Bounds check
            if not (0 <= px < self.image_width and 0 <= py < self.image_height):
                continue

            # 3. Compute selection criteria
            incidence_angle = self._compute_incidence_angle(
                world_point, shot, surface_normal
            )
            camera_distance = self._compute_camera_distance(world_point, shot)

            # Distance from image center (normalized)
            cx, cy = self.image_width / 2, self.image_height / 2
            max_dist = np.hypot(cx, cy)
            distance_from_center = np.hypot(px - cx, py - cy) / max_dist

            matches.append(RawImageMatch(
                image_name=image_name,
                pixel_x=px,
                pixel_y=py,
                distance_from_center=distance_from_center,
                incidence_angle=incidence_angle,
                camera_distance=camera_distance,
            ))

        except Exception as e:
            logger.debug(f"Projection failed for {image_name}: {e}")
            continue

    # DETERMINISTIC SORT using selection_key
    matches.sort(key=lambda m: m.selection_key())

    if matches:
        best = matches[0]
        logger.debug(
            f"Best match: {best.image_name} "
            f"(angle={np.degrees(best.incidence_angle):.1f}°, "
            f"dist={best.camera_distance:.1f}m, "
            f"center={1-best.distance_from_center:.2f})"
        )

    return matches
```

### Step 6: Add Helper for World Point Elevation
**File:** `camera_projector.py`

Add method to get elevation at arbitrary world coordinates (not just ortho pixels):

```python
def get_elevation_at_world_point(self, world_x: float, world_y: float) -> float:
    """
    Get elevation at world coordinates (local ENU) from DSM.

    This is needed for surface normal computation where we sample
    neighboring points that may not correspond to ortho pixels.
    """
    if self._dsm_data is None:
        return self._ground_elevation

    # Convert local ENU back to UTM
    utm_x = world_x + self._ref_utm_x
    utm_y = world_y + self._ref_utm_y

    # Convert UTM to DSM pixel coordinates
    dsm_col = (utm_x - self._dsm_transform.c) / self._dsm_transform.a
    dsm_row = (utm_y - self._dsm_transform.f) / self._dsm_transform.e

    dsm_h, dsm_w = self._dsm_data.shape

    if not (0 <= dsm_col < dsm_w and 0 <= dsm_row < dsm_h):
        return self._ground_elevation

    # Nearest neighbor for simplicity (bilinear would be better but more code)
    elevation_utm = self._dsm_data[int(dsm_row), int(dsm_col)]

    # Convert to local ENU Z
    if self._dsm_nodata is not None and elevation_utm == self._dsm_nodata:
        return self._ground_elevation

    return float(elevation_utm - self._dsm_to_enu_offset)
```

### Step 7: Update gps_matcher.py
**File:** `gps_matcher.py`

Remove temperature-based camera SELECTION (keep it for validation only):

```python
def _find_best_match(self, ortho_x: float, ortho_y: float, temp_dir: Path) -> Optional[DefectMatch]:
    """
    Find the best raw image match using deterministic geometric selection.

    Temperature is NOT used for selection (non-deterministic).
    The geometric criteria ensure we always pick the same image.
    """
    if not self.camera_projector.available or self._use_gps_fallback:
        return None

    # Get ALL candidate matches sorted by deterministic criteria
    all_matches = self.camera_projector.project_ortho_to_raw(ortho_x, ortho_y)

    if not all_matches:
        return None

    # Take the FIRST match (deterministically best)
    best = all_matches[0]
    image_path = temp_dir / best.image_name

    if not image_path.exists():
        # Try next best if file missing
        for match in all_matches[1:]:
            alt_path = temp_dir / match.image_name
            if alt_path.exists():
                best = match
                image_path = alt_path
                break
        else:
            return None

    logger.info(
        f"Deterministic match: {best.image_name} "
        f"(angle={np.degrees(best.incidence_angle):.1f}°, "
        f"dist={best.camera_distance:.1f}m)"
    )

    return DefectMatch(
        image_name=best.image_name,
        image_path=image_path,
        method="reprojection",
        pixel_x=best.pixel_x,
        pixel_y=best.pixel_y,
    )
```

---

## Testing Plan

### Unit Tests
1. **Surface normal computation**: Verify flat surface returns [0,0,1]
2. **Incidence angle**: Verify 0° for camera directly above, 45° for 45° tilt
3. **Selection key ordering**: Verify lower angle beats closer distance
4. **Tie-breaker**: Verify same criteria → lexicographic filename wins

### Integration Tests
1. Run on existing project, verify same defect → same raw image every time
2. Compare pixel coordinates across runs (should be identical)
3. Verify hotspot is within 5px of detected location

### Validation
1. Visual inspection: projected point should land on the defect
2. Temperature check: projected point should be hot (but not used for selection)

---

## Files to Modify

| File | Changes |
|------|---------|
| `camera_projector.py` | Add incidence angle, camera distance, surface normal, update sorting |
| `gps_matcher.py` | Remove temperature-based selection, use deterministic match |

## Estimated Effort
- Implementation: ~2 hours
- Testing: ~1 hour
- Total: ~3 hours

## Risks
- DSM gradient at panel edges may be noisy → mitigate by clamping normal to reasonable range
- Very oblique angles may still produce poor matches → incidence angle filter (reject >60°)
