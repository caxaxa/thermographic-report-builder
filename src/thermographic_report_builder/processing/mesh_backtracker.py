"""Deterministic backtracking from orthophoto pixel to raw image pixel.

Uses:
1) Textured mesh (local ENU coordinates)
2) labeling.vec (face -> view index)
3) reconstruction.nvm (view index -> image name)
4) reconstruction.json (camera intrinsics for projection)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from ..utils.logger import get_logger
from ..utils.geospatial import PixelToLatLonConverter
from .camera_projector import CameraProjector

logger = get_logger(__name__)


@dataclass
class BacktrackResult:
    image_name: Optional[str]
    pixel_x: Optional[float]
    pixel_y: Optional[float]
    face_id: int
    view_index: Optional[int]
    world_x: float
    world_y: float
    world_z: float


class MeshBacktracker:
    """Backtrack ortho pixel to raw pixel using mesh + labeling + NVM."""

    def __init__(
        self,
        mesh_path: Path,
        labeling_path: Optional[Path],
        nvm_path: Optional[Path],
        geo_converter: PixelToLatLonConverter,
        camera_projector: CameraProjector,
        ortho_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.geo_converter = geo_converter
        self.camera_projector = camera_projector
        self.available = False
        self.labels = None
        self.view_names = []
        self._labeling_available = False

        try:
            self.mesh = self._load_mesh(mesh_path)
        except Exception as exc:
            logger.warning(f"Failed to load mesh: {exc}")
            return

        if self.mesh.faces is None or len(self.mesh.faces) == 0:
            logger.warning("Mesh has no faces; backtracking disabled")
            return

        # Try to load labeling.vec (optional - only needed for labeling-based backtracking)
        if labeling_path and labeling_path.exists():
            try:
                self.labels = self._load_labeling(labeling_path, len(self.mesh.faces))
                self._labeling_available = True
            except Exception as exc:
                logger.info(f"Labeling vec not usable (will use all-cameras mode): {exc}")

        # Try to load NVM views (optional - only needed for labeling-based backtracking)
        if nvm_path and nvm_path.exists():
            try:
                self.view_names = self._load_nvm_views(nvm_path)
            except Exception as exc:
                logger.info(f"NVM views not loaded: {exc}")

        if not self.camera_projector.available:
            logger.warning("Camera projector unavailable; backtracking disabled")
            return

        if not self._validate_alignment(ortho_size):
            logger.warning("Mesh/orthophoto alignment check failed; backtracking disabled")
            return

        self.available = True

        if self._labeling_available and self.view_names:
            logger.info(
                f"Mesh backtracker ready: faces={len(self.mesh.faces)}, views={len(self.view_names)}, "
                f"labeling available (both modes supported)"
            )
        else:
            logger.info(
                f"Mesh backtracker ready: faces={len(self.mesh.faces)}, "
                f"all-cameras mode only (labeling unavailable)"
            )

    def backtrack(self, ortho_x: float, ortho_y: float, use_all_cameras: bool = True) -> Optional[BacktrackResult]:
        """Return a raycast hit with optional view/pixel for an orthophoto location.

        Args:
            ortho_x: X coordinate in orthophoto (pixels)
            ortho_y: Y coordinate in orthophoto (pixels)
            use_all_cameras: If True, skip labeling.vec and try all cameras to find the best match.
                            This is more reliable as labeling.vec indices may not match camera order.
                            If False, use the original labeling.vec-based approach.
        """
        if not self.available:
            return None

        # Convert ortho pixel to local ENU XY (same frame as mesh)
        local_x, local_y = self.camera_projector.ortho_to_local(ortho_x, ortho_y)

        # Cast a vertical ray down onto the mesh
        z_max = float(self.mesh.bounds[1][2])
        origin = np.array([[local_x, local_y, z_max + 10.0]])
        direction = np.array([[0.0, 0.0, -1.0]])

        # Log the raycast parameters for debugging
        mesh_bounds = self.mesh.bounds
        logger.debug(
            f"Raycast: ortho=({ortho_x:.1f}, {ortho_y:.1f}) -> local=({local_x:.1f}, {local_y:.1f}), "
            f"mesh_x=[{mesh_bounds[0][0]:.1f}, {mesh_bounds[1][0]:.1f}], "
            f"mesh_y=[{mesh_bounds[0][1]:.1f}, {mesh_bounds[1][1]:.1f}]"
        )

        try:
            locations, _, face_ids = self.mesh.ray.intersects_location(
                origin, direction, multiple_hits=True
            )
            logger.debug(f"Raycast result: {len(locations)} hits from local=({local_x:.1f}, {local_y:.1f})")
        except Exception as exc:
            logger.debug(f"Raycast exception: {exc}")
            return None

        if len(locations) == 0:
            logger.debug(f"Raycast miss: local=({local_x:.1f}, {local_y:.1f}), no intersections found")
            return None

        # Choose the closest intersection (highest Z)
        idx = int(np.argmax(locations[:, 2]))
        hit = locations[idx]
        face_id = int(face_ids[idx])
        logger.debug(f"Raycast hit: face_id={face_id}, hit=({hit[0]:.1f}, {hit[1]:.1f}, {hit[2]:.1f})")

        # If use_all_cameras is True, skip labeling.vec and find best camera directly
        if use_all_cameras:
            return self._backtrack_all_cameras(hit, face_id)

        # Original labeling.vec-based approach (kept for backwards compatibility)
        return self._backtrack_via_labeling(hit, face_id)

    def _backtrack_all_cameras(self, hit: np.ndarray, face_id: int) -> Optional[BacktrackResult]:
        """Find the best camera by trying all cameras with the 3D hit point.

        This bypasses labeling.vec entirely and uses camera_projector to project
        the world point to all cameras, selecting the one where the point
        appears most centrally (best view quality).
        """
        world_point = np.array([hit[0], hit[1], hit[2]])

        # Use camera_projector to find all cameras that can see this point
        matches = []
        for image_name, shot in self.camera_projector.shots.items():
            pixel = self.camera_projector.project_world_to_image(image_name, world_point)
            if pixel is not None:
                px, py = pixel

                # Calculate distance from image center (for ranking)
                img_w = self.camera_projector.image_width
                img_h = self.camera_projector.image_height
                cx, cy = img_w / 2, img_h / 2
                max_dist = np.hypot(cx, cy)
                dist_from_center = np.hypot(px - cx, py - cy) / max_dist

                matches.append({
                    'image_name': image_name,
                    'pixel_x': px,
                    'pixel_y': py,
                    'dist_from_center': dist_from_center,
                })

        if not matches:
            logger.debug(f"No camera sees point at ({hit[0]:.1f}, {hit[1]:.1f}, {hit[2]:.1f})")
            return BacktrackResult(
                image_name=None,
                pixel_x=None,
                pixel_y=None,
                face_id=face_id,
                view_index=None,
                world_x=float(hit[0]),
                world_y=float(hit[1]),
                world_z=float(hit[2]),
            )

        # Pick the best match (closest to center)
        best = min(matches, key=lambda m: m['dist_from_center'])

        logger.info(
            f"Mesh backtrack (all cameras): face={face_id} -> image={best['image_name']} "
            f"-> pixel=({best['pixel_x']:.1f}, {best['pixel_y']:.1f}), "
            f"centrality={1-best['dist_from_center']:.2f}"
        )

        return BacktrackResult(
            image_name=best['image_name'],
            pixel_x=best['pixel_x'],
            pixel_y=best['pixel_y'],
            face_id=face_id,
            view_index=None,  # Not using labeling.vec
            world_x=float(hit[0]),
            world_y=float(hit[1]),
            world_z=float(hit[2]),
        )

    def _backtrack_via_labeling(self, hit: np.ndarray, face_id: int) -> Optional[BacktrackResult]:
        """Original backtracking using labeling.vec face->view mapping.

        Note: This may not work correctly if labeling.vec indices don't match
        the camera order in reconstruction.nvm. Use _backtrack_all_cameras instead.
        """
        if self.labels is None or not self._labeling_available:
            logger.debug("Labeling not available, falling back to all-cameras mode")
            return self._backtrack_all_cameras(hit, face_id)

        if face_id < 0 or face_id >= len(self.labels):
            logger.debug(f"Invalid face_id {face_id} (labels has {len(self.labels)} entries)")
            return None

        view_idx = int(self.labels[face_id])
        view_index: Optional[int] = None
        image_name: Optional[str] = None
        pixel_x: Optional[float] = None
        pixel_y: Optional[float] = None

        if self._is_valid_view_index(view_idx):
            view_index = view_idx
            view_name = self.view_names[view_idx]
            image_name = self._resolve_image_name(view_name)
            if image_name is None:
                logger.debug(f"Could not resolve image name for view '{view_name}'")
            else:
                pixel = self.camera_projector.project_world_to_image(
                    image_name, np.array([hit[0], hit[1], hit[2]])
                )
                if pixel is None:
                    logger.debug(f"Projection to image '{image_name}' failed")
                else:
                    pixel_x, pixel_y = float(pixel[0]), float(pixel[1])
                    logger.info(
                        "Mesh backtrack (labeling): face=%s -> view=%s -> image=%s -> pixel=(%.1f, %.1f)",
                        face_id,
                        view_idx,
                        image_name,
                        pixel_x,
                        pixel_y,
                    )
        else:
            logger.debug(
                f"Invalid view_idx {view_idx} for face_id {face_id} (returning world point only)"
            )

        return BacktrackResult(
            image_name=image_name,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            face_id=face_id,
            view_index=view_index,
            world_x=float(hit[0]),
            world_y=float(hit[1]),
            world_z=float(hit[2]),
        )

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump()
            if not meshes:
                raise ValueError("Empty mesh scene")
            mesh = trimesh.util.concatenate(meshes)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Unsupported mesh type")
        return mesh

    def _load_labeling(self, labeling_path: Path, face_count: int) -> np.ndarray:
        data = labeling_path.read_bytes()
        if face_count <= 0:
            raise ValueError("Mesh has no faces")
        if len(data) == face_count * 8:
            dtype = np.dtype("<u8")
        elif len(data) == face_count * 4:
            dtype = np.dtype("<u4")
        else:
            raise ValueError(
                f"Labeling vec size mismatch (faces={face_count}, bytes={len(data)})"
            )
        labels = np.frombuffer(data, dtype=dtype)
        return labels

    def _load_nvm_views(self, nvm_path: Path) -> List[str]:
        lines = [line.strip() for line in nvm_path.read_text().splitlines() if line.strip()]
        if not lines or not lines[0].startswith("NVM"):
            raise ValueError("Invalid NVM header")
        if len(lines) < 2:
            raise ValueError("Invalid NVM file")
        camera_count = int(lines[1])
        view_names = []
        for i in range(camera_count):
            idx = 2 + i
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if parts:
                view_names.append(parts[0])
        return view_names

    def _resolve_image_name(self, view_name: str) -> Optional[str]:
        """Map NVM view name to reconstruction.json shot name.

        NVM files from ODM have names like 'images/DJI_xxx.JPG.tif'
        while reconstruction.json uses 'DJI_xxx.JPG'.
        """
        # Direct match
        if view_name in self.camera_projector.shots:
            return view_name

        # Try basename (strip directory)
        basename = Path(view_name).name
        if basename in self.camera_projector.shots:
            return basename

        # Try stripping .tif suffix (ODM adds this internally)
        if basename.endswith('.tif'):
            basename_no_tif = basename[:-4]  # Remove '.tif'
            if basename_no_tif in self.camera_projector.shots:
                return basename_no_tif

        return None

    def _is_valid_view_index(self, view_idx: int) -> bool:
        if view_idx < 0:
            return False
        if view_idx >= len(self.view_names):
            return False
        sentinel = int(np.iinfo(self.labels.dtype).max)
        if view_idx == sentinel:
            return False
        return True

    def _validate_alignment(self, ortho_size: Optional[Tuple[int, int]]) -> bool:
        if ortho_size is None:
            return True

        ortho_w, ortho_h = ortho_size
        corners = [
            (0, 0),
            (ortho_w, 0),
            (0, ortho_h),
            (ortho_w, ortho_h),
        ]
        local_pts = [self.camera_projector.ortho_to_local(x, y) for x, y in corners]
        xs = [p[0] for p in local_pts]
        ys = [p[1] for p in local_pts]
        ortho_min_x, ortho_max_x = min(xs), max(xs)
        ortho_min_y, ortho_max_y = min(ys), max(ys)

        mesh_min, mesh_max = self.mesh.bounds
        mesh_min_x, mesh_min_y = mesh_min[0], mesh_min[1]
        mesh_max_x, mesh_max_y = mesh_max[0], mesh_max[1]

        overlap_x = not (ortho_max_x < mesh_min_x or ortho_min_x > mesh_max_x)
        overlap_y = not (ortho_max_y < mesh_min_y or ortho_min_y > mesh_max_y)

        if not (overlap_x and overlap_y):
            logger.warning(
                "Mesh/ortho bounds do not overlap: "
                f"ortho_x=[{ortho_min_x:.1f},{ortho_max_x:.1f}], "
                f"ortho_y=[{ortho_min_y:.1f},{ortho_max_y:.1f}], "
                f"mesh_x=[{mesh_min_x:.1f},{mesh_max_x:.1f}], "
                f"mesh_y=[{mesh_min_y:.1f},{mesh_max_y:.1f}]"
            )
            return False
        return True
