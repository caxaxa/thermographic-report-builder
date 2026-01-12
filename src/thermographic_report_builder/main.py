"""Main entrypoint for AWS Batch job - orchestrates the entire report generation pipeline."""

import sys
import time
from pathlib import Path

from .config import settings
from .io import S3Client, load_defect_labels
from .io.image_loader import load_orthophoto
from .processing import DefectMapper, annotate_orthophoto, create_layer_image, crop_defect_regions, GPSMatcher, create_dxf_layers, CropTransform
from .processing.camera_projector import CameraProjector
from .report import ReportBuilder, export_metrics_json, export_metrics_csv, export_panel_grid_json
from .models.report import ReportConfig
from .models.job import JobOutput
from .utils import setup_logging, get_logger, PixelToLatLonConverter, reverse_geocode, get_orthophoto_center
from .utils.exceptions import ProcessingError
from .utils.thermal_alignment import log_alignment_config

logger = get_logger(__name__)


def _compile_only_mode(work_dir: Path, images_dir: Path, start_time: float) -> int:
    """
    Run in compile-only mode: download tex_bundle and recompile PDF.

    This mode is used when the user has manually edited the TeX file and wants
    to regenerate the PDF without re-running the full 11-step pipeline.

    Args:
        work_dir: Working directory for temporary files
        images_dir: Directory for report images
        start_time: Pipeline start time for duration logging

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import subprocess

    s3_client = S3Client()

    # Download tex_bundle from S3
    logger.info("Downloading TeX bundle from S3...")
    if not s3_client.download_tex_bundle(work_dir):
        logger.error("Failed to download TeX bundle - cannot recompile")
        return 1

    tex_path = work_dir / "report.tex"
    if not tex_path.exists():
        logger.error(f"report.tex not found in downloaded bundle at {tex_path}")
        return 1

    logger.info(f"Found report.tex ({tex_path.stat().st_size / 1024:.1f} KB)")

    # Count images
    if images_dir.exists():
        image_count = len(list(images_dir.iterdir()))
        logger.info(f"Found {image_count} images in bundle")

    # Compile PDF (same logic as main pipeline STEP 10)
    logger.info("=" * 80)
    logger.info("Compiling LaTeX to PDF")
    logger.info("=" * 80)

    pdf_full_path = work_dir / "report.pdf"
    try:
        # First pass
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(work_dir), str(tex_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"First pdflatex pass had warnings (exit code {result.returncode})")
            if result.stdout:
                logger.error(f"pdflatex stdout (last 50 lines): {chr(10).join(result.stdout.splitlines()[-50:])}")

        # Second pass (for cross-references)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(work_dir), str(tex_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"Second pdflatex pass had warnings (exit code {result.returncode})")

        if pdf_full_path.exists():
            pdf_size_mb = pdf_full_path.stat().st_size / 1024 / 1024
            logger.info(f"Generated full-resolution PDF: {pdf_size_mb:.1f} MB")
        else:
            logger.error("PDF generation failed - output file not found")
            return 1

    except subprocess.TimeoutExpired:
        logger.error("PDF compilation timed out after 5 minutes")
        return 1
    except Exception as e:
        logger.error(f"PDF compilation failed: {e}")
        return 1

    # Generate low-res PDF using Ghostscript
    pdf_lowres_path = work_dir / "report-lowres.pdf"
    try:
        result = subprocess.run([
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/ebook", "-dNOPAUSE", "-dQUIET", "-dBATCH",
            f"-sOutputFile={pdf_lowres_path}", str(pdf_full_path)
        ], capture_output=True, text=True, timeout=120)

        if pdf_lowres_path.exists():
            lowres_size_mb = pdf_lowres_path.stat().st_size / 1024 / 1024
            logger.info(f"Generated low-res PDF: {lowres_size_mb:.1f} MB")
        else:
            logger.warning("Low-res PDF generation failed, will upload full PDF only")
            pdf_lowres_path = None
    except Exception as e:
        logger.warning(f"Low-res PDF generation failed: {e}")
        pdf_lowres_path = None

    # Upload PDFs to S3
    logger.info("=" * 80)
    logger.info("Uploading recompiled PDFs to S3")
    logger.info("=" * 80)

    pdf_full_s3 = s3_client.upload_report(pdf_full_path, "report-full.pdf")
    if pdf_lowres_path and pdf_lowres_path.exists():
        pdf_lowres_s3 = s3_client.upload_report(pdf_lowres_path, "report-lowres.pdf")
    else:
        pdf_lowres_s3 = pdf_full_s3

    # Delete the .edited marker since we've successfully recompiled
    s3_client.delete_tex_edited_marker()

    # Success!
    duration = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"TeX recompilation completed successfully in {duration:.1f}s")
    logger.info(f"Full PDF: {pdf_full_s3}")
    logger.info(f"Low-res PDF: {pdf_lowres_s3}")
    logger.info("=" * 80)

    return 0


def main() -> int:
    """
    Main entrypoint for AWS Batch job.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    setup_logging(settings.log_level, json_format=settings.log_json)
    logger.info("=" * 80)
    logger.info(f"Starting Thermographic Report Builder v0.1.0")
    logger.info(f"Project: {settings.project_id}, User: {settings.user_id}")
    log_alignment_config()
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Ensure work directory exists
        work_dir = settings.work_dir
        images_dir = work_dir / "report_images"
        raw_images_dir = work_dir / "raw_images"
        images_dir.mkdir(exist_ok=True)
        raw_images_dir.mkdir(exist_ok=True)

        # ===== COMPILE_ONLY MODE =====
        # Skip full pipeline and only recompile existing TeX bundle
        if settings.compile_only:
            logger.info("=" * 80)
            logger.info("COMPILE_ONLY MODE: Recompiling existing TeX bundle")
            logger.info("=" * 80)
            exit_code = _compile_only_mode(work_dir, images_dir, start_time)
            return exit_code

        # Copy static assets (logo) to images directory
        import shutil
        assets_dir = Path(__file__).parent / "assets"
        logo_src = assets_dir / "aisol_logo.png"
        if logo_src.exists():
            shutil.copy(logo_src, images_dir / "aisol_logo.png")
            logger.info(f"Copied logo to {images_dir / 'aisol_logo.png'}")

        # ===== STEP 1: Download inputs from S3 =====
        logger.info("=" * 80)
        logger.info("STEP 1: Downloading inputs from S3")
        logger.info("=" * 80)

        s3_client = S3Client()

        # Download orthophoto - prefer resampled (1.6cm) which is ALREADY cropped/rotated
        # by the inference pipeline. This matches the defect label coordinate space.
        visualization_ortho_path, ortho_version = s3_client.download_orthophoto_resampled(
            work_dir / "odm_orthophoto.tif",
            prefer_resampled=True
        )
        logger.info(f"Using orthophoto version: {ortho_version}")

        labels_path = s3_client.download_defect_labels(work_dir / "defect_labels.json")

        # Download crop annotation for layout configuration and coordinate transform
        is_horizontal = False
        is_double_tracker = False
        crop_data = None
        crop_metadata = None

        crop_annotation_path = s3_client.download_crop_annotation(work_dir / "crop_annotation.json")
        crop_metadata_path = s3_client.download_crop_annotation_metadata(work_dir / "crop_annotation_metadata.json")

        if crop_annotation_path:
            import json
            try:
                with open(crop_annotation_path, 'r') as f:
                    crop_data = json.load(f)
                # Check for isHorizontal first (new format), fallback to isVertical (old format)
                # Note: Old frontend confusingly used isVertical=true to mean horizontal layout
                # We fixed the frontend to use isHorizontal, but support both for backwards compat
                if 'isHorizontal' in crop_data:
                    is_horizontal = crop_data.get('isHorizontal', False)
                elif 'isVertical' in crop_data:
                    # Old format: isVertical=true meant horizontal layout (confusing but true)
                    is_horizontal = crop_data.get('isVertical', False)
                is_double_tracker = crop_data.get('isDouble', False)
                logger.info(f"Crop annotation: isHorizontal={is_horizontal}, isDouble={is_double_tracker}")
            except Exception as e:
                logger.warning(f"Failed to parse crop annotation: {e}")
                crop_data = None

        if crop_metadata_path:
            import json
            try:
                with open(crop_metadata_path, 'r') as f:
                    crop_metadata = json.load(f)
                logger.info(f"Crop metadata: preview={crop_metadata.get('width')}x{crop_metadata.get('height')}")
            except Exception as e:
                logger.warning(f"Failed to parse crop metadata: {e}")
                crop_metadata = None

        # ===== STEP 2: Load and parse data =====
        logger.info("=" * 80)
        logger.info("STEP 2: Loading orthophoto and defect labels")
        logger.info("=" * 80)

        # Load defect labels
        defect_labels = load_defect_labels(labels_path)

        # The resampled orthophoto (odm_orthophoto_1.6cm.tif) is ALREADY cropped/rotated
        # by the inference pipeline, so it matches defect label coordinates directly
        _, viz_transform, viz_crs, (img_h, img_w) = load_orthophoto(visualization_ortho_path)
        logger.info(f"Visualization orthophoto: {img_w}x{img_h} pixels (already cropped/rotated)")
        logger.info(f"Labels: {len(defect_labels.get_defects())} defects, {len(defect_labels.get_panels())} panels")

        # Download ORIGINAL uncropped orthophoto for mesh backtracking
        # The mesh and geo-transform are in the uncropped coordinate space
        uncropped_ortho_path = work_dir / "odm_orthophoto_original.tif"
        has_original_ortho = False
        try:
            s3_client.download_orthophoto(uncropped_ortho_path)
            has_original_ortho = True
            logger.info("Downloaded original uncropped orthophoto for mesh backtracking")
        except Exception as e:
            logger.warning(f"Failed to download original orthophoto: {e}")
            uncropped_ortho_path = visualization_ortho_path  # Fall back

        # Get uncropped dimensions and geo-transform for mesh backtracking
        if has_original_ortho:
            _, uncropped_transform, crs, (uncropped_h, uncropped_w) = load_orthophoto(uncropped_ortho_path)
            logger.info(f"Original orthophoto: {uncropped_w}x{uncropped_h} pixels (uncropped)")
        else:
            # Fall back to visualization ortho dimensions (no crop transform needed)
            uncropped_transform, crs = viz_transform, viz_crs
            uncropped_h, uncropped_w = img_h, img_w
            logger.info("Using visualization ortho as fallback (no mesh backtracking available)")

        geo_converter = PixelToLatLonConverter(uncropped_transform, crs)

        # Create crop transform to map coordinates from resampled/cropped/rotated space (defect labels)
        # to uncropped space (mesh backtracking). Only needed if we have the original ortho.
        if has_original_ortho and crop_data and crop_metadata:
            crop_transform = CropTransform(
                crop_annotation=crop_data,
                crop_metadata=crop_metadata,
                ortho_width=uncropped_w,
                ortho_height=uncropped_h,
                resampled_width=img_w,
                resampled_height=img_h,
            )
        else:
            # No transform needed - either no original ortho or no crop annotation
            crop_transform = CropTransform()
            if not has_original_ortho:
                logger.info("Crop transform disabled (no original orthophoto)")
            elif not crop_data:
                logger.info("Crop transform disabled (no crop annotation)")

        # ===== STEP 3: Map defects to panels =====
        logger.info("=" * 80)
        logger.info("STEP 3: Mapping defects to panel grid")
        logger.info("=" * 80)

        mapper = DefectMapper(
            img_w, img_h, geo_converter,
            is_horizontal=is_horizontal,
            is_double_tracker=is_double_tracker
        )
        panel_grid = mapper.map_defects_to_panels(
            panel_boxes=defect_labels.get_panels(), defect_boxes=defect_labels.get_defects()
        )

        panels_with_defects = sum(1 for p in panel_grid.values() if p.has_defects)
        total_defects = sum(p.defect_count for p in panel_grid.values())
        logger.info(f"Result: {len(panel_grid)} panels, {panels_with_defects} with defects, {total_defects} total defects")

        # ===== STEP 4: Generate annotated overview image =====
        logger.info("=" * 80)
        logger.info("STEP 4: Generating annotated orthophoto")
        logger.info("=" * 80)

        annotate_orthophoto(
            ortho_path=visualization_ortho_path,
            panel_grid=panel_grid,
            output_path=images_dir / "ortho.png",
            scale_factor=settings.orthophoto_downscale_factor,
        )

        # ===== STEP 5: Create layer image (vector PDF) =====
        logger.info("=" * 80)
        logger.info("STEP 5: Creating layer image")
        logger.info("=" * 80)

        create_layer_image(
            panel_grid=panel_grid,
            img_width=img_w,
            img_height=img_h,
            output_path=images_dir / "layer_img.pdf",
        )

        # ===== STEP 5.5: Create DXF layers (georeferenced CAD file) =====
        logger.info("=" * 80)
        logger.info("STEP 5.5: Creating DXF layers (CAD file)")
        logger.info("=" * 80)

        # DXF needs geo_converter matching the panel_grid coordinate space (visualization ortho)
        viz_geo_converter = PixelToLatLonConverter(viz_transform, viz_crs)

        dxf_path = work_dir / "layers.dxf"
        try:
            create_dxf_layers(
                panel_grid=panel_grid,
                geo_converter=viz_geo_converter,
                output_path=dxf_path,
                area_name=settings.area_name,
            )
            logger.info(f"Created DXF file: {dxf_path}")
        except Exception as e:
            logger.warning(f"Failed to create DXF file (continuing without it): {e}")
            dxf_path = None

        # ===== STEP 6: Crop defect regions =====
        logger.info("=" * 80)
        logger.info("STEP 6: Cropping defect regions")
        logger.info("=" * 80)

        crop_defect_regions(
            ortho_path=visualization_ortho_path,
            panel_grid=panel_grid,
            output_dir=images_dir,
            layer_pdf_path=images_dir / "layer_img.pdf",
            scale_factor=settings.crop_downscale_factor,
        )

        # ===== STEP 7: Match raw thermal images (reprojection or GPS fallback) =====
        logger.info("=" * 80)
        logger.info("STEP 7: Matching raw thermal images to defects")
        logger.info("=" * 80)

        # Try to download reconstruction.json for camera reprojection
        # This enables precise pixel-to-pixel mapping and temperature extraction
        reconstruction_data = None
        reconstruction_path = work_dir / "reconstruction.json"
        if s3_client.download_reconstruction_json(reconstruction_path):
            import json
            try:
                with open(reconstruction_path, 'r') as f:
                    reconstruction_data = json.load(f)
                logger.info("Loaded reconstruction.json for camera reprojection")
            except Exception as e:
                logger.warning(f"Failed to parse reconstruction.json: {e}")
                reconstruction_data = None

        # Try to download OpenSfM tracks.csv for feature-based calibration
        tracks_path = work_dir / "tracks.csv"
        tracks_path = s3_client.download_tracks_csv(tracks_path)

        # Try to download DSM for precise elevation
        # Using DSM elevation instead of estimated ground level significantly
        # improves projection accuracy - it's the same elevation ODM used
        dsm_path = work_dir / "dsm.tif"
        if not s3_client.download_dsm(dsm_path):
            dsm_path = None

        # Initialize camera projector (used for both reprojection + mesh backtracking)
        camera_projector = CameraProjector(
            reconstruction_json=reconstruction_data,
            geo_converter=geo_converter,
            dsm_path=dsm_path,
        )

        # Try to download deterministic backtracking artifacts
        mesh_path = work_dir / "odm_textured_model_geo.obj"
        labeling_path = work_dir / "odm_textured_model_geo_labeling.vec"
        nvm_path = work_dir / "reconstruction.nvm"

        mesh_path = s3_client.download_textured_mesh(mesh_path)
        labeling_path = s3_client.download_labeling_vec(labeling_path)
        nvm_path = s3_client.download_reconstruction_nvm(nvm_path)

        mesh_backtracker = None
        # Mesh backtracker can work WITHOUT labeling_path and nvm_path
        # In that case, it uses "all cameras" mode which tries all cameras and picks the best one
        if mesh_path:
            from .processing.mesh_backtracker import MeshBacktracker
            # Use uncropped ortho size for mesh backtracker (mesh is aligned to uncropped geo-transform)
            mesh_backtracker = MeshBacktracker(
                mesh_path=mesh_path,
                labeling_path=labeling_path,  # Optional - if None, uses all-cameras mode
                nvm_path=nvm_path,            # Optional - if None, uses all-cameras mode
                geo_converter=geo_converter,
                camera_projector=camera_projector,
                ortho_size=(uncropped_w, uncropped_h),
            )

        # Try to download source-map for authoritative ortho->raw pixel mapping
        # The source-map is generated by ODM and contains the exact pixel correspondence
        # used during orthophoto creation - more reliable than camera projection
        source_map_path = work_dir / "odm_orthophoto_sources.tif"
        source_map_backtracker = None
        if s3_client.download_source_map(source_map_path):
            from .processing.source_map_backtracker import SourceMapBacktracker
            source_map_backtracker = SourceMapBacktracker(
                source_map_path=source_map_path,
                nvm_path=nvm_path,  # For view_id -> image_name mapping
            )
            if source_map_backtracker.available:
                stats = source_map_backtracker.get_coverage_stats()
                logger.info(f"Source-map loaded: {stats['coverage_percent']}% coverage")
            else:
                source_map_backtracker = None
                logger.warning("Source-map failed to load, using camera projection fallback")

        gps_matcher = GPSMatcher(
            s3_client=s3_client,
            geo_converter=geo_converter,
            reconstruction_data=reconstruction_data,
            dsm_path=dsm_path,
            camera_projector=camera_projector,
            mesh_backtracker=mesh_backtracker,
            source_map_backtracker=source_map_backtracker,
            crop_transform=crop_transform,
            enable_temperature=True,
            tracks_path=tracks_path,
        )
        matched_count, defect_matches = gps_matcher.match_images_to_panels(
            panel_grid=panel_grid, temp_dir=raw_images_dir, output_dir=images_dir
        )
        logger.info(f"Matched {matched_count} raw images")

        # ===== STEP 7.1: Export annotation manifest =====
        logger.info("=" * 80)
        logger.info("STEP 7.1: Exporting annotation manifest")
        logger.info("=" * 80)

        manifest = gps_matcher.export_annotation_manifest()
        manifest_path = work_dir / "annotation_manifest.json"
        manifest.save(manifest_path)
        logger.info(f"Exported {len(manifest.annotations)} annotations to manifest")

        # ===== STEP 7.2: Check for and apply annotation overrides =====
        logger.info("=" * 80)
        logger.info("STEP 7.2: Checking for annotation overrides")
        logger.info("=" * 80)

        from .models.annotation_manifest import OverrideManifest
        override_path = work_dir / "annotation_overrides.json"
        if s3_client.download_annotation_overrides(override_path):
            try:
                overrides = OverrideManifest.load(override_path)
                logger.info(f"Found {len(overrides.overrides)} overrides, re-rendering...")
                re_rendered = gps_matcher.apply_overrides(
                    overrides=overrides,
                    raw_images_dir=raw_images_dir,
                    output_dir=images_dir,
                )
                logger.info(f"Re-rendered {re_rendered} annotations with overrides")
            except Exception as e:
                logger.warning(f"Failed to apply annotation overrides: {e}")
        else:
            logger.info("No annotation overrides found, using automatic annotations")

        # ===== STEP 7.5: Download ODM statistics (optional) =====
        logger.info("=" * 80)
        logger.info("STEP 7.5: Downloading ODM statistics (optional)")
        logger.info("=" * 80)

        odm_stats_dir = work_dir / "odm_stats"
        odm_stats = None
        try:
            s3_client.download_odm_stats(odm_stats_dir)

            # Try to load stats.json if it exists
            stats_json_path = odm_stats_dir / "stats.json"
            if stats_json_path.exists():
                import json
                with open(stats_json_path, 'r') as f:
                    odm_stats = json.load(f)
                logger.info("ODM stats loaded successfully")
            else:
                logger.warning("stats.json not found in ODM stats directory")
        except Exception as e:
            logger.warning(f"Could not download ODM stats (will continue without appendix): {e}")
            odm_stats = None

        # ===== STEP 8: Generate LaTeX report =====
        logger.info("=" * 80)
        logger.info("STEP 8: Generating LaTeX report")
        logger.info("=" * 80)

        # Derive location from orthophoto center if not explicitly set
        report_location = settings.location
        if not report_location:
            try:
                center_lat, center_lon = get_orthophoto_center(viz_transform, img_w, img_h, viz_crs)
                logger.info(f"Orthophoto center: ({center_lat:.6f}, {center_lon:.6f})")
                report_location = reverse_geocode(center_lat, center_lon)
            except Exception as e:
                logger.warning(f"Could not derive location from orthophoto: {e}")
                report_location = ""

        report_config = ReportConfig(
            area_name=settings.area_name,
            client_name=settings.client_name,
            engineer_name=settings.engineer_name,
            crea_number=settings.crea_number,
            location=report_location,
            address=settings.address,
        )

        builder = ReportBuilder(
            panel_grid=panel_grid,
            images_dir=images_dir,
            config=report_config,
            odm_stats=odm_stats,
            odm_stats_dir=odm_stats_dir,
            defect_matches=defect_matches,
            annotation_manifest=manifest,
        )

        tex_path = work_dir / "report.tex"
        builder.generate_tex(tex_path)

        # ===== STEP 9: Export metrics =====
        logger.info("=" * 80)
        logger.info("STEP 9: Exporting metrics")
        logger.info("=" * 80)

        metrics_json_path = export_metrics_json(
            panel_grid,
            work_dir / "metrics.json",
            ortho_width=img_w,
            ortho_height=img_h,
            ortho_scale_factor=settings.orthophoto_downscale_factor,
        )
        metrics_csv_path = export_metrics_csv(panel_grid, work_dir / "metrics.csv")

        # Export panel_grid.json for interactive defect map (with report-aligned panel IDs)
        # Pass images_dir so it can check for annotated thermal images
        # Pass annotation_manifest so it can include temperature data
        panel_grid_json_path = export_panel_grid_json(
            panel_grid,
            work_dir / "panel_grid.json",
            ortho_width=img_w,
            ortho_height=img_h,
            ortho_scale_factor=settings.orthophoto_downscale_factor,
            images_dir=images_dir,
            annotation_manifest=manifest,
        )

        # ===== STEP 10: Compile LaTeX to PDF =====
        logger.info("=" * 80)
        logger.info("STEP 10: Compiling LaTeX to PDF")
        logger.info("=" * 80)

        import subprocess
        import shutil

        # Compile PDF (run pdflatex twice for references)
        pdf_full_path = work_dir / "report.pdf"
        try:
            # First pass
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(work_dir), str(tex_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding='latin-1',
                errors='replace',
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"First pdflatex pass had warnings (exit code {result.returncode})")
                # Log last lines of output to see the error
                if result.stderr:
                    logger.error(f"pdflatex stderr: {result.stderr[-2000:]}")
                if result.stdout:
                    logger.error(f"pdflatex stdout (last 50 lines): {chr(10).join(result.stdout.splitlines()[-50:])}")

            # Second pass (for cross-references)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(work_dir), str(tex_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding='latin-1',
                errors='replace',
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"Second pdflatex pass had warnings (exit code {result.returncode})")
                if result.stderr:
                    logger.error(f"pdflatex stderr: {result.stderr[-2000:]}")
                if result.stdout:
                    logger.error(f"pdflatex stdout (last 50 lines): {chr(10).join(result.stdout.splitlines()[-50:])}")

            if pdf_full_path.exists():
                pdf_size_mb = pdf_full_path.stat().st_size / 1024 / 1024
                logger.info(f"✓ Generated full-resolution PDF: {pdf_size_mb:.1f} MB")
            else:
                raise ProcessingError("PDF generation failed - output file not found")

        except subprocess.TimeoutExpired:
            raise ProcessingError("PDF compilation timed out after 5 minutes")
        except Exception as e:
            raise ProcessingError(f"PDF compilation failed: {e}")

        # Generate low-res PDF using Ghostscript
        pdf_lowres_path = work_dir / "report-lowres.pdf"
        try:
            result = subprocess.run([
                "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/ebook", "-dNOPAUSE", "-dQUIET", "-dBATCH",
                f"-sOutputFile={pdf_lowres_path}", str(pdf_full_path)
            ], capture_output=True, text=True, timeout=120)

            if pdf_lowres_path.exists():
                lowres_size_mb = pdf_lowres_path.stat().st_size / 1024 / 1024
                logger.info(f"✓ Generated low-res PDF: {lowres_size_mb:.1f} MB")
            else:
                logger.warning("Low-res PDF generation failed, will upload full PDF only")
                pdf_lowres_path = None
        except Exception as e:
            logger.warning(f"Low-res PDF generation failed: {e}")
            pdf_lowres_path = None

        # ===== STEP 11: Upload LaTeX bundle, PDFs, and metrics to S3 =====
        logger.info("=" * 80)
        logger.info("STEP 11: Uploading reports and metrics to S3")
        logger.info("=" * 80)

        # Upload the complete LaTeX bundle (tex + images) for archival
        tex_bundle_s3 = s3_client.upload_tex_bundle(work_dir)

        # Upload PDFs
        pdf_full_s3 = s3_client.upload_report(pdf_full_path, "report-full.pdf")
        if pdf_lowres_path and pdf_lowres_path.exists():
            pdf_lowres_s3 = s3_client.upload_report(pdf_lowres_path, "report-lowres.pdf")
        else:
            pdf_lowres_s3 = pdf_full_s3  # Fallback to full PDF

        # Upload metrics
        metrics_json_s3 = s3_client.upload_report(metrics_json_path, "metrics.json")
        metrics_csv_s3 = s3_client.upload_report(metrics_csv_path, "metrics.csv")

        # Upload panel_grid.json for interactive defect map
        panel_grid_s3 = s3_client.upload_report(panel_grid_json_path, "panel_grid.json")
        logger.info(f"Uploaded panel_grid.json for defect map: {panel_grid_s3}")

        # Upload ortho.png for interactive defect map
        ortho_png_path = images_dir / "ortho.png"
        if ortho_png_path.exists():
            ortho_png_s3 = s3_client.upload_report(ortho_png_path, "ortho.png")
            logger.info(f"Uploaded ortho.png for defect map: {ortho_png_s3}")

        # Upload DXF layers file (if generated)
        dxf_s3 = None
        if dxf_path and dxf_path.exists():
            dxf_s3 = s3_client.upload_report(dxf_path, "layers.dxf")
            logger.info(f"Uploaded DXF layers: {dxf_s3}")

        # Upload annotation manifest for human-in-the-loop review
        if manifest_path.exists():
            manifest_s3 = s3_client.upload_report(manifest_path, "annotation_manifest.json")
            logger.info(f"Uploaded annotation manifest: {manifest_s3}")

        # ===== Success! =====
        duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ Thermographic report generated successfully in {duration:.1f}s")
        logger.info(f"Full PDF: {pdf_full_s3}")
        logger.info(f"Low-res PDF: {pdf_lowres_s3}")
        logger.info("=" * 80)

        # Create job output summary
        job_output = JobOutput(
            report_full_pdf_s3=pdf_full_s3,
            report_lowres_pdf_s3=pdf_lowres_s3,
            metrics_json_s3=metrics_json_s3,
            metrics_csv_s3=metrics_csv_s3,
            layers_dxf_s3=dxf_s3,
            total_panels=len(panel_grid),
            panels_with_defects=panels_with_defects,
            total_defects=total_defects,
            processing_duration_seconds=duration,
        )

        logger.info(f"Job output: {job_output.to_dict()}")
        return 0

    except ProcessingError as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
