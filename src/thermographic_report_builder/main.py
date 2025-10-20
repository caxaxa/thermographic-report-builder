"""Main entrypoint for AWS Batch job - orchestrates the entire report generation pipeline."""

import sys
import time
from pathlib import Path

from .config import settings
from .io import S3Client, load_defect_labels
from .io.image_loader import load_orthophoto
from .processing import DefectMapper, annotate_orthophoto, create_layer_image, crop_defect_regions, GPSMatcher
from .report import ReportBuilder, export_metrics_json, export_metrics_csv
from .models.report import ReportConfig
from .models.job import JobOutput
from .utils import setup_logging, get_logger
from .utils.exceptions import ProcessingError

logger = get_logger(__name__)


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
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Ensure work directory exists
        work_dir = settings.work_dir
        images_dir = work_dir / "report_images"
        raw_images_dir = work_dir / "raw_images"
        images_dir.mkdir(exist_ok=True)
        raw_images_dir.mkdir(exist_ok=True)

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

        ortho_path = s3_client.download_orthophoto(work_dir / "odm_orthophoto.tif")
        labels_path = s3_client.download_defect_labels(work_dir / "defect_labels.json")

        # ===== STEP 2: Load and parse data =====
        logger.info("=" * 80)
        logger.info("STEP 2: Loading orthophoto and defect labels")
        logger.info("=" * 80)

        ortho_img, transform, (img_h, img_w) = load_orthophoto(ortho_path)
        defect_labels = load_defect_labels(labels_path)

        logger.info(f"Orthophoto: {img_w}x{img_h} pixels")
        logger.info(f"Labels: {len(defect_labels.get_defects())} defects, {len(defect_labels.get_panels())} panels")

        # ===== STEP 3: Map defects to panels =====
        logger.info("=" * 80)
        logger.info("STEP 3: Mapping defects to panel grid")
        logger.info("=" * 80)

        mapper = DefectMapper(img_w, img_h, transform)
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
            ortho_path=ortho_path,
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

        # ===== STEP 6: Crop defect regions =====
        logger.info("=" * 80)
        logger.info("STEP 6: Cropping defect regions")
        logger.info("=" * 80)

        crop_defect_regions(
            ortho_path=ortho_path,
            panel_grid=panel_grid,
            output_dir=images_dir,
            scale_factor=settings.crop_downscale_factor,
        )

        # ===== STEP 7: Match raw thermal images via GPS =====
        logger.info("=" * 80)
        logger.info("STEP 7: Matching raw thermal images via GPS")
        logger.info("=" * 80)

        gps_matcher = GPSMatcher(s3_client)
        matched_count = gps_matcher.match_images_to_panels(
            panel_grid=panel_grid, temp_dir=raw_images_dir, output_dir=images_dir
        )
        logger.info(f"Matched {matched_count} raw images")

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

        report_config = ReportConfig(
            area_name=settings.area_name,
            client_name=settings.client_name,
            engineer_name=settings.engineer_name,
            crea_number=settings.crea_number,
            location=settings.location,
            address=settings.address,
        )

        builder = ReportBuilder(
            panel_grid=panel_grid,
            images_dir=images_dir,
            config=report_config,
            odm_stats=odm_stats,
            odm_stats_dir=odm_stats_dir,
        )

        tex_path = work_dir / "report.tex"
        builder.generate_tex(tex_path)

        # ===== STEP 9: Export metrics =====
        logger.info("=" * 80)
        logger.info("STEP 9: Exporting metrics")
        logger.info("=" * 80)

        metrics_json_path = export_metrics_json(panel_grid, work_dir / "metrics.json")
        metrics_csv_path = export_metrics_csv(panel_grid, work_dir / "metrics.csv")

        # ===== STEP 10: Upload LaTeX bundle and metrics to S3 =====
        logger.info("=" * 80)
        logger.info("STEP 10: Uploading LaTeX bundle and metrics to S3")
        logger.info("=" * 80)

        # Upload the complete LaTeX bundle (tex + images) for compilation
        tex_bundle_s3 = s3_client.upload_tex_bundle(work_dir)

        # Upload metrics
        metrics_json_s3 = s3_client.upload_report(metrics_json_path, "metrics.json")
        metrics_csv_s3 = s3_client.upload_report(metrics_csv_path, "metrics.csv")

        # ===== Success! =====
        duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ Report data processing completed successfully in {duration:.1f}s")
        logger.info(f"LaTeX bundle ready at: {tex_bundle_s3}")
        logger.info("=" * 80)

        # Create job output summary
        job_output = JobOutput(
            report_full_pdf_s3=tex_bundle_s3,  # Reusing field to store tex bundle location
            report_lowres_pdf_s3=tex_bundle_s3,
            metrics_json_s3=metrics_json_s3,
            metrics_csv_s3=metrics_csv_s3,
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
