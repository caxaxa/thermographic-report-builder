"""S3 client abstraction for downloading and uploading files."""

import boto3
from pathlib import Path
from typing import BinaryIO
from botocore.exceptions import ClientError

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import S3DownloadError, S3UploadError

logger = get_logger(__name__)


class S3Client:
    """Abstraction for S3 operations."""

    def __init__(self, region: str | None = None):
        """
        Initialize S3 client.

        Args:
            region: AWS region (defaults to settings.aws_region)
        """
        self.region = region or settings.aws_region
        self.s3 = boto3.client("s3", region_name=self.region)
        logger.info(f"Initialized S3 client for region {self.region}")

    def download_orthophoto(self, local_path: Path) -> Path:
        """
        Download orthophoto from S3.

        Args:
            local_path: Local path to save the file

        Returns:
            Path to downloaded file

        Raises:
            S3DownloadError: If download fails
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/odm_orthophoto/odm_orthophoto.tif"
        bucket = settings.orthos_bucket

        logger.info(f"Downloading orthophoto from s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_mb = local_path.stat().st_size / 1_000_000
            logger.info(f"Downloaded orthophoto: {size_mb:.1f} MB")
            return local_path
        except ClientError as e:
            error_msg = f"Failed to download orthophoto from s3://{bucket}/{key}: {e}"
            logger.error(error_msg)
            raise S3DownloadError(error_msg) from e

    def download_orthophoto_resampled(self, local_path: Path, prefer_resampled: bool = True) -> tuple[Path, str]:
        """
        Download orthophoto from S3, preferring resampled version if available.

        This method tries to download the resampled (1.6cm GSD) version first,
        which ensures coordinate consistency with the inference pipeline.
        Falls back to original if resampled version doesn't exist.

        Args:
            local_path: Local path to save the file
            prefer_resampled: If True, try to download resampled version first

        Returns:
            Tuple of (downloaded_path, version_used)
            where version_used is either "resampled_1.6cm" or "original"

        Raises:
            S3DownloadError: If download fails
        """
        bucket = settings.orthos_bucket

        # Try resampled version first if preferred
        if prefer_resampled:
            resampled_key = f"{settings.user_id}/projects/{settings.project_id}/odm_orthophoto/odm_orthophoto_1.6cm.tif"
            logger.info(f"Attempting to download resampled orthophoto from s3://{bucket}/{resampled_key}")

            try:
                self.s3.download_file(Bucket=bucket, Key=resampled_key, Filename=str(local_path))
                size_mb = local_path.stat().st_size / 1_000_000
                logger.info(f"Downloaded resampled orthophoto (1.6cm GSD): {size_mb:.1f} MB")
                return local_path, "resampled_1.6cm"
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'NoSuchKey' or error_code == '404':
                    logger.info("Resampled orthophoto not found, falling back to original")
                else:
                    logger.warning(f"Failed to download resampled orthophoto: {e}")

        # Fall back to original
        logger.info("Downloading original orthophoto")
        return self.download_orthophoto(local_path), "original"

    def download_defect_labels(self, local_path: Path) -> Path:
        """
        Download defect_labels.json from S3.

        Downloads from groundtruth bucket (human-reviewed annotations from annotation tool).
        This ensures the report uses the latest reviewed annotations, not the raw inference output.

        Args:
            local_path: Local path to save the file

        Returns:
            Path to downloaded file

        Raises:
            S3DownloadError: If download fails
        """
        # Use groundtruth bucket - this has the human-reviewed annotations from the annotation tool
        key = f"{settings.user_id}/projects/{settings.project_id}/groundtruth/defect_labels.json"
        bucket = settings.groundtruth_bucket

        logger.info(f"Downloading defect labels from s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_kb = local_path.stat().st_size / 1_000
            logger.info(f"Downloaded defect labels: {size_kb:.1f} KB")
            return local_path
        except ClientError as e:
            error_msg = f"Failed to download defect labels from s3://{bucket}/{key}: {e}"
            logger.error(error_msg)
            raise S3DownloadError(error_msg) from e

    def list_raw_images(self) -> list[str]:
        """
        List all thermal images for this project.

        Returns:
            List of S3 keys for thermal images

        Raises:
            S3DownloadError: If listing fails
        """
        prefix = f"{settings.user_id}/projects/{settings.project_id}/images/"
        bucket = settings.uploads_bucket

        logger.info(f"Listing thermal images from s3://{bucket}/{prefix}")

        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            keys = [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].endswith("_T.JPG")
            ]

            logger.info(f"Found {len(keys)} thermal images in uploads bucket")
            return keys
        except ClientError as e:
            error_msg = f"Failed to list raw images from s3://{bucket}/{prefix}: {e}"
            logger.error(error_msg)
            raise S3DownloadError(error_msg) from e

    def download_raw_image(self, s3_key: str, local_path: Path) -> Path:
        """
        Download a single raw thermal image.

        Args:
            s3_key: S3 key of the image
            local_path: Local path to save the file

        Returns:
            Path to downloaded file

        Raises:
            S3DownloadError: If download fails
        """
        bucket = settings.uploads_bucket

        try:
            self.s3.download_file(Bucket=bucket, Key=s3_key, Filename=str(local_path))
            return local_path
        except ClientError as e:
            error_msg = f"Failed to download raw image s3://{bucket}/{s3_key}: {e}"
            logger.error(error_msg)
            raise S3DownloadError(error_msg) from e

    def upload_report(self, local_path: Path, report_filename: str) -> str:
        """
        Upload generated report to S3.

        Args:
            local_path: Local path of the file to upload
            report_filename: Filename to use in S3 (e.g., 'report-full.pdf')

        Returns:
            S3 URI of uploaded file

        Raises:
            S3UploadError: If upload fails
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/thermographic-report/{report_filename}"
        bucket = settings.reports_bucket

        size_mb = local_path.stat().st_size / 1_000_000
        logger.info(f"Uploading {report_filename} ({size_mb:.1f} MB) to s3://{bucket}/{key}")

        try:
            self.s3.upload_file(Filename=str(local_path), Bucket=bucket, Key=key)
            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded successfully: {s3_uri}")
            return s3_uri
        except ClientError as e:
            error_msg = f"Failed to upload {report_filename} to s3://{bucket}/{key}: {e}"
            logger.error(error_msg)
            raise S3UploadError(error_msg) from e

    def upload_file(self, local_path: Path, s3_key: str, bucket: str | None = None) -> str:
        """
        Upload any file to S3 with custom key.

        Args:
            local_path: Local path of the file to upload
            s3_key: S3 key to use
            bucket: S3 bucket (defaults to reports bucket)

        Returns:
            S3 URI of uploaded file

        Raises:
            S3UploadError: If upload fails
        """
        bucket = bucket or settings.reports_bucket

        try:
            self.s3.upload_file(Filename=str(local_path), Bucket=bucket, Key=s3_key)
            s3_uri = f"s3://{bucket}/{s3_key}"
            logger.info(f"Uploaded file to {s3_uri}")
            return s3_uri
        except ClientError as e:
            error_msg = f"Failed to upload to s3://{bucket}/{s3_key}: {e}"
            logger.error(error_msg)
            raise S3UploadError(error_msg) from e

    def upload_tex_bundle(self, work_dir: Path) -> str:
        """
        Upload LaTeX bundle (tex file + all assets) to S3 for compilation.

        Uploads all files from work_dir to a temporary location in S3
        for the LaTeX compiler job to download and process.

        Args:
            work_dir: Directory containing report.tex and report_images/

        Returns:
            S3 prefix where bundle was uploaded

        Raises:
            S3UploadError: If upload fails
        """
        base_prefix = f"{settings.user_id}/projects/{settings.project_id}/tex_bundle"
        bucket = settings.reports_bucket

        logger.info(f"Uploading LaTeX bundle to s3://{bucket}/{base_prefix}")

        try:
            file_count = 0

            # Upload .tex file
            tex_file = work_dir / "report.tex"
            if tex_file.exists():
                self.upload_file(tex_file, f"{base_prefix}/report.tex", bucket)
                file_count += 1

            # Upload all images
            images_dir = work_dir / "report_images"
            if images_dir.exists():
                for img_file in images_dir.iterdir():
                    if img_file.is_file():
                        self.upload_file(img_file, f"{base_prefix}/report_images/{img_file.name}", bucket)
                        file_count += 1

            logger.info(f"Uploaded {file_count} files to LaTeX bundle")
            return f"s3://{bucket}/{base_prefix}"

        except Exception as e:
            error_msg = f"Failed to upload LaTeX bundle: {e}"
            logger.error(error_msg)
            raise S3UploadError(error_msg) from e

    def download_reconstruction_json(self, local_path: Path) -> Path | None:
        """
        Download OpenSfM reconstruction.json for camera reprojection.

        This file contains camera poses and calibration data that enables
        precise mapping from orthophoto coordinates back to raw image pixels.
        Only available for projects processed after the reconstruction upload
        feature was enabled.

        Args:
            local_path: Local path to save the file

        Returns:
            Path to downloaded file, or None if not available

        Note:
            This is optional - older projects won't have this file.
            The report builder falls back to GPS matching when not available.
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/opensfm/reconstruction.json"
        bucket = settings.orthos_bucket

        logger.info(f"Attempting to download reconstruction.json from s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_mb = local_path.stat().st_size / 1_000_000
            logger.info(f"Downloaded reconstruction.json: {size_mb:.2f} MB (camera reprojection enabled)")
            return local_path
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                logger.info("reconstruction.json not found - will use GPS-only matching (older project)")
                return None
            else:
                logger.warning(f"Failed to download reconstruction.json: {e}")
                return None

    def download_dsm(self, local_path: Path) -> Path | None:
        """
        Download Digital Surface Model (DSM) from S3.

        The DSM contains elevation data for each orthophoto pixel, enabling
        precise back-projection from orthophoto coordinates to raw image pixels.
        Using the actual DSM elevation instead of estimated ground level
        significantly improves projection accuracy.

        Args:
            local_path: Local path to save the DSM file

        Returns:
            Path to downloaded file, or None if not available

        Note:
            The DSM is stored in the intermediate bucket as odm/dsm.tif
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/odm/dsm.tif"
        bucket = settings.intermediate_bucket

        logger.info(f"Attempting to download DSM from s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_mb = local_path.stat().st_size / 1_000_000
            logger.info(f"Downloaded DSM: {size_mb:.2f} MB (precise elevation enabled)")
            return local_path
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                logger.info("DSM not found - will use estimated ground elevation")
                return None
            else:
                logger.warning(f"Failed to download DSM: {e}")
                return None

    def download_odm_stats(self, output_dir: Path) -> Path:
        """
        Download ODM statistics and visualizations from S3.

        Downloads all files from the odm_stats/ directory including:
        - stats.json (processing statistics)
        - topview.png (flight path visualization)
        - matchgraph.png (feature matching graph)
        - overlap.png (image overlap diagram)
        - residual_histogram.png (reconstruction residuals)
        - And other diagnostic images

        Args:
            output_dir: Local directory to save ODM stats files

        Returns:
            Path to the output directory

        Raises:
            S3DownloadError: If download fails
        """
        output_dir.mkdir(exist_ok=True, parents=True)

        # ODM stats are stored in the orthos bucket
        prefix = f"{settings.user_id}/projects/{settings.project_id}/odm_stats/"
        bucket = settings.orthos_bucket

        logger.info(f"Downloading ODM stats from s3://{bucket}/{prefix}")

        try:
            # List all files in the odm_stats directory
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' not in response:
                logger.warning(f"No ODM stats found at s3://{bucket}/{prefix}")
                return output_dir

            file_count = 0
            for obj in response['Contents']:
                s3_key = obj['Key']

                # Skip the directory itself
                if s3_key.endswith('/'):
                    continue

                # Get filename relative to prefix
                filename = s3_key.replace(prefix, '')
                local_path = output_dir / filename

                # Download file
                logger.debug(f"Downloading {s3_key} to {local_path}")
                self.s3.download_file(bucket, s3_key, str(local_path))
                file_count += 1

            logger.info(f"Downloaded {file_count} ODM stats files to {output_dir}")
            return output_dir

        except ClientError as e:
            error_msg = f"Failed to download ODM stats from s3://{bucket}/{prefix}: {e}"
            logger.error(error_msg)
            raise S3DownloadError(error_msg) from e

    def download_annotation_overrides(self, local_path: Path) -> bool:
        """
        Download annotation_overrides.json if it exists.

        This file is created by the frontend when a user manually corrects
        thermal annotation positions. If present, the report builder will
        use these override coordinates instead of automatic detection.

        Args:
            local_path: Local path to save the overrides file

        Returns:
            True if overrides were found and downloaded, False otherwise
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/annotation_overrides.json"
        bucket = settings.reports_bucket

        logger.info(f"Checking for annotation overrides at s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_bytes = local_path.stat().st_size
            logger.info(f"Downloaded annotation overrides: {size_bytes} bytes")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                logger.info("No annotation overrides found - using automatic annotations")
                return False
            else:
                logger.warning(f"Failed to download annotation overrides: {e}")
                return False

    def download_crop_annotation(self, local_path: Path) -> Path | None:
        """
        Download crop_annotation.json from S3.

        This file contains crop region and layout configuration flags set by the
        admin annotation tool:
        - polygon: Crop boundary points
        - rotationLine: Alignment reference line
        - isDouble: Whether panels are double trackers
        - isHorizontal: Whether to use row-first numbering (horizontal layout)
        - is2H: Whether panels are 2H configuration

        Args:
            local_path: Local path to save the file

        Returns:
            Path to downloaded file, or None if not available
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/groundtruth/crop_annotation.json"
        bucket = settings.groundtruth_bucket

        logger.info(f"Attempting to download crop annotation from s3://{bucket}/{key}")

        try:
            self.s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
            size_bytes = local_path.stat().st_size
            logger.info(f"Downloaded crop annotation: {size_bytes} bytes")
            return local_path
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                logger.info("Crop annotation not found - using default layout (vertical, single tracker)")
                return None
            else:
                logger.warning(f"Failed to download crop annotation: {e}")
                return None

    def download_tex_bundle(self, work_dir: Path) -> bool:
        """
        Download the LaTeX bundle from S3 for recompilation.

        Downloads the tex_bundle/ directory contents (report.tex + report_images/)
        that was previously uploaded during report generation. This enables
        recompiling the report after manual TeX edits.

        Args:
            work_dir: Local directory to save bundle files

        Returns:
            True if bundle was found and downloaded, False otherwise
        """
        base_prefix = f"{settings.user_id}/projects/{settings.project_id}/tex_bundle/"
        bucket = settings.reports_bucket

        logger.info(f"Downloading TeX bundle from s3://{bucket}/{base_prefix}")

        try:
            # Ensure work directory exists
            work_dir.mkdir(exist_ok=True, parents=True)
            images_dir = work_dir / "report_images"
            images_dir.mkdir(exist_ok=True)

            # List all files in the tex_bundle directory
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix)

            if 'Contents' not in response:
                logger.warning(f"No TeX bundle found at s3://{bucket}/{base_prefix}")
                return False

            file_count = 0
            for obj in response['Contents']:
                s3_key = obj['Key']

                # Skip the directory itself and the .edited marker
                if s3_key.endswith('/') or s3_key.endswith('.edited'):
                    continue

                # Get filename relative to prefix
                rel_path = s3_key.replace(base_prefix, '')

                # Skip if empty rel_path
                if not rel_path:
                    continue

                # Determine local path
                if rel_path.startswith('report_images/'):
                    local_path = work_dir / rel_path
                else:
                    local_path = work_dir / rel_path

                # Ensure parent directory exists
                local_path.parent.mkdir(exist_ok=True, parents=True)

                # Download file
                logger.debug(f"Downloading {s3_key} to {local_path}")
                self.s3.download_file(bucket, s3_key, str(local_path))
                file_count += 1

            logger.info(f"Downloaded {file_count} files from TeX bundle")
            return file_count > 0

        except ClientError as e:
            logger.error(f"Failed to download TeX bundle: {e}")
            return False

    def delete_tex_edited_marker(self) -> bool:
        """
        Delete the .edited marker file from tex_bundle after recompilation.

        This marker indicates the TeX was manually edited. Deleting it after
        successful recompilation clears the "edited" state.

        Returns:
            True if marker was deleted or didn't exist, False on error
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/tex_bundle/.edited"
        bucket = settings.reports_bucket

        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted TeX edit marker: s3://{bucket}/{key}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', '404'):
                logger.debug("TeX edit marker didn't exist")
                return True
            logger.warning(f"Failed to delete TeX edit marker: {e}")
            return False
