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
        key = f"{settings.user_id}/projects/{settings.project_id}/odm_orthophoto.tif"
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

    def download_defect_labels(self, local_path: Path) -> Path:
        """
        Download defect_labels.json from S3.

        Args:
            local_path: Local path to save the file

        Returns:
            Path to downloaded file

        Raises:
            S3DownloadError: If download fails
        """
        key = f"{settings.user_id}/projects/{settings.project_id}/defect_labels.json"
        bucket = settings.reports_bucket

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
