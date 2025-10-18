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
