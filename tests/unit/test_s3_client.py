"""Tests for S3Client — download/upload operations.

Tests focus on the fallback logic (resampled → original orthophoto),
optional file downloads (return None on missing), and error handling.
The S3 client uses `self.s3` as its internal boto3 client attribute.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from botocore.exceptions import ClientError


def _make_no_such_key_error():
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
        "GetObject",
    )


class TestS3ClientInit:
    def test_creates_client(self):
        with patch("thermographic_report_builder.io.s3_client.boto3") as mock_boto:
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            assert client is not None
            assert client.region == "us-east-1"


class TestDownloadOrthophotoResampled:
    """Test the fallback logic in download_orthophoto_resampled."""

    def test_tries_resampled_first(self, tmp_path):
        """Should attempt resampled version before falling back to original."""
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            call_log = []

            def mock_download_file(Bucket, Key, Filename):
                call_log.append(Key)
                if "1.6cm" in Key:
                    raise _make_no_such_key_error()
                # Create a fake file for the original download
                Path(Filename).write_bytes(b"fake tif")

            client.s3 = MagicMock()
            client.s3.download_file = mock_download_file

            local_path = tmp_path / "ortho.tif"
            result_path, version = client.download_orthophoto_resampled(local_path)

            # Should have tried resampled first, then fallen back
            assert len(call_log) == 2
            assert "1.6cm" in call_log[0]
            assert version == "original"


class TestOptionalDownloads:
    """Test methods that return None when files don't exist in S3."""

    def test_download_reconstruction_json_returns_none_on_missing(self):
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            client.s3 = MagicMock()
            client.s3.download_file.side_effect = _make_no_such_key_error()

            result = client.download_reconstruction_json(local_path=Path("/tmp/test/recon.json"))
            assert result is None

    def test_download_source_map_returns_none_on_missing(self):
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            client.s3 = MagicMock()
            client.s3.download_file.side_effect = _make_no_such_key_error()

            result = client.download_source_map(local_path=Path("/tmp/test/sourcemap.tif"))
            assert result is None

    def test_download_crop_annotation_returns_none_on_missing(self):
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            client.s3 = MagicMock()
            client.s3.download_file.side_effect = _make_no_such_key_error()

            result = client.download_crop_annotation(local_path=Path("/tmp/test/crop.json"))
            assert result is None

    def test_download_tracks_csv_returns_none_on_missing(self):
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            client.s3 = MagicMock()
            client.s3.download_file.side_effect = _make_no_such_key_error()

            result = client.download_tracks_csv(local_path=Path("/tmp/test/tracks.csv"))
            assert result is None


class TestUploadReport:
    def test_upload_report_calls_s3(self, tmp_path):
        with patch("thermographic_report_builder.io.s3_client.boto3"):
            from thermographic_report_builder.io.s3_client import S3Client

            client = S3Client(region="us-east-1")
            client.s3 = MagicMock()

            # Create a fake report file
            report_file = tmp_path / "report.pdf"
            report_file.write_bytes(b"fake pdf content")

            result = client.upload_report(
                local_path=report_file,
                report_filename="report.pdf",
            )

            client.s3.upload_file.assert_called_once()
            assert "s3://" in result
