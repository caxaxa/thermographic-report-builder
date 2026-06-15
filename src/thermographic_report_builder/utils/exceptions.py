"""Custom exception classes for error handling.

The ``ProcessingError`` hierarchy now lives in the shared ``solar_report_utils``
library; this module re-exports it so existing
``from ...utils.exceptions import X`` call sites keep working.
"""

from solar_report_utils import (
    ImageProcessingError,
    InvalidInputError,
    ProcessingError,
    ReportGenerationError,
    S3DownloadError,
    S3UploadError,
)

__all__ = [
    "ProcessingError",
    "S3DownloadError",
    "S3UploadError",
    "ImageProcessingError",
    "ReportGenerationError",
    "InvalidInputError",
]
