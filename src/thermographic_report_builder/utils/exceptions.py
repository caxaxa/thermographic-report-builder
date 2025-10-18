"""Custom exception classes for error handling."""


class ProcessingError(Exception):
    """Base exception for all processing errors."""

    pass


class S3DownloadError(ProcessingError):
    """Error downloading files from S3."""

    pass


class S3UploadError(ProcessingError):
    """Error uploading files to S3."""

    pass


class ImageProcessingError(ProcessingError):
    """Error during image processing operations."""

    pass


class ReportGenerationError(ProcessingError):
    """Error during report generation."""

    pass


class InvalidInputError(ProcessingError):
    """Invalid input data or parameters."""

    pass
