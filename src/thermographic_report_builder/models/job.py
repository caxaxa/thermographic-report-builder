"""Data models for AWS Batch job inputs and outputs."""

from pydantic import BaseModel, Field


class JobInput(BaseModel):
    """Input parameters for the report builder job."""

    project_id: str = Field(description="Unique project identifier")
    user_id: str = Field(description="User/customer identifier")
    area_name: str = Field(default="Solar Farm", description="Name of the solar farm")
    orthophoto_bucket: str = Field(description="S3 bucket containing orthophoto")
    uploads_bucket: str = Field(description="S3 bucket containing raw thermal images")
    reports_bucket: str = Field(description="S3 bucket for report outputs")

    @property
    def orthophoto_key(self) -> str:
        """S3 key for orthophoto."""
        return f"{self.user_id}/projects/{self.project_id}/odm_orthophoto.tif"

    @property
    def defect_labels_key(self) -> str:
        """S3 key for defect labels JSON."""
        return f"{self.user_id}/projects/{self.project_id}/defect_labels.json"

    @property
    def raw_images_prefix(self) -> str:
        """S3 prefix for raw thermal images."""
        return f"{self.user_id}/projects/{self.project_id}/images/"

    @property
    def report_output_prefix(self) -> str:
        """S3 prefix for report outputs."""
        return f"{self.user_id}/projects/{self.project_id}/thermographic-report/"


class JobOutput(BaseModel):
    """Output artifacts from the report builder job."""

    report_full_pdf_s3: str = Field(description="S3 URI for full-resolution PDF")
    report_lowres_pdf_s3: str = Field(description="S3 URI for low-resolution PDF")
    metrics_json_s3: str = Field(description="S3 URI for metrics JSON")
    metrics_csv_s3: str = Field(description="S3 URI for metrics CSV")
    layers_dxf_s3: str | None = Field(default=None, description="S3 URI for DXF layers (optional)")

    total_panels: int
    panels_with_defects: int
    total_defects: int
    processing_duration_seconds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for DynamoDB update."""
        return {
            "artifacts": {
                "report_full_pdf": self.report_full_pdf_s3,
                "report_lowres_pdf": self.report_lowres_pdf_s3,
                "metrics_json": self.metrics_json_s3,
                "metrics_csv": self.metrics_csv_s3,
                "layers_dxf": self.layers_dxf_s3,
            },
            "metrics": {
                "total_panels": self.total_panels,
                "panels_with_defects": self.panels_with_defects,
                "total_defects": self.total_defects,
            },
            "processing_duration_seconds": self.processing_duration_seconds,
        }
