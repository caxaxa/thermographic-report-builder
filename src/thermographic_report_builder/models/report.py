"""Data models for report configuration and metadata."""

from datetime import datetime
from pydantic import BaseModel, Field


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    area_name: str = Field(description="Name of the solar farm area")
    client_name: str = Field(default="Anonimizado", description="Client name")
    language: str = Field(default="pt-BR", description="Report language")
    company_name: str = Field(default="Aisol", description="Company name")
    engineer_name: str = Field(default="Anonimizado", description="Responsible engineer")
    crea_number: str = Field(default="12345678", description="Engineering license number")
    location: str = Field(default="Campo Grande, MS. Brasil", description="Farm location")
    address: str = Field(
        default="Rua Manoel Inácio de Souza, n. 24, C.E.P : 79.020-220",
        description="Farm address",
    )


class DefectMetrics(BaseModel):
    """Metrics and statistics for defects."""

    total_panels: int = Field(description="Total number of panels")
    panels_with_defects: int = Field(description="Number of panels with defects")
    total_defects: int = Field(description="Total number of defects")
    hotspots_count: int = Field(default=0, description="Number of hotspots")
    faulty_diodes_count: int = Field(default=0, description="Number of faulty diodes")
    offline_panels_count: int = Field(default=0, description="Number of offline panels")

    @property
    def defect_rate(self) -> float:
        """Percentage of panels with defects."""
        if self.total_panels == 0:
            return 0.0
        return (self.panels_with_defects / self.total_panels) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            "total_panels": self.total_panels,
            "panels_with_defects": self.panels_with_defects,
            "total_defects": self.total_defects,
            "defect_rate_percent": round(self.defect_rate, 2),
            "hotspots": self.hotspots_count,
            "faulty_diodes": self.faulty_diodes_count,
            "offline_panels": self.offline_panels_count,
        }


class ReportMetadata(BaseModel):
    """Metadata about the generated report."""

    project_id: str
    user_id: str
    generation_date: datetime = Field(default_factory=datetime.utcnow)
    config: ReportConfig
    metrics: DefectMetrics
    orthophoto_path: str
    defect_labels_path: str
    pdf_output_path: str
    processing_duration_seconds: float = Field(default=0.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "generation_date": self.generation_date.isoformat(),
            "processing_duration_seconds": self.processing_duration_seconds,
            "config": {
                "area_name": self.config.area_name,
                "client_name": self.config.client_name,
                "language": self.config.language,
            },
            "metrics": self.metrics.to_dict(),
            "files": {
                "orthophoto": self.orthophoto_path,
                "defect_labels": self.defect_labels_path,
                "pdf_output": self.pdf_output_path,
            },
        }
