"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    All settings can be overridden via environment variables prefixed with SOLAR_
    For example: SOLAR_PROJECT_ID=abc123
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SOLAR_",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== AWS Configuration =====
    aws_region: str = "us-east-2"
    orthos_bucket: str = "solar-orthos-prod"
    uploads_bucket: str = "solar-uploads-prod"
    reports_bucket: str = "solar-reports-prod"

    # ===== Job Parameters (from AWS Batch environment) =====
    project_id: str
    user_id: str
    area_name: str = "Solar Farm"

    # ===== Processing Parameters =====
    orthophoto_downscale_factor: float = 0.25
    crop_downscale_factor: float = 0.5
    default_panel_width_px: int = 127
    crop_panel_size: int = 5  # Number of panels to include in crop

    # ===== Report Configuration =====
    report_language: str = "pt-BR"
    company_name: str = "Aisol"
    client_name: str = "Anonimizado"
    engineer_name: str = "Anonimizado"
    crea_number: str = "12345678"
    location: str = "Campo Grande, MS. Brasil"
    address: str = "Rua Manoel Inácio de Souza, n. 24, C.E.P : 79.020-220"

    # ===== PDF Generation =====
    pdf_quality: int = 90
    jpeg_quality: int = 70
    generate_lowres_pdf: bool = True

    # ===== Logging =====
    log_level: str = "INFO"
    log_json: bool = True  # JSON format for CloudWatch

    # ===== Working Directory =====
    work_dir: Path = Path("/tmp/report_work")

    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
