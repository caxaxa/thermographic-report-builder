"""Application settings loaded from environment variables."""

import os
from pathlib import Path

from solar_report_utils import BaseReportSettings, get_bucket_suffix


class Settings(BaseReportSettings):
    """
    Application configuration loaded from environment variables.

    All settings can be overridden via environment variables prefixed with SOLAR_
    For example: SOLAR_PROJECT_ID=abc123
    """

    # model_config (env_file/env_prefix/case_sensitive/extra) and the
    # ``aws_region`` default are inherited from BaseReportSettings, which is
    # byte-identical to the previous local definitions.

    # ===== AWS Configuration =====
    # Bucket names are determined by ENVIRONMENT variable (dev/prod)
    orthos_bucket: str = f"solar-orthos-{get_bucket_suffix()}"
    uploads_bucket: str = f"solar-uploads-{get_bucket_suffix()}"
    reports_bucket: str = f"solar-reports-{get_bucket_suffix()}"
    groundtruth_bucket: str = f"solar-groundtruth-{get_bucket_suffix()}"
    intermediate_bucket: str = f"solar-intermediate-{get_bucket_suffix()}"

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
    client_name: str = ""  # Will be set from user email
    engineer_name: str = "Automático"
    crea_number: str = ""  # Left blank for automated reports
    location: str = ""  # Will be derived from orthophoto GPS coordinates
    address: str = "Rua Manoel Inácio de Souza, n. 24, C.E.P : 79.020-220"

    # ===== PDF Generation =====
    pdf_quality: int = 90
    jpeg_quality: int = 70
    generate_preview_pdf: bool = True

    # ===== Thermal Calibration Parameters =====
    # These parameters affect temperature accuracy in thermal analysis
    # See DJI Thermal SDK documentation for details
    thermal_emissivity: float = 0.60  # Effective emissivity for solar panels
    thermal_object_distance: float = 25.0  # Default flight altitude in meters (1-25m range)
    thermal_relative_humidity: float = 50.0  # Relative humidity % (20-100%)
    thermal_reflected_temp: float = 20.0  # Reflected apparent temperature in Celsius
    thermal_auto_altitude: bool = True  # Extract altitude from EXIF if available

    # ===== Thermal-Visual Alignment Parameters =====
    # The DJI M30T R-JPEG embeds a visual channel (1280x1024) that is pre-aligned with
    # the thermal data (640x512). However, there can be residual misalignment due to
    # manufacturing tolerances or temperature drift. These offsets correct for that.
    # Units are in VISUAL pixels (1280x1024 space), applied BEFORE scaling to thermal.
    # Positive X = shift thermal lookup RIGHT, Positive Y = shift thermal lookup DOWN.
    thermal_visual_offset_x: float = 0.0  # Horizontal offset (visual pixels)
    thermal_visual_offset_y: float = 0.0  # Vertical offset (visual pixels)

    # ===== Mesh Backtracking Configuration =====
    # Mesh backtracking is disabled by default to save bandwidth (mesh artifacts are 100MB+).
    # Mesh uses the same centrality-based camera selection as source map (redundant) and is
    # demoted to last-resort fallback after source map, camera reprojection, and GPS methods.
    # Enable only if you need the mesh fallback for edge cases where other methods fail.
    enable_mesh_fallback: bool = False

    # ===== Logging =====
    log_level: str = "INFO"
    log_json: bool = True  # JSON format for CloudWatch

    # ===== Working Directory =====
    work_dir: Path = Path("/tmp/report_work")

    # ===== Compile-Only Mode =====
    # When True, skip report generation and only recompile existing tex_bundle
    # Used when user has edited the TeX file and wants to regenerate PDF without
    # re-running the full pipeline
    compile_only: bool = False

    def __init__(self, **kwargs):  # type: ignore
        # Handle backward compatibility with old environment variable names
        # Old: PROJECT_ID, ORG_ID -> New: SOLAR_PROJECT_ID, SOLAR_USER_ID
        if 'project_id' not in kwargs and not os.getenv('SOLAR_PROJECT_ID'):
            project_id = os.getenv('PROJECT_ID')
            if project_id:
                kwargs['project_id'] = project_id

        if 'user_id' not in kwargs and not os.getenv('SOLAR_USER_ID'):
            user_id = os.getenv('ORG_ID') or os.getenv('USER_ID')
            if user_id:
                kwargs['user_id'] = user_id

        # Handle backward compatibility for bucket names (without SOLAR_ prefix)
        if 'orthos_bucket' not in kwargs and not os.getenv('SOLAR_ORTHOS_BUCKET'):
            orthos_bucket = os.getenv('ORTHOS_BUCKET')
            if orthos_bucket:
                kwargs['orthos_bucket'] = orthos_bucket

        if 'uploads_bucket' not in kwargs and not os.getenv('SOLAR_UPLOADS_BUCKET'):
            uploads_bucket = os.getenv('UPLOADS_BUCKET')
            if uploads_bucket:
                kwargs['uploads_bucket'] = uploads_bucket

        if 'reports_bucket' not in kwargs and not os.getenv('SOLAR_REPORTS_BUCKET'):
            reports_bucket = os.getenv('REPORTS_BUCKET')
            if reports_bucket:
                kwargs['reports_bucket'] = reports_bucket

        if 'aws_region' not in kwargs and not os.getenv('SOLAR_AWS_REGION'):
            aws_region = os.getenv('AWS_DEFAULT_REGION')
            if aws_region:
                kwargs['aws_region'] = aws_region

        super().__init__(**kwargs)
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
