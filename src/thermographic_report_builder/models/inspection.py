"""Who inspected, with what equipment, and under which conditions.

IEC TS 62446-3 clause 8 asks the report for the inspector, the camera, the date and place,
the ambient conditions and every deviation from the procedure. These models carry that record
from the platform (InspectionDetails), the raw frames (FlightMetadata) and a weather dataset
(AmbientConditions) to the report builder.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InspectionDetails(BaseModel):
    """Inspection data the client enters on the platform (inspection.json beside the raw images)."""

    model_config = ConfigDict(extra="ignore")

    pilot_name: Optional[str] = Field(default=None, description="Pilot or thermographer")
    pilot_qualification: Optional[str] = Field(
        default=None, description="Thermography qualification, e.g. level 1 per ISO 9712"
    )
    engineer_name: Optional[str] = Field(default=None, description="Responsible engineer")
    engineer_crea: Optional[str] = Field(default=None, description="CREA registration of the engineer")
    plant_address: Optional[str] = Field(default=None, description="Address of the inspected plant")
    calibration_lab: Optional[str] = Field(default=None, description="Lab that calibrated the thermal camera")
    calibration_certificate: Optional[str] = Field(default=None, description="Calibration certificate number")
    calibration_date: Optional[date] = Field(default=None, description="Date of the camera calibration")
    onsite_irradiance_wm2: Optional[float] = Field(
        default=None, ge=0, le=1600, description="Irradiance measured on site during the flight (W/m²)"
    )
    onsite_irradiance_source: Optional[str] = Field(
        default=None, description="Sensor that measured it, e.g. the plant's pyranometer"
    )
    reflected_temp_c: Optional[float] = Field(
        default=None, ge=-60, le=60, description="Reflected apparent temperature measured on site (°C)"
    )

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_missing(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value


class FlightMetadata(BaseModel):
    """What the raw thermal frames record about the flight (EXIF and DJI XMP)."""

    start: Optional[datetime] = Field(default=None, description="First capture, drone clock (local time)")
    end: Optional[datetime] = Field(default=None, description="Last capture, drone clock (local time)")
    frame_count: int = 0
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    drone_model: Optional[str] = None
    drone_serial: Optional[str] = None
    gimbal_pitch_deg: Optional[float] = Field(default=None, description="Median gimbal pitch, -90 = straight down")
    rangefinder_distance_m: Optional[float] = Field(default=None, description="Median laser-measured distance")

    @property
    def midpoint(self) -> Optional[datetime]:
        if self.start and self.end:
            return self.start + (self.end - self.start) / 2
        return self.start or self.end


class AmbientConditions(BaseModel):
    """Ambient conditions at the plant during the flight, estimated from a weather dataset."""

    dataset: str = Field(description="Dataset key: 'ecmwf_ifs' or 'era5'")
    hour_start: datetime = Field(description="Local start of the hour whose mean irradiance is reported")
    irradiance_wm2: Optional[float] = Field(default=None, description="Global horizontal irradiance, hourly mean")
    air_temperature_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    grid_latitude: Optional[float] = None
    grid_longitude: Optional[float] = None

    @property
    def cloud_cover_okta(self) -> Optional[int]:
        if self.cloud_cover_pct is None:
            return None
        return int(round(self.cloud_cover_pct / 12.5))


class SatelliteConditions(BaseModel):
    """Irradiance and cloud cover observed by NOAA's GOES-East satellite at the plant during the flight."""

    satellite: Optional[str] = Field(default=None, description="e.g. GOES-19; None when no scan was found")
    irradiance_wm2: Optional[float] = Field(default=None, description="Global horizontal (DSR), mean over the flight")
    irradiance_resolution: Optional[str] = Field(default=None, description="DSR grid, e.g. '2 km' or '0.5°'")
    irradiance_scans: list[datetime] = Field(default_factory=list, description="DSR scan starts used, local time")
    cloud_fraction: Optional[float] = Field(default=None, description="Share of cloudy 2 km pixels around the plant")
    cloud_scans: list[datetime] = Field(default_factory=list, description="Cloud-mask scan starts used, local time")
    clear_sky_wm2: Optional[float] = Field(default=None, description="Clear-sky global horizontal from date, time and place")

    @property
    def cloud_okta(self) -> Optional[int]:
        if self.cloud_fraction is None:
            return None
        return int(round(self.cloud_fraction * 8))


class MeasurementParameters(BaseModel):
    """Object parameters passed to the DJI Thermal SDK for every temperature in the report."""

    emissivity: float
    reflected_temp_c: float
    reflected_source: str = Field(default="default", description="measured, sky_estimate or default")
    relative_humidity_pct: float
    humidity_source: str = Field(default="default", description="dataset or default")
    object_distance_m: float = Field(description="Default distance and the SDK's 25 m cap")
    distance_from_altitude: bool = Field(
        default=True, description="True when each frame's laser range (or altitude) replaces the default"
    )
