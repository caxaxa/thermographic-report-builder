"""Tests for the inspection record in the generated LaTeX (IEC TS 62446-3 clause 8)."""
from datetime import date, datetime
from pathlib import Path

from thermographic_report_builder.models.defect import BoundingBox, Defect, GeospatialCoordinate, Panel
from thermographic_report_builder.models.inspection import (
    AmbientConditions,
    FlightMetadata,
    InspectionDetails,
    MeasurementParameters,
    SatelliteConditions,
)
from thermographic_report_builder.models.report import ReportConfig
from thermographic_report_builder.report.builder import ReportBuilder


def _panel():
    bbox = BoundingBox(left=100, top=200, width=50, height=80, label="solarpanels")
    hotspot = Defect(
        bbox=BoundingBox(left=110, top=220, width=10, height=15, label="hotspots"),
        defect_center_px=(115, 227.5),
        panel_centroid_geospatial=GeospatialCoordinate(longitude=-54.0, latitude=-20.0),
        defect_type="hotspots",
    )
    return Panel(row=1, column=1, bbox=bbox, hotspots=[hotspot])


def _tex(tmp_path, language="pt-BR", **config):
    images_dir = Path(tmp_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    builder = ReportBuilder(
        panel_grid={(1, 1): _panel()},
        images_dir=images_dir,
        config=ReportConfig(area_name="Usina Teste", language=language, **config),
        defect_matches={},
        odm_stats_dir=Path(tmp_path) / "odm_stats",
    )
    out = Path(tmp_path) / "report.tex"
    builder.generate_tex(out)
    return out.read_text()


FULL = dict(
    engineer_name="Carlos Lima",
    crea_number="MS-12345",
    address="Rodovia MS-080, km 12",
    inspection=InspectionDetails(
        pilot_name="Ana Souza",
        pilot_qualification="Nível 1",
        calibration_lab="Lab X",
        calibration_certificate="C-001",
        calibration_date=date(2026, 3, 1),
    ),
    flight=FlightMetadata(
        start=datetime(2026, 11, 10, 11, 2),
        end=datetime(2026, 11, 10, 11, 19),
        frame_count=511,
        camera_make="DJI",
        camera_model="M30T",
        camera_serial="CAM_456",
        drone_model="M30T",
        drone_serial="DRONE789",
    ),
    conditions=AmbientConditions(
        dataset="ecmwf_ifs",
        hour_start=datetime(2026, 11, 10, 11),
        irradiance_wm2=914,
        air_temperature_c=29.0,
        relative_humidity_pct=52,
        wind_speed_kmh=10.6,
        cloud_cover_pct=64,
        grid_latitude=-20.49,
        grid_longitude=-54.65,
    ),
    measurement=MeasurementParameters(
        emissivity=0.85, reflected_temp_c=12.2, reflected_source="sky_estimate",
        relative_humidity_pct=51, humidity_source="dataset", object_distance_m=25,
    ),
)


class TestInspectionRecord:

    def test_cover_uses_inspection_data(self, tmp_path):
        tex = _tex(tmp_path, **FULL)
        assert "Carlos Lima" in tex and "MS{-}12345" in tex
        assert "Data da inspeção:} 10/11/2026" in tex
        assert "Rodovia MS{-}080, km 12" in tex
        assert "Rua Manoel Inácio" not in tex

    def test_inspection_section(self, tmp_path):
        tex = _tex(tmp_path, **FULL)
        assert "\\section{Dados da Inspeção}" in tex
        assert "10/11/2026, das 11:02 às 11:19" in tex
        assert "Ana Souza (Nível 1)" in tex
        assert "DJI M30T, câmera radiométrica de onda longa (LWIR), nº de série CAM\\_456" in tex
        assert "Lab X, certificado C{-}001, de 01/03/2026" in tex
        assert "914 W/m$^2$, global horizontal estimada, média das 11:00 às 12:00" in tex
        assert "5/8" in tex
        assert "ECMWF IFS" in tex

    def test_measurement_parameters_show_their_source(self, tmp_path):
        tex = _tex(tmp_path, **FULL)
        assert "0.85 (vidro do módulo, onda longa)" in tex
        assert "12.2 °C, temperatura do céu estimada pelas condições do dia" in tex
        assert "51\\%, do conjunto de dados meteorológicos" in tex
        assert "telêmetro laser de cada imagem (ou altitude de voo), até 25 m" in tex
        assert "A temperatura refletida (12.2 °C) foi estimada pela temperatura do céu" in tex

    def test_missing_data_is_marked_and_listed(self, tmp_path):
        tex = _tex(tmp_path, engineer_name="", crea_number="", address="")
        assert "Responsável Técnico:} Não informado" in tex
        assert "\\section{Desvios e Limitações}" in tex
        assert "O piloto ou termografista e sua qualificação não foram informados." in tex
        assert "A irradiância não foi medida nem estimada" in tex
        assert "Temperatura do ar, umidade e vento não foram obtidos" in tex

    def test_deviations_follow_the_conditions(self, tmp_path):
        full = dict(FULL)
        full["conditions"] = FULL["conditions"].model_copy(update={"irradiance_wm2": 480})
        tex = _tex(tmp_path, **full)
        assert "A irradiância estimada de 480 W/m$^2$ ficou abaixo do mínimo" in tex
        assert "A nebulosidade estimada de 5/8" in tex

    def test_finding_has_cause_and_action(self, tmp_path):
        tex = _tex(tmp_path, **FULL)
        assert "Causa provável:} aquecimento localizado" in tex
        assert "Ação recomendada:} verificar sujeira e sombreamento" in tex

    def test_english_report(self, tmp_path):
        tex = _tex(tmp_path, language="en-US", **FULL)
        assert "\\section{Inspection Data}" in tex
        assert "\\section{Deviations and Limitations}" in tex
        assert "Inspection date:} Nov 10, 2026" in tex


class TestSatelliteRecord:

    def test_satellite_rows(self, tmp_path):
        full = dict(FULL)
        full["satellite"] = SatelliteConditions(
            satellite="GOES-19",
            irradiance_wm2=905.4,
            irradiance_resolution="2 km",
            irradiance_scans=[datetime(2026, 11, 10, 11, 0), datetime(2026, 11, 10, 11, 10)],
            cloud_fraction=0.0,
            clear_sky_wm2=931.2,
        )
        tex = _tex(tmp_path, **full)
        assert "905 W/m$^2$, global horizontal observada pelo satélite GOES-19 (NOAA, grade de 2 km), média das imagens das 11:00 às 11:20" in tex
        assert "931 W/m$^2$, calculada pela data, hora e local da inspeção" in tex
        assert "0/8, observada pelo satélite GOES-19" in tex
        assert "Fonte da temperatura, umidade e vento" in tex
        assert "A irradiância foi observada pelo satélite GOES-19 da NOAA" in tex
        assert "Temperatura do ar e vento foram estimados a partir de ECMWF IFS" in tex


class TestSummaryLabels:

    def test_summary_table_is_localized(self, tmp_path):
        tex = _tex(tmp_path)
        assert "Diodos de Bypass Queimados" in tex
        assert "Faulty Bypass Diodes" not in tex
