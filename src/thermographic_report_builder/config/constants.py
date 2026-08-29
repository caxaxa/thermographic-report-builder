"""Constants and configuration values."""

from ..models.defect import DefectType

# Defect visualization colors (BGR format for OpenCV)
DEFECT_COLORS = {
    DefectType.HOTSPOTS.value: (0, 0, 255),  # Red
    DefectType.FAULTY_DIODES.value: (255, 0, 0),  # Blue
    DefectType.OFFLINE_PANELS.value: (0, 255, 0),  # Green
}

# Portuguese labels for report
DEFECT_LABELS_PT = {
    DefectType.HOTSPOTS.value: "Pontos Quentes (Hotspots)",
    DefectType.FAULTY_DIODES.value: "Diodos de Bypass Queimados",
    DefectType.OFFLINE_PANELS.value: "Painéis Inativos",
}

# English labels for report
DEFECT_LABELS_EN = {
    DefectType.HOTSPOTS.value: "Hotspots",
    DefectType.FAULTY_DIODES.value: "Faulty Bypass Diodes",
    DefectType.OFFLINE_PANELS.value: "Offline Panels",
}


def get_defect_labels(language: str = "pt-BR") -> dict:
    """Get defect labels for the specified language."""
    return DEFECT_LABELS_EN if language in ("en-US", "en") else DEFECT_LABELS_PT

# Image processing defaults
DEFAULT_PANEL_WIDTH_PX = 127  # Average panel width in pixels
ORTHOPHOTO_DOWNSCALE_FACTOR = 0.25  # Downscale to 25% for overview image
CROP_DOWNSCALE_FACTOR = 0.5  # Downscale to 50% for detail crops

# Report text templates (Portuguese)
REPORT_ABSTRACT_PT = (
    "As inspeções termográficas tornaram-se uma ferramenta essencial para avaliar o desempenho e a confiabilidade "
    "de usinas solares. Este estudo foca na detecção e análise de "
    "\\textbf{pontos quentes (hotspots), diodos de bypass queimados e painéis ou strings inativos}, "
    "indicadores críticos de ineficiências operacionais. \\textbf{Hotspots} aparecem como regiões de alta "
    "temperatura localizadas nos painéis solares, frequentemente causadas por sombreamento, acúmulo de sujeira ou "
    "células fotovoltaicas defeituosas, podendo levar à degradação do desempenho e danos a longo prazo. "
    "\\textbf{Diodos de bypass queimados} interrompem o fluxo elétrico esperado, causando o superaquecimento de fileiras "
    "inteiras de células (\\textbf{hot lines}), o que pode reduzir significativamente a eficiência do sistema. "
    "Além disso, \\textbf{painéis ou strings inativos} — identificados como regiões mais frias do que o esperado nas "
    "imagens térmicas — indicam possíveis falhas em inversores, desconexões ou problemas elétricos. "
    "A detecção precoce e a implementação de ações corretivas direcionadas podem otimizar a geração de energia, "
    "prolongar a vida útil do sistema e prevenir falhas onerosas."
)

# Report text templates (English)
REPORT_ABSTRACT_EN = (
    "Thermographic inspections have become an essential tool for evaluating the performance and reliability "
    "of solar power plants. This study focuses on the detection and analysis of "
    "\\textbf{hotspots, faulty bypass diodes, and offline panels or strings}, "
    "critical indicators of operational inefficiencies. \\textbf{Hotspots} appear as localized high-temperature "
    "regions on solar panels, often caused by shading, dirt accumulation, or defective photovoltaic cells, "
    "potentially leading to performance degradation and long-term damage. "
    "\\textbf{Faulty bypass diodes} disrupt the expected electrical flow, causing entire rows of cells to overheat "
    "(\\textbf{hot lines}), which can significantly reduce system efficiency. "
    "Additionally, \\textbf{offline panels or strings} — identified as regions cooler than expected in thermal images — "
    "indicate potential inverter failures, disconnections, or electrical issues. "
    "Early detection and implementation of targeted corrective actions can optimize power generation, "
    "extend system lifespan, and prevent costly failures."
)

REPORT_INTRO_PT = (
    "A termografia, utilizando tecnologia infravermelha, é uma ferramenta fundamental para identificar discrepâncias "
    "térmicas em instalações solares. Este relatório emprega metodologias termográficas avançadas para detectar e localizar "
    "\\textbf{pontos quentes (hotspots)} em painéis solares e rastreadores. Esses hotspots, caracterizados por regiões de temperatura elevada, "
    "geralmente indicam anomalias operacionais ou ineficiências materiais na infraestrutura. "
    "\n\n"
    "A primeira seção do relatório, \\textbf{Dados do Cliente}, apresenta um conjunto de dados que inclui identificadores específicos do cliente "
    "e especificações dos equipamentos. Em seguida, a \\textbf{Visão Geral da Área} oferece uma representação espacial do local da instalação. "
    "Utilizando coordenadas geoespaciais, esta seção fornece um layout escalado dos painéis solares e rastreadores, estabelecendo uma matriz de referência. "
    "\n\n"
)

REPORT_INTRO_EN = (
    "Thermography, using infrared technology, is a fundamental tool for identifying thermal discrepancies "
    "in solar installations. This report employs advanced thermographic methodologies to detect and locate "
    "\\textbf{hotspots} in solar panels and trackers. These hotspots, characterized by elevated temperature regions, "
    "typically indicate operational anomalies or material inefficiencies in the infrastructure. "
    "\n\n"
    "The first section of the report, \\textbf{Client Data}, presents a dataset that includes client-specific identifiers "
    "and equipment specifications. Following that, the \\textbf{Area Overview} provides a spatial representation of the installation site. "
    "Using geospatial coordinates, this section provides a scaled layout of solar panels and trackers, establishing a reference matrix. "
    "\n\n"
)

AREA_OVERVIEW_TEXT_PT = (
    " Nesta seção, apresentamos uma visão abrangente da área inspecionada. A \\textbf{Figura 1} exibe a \\textbf{ortofoto} montada a partir de todas as imagens capturadas "
    "da região. A \\textbf{Figura 2} apresenta uma representação esquemática da área, destacando as localizações dos rastreadores e dos hotspots detectados. "
    "A numeração dos rastreadores segue o padrão: da esquerda para a direita (de oeste para leste) e de cima para baixo (de norte para sul)."
)

AREA_OVERVIEW_TEXT_EN = (
    " In this section, we present a comprehensive overview of the inspected area. \\textbf{Figure 1} displays the \\textbf{orthophoto} assembled from all captured images "
    "of the region. \\textbf{Figure 2} presents a schematic representation of the area, highlighting the locations of trackers and detected hotspots. "
    "Tracker numbering follows the pattern: left to right (west to east) and top to bottom (north to south)."
)


def get_report_abstract(language: str = "pt-BR") -> str:
    """Get report abstract for the specified language."""
    return REPORT_ABSTRACT_EN if language in ("en-US", "en") else REPORT_ABSTRACT_PT


def get_report_intro(language: str = "pt-BR") -> str:
    """Get report introduction for the specified language."""
    return REPORT_INTRO_EN if language in ("en-US", "en") else REPORT_INTRO_PT


def get_area_overview_text(language: str = "pt-BR") -> str:
    """Get area overview text for the specified language."""
    return AREA_OVERVIEW_TEXT_EN if language in ("en-US", "en") else AREA_OVERVIEW_TEXT_PT


# Report section titles and labels
REPORT_TEXTS = {
    "pt-BR": {
        "report_title": "Relatório Termográfico",
        "abstract_title": "Resumo",
        "introduction_title": "Introdução",
        "client_data_title": "Dados do Cliente",
        "area_overview_title": "Visão Geral da Área",
        "defect_summary_title": "Resumo Geral dos Defeitos Identificados",
        "defect_details_title": "Detalhes dos Defeitos",
        "appendix_title": "Apêndice",
        "defect_type": "Tipo de Defeito",
        "quantity": "Quantidade",
        "total": "Total",
        "panel": "Painel",
        "tracker": "Rastreador",
        "row": "Linha",
        "column": "Coluna",
        "temperature": "Temperatura",
        "figure": "Figura",
        "table": "Tabela",
        "client_name": "Nome do Cliente",
        "engineer_name": "Engenheiro Responsável",
        "crea_number": "Número CREA",
        "location": "Localização",
        "inspection_date": "Data da Inspeção",
        "footer_text": "GreTA® - 2026 \\quad Desenvolvido por Aisol",
    },
    "en-US": {
        "report_title": "Thermographic Report",
        "abstract_title": "Abstract",
        "introduction_title": "Introduction",
        "client_data_title": "Client Data",
        "area_overview_title": "Area Overview",
        "defect_summary_title": "General Summary of Identified Defects",
        "defect_details_title": "Defect Details",
        "appendix_title": "Appendix",
        "defect_type": "Defect Type",
        "quantity": "Quantity",
        "total": "Total",
        "panel": "Panel",
        "tracker": "Tracker",
        "row": "Row",
        "column": "Column",
        "temperature": "Temperature",
        "figure": "Figure",
        "table": "Table",
        "client_name": "Client Name",
        "engineer_name": "Responsible Engineer",
        "crea_number": "CREA Number",
        "location": "Location",
        "inspection_date": "Inspection Date",
        "footer_text": "GreTA® - 2026 \\quad Developed by Aisol",
    },
}


def get_report_text(key: str, language: str = "pt-BR") -> str:
    """Get a report text for the specified language."""
    lang = language if language in REPORT_TEXTS else "pt-BR"
    return REPORT_TEXTS[lang].get(key, key)


# Title page texts
TITLE_PAGE_TEXTS = {
    "pt-BR": {
        "report_subtitle": "Relatório Físico",
        "report_subtitle_preview": "Prévia do Relatório",
        "developed_by": "Desenvolvido por:",
        "thermal_inspection_report": "Relatório de Inspeção por Imagem Térmica",
        "technical_responsible": "Responsável Técnico:",
        "date": "Data:",
        "location": "Localização:",
        "address": "Endereço:",
        "software": "Software:",
        "software_desc": "GreTA® - Sistema de Análise Termográfica",
        "copyright": "Copyright © 2026 Aisol Soluções em Inteligência Artificial.",
        "rights_reserved": "Todos os direitos reservados. Nenhuma parte desta publicação pode ser reproduzida, distribuída ou transmitida sem autorização prévia.",
        "isbn": "ISBN:",
        "isbn_value": "A definir.",
    },
    "en-US": {
        "report_subtitle": "Physical Report",
        "report_subtitle_preview": "Report Preview",
        "developed_by": "Developed by:",
        "thermal_inspection_report": "Thermal Imaging Inspection Report",
        "technical_responsible": "Technical Responsible:",
        "date": "Date:",
        "location": "Location:",
        "address": "Address:",
        "software": "Software:",
        "software_desc": "GreTA® - Thermographic Analysis System",
        "copyright": "Copyright © 2026 Aisol Artificial Intelligence Solutions.",
        "rights_reserved": "All rights reserved. No part of this publication may be reproduced, distributed, or transmitted without prior authorization.",
        "isbn": "ISBN:",
        "isbn_value": "To be defined.",
    },
}


CAPACITY_TEXTS = {
    "pt-BR": {
        "title": "Resumo de Capacidade Afetada",
        "intro": (
            "Esta se\\c{c}\\~ao consolida as quantidades f\\'isicas verific\\'aveis da "
            "inspe\\c{c}\\~ao: m\\'odulos comprometidos, capacidade DC afetada e a "
            "classifica\\c{c}\\~ao de severidade t\\'ermica de cada anomalia. Todos os "
            "valores s\\~ao rastre\\'aveis \\`as detec\\c{c}\\~oes e \\`as imagens "
            "originais que as evidenciam."
        ),
        "headline": "{compromised} de {total} m\\'odulos comprometidos ({pct}\\%) --- "
        "aproximadamente {kwp} kWp de capacidade DC afetada (base {wp} Wp/m\\'odulo).",
        "class_table_caption": "Capacidade afetada por classe de defeito",
        "col_class": "Classe de defeito",
        "col_panels": "M\\'odulos",
        "col_kwp": "kWp afetados",
        "severity_para": "Distribui\\c{c}\\~ao de severidade t\\'ermica ($\\Delta$T, conforme pr\\'atica IEC TS 62446-3): {breakdown}. Maior $\\Delta$T registrado: {max_dt}$^{{\\circ}}$C.",
        "severity_critical": "cr\\'itica",
        "severity_attention": "aten\\c{c}\\~ao",
        "severity_monitoring": "monitoramento",
        "priority_caption": "Plano de manuten\\c{c}\\~ao --- prioriza\\c{c}\\~ao por severidade (top {n})",
        "col_panel": "Painel",
        "col_defect": "Defeito",
        "col_dt": "$\\Delta$T ($^{{\\circ}}$C)",
        "col_severity": "Severidade",
        "dependencies_para": (
            "\\textbf{Sobre impacto financeiro e planejamento de manuten\\c{c}\\~ao:} "
            "a convers\\~ao das quantidades acima em perda de receita ou em cronograma "
            "de reparo depende de fatores espec\\'ificos do ativo que n\\~ao s\\~ao "
            "observ\\'aveis pela inspe\\c{c}\\~ao a\\'erea: os termos e pre\\c{c}os do "
            "PPA ou tarifa contratada; a rela\\c{c}\\~ao de sobredimensionamento DC/AC "
            "(em usinas com clipping, parte das perdas DC pode n\\~ao se converter em "
            "perda de energia); a tecnologia, pot\\^encia nominal e estado de "
            "degrada\\c{c}\\~ao dos m\\'odulos; a topologia de strings e inversores; a "
            "sazonalidade de irradia\\c{c}\\~ao; e os termos de garantia e seguro. Por "
            "esse motivo, este relat\\'orio apresenta apenas grandezas f\\'isicas "
            "verific\\'aveis. An\\'alise financeira espec\\'ifica do ativo, combinando "
            "estes dados com os par\\^ametros comerciais da usina, pode ser elaborada "
            "sob consulta."
        ),
    },
    "en-US": {
        "title": "Affected Capacity Summary",
        "intro": (
            "This section consolidates the verifiable physical quantities of the "
            "inspection: compromised modules, affected DC capacity and the thermal "
            "severity classification of each anomaly. Every figure is traceable to "
            "the detections and to the original images that evidence them."
        ),
        "headline": "{compromised} of {total} modules compromised ({pct}\\%) --- "
        "approximately {kwp} kWp of affected DC capacity (assuming {wp} Wp/module).",
        "class_table_caption": "Affected capacity by defect class",
        "col_class": "Defect class",
        "col_panels": "Modules",
        "col_kwp": "Affected kWp",
        "severity_para": "Thermal severity distribution ($\\Delta$T, per IEC TS 62446-3 practice): {breakdown}. Highest recorded $\\Delta$T: {max_dt}$^{{\\circ}}$C.",
        "severity_critical": "critical",
        "severity_attention": "attention",
        "severity_monitoring": "monitoring",
        "priority_caption": "Maintenance plan --- severity-ranked priorities (top {n})",
        "col_panel": "Panel",
        "col_defect": "Defect",
        "col_dt": "$\\Delta$T ($^{{\\circ}}$C)",
        "col_severity": "Severity",
        "dependencies_para": (
            "\\textbf{On financial impact and maintenance planning:} converting the "
            "quantities above into revenue loss or a repair schedule depends on "
            "asset-specific factors that aerial inspection cannot observe: the PPA "
            "terms or contracted tariff; the DC/AC oversizing ratio (in clipped "
            "plants, part of the DC loss may not translate into energy loss); module "
            "technology, nameplate rating and degradation state; string and inverter "
            "topology; irradiance seasonality; and warranty and insurance terms. For "
            "this reason, this report presents only verifiable physical quantities. "
            "An asset-specific financial analysis, combining these findings with the "
            "plant's commercial parameters, is available upon request."
        ),
    },
}


def get_capacity_text(key: str, language: str = "pt-BR") -> str:
    """Get an affected-capacity section text for the specified language."""
    lang = language if language in CAPACITY_TEXTS else "pt-BR"
    return CAPACITY_TEXTS[lang].get(key, key)


def get_title_page_text(key: str, language: str = "pt-BR") -> str:
    """Get title page text for the specified language."""
    lang = language if language in TITLE_PAGE_TEXTS else "pt-BR"
    return TITLE_PAGE_TEXTS[lang].get(key, key)


# Defect details section texts
DEFECT_DETAILS_TEXTS = {
    "pt-BR": {
        "hotspots_title": "Pontos Quentes (Hot Spots)",
        "hotspots_intro": "Foram detectados pontos quentes nas placas abaixo.",
        "hotspots_panel_text": "No painel {panel_id}, há sinais de pontos quentes conforme as figuras acima.",
        "hotspots_none": "Não foram encontrados problemas de pontos quentes na área inspecionada.",
        "faulty_diodes_title": "Diodos de Bypass Queimados",
        "faulty_diodes_intro": "Foram detectados diodos de bypass queimados nas placas abaixo.",
        "faulty_diodes_panel_text": "No painel {panel_id}, foram detectados diodos de bypass queimados.",
        "faulty_diodes_none": "Não foram encontrados problemas de diodos de bypass queimados na área inspecionada.",
        "offline_panels_title": "Painéis Desligados",
        "offline_panels_intro": "Foram detectadas anomalias indicando painel(es) desligados nas placas abaixo.",
        "offline_panels_panel_text": "No painel {panel_id}, foram detectadas anomalias indicando painel(es) desligados.",
        "offline_panels_none": "Não foram encontrados problemas de painéis desligados na área inspecionada.",
        "panel_subsection": "Painel {panel_id}",
        "panel_images_caption": "Imagens do Painel n. {row} da coluna n. {col}.",
        "context_map": "Mapa de Contexto",
        "problem_location": "Localização do Problema",
        "drone_image": "Imagem Original do Drone",
        "thermal_analysis": "Análise Termográfica - Painel {panel_id}",
        "thermal_analysis_defect": "Análise Termográfica - Painel {panel_id} (Defeito {defect_num})",
    },
    "en-US": {
        "hotspots_title": "Hotspots",
        "hotspots_intro": "Hotspots were detected on the panels below.",
        "hotspots_panel_text": "Panel {panel_id} shows signs of hotspots as shown in the figures above.",
        "hotspots_none": "No hotspot issues were found in the inspected area.",
        "faulty_diodes_title": "Faulty Bypass Diodes",
        "faulty_diodes_intro": "Faulty bypass diodes were detected on the panels below.",
        "faulty_diodes_panel_text": "Panel {panel_id} has faulty bypass diodes detected.",
        "faulty_diodes_none": "No faulty bypass diode issues were found in the inspected area.",
        "offline_panels_title": "Offline Panels",
        "offline_panels_intro": "Anomalies indicating offline panel(s) were detected on the panels below.",
        "offline_panels_panel_text": "Panel {panel_id} shows anomalies indicating offline panel(s).",
        "offline_panels_none": "No offline panel issues were found in the inspected area.",
        "panel_subsection": "Panel {panel_id}",
        "panel_images_caption": "Images of Panel row {row}, column {col}.",
        "context_map": "Context Map",
        "problem_location": "Problem Location",
        "drone_image": "Original Drone Image",
        "thermal_analysis": "Thermal Analysis - Panel {panel_id}",
        "thermal_analysis_defect": "Thermal Analysis - Panel {panel_id} (Defect {defect_num})",
    },
}


def get_defect_details_text(key: str, language: str = "pt-BR") -> str:
    """Get defect details text for the specified language."""
    lang = language if language in DEFECT_DETAILS_TEXTS else "pt-BR"
    return DEFECT_DETAILS_TEXTS[lang].get(key, key)


# Summary table texts
SUMMARY_TABLE_TEXTS = {
    "pt-BR": {
        "defect_location_title": "Localização de Cada Defeito Identificado",
        "defect_location_cont": "Localização de Cada Defeito Identificado (cont.)",
        "problem_type": "Tipo de Problema",
        "panel_location": "Local do Painel",
        "coordinates": "Coordenadas",
        "total_panels_with_defects": "Total de {count} painéis com defeitos identificados.",
    },
    "en-US": {
        "defect_location_title": "Location of Each Identified Defect",
        "defect_location_cont": "Location of Each Identified Defect (cont.)",
        "problem_type": "Problem Type",
        "panel_location": "Panel Location",
        "coordinates": "Coordinates",
        "total_panels_with_defects": "Total of {count} panels with identified defects.",
    },
}


def get_summary_table_text(key: str, language: str = "pt-BR") -> str:
    """Get summary table text for the specified language."""
    lang = language if language in SUMMARY_TABLE_TEXTS else "pt-BR"
    return SUMMARY_TABLE_TEXTS[lang].get(key, key)


# Thermal legend table texts
THERMAL_LEGEND_TEXTS = {
    "pt-BR": {
        "parameter": "Parâmetro",
        "value": "Valor",
        "hotpoint": "Ponto Quente",
        "reference": "Referência",
        "delta_t": "$\\Delta$T (Diferença)",
        "classification": "Classificação",
        "severity_critical": "Crítico",
        "severity_high": "Alto",
        "severity_medium": "Médio",
        "severity_low": "Baixo",
        "severity_minimal": "Mínimo",
    },
    "en-US": {
        "parameter": "Parameter",
        "value": "Value",
        "hotpoint": "Hot Point",
        "reference": "Reference",
        "delta_t": "$\\Delta$T (Difference)",
        "classification": "Classification",
        "severity_critical": "Critical",
        "severity_high": "High",
        "severity_medium": "Medium",
        "severity_low": "Low",
        "severity_minimal": "Minimal",
    },
}


def get_thermal_legend_text(key: str, language: str = "pt-BR") -> str:
    """Get thermal legend text for the specified language."""
    lang = language if language in THERMAL_LEGEND_TEXTS else "pt-BR"
    return THERMAL_LEGEND_TEXTS[lang].get(key, key)


# Appendix texts
APPENDIX_TEXTS = {
    "pt-BR": {
        "flight_info_title": "Informações do Voo e Ortofoto",
        "orthophoto_data_title": "Dados da Ortofoto",
        "flight_path_caption": "Trajetória de Voo do Drone",
        "dashboard_caption": "Dashboard do Voo - Estatísticas e Perfis",
        "flight_data_unavailable": "Dados de voo não disponíveis para este projeto.",
        "flight_viz_error": "Erro ao gerar visualizações de voo.",
        "matchgraph_caption": "Grafo de Correspondência de Características",
        "overlap_caption": "Diagrama de Sobreposição de Imagens",
        "residual_histogram_caption": "Histograma de Resíduos do Modelo",
        "processing_step": "Etapa de Processamento",
        "time": "Tempo",
        "processing_stats_caption": "Estatísticas de Processamento",
        "flight_metric": "Métrica de Voo",
        "flight_stats_caption": "Estatísticas do Voo",
        "total_images": "Total de Imagens",
        "flight_duration": "Duração do Voo",
        "total_distance": "Distância Total",
        "min_altitude": "Altitude Mínima",
        "max_altitude": "Altitude Máxima",
        "mean_altitude": "Altitude Média",
        "coverage_area": "Área de Cobertura",
        "mean_speed": "Velocidade Média",
        "mean_gsd": "GSD Médio",
        "gps_error_metric": "Métrica de Erro GPS",
        "gps_error_caption": "Erros de Predição GPS (Precisão da Reconstrução)",
        "mean_x": "Média X",
        "mean_y": "Média Y",
        "mean_z": "Média Z",
        "std_x": "Desvio Padrão X",
        "std_y": "Desvio Padrão Y",
        "std_z": "Desvio Padrão Z",
        "error_3d_mean": "Erro 3D Médio",
        "error_3d_max": "Erro 3D Máximo",
        "reconstruction_metric": "Métrica de Reconstrução",
        "reconstruction_stats_caption": "Estatísticas de Reconstrução",
        "components": "Componentes",
        "has_gps": "Possui GPS",
        "initial_points": "Pontos Iniciais",
        "reconstructed_points": "Pontos Reconstruídos",
        "features_metric": "Métrica de Características",
        "features_stats_caption": "Estatísticas de Características",
        "detected_features": "Características Detectadas",
        "reconstructed_features": "Características Reconstruídas",
        "minimum": "Mínimo",
        "maximum": "Máximo",
        "mean": "Média",
        "median": "Mediana",
    },
    "en-US": {
        "flight_info_title": "Flight and Orthophoto Information",
        "orthophoto_data_title": "Orthophoto Data",
        "flight_path_caption": "Drone Flight Path",
        "dashboard_caption": "Flight Dashboard - Statistics and Profiles",
        "flight_data_unavailable": "Flight data not available for this project.",
        "flight_viz_error": "Error generating flight visualizations.",
        "matchgraph_caption": "Feature Correspondence Graph",
        "overlap_caption": "Image Overlap Diagram",
        "residual_histogram_caption": "Model Residual Histogram",
        "processing_step": "Processing Step",
        "time": "Time",
        "processing_stats_caption": "Processing Statistics",
        "flight_metric": "Flight Metric",
        "flight_stats_caption": "Flight Statistics",
        "total_images": "Total Images",
        "flight_duration": "Flight Duration",
        "total_distance": "Total Distance",
        "min_altitude": "Minimum Altitude",
        "max_altitude": "Maximum Altitude",
        "mean_altitude": "Mean Altitude",
        "coverage_area": "Coverage Area",
        "mean_speed": "Mean Speed",
        "mean_gsd": "Mean GSD",
        "gps_error_metric": "GPS Error Metric",
        "gps_error_caption": "GPS Prediction Errors (Reconstruction Accuracy)",
        "mean_x": "Mean X",
        "mean_y": "Mean Y",
        "mean_z": "Mean Z",
        "std_x": "Std Dev X",
        "std_y": "Std Dev Y",
        "std_z": "Std Dev Z",
        "error_3d_mean": "Mean 3D Error",
        "error_3d_max": "Max 3D Error",
        "reconstruction_metric": "Reconstruction Metric",
        "reconstruction_stats_caption": "Reconstruction Statistics",
        "components": "Components",
        "has_gps": "Has GPS",
        "initial_points": "Initial Points",
        "reconstructed_points": "Reconstructed Points",
        "features_metric": "Features Metric",
        "features_stats_caption": "Features Statistics",
        "detected_features": "Detected Features",
        "reconstructed_features": "Reconstructed Features",
        "minimum": "Minimum",
        "maximum": "Maximum",
        "mean": "Mean",
        "median": "Median",
    },
}


def get_appendix_text(key: str, language: str = "pt-BR") -> str:
    """Get appendix text for the specified language."""
    lang = language if language in APPENDIX_TEXTS else "pt-BR"
    return APPENDIX_TEXTS[lang].get(key, key)


# Preview mode texts (suppression messages)
PREVIEW_TEXTS = {
    "pt-BR": {
        "table_suppressed": "Suprimida — Veja Tabela na Versão Completa",
        "images_suppressed": "Suprimida — Veja Imagens na Versão Completa",
        "appendix_suppressed": "Suprimida — Veja Apêndice na Versão Completa",
    },
    "en-US": {
        "table_suppressed": "Suppressed — See Table in the Full Report Version",
        "images_suppressed": "Suppressed — See Images in the Full Report Version",
        "appendix_suppressed": "Suppressed — See Appendix in the Full Report Version",
    },
}


def get_preview_text(key: str, language: str = "pt-BR") -> str:
    """Get preview-mode suppression text for the specified language."""
    lang = language if language in PREVIEW_TEXTS else "pt-BR"
    return PREVIEW_TEXTS[lang].get(key, key)
