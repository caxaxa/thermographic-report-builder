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

AREA_OVERVIEW_TEXT_PT = (
    " Nesta seção, apresentamos uma visão abrangente da área inspecionada. A \\textbf{Figura 1} exibe a \\textbf{ortofoto} montada a partir de todas as imagens capturadas "
    "da região. A \\textbf{Figura 2} apresenta uma representação esquemática da área, destacando as localizações dos rastreadores e dos hotspots detectados. "
    "A numeração dos rastreadores segue o padrão: da esquerda para a direita (de oeste para leste) e de cima para baixo (de norte para sul)."
)
