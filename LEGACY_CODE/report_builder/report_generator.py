import pylatex as pl
from pylatex.utils import NoEscape, bold
from datetime import datetime
import json
import os

def generate_report(defects_dict, area_name, current_dir):
    """
    Generate a thermographic inspection report for solar power plants.
    The source code is in English, but the client report is in Portuguese.
    """
    # Format the area name for LaTeX and define paths
    area_name = area_name.replace("_", "\\_")
    report_images_dir = "report_images"
    orthophoto_path_img = os.path.join(report_images_dir, 'ortho.png')
    aisol_logo_path = os.path.join(report_images_dir, 'aisol_logo.png')
    aisol_logo_2_path = os.path.join(report_images_dir, 'logo_2.png')
    layer_img_path = os.path.join(report_images_dir, 'layer_img.pdf')
    top_view = os.path.join(report_images_dir, "topview.png")
    match = os.path.join(report_images_dir, "matchgraph.png")
    overlap = os.path.join(report_images_dir, "overlap.png")
    residual = os.path.join(report_images_dir, "residual_histogram.png")
    
    current_date = datetime.now()

    # Client and report texts
    client_data = "Anonimizado"
    abstract = ("As inspeções termográficas tornaram-se uma ferramenta essencial para avaliar o desempenho e a confiabilidade "
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
                "prolongar a vida útil do sistema e prevenir falhas onerosas.")
    
    intro_text_pt = ("A termografia, utilizando tecnologia infravermelha, é uma ferramenta fundamental para identificar discrepâncias "
                     "térmicas em instalações solares. Este relatório emprega metodologias termográficas avançadas para detectar e localizar "
                     "\\textbf{pontos quentes (hotspots)} em painéis solares e rastreadores. Esses hotspots, caracterizados por regiões de temperatura elevada, "
                     "geralmente indicam anomalias operacionais ou ineficiências materiais na infraestrutura. "
                     "\n\n"
                     "A primeira seção do relatório, \\textbf{Dados do Cliente}, apresenta um conjunto de dados que inclui identificadores específicos do cliente "
                     "e especificações dos equipamentos. Em seguida, a \\textbf{Visão Geral da Área} oferece uma representação espacial do local da instalação. "
                      "Utilizando coordenadas geoespaciais, esta seção fornece um layout escalado dos painéis solares e rastreadores, estabelecendo uma matriz de referência. "
                     "\n\n")

    drone_intro = ("A última seção aborda o \\textbf{Voo do Drone e a Construção da Imagem Ortorretificada}, onde são documentados os parâmetros do voo do drone, "
                     "incluindo altitude, velocidade e trajetória, além dos detalhes dos sensores infravermelhos empregados.")
    
    text_3_1_pt = (" Nesta seção, apresentamos uma visão abrangente da área inspecionada. A \\textbf{Figura 1} exibe a \\textbf{ortofoto} montada a partir de todas as imagens capturadas "
                   "da região. A \\textbf{Figura 2} apresenta uma representação esquemática da área, destacando as localizações dos rastreadores e dos hotspots detectados. "
                   "A numeração dos rastreadores segue o padrão: da esquerda para a direita (de oeste para leste) e de cima para baixo (de norte para sul).")

    # Load processing statistics if available
    stats_file = os.path.join(current_dir, 'Output', 'report', 'report_images', 'stats.json')
    stats = None
    if os.path.exists(stats_file):
        with open(stats_file, "r") as file:
            stats = json.load(file)

    # Document setup with additional academic packages
    doc = pl.Document(documentclass="article", document_options='dvipsnames')
    doc.preamble.append(pl.Command('usepackage', options='utf8', arguments='inputenc'))
    doc.preamble.append(pl.Command('usepackage', options='brazil', arguments='babel'))
    doc.packages.append(pl.Package("graphicx"))
    doc.packages.append(pl.Package('placeins'))
    doc.packages.append(pl.Package('calc'))
    doc.packages.append(pl.Package('tikz'))
    doc.packages.append(pl.Package('xcolor'))
    doc.packages.append(pl.Package('fancyhdr'))
    doc.packages.append(pl.Package('subfig'))
    doc.packages.append(pl.Package('geometry'))


    # Add booktabs for a cleaner table layout
    doc.packages.append(pl.Package('booktabs'))

    # Adjust header and footer margins
    doc.preamble.append(NoEscape(r'\setlength{\headsep}{3cm}'))
    doc.preamble.append(NoEscape(r'\setlength{\footskip}{1cm}'))
    doc.preamble.append(NoEscape(r'\geometry{top=4cm}'))

    # Configure fancyhdr
    logo_path_fixed = aisol_logo_2_path.replace('\\', '/') # NECESSARY TO INVERT THE '/' IN LATEX
    doc.preamble.append(NoEscape(r'\fancyhead[L]{{\includegraphics[width=0.1\paperwidth]{{{}}}}}'.format(logo_path_fixed)))

    doc.preamble.append(NoEscape(r'\fancyhead[R]{Relatóriio Termográfico}'))
    doc.preamble.append(NoEscape(r'\fancyfoot[C]{GreTA®, Versão Beta - 2025 \quad Desenvolvido por Aisol}'))
    doc.preamble.append(NoEscape(r'\usetikzlibrary{calc}'))

    # Title page with TikZ content from external file
    tikz_template = os.path.join(current_dir, "report_builder", "tikz_code.txt")
    with open(tikz_template, "r", encoding="utf-8") as f:
        tikz_code = f.read()
    doc.append(NoEscape(tikz_code))
    doc.append(NoEscape(r'\newpage'))

    # Title & basic report details
    doc.append(NoEscape(r'\thispagestyle{empty}'))
    doc.append(NoEscape(r'\vspace*{0.4cm}'))
    doc.append(NoEscape(r'\rule{\linewidth}{0.5pt}'))
    doc.append(NoEscape(r'\begin{center}'))
    doc.append(NoEscape(r'{\large\bfseries Relatório de Inspeção por Imagem Térmica.  }'))
    doc.append(NoEscape(r'\vspace*{0.5cm}'))
    doc.append(NoEscape(r'\textbf{Responsável Técnico:} ANONIMIZADO, Engineer.  '))
    doc.append(NoEscape(r'\textbf{CREA:} 12345678  '))
    doc.append(NoEscape(r'\textbf{{Date:}} {}'.format(current_date.strftime("%B %Y"))))
    doc.append(NoEscape(r'\textbf{Localização:} Campo Grande, MS. Brasil.  '))
    doc.append(NoEscape(r'\textbf{Endereço:} Rua Manoel Inácio de Souza, n. 24, C.E.P : 79.020-220  '))
    doc.append(NoEscape(r'\textbf{Software:} GreTA® - Georeferenced Thermographic Analysis System, Versão Beta.  '))
    doc.append(NoEscape(r'\textbf{Versão:} Versão ANONIMIZADA  '))
    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\vspace*{0.4cm}'))
    doc.append(NoEscape(r'\rule{\linewidth}{0.5pt}'))
    doc.append(NoEscape(r'\vfill'))
    doc.append(NoEscape(r'\noindent\textbf{Copyright © 2025 Aisol Soluções em Inteligência Artificial.  }'))
    doc.append(NoEscape(r'Todos os direitos reservados. Nenhuma parte desta publicação pode ser reproduzida, distribuída ou transmitida sem autorização prévia.  '))
    doc.append(NoEscape(r'\vspace*{0.2cm}'))
    doc.append(NoEscape(r'\noindent\textbf{ISBN:} xxxxxxxx.  '))



    
    # Additional report data
    doc.append(bold("Location:"))
    doc.append("Campo Grande, MS. Brasil.  ")
    doc.append("Rua Manoel Inácio de Souza, n. 24.  ")
    doc.append("C.E.P : 79.020-220.  ")
    doc.append(bold("Copyrights:"))
    doc.append("Aisol, 2023.  ")
    doc.append(bold("Release:"))
    doc.append("VERSAO ANONIMIZADA.  ")
    doc.append(bold("Company:"))
    doc.append("Aisol Soluções em Inteligência Artificial em parceria com PVX Engenharia.  ")
    doc.append(NoEscape(r'\newpage'))
            
    # Table of contents
    doc.append(NoEscape(r'\tableofcontents'))
    doc.append(NoEscape(r'\newpage'))

    # Abstract page
    doc.append(NoEscape(r'\newpage'))
    doc.append(NoEscape(r'\begin{abstract}'))
    doc.append(NoEscape(abstract))
    doc.append(NoEscape(r'\end{abstract}'))
    doc.append(NoEscape(r'\pagestyle{fancy}'))

    # Main Content Sections
    doc.append(NoEscape(r'\section{Introduction}'))
    doc.append(NoEscape(intro_text_pt))
    if stats:
        doc.append(NoEscape(drone_intro))
    doc.append(NoEscape(r'\section{Dados do Cliente}'))
    doc.append(client_data)
    doc.append(NoEscape(r'\section{Visão Geral da Área}'))
    doc.append(NoEscape(text_3_1_pt))
    doc.append(NoEscape(r'\FloatBarrier'))
    with doc.create(pl.Figure(position='h!')) as fig:
        fig.add_image(orthophoto_path_img, width=NoEscape(r'0.6\linewidth'))
        fig.add_caption("Ortofoto")

    doc.append(NoEscape(r'\FloatBarrier'))

    with doc.create(pl.Figure(position='h!')) as fig:
        fig.add_image(layer_img_path, width=NoEscape(r'0.6\linewidth'))
        fig.add_caption("Máscara dos Painéis")

    doc.append(NoEscape(r'\FloatBarrier'))


        # --- Defect Presentation by Issue Type ---
    expected_types = {
        "hotspots": "Pontos Quentes (Hot Spots)",
        "offlinepanels": "Painéis Desligados",
        "faultydiodes": "Diodos de Bypass Queimados"
    }


    # --- Defect Summary Table in academic style ---
    if defects_dict:
        rows_per_table = 35
        defects_items = list(defects_dict.items())
        total_rows = len(defects_items)

        for batch_idx in range(0, total_rows, rows_per_table):
            with doc.create(pl.Table(position='h!')) as table:
                caption = ("Resumo dos Defeitos Identificados" 
                        if batch_idx == 0 
                        else "Resumo dos Defeitos Identificados (cont.)")
                table.add_caption(caption)
                table.append(NoEscape(r'\centering'))  # Center the whole table
                with doc.create(pl.Tabular("lcl")) as tabular:  # Center the 2nd column
                    tabular.append(NoEscape(r'\toprule'))
                    tabular.add_row(["Tipo de Problema", "Local do Painel", "Coordenadas"], escape=False)
                    tabular.append(NoEscape(r'\midrule'))
                    for key, defect in defects_items[batch_idx:batch_idx + rows_per_table]:
                        parts = key.split("_")
                        if len(parts) >= 2:
                            local = parts[0]
                            issue = parts[1].lower()
                        else:
                            local = key
                            issue = defect["issue_type"].lower()
                        tipo_problema = expected_types.get(issue, issue)
                        tabular.add_row([tipo_problema, local, str('ANONIMIZADO')])
                    tabular.append(NoEscape(r'\bottomrule'))
            doc.append(NoEscape(r'\FloatBarrier'))
    else:
        doc.append("Nenhum defeito identificado.")












    # Group defects by (issue, local) so that each panel appears once per defect type.
    defects_by_type_local = {}
    for key, defect in defects_dict.items():
        parts = key.split("_")
        if len(parts) >= 2:
            local = parts[0]         # e.g., "1-2"
            issue = parts[1].lower()   # e.g., "hotspots"
        else:
            local = key
            issue = defect["issue_type"].lower()
        defects_by_type_local.setdefault((issue, local), []).append(defect)

    # Now, for each expected issue type, create a section.
    for issue in expected_types:
        doc.append(NoEscape(r'\newpage'))
        section_title = expected_types[issue]
        doc.append(NoEscape(r'\section{' + section_title + '}'))
        
        # Get all groups for this issue type.
        groups = [(local, defects) for (iss, local), defects in defects_by_type_local.items() if iss == issue]
        
        if groups:
            # Append introductory text for the section.
            if issue == "hotspots":
                intro_text = "Foram detectados pontos quentes nas placas abaixo."
            elif issue == "offlinepanels":
                intro_text = "Foram detectadas anomalias indicando painel(es) desligados nas placas abaixo."
            elif issue == "faultydiodes":
                intro_text = "Foram detectados diodos de bypass queimados nas placas abaixo."
            else:
                intro_text = ""
            doc.append(intro_text + "\n")
            
            # Process each panel group for this issue.
            for local, defects_list in groups:
                # Create a subsubsection for the panel.
                doc.append(NoEscape(r'\subsubsection{Painel ' + local + '}'))
                
                # Parse local string (expected "col-row") into column and row.
                try:
                    col, row = local.split("-")
                except Exception:
                    col, row = "?", "?"
                overall_caption = f"Imagens do Painel n. {row} da coluna n. {col}."
                
                # Build filenames using the naming convention.
                # E.g.: hotspots_(1-2)_layer.pdf, hotspots_(1-2)_cropped.jpg, hotspots_(1-2).jpg
                defect_map  = os.path.join(report_images_dir, f"{issue}_({local})_layer.pdf").replace("\\", "/")
                defect_crop = os.path.join(report_images_dir, f"{issue}_({local})_cropped.jpg").replace("\\", "/")
                drone_img   = os.path.join(report_images_dir, f"{issue}_({local}).jpg").replace("\\", "/")
                
                # Create a figure using subfloats.
                with doc.create(pl.Figure(position='h!')) as fig:
                    fig.append(NoEscape(r'\centering'))
                    fig.append(NoEscape(r'\subfloat[Recorte do Painel em Análise]{\includegraphics[width=0.31\linewidth]{' + defect_map + r'}}'))
                    fig.append(NoEscape(r'\hfill'))
                    fig.append(NoEscape(r'\subfloat[Localização do Problema]{\includegraphics[width=0.31\linewidth]{' + defect_crop + r'}}'))
                    fig.append(NoEscape(r'\hfill'))
                    fig.append(NoEscape(r'\subfloat[Imagem Original do Drone]{\includegraphics[width=0.31\linewidth]{' + drone_img + r'}}'))
                    fig.append(NoEscape(r'\caption{' + overall_caption + r'}'))
                doc.append(NoEscape(r'\FloatBarrier'))
                
                # Add optional descriptive text per defect type.
                if issue == "hotspots":
                    doc.append(f"No painel {local}, há sinais de pontos quentes conforme as figuras acima.\n")
                elif issue == "offlinepanels":
                    doc.append(f"No painel {local}, foram detectadas anomalias indicando painel(es) desligados.\n")
                elif issue == "faultydiodes":
                    doc.append(f"No painel {local}, foram detectados diodos de bypass queimados.\n")
        else:
            # No defects found for this issue type.
            if issue == "hotspots":
                doc.append("Não foram encontrados problemas de pontos quentes na área inspecionada.\n")
            elif issue == "offlinepanels":
                doc.append("Não foram encontrados problemas de painéis desligados na área inspecionada.\n")
            elif issue == "faultydiodes":
                doc.append("Não foram encontrados problemas de diodos de bypass queimados na área inspecionada.\n")




    # --- Append additional statistics if available ---
    if stats:
        # Functions to create various statistics tables are defined below.
        def create_processing_stats_table(stats):
            processing_stats = stats['processing_statistics']
            data = [
                ['Feature Extraction', processing_stats['steps_times']['Feature Extraction']],
                ['Features Matching', processing_stats['steps_times']['Features Matching']],
                ['Tracks Merging', processing_stats['steps_times']['Tracks Merging']],
                ['Reconstruction', processing_stats['steps_times']['Reconstruction']],
                ['Total Time', processing_stats['steps_times']['Total Time']]
            ]
            table = pl.Tabular('lr')
            table.append(NoEscape(r'\toprule'))
            for row in data:
                table.add_row(row)
            table.append(NoEscape(r'\bottomrule'))
            with doc.create(pl.Center()) as centered:
                with centered.create(pl.Table(position='h!')) as table_with_caption:
                    table_with_caption.add_caption("Processing Statistics")
                    table_with_caption.append(NoEscape(r'\begin{center}'))
                    table_with_caption.append(table)
                    table_with_caption.append(NoEscape(r'\end{center}'))
            return centered

        def create_features_stats_table(stats):
            features_stats = stats['features_statistics']
            detected_features = features_stats['detected_features']
            reconstructed_features = features_stats['reconstructed_features']
            data = [
                ['Detected Features - Min', detected_features['min']],
                ['Detected Features - Max', detected_features['max']],
                ['Detected Features - Mean', detected_features['mean']],
                ['Detected Features - Median', detected_features['median']],
                ['Reconstructed Features - Min', reconstructed_features['min']],
                ['Reconstructed Features - Max', reconstructed_features['max']],
                ['Reconstructed Features - Mean', reconstructed_features['mean']],
                ['Reconstructed Features - Median', reconstructed_features['median']]
            ]
            table = pl.Tabular('lr')
            table.append(NoEscape(r'\toprule'))
            for row in data:
                table.add_row(row)
            table.append(NoEscape(r'\bottomrule'))
            with doc.create(pl.Center()) as centered:
                with centered.create(pl.Table(position='h!')) as table_with_caption:
                    table_with_caption.add_caption("Feature Statistics")
                    table_with_caption.append(NoEscape(r'\begin{center}'))
                    table_with_caption.append(table)
                    table_with_caption.append(NoEscape(r'\end{center}'))
            return centered

        def create_reconstruction_stats_table(stats):
            reconstruction_stats = stats['reconstruction_statistics']
            data = [
                ['Components', reconstruction_stats['components']],
                ['Has GPS', reconstruction_stats['has_gps']],
                ['Initial Points Count', reconstruction_stats['initial_points_count']],
                ['Reconstructed Points Count', reconstruction_stats['reconstructed_points_count']]
            ]
            table = pl.Tabular('lr')
            table.append(NoEscape(r'\toprule'))
            for row in data:
                table.add_row(row)
            table.append(NoEscape(r'\bottomrule'))
            with doc.create(pl.Center()) as centered:
                with centered.create(pl.Table(position='h!')) as table_with_caption:
                    table_with_caption.add_caption("Reconstruction Statistics")
                    table_with_caption.append(NoEscape(r'\begin{center}'))
                    table_with_caption.append(table)
                    table_with_caption.append(NoEscape(r'\end{center}'))
            return centered

        def create_gps_errors_table(stats):
            gps_errors = stats['gps_errors']
            data = [
                ['Mean X', gps_errors['mean']['x']],
                ['Mean Y', gps_errors['mean']['y']],
                ['Mean Z', gps_errors['mean']['z']],
                ['STD X', gps_errors['std']['x']],
                ['STD Y', gps_errors['std']['y']],
                ['STD Z', gps_errors['std']['z']],
                ['Error X', gps_errors['error']['x']],
                ['Error Y', gps_errors['error']['y']],
                ['Error Z', gps_errors['error']['z']],
                ['Average Error', gps_errors['average_error']],
                ['CE90', gps_errors['ce90']],
                ['LE90', gps_errors['le90']]
            ]
            table = pl.Tabular('lr')
            table.append(NoEscape(r'\toprule'))
            for row in data:
                table.add_row(row)
            table.append(NoEscape(r'\bottomrule'))
            with doc.create(pl.Center()) as centered:
                with centered.create(pl.Table(position='h!')) as table_with_caption:
                    table_with_caption.add_caption("GPS Errors")
                    table_with_caption.append(NoEscape(r'\begin{center}'))
                    table_with_caption.append(table)
                    table_with_caption.append(NoEscape(r'\end{center}'))
            return centered

        doc.append(NoEscape(r'\newpage'))
        doc.append(NoEscape(r'\appendix'))
        doc.append(NoEscape(r'\section{Drone and Flight Information}'))
        doc.append(NoEscape(r'\newcommand{\rotatedimage}[1]{\includegraphics[angle=90, width=0.8\textwidth]{#1}}'))
        top_view_fixed = top_view.replace('\\', '/')
        with doc.create(pl.Figure(position='h!')) as fig:
            fig.append(NoEscape(r'\rotatedimage{' + top_view_fixed + '}'))
        doc.append("Drone Flight Information.")
        create_processing_stats_table(stats)

        # Orthophoto Data section
        doc.append(NoEscape(r'\section{Orthophoto Data}'))
        with doc.create(pl.Figure(position='h!')) as fig:
            fig.add_image(match)
            fig.add_caption("Match Graph")
        doc.append("Descrição do gráfico.")
        create_gps_errors_table(stats)
        with doc.create(pl.Figure(position='h!')) as fig:
            fig.add_image(overlap)
            fig.add_caption("Overlap Graph")
        doc.append("Descrição do gráfico.")
        create_reconstruction_stats_table(stats)
        with doc.create(pl.Figure(position='h!')) as fig:
            fig.add_image(residual)
            fig.add_caption("Model Residual")
        doc.append("Descrição do gráfico.")
        create_features_stats_table(stats)

    # Return the full LaTeX source as a string
    return doc.dumps()

