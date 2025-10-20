"""Main report builder orchestrating PDF generation."""

from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import subprocess

import pylatex as pl
from pylatex.utils import NoEscape, bold

from ..models.defect import Panel
from ..models.report import ReportConfig
from ..config import constants
from ..utils.logger import get_logger
from ..utils.exceptions import ReportGenerationError

logger = get_logger(__name__)


class ReportBuilder:
    """Orchestrates thermographic report PDF generation."""

    def __init__(
        self,
        panel_grid: Dict[Tuple[int, int], Panel],
        images_dir: Path,
        config: ReportConfig,
        odm_stats: dict = None,
        odm_stats_dir: Path = None,
    ):
        """
        Initialize report builder.

        Args:
            panel_grid: Dictionary of panels with defects
            images_dir: Directory containing report images
            config: Report configuration
            odm_stats: ODM statistics dictionary (optional)
            odm_stats_dir: Directory containing ODM stats images (optional)
        """
        self.panel_grid = panel_grid
        self.images_dir = images_dir
        self.config = config
        self.odm_stats = odm_stats
        self.odm_stats_dir = odm_stats_dir

        # Count defects
        self.panels_with_defects = [p for p in panel_grid.values() if p.has_defects]
        self.total_defects = sum(p.defect_count for p in panel_grid.values())

        logger.info(
            f"Initialized report builder: {len(panel_grid)} panels, "
            f"{len(self.panels_with_defects)} with defects, {self.total_defects} total defects, "
            f"ODM stats: {'available' if odm_stats else 'not available'}"
        )

    def generate_tex(self, output_path: Path) -> Path:
        """
        Generate LaTeX file without compiling to PDF.

        Args:
            output_path: Path where .tex file should be saved

        Returns:
            Path to generated .tex file
        """
        logger.info(f"Generating LaTeX file: {output_path}")

        try:
            # Create LaTeX document
            doc = self._create_latex_document()

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.dumps())

            logger.info(f"Generated LaTeX file: {output_path.stat().st_size / 1_000:.1f} KB")
            return output_path

        except Exception as e:
            error_msg = f"Failed to generate LaTeX: {e}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e

    def generate_pdf(self, output_path: Path) -> Path:
        """
        Generate full-resolution PDF report.

        Args:
            output_path: Path to save PDF

        Returns:
            Path to generated PDF

        Raises:
            ReportGenerationError: If PDF generation fails
        """
        logger.info(f"Generating PDF report: {output_path}")

        try:
            # Generate LaTeX document
            doc = self._create_latex_document()

            # Compile to PDF
            tex_path = output_path.with_suffix(".tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(doc.dumps())

            logger.info(f"Generated LaTeX file: {tex_path}")

            # Compile twice for proper references
            self._compile_latex(tex_path)
            self._compile_latex(tex_path)

            logger.info(f"PDF report generated: {output_path.stat().st_size / 1_000_000:.1f} MB")
            return output_path

        except Exception as e:
            error_msg = f"Failed to generate PDF: {e}"
            logger.error(error_msg)
            raise ReportGenerationError(error_msg) from e

    def _create_latex_document(self) -> pl.Document:
        """Create LaTeX document with report content."""
        doc = pl.Document(documentclass="article", document_options="dvipsnames")

        # Preamble
        self._add_preamble(doc)

        # Title page
        self._add_title_page(doc)

        # Table of contents
        doc.append(NoEscape(r"\tableofcontents"))
        doc.append(NoEscape(r"\newpage"))

        # Abstract
        self._add_abstract(doc)

        # Enable fancy headers/footers
        doc.append(NoEscape(r"\pagestyle{fancy}"))

        # Section 1: Client data
        self._add_client_section(doc)

        # Section 2: Area overview
        self._add_area_overview(doc)

        # Section 3: Defect details
        self._add_defect_details(doc)

        # Appendix: Flight data and ODM statistics (if available)
        self._add_appendix(doc)

        return doc

    def _add_preamble(self, doc: pl.Document) -> None:
        """Add LaTeX preamble with packages and configuration."""
        doc.preamble.append(pl.Command("usepackage", options="utf8", arguments="inputenc"))
        doc.preamble.append(pl.Command("usepackage", options="brazil", arguments="babel"))
        doc.packages.append(pl.Package("graphicx"))
        doc.packages.append(pl.Package("placeins"))
        doc.packages.append(pl.Package("subfig"))
        doc.packages.append(pl.Package("fancyhdr"))
        doc.packages.append(pl.Package("geometry"))
        doc.packages.append(pl.Package("booktabs"))

        # Additional packages for fancy cover and appendix
        doc.packages.append(pl.Package("calc"))
        doc.packages.append(pl.Package("tikz"))
        doc.packages.append(pl.Package("xcolor"))
        doc.preamble.append(NoEscape(r"\usetikzlibrary{calc}"))

        # Geometry
        doc.preamble.append(NoEscape(r"\setlength{\headsep}{3cm}"))
        doc.preamble.append(NoEscape(r"\setlength{\footskip}{1cm}"))
        doc.preamble.append(NoEscape(r"\geometry{top=4cm}"))

        # Header and footer
        # Use relative path for LaTeX compilation in different container
        logo_path = "report_images/aisol_logo.png"
        doc.preamble.append(
            NoEscape(
                r"\fancyhead[L]{{\includegraphics[width=0.1\paperwidth]{" + logo_path + r"}}}"
            )
        )
        doc.preamble.append(NoEscape(r"\fancyhead[R]{Relatório Termográfico}"))
        doc.preamble.append(
            NoEscape(r"\fancyfoot[C]{GreTA®, Versão Beta - 2025 \quad Desenvolvido por Aisol}")
        )

    def _add_title_page(self, doc: pl.Document) -> None:
        """Add fancy TikZ cover page."""
        doc.append(NoEscape(r"\thispagestyle{empty}"))

        # Fancy TikZ cover with geometric shapes
        doc.append(NoEscape(r"""
\begin{tikzpicture}[overlay,remember picture]

% Rectangles
\shade[
left color=lightgray,
right color=NavyBlue!40,
transform canvas ={rotate around ={45:($(current page.north west)+(0,-6)$)}}]
($(current page.north west)+(0,-6)$) rectangle ++(9,1.5);

\shade[
left color=lightgray,
right color=lightgray!50,
rounded corners=0.75cm,
transform canvas ={rotate around ={45:($(current page.north west)+(.5,-10)$)}}]
($(current page.north west)+(0.5,-10)$) rectangle ++(15,1.5);

\shade[
left color=lightgray,
rounded corners=0.3cm,
transform canvas ={rotate around ={45:($(current page.north west)+(.5,-10)$)}}] ($(current page.north west)+(1.5,-9.55)$) rectangle ++(7,.6);

\shade[
left color=lightgray!80,
right color=blue!60,
rounded corners=0.4cm,
transform canvas ={rotate around ={45:($(current page.north)+(-1.5,-3)$)}}]
($(current page.north)+(-1.5,-3)$) rectangle ++(9,0.8);

\shade[
left color=RoyalBlue!80,
right color=blue!80,
rounded corners=0.9cm,
transform canvas ={rotate around ={45:($(current page.north)+(-3,-8)$)}}] ($(current page.north)+(-3,-8)$) rectangle ++(15,1.8);

\shade[
left color=lightgray,
right color=RoyalBlue,
rounded corners=0.9cm,
transform canvas ={rotate around ={45:($(current page.north west)+(4,-15.5)$)}}]
($(current page.north west)+(4,-15.5)$) rectangle ++(30,1.8);

\shade[
left color=RoyalBlue,
right color=Emerald,
rounded corners=0.75cm,
transform canvas ={rotate around ={45:($(current page.north west)+(13,-10)$)}}]
($(current page.north west)+(13,-10)$) rectangle ++(15,1.5);

\shade[
left color=lightgray,
rounded corners=0.3cm,
transform canvas ={rotate around ={45:($(current page.north west)+(18,-8)$)}}]
($(current page.north west)+(18,-8)$) rectangle ++(15,0.6);

\shade[
left color=lightgray,
rounded corners=0.4cm,
transform canvas ={rotate around ={45:($(current page.north west)+(19,-5.65)$)}}]
($(current page.north west)+(19,-5.65)$) rectangle ++(15,0.8);

\shade[
left color=RoyalBlue,
right color=red!80,
rounded corners=0.6cm,
transform canvas ={rotate around ={45:($(current page.north west)+(20,-9)$)}}]
($(current page.north west)+(20,-9)$) rectangle ++(14,1.2);

% Year
\draw[ultra thick,gray]
($(current page.center)+(5,2)$) -- ++(0,-3cm)
node[
midway,
left=0.25cm,
text width=5cm,
align=right,
black!75
]
{
{\fontsize{25}{30} \selectfont \bf """ + self.config.area_name + r""" \\[10pt]}
}
node[
midway,
right=0.25cm,
text width=6cm,
align=left,
RoyalBlue]
{
{\fontsize{72}{86.4} \selectfont """ + datetime.now().strftime("%Y") + r"""}
};

% Title
\node[align=center] at ($(current page.center)+(0,-5)$)
{
{\fontsize{60}{72} \selectfont {{Relatório Termográfico}}} \\[1cm]
{\fontsize{16}{19.2} \selectfont \textcolor{RoyalBlue}{ \bf Relatório Físico}}\\[3pt]};
\node[align=center] at ($(current page.center)+(0,-9.5)$)
{ Desenvolvido por:};
\node[align=center] at ($(current page.center)+(0,-11)$)
{\includegraphics[width=0.4\paperwidth]{report_images/aisol_logo.png}};

\end{tikzpicture}%
"""))
        doc.append(NoEscape(r"\newpage"))

        # Second title page with details
        doc.append(NoEscape(r"\thispagestyle{empty}"))
        doc.append(NoEscape(r"\vspace*{0.4cm}"))
        doc.append(NoEscape(r"\rule{\linewidth}{0.5pt}"))
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"{\large\bfseries Relatório de Inspeção por Imagem Térmica}\\"))
        doc.append(NoEscape(r"\vspace*{0.5cm}"))
        doc.append(NoEscape(f"\\textbf{{Responsável Técnico:}} {self.config.engineer_name}\\\\"))
        doc.append(NoEscape(f"\\textbf{{CREA:}} {self.config.crea_number}\\\\"))
        doc.append(NoEscape(f"\\textbf{{Data:}} {datetime.now().strftime('%B %Y')}\\\\"))
        doc.append(NoEscape(f"\\textbf{{Localização:}} {self.config.location}\\\\"))
        doc.append(NoEscape(f"\\textbf{{Endereço:}} {self.config.address}\\\\"))
        doc.append(NoEscape(r"\textbf{Software:} GreTA® - Sistema de Análise Termográfica\\"))
        doc.append(NoEscape(r"\end{center}"))
        doc.append(NoEscape(r"\rule{\linewidth}{0.5pt}"))
        doc.append(NoEscape(r"\newpage"))

    def _add_abstract(self, doc: pl.Document) -> None:
        """Add abstract page."""
        doc.append(NoEscape(r"\newpage"))
        doc.append(NoEscape(r"\begin{abstract}"))
        doc.append(NoEscape(constants.REPORT_ABSTRACT_PT))
        doc.append(NoEscape(r"\end{abstract}"))

    def _add_client_section(self, doc: pl.Document) -> None:
        """Add client data section."""
        with doc.create(pl.Section("Dados do Cliente")):
            doc.append(bold("Cliente: "))
            doc.append(f"{self.config.client_name}\n\n")
            doc.append(bold("Área: "))
            doc.append(f"{self.config.area_name}\n\n")
            doc.append(bold("Localização: "))
            doc.append(f"{self.config.location}\n\n")

    def _add_area_overview(self, doc: pl.Document) -> None:
        """Add area overview section with orthophoto images."""
        with doc.create(pl.Section("Visão Geral da Área")):
            doc.append(NoEscape(constants.AREA_OVERVIEW_TEXT_PT))

            # Orthophoto overview (use relative path)
            ortho_path = "report_images/ortho.png"
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(ortho_path, width=NoEscape(r"0.9\textwidth"))
                fig.add_caption("Ortofoto da área inspecionada")

            # Layer map (use relative path)
            layer_path = "report_images/layer_img.pdf"
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(layer_path, width=NoEscape(r"0.9\textwidth"))
                fig.add_caption("Mapa de rastreadores e defeitos detectados")

    def _add_defect_details(self, doc: pl.Document) -> None:
        """Add detailed section for each panel with defects."""
        with doc.create(pl.Section("Detalhes dos Defeitos")):
            doc.append(
                f"Total de {len(self.panels_with_defects)} painéis com defeitos identificados.\n\n"
            )

            for panel in sorted(self.panels_with_defects, key=lambda p: (p.column, p.row)):
                self._add_panel_defect_page(doc, panel)

    def _add_panel_defect_page(self, doc: pl.Document, panel: Panel) -> None:
        """Add defect detail page for a single panel."""
        with doc.create(pl.Subsection(f"Painel {panel.panel_id}")):
            # Summary
            doc.append(f"Localização: Coluna {panel.column}, Linha {panel.row}\n\n")
            doc.append(f"Total de defeitos: {panel.defect_count}\n\n")

            if panel.hotspots:
                doc.append(f"- Pontos quentes: {len(panel.hotspots)}\n\n")
            if panel.faulty_diodes:
                doc.append(f"- Diodos queimados: {len(panel.faulty_diodes)}\n\n")
            if panel.offline_panels:
                doc.append(f"- Painéis inativos: {len(panel.offline_panels)}\n\n")

            # Add images for each defect type
            for defect_type in ["hotspots", "faultydiodes", "offlinepanels"]:
                defects = getattr(panel, defect_type, [])
                if not defects:
                    continue

                # Three images: layer map, crop, and raw drone image (check existence using full path)
                layer_img_path = self.images_dir / f"{defect_type}_({panel.panel_id})_layer.pdf"
                crop_img_path = self.images_dir / f"{defect_type}_({panel.panel_id})_cropped.jpg"
                drone_img_path = self.images_dir / f"{defect_type}_({panel.panel_id}).jpg"

                # Use relative paths for LaTeX
                layer_img = f"report_images/{defect_type}_({panel.panel_id})_layer.pdf"
                crop_img = f"report_images/{defect_type}_({panel.panel_id})_cropped.jpg"
                drone_img = f"report_images/{defect_type}_({panel.panel_id}).jpg"

                with doc.create(pl.Figure(position="h!")) as fig:
                    fig.append(NoEscape(r"\centering"))

                    # Layer map
                    if layer_img_path.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Localização]{\includegraphics[width=0.31\linewidth]{"
                                + layer_img
                                + r"}}"
                            )
                        )
                        fig.append(NoEscape(r"\hfill"))

                    # Crop
                    if crop_img_path.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Detalhe]{\includegraphics[width=0.31\linewidth]{"
                                + crop_img
                                + r"}}"
                            )
                        )
                        fig.append(NoEscape(r"\hfill"))

                    # Drone image
                    if drone_img_path.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Imagem Térmica]{\includegraphics[width=0.31\linewidth]{"
                                + drone_img
                                + r"}}"
                            )
                        )

                    fig.add_caption(f"Defeito tipo {defect_type} no painel {panel.panel_id}")

            doc.append(NoEscape(r"\FloatBarrier"))

    def _compile_latex(self, tex_path: Path) -> None:
        """
        Compile LaTeX to PDF using pdflatex.

        Args:
            tex_path: Path to .tex file

        Raises:
            ReportGenerationError: If compilation fails
        """
        logger.info(f"Compiling LaTeX: {tex_path}")

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(tex_path)],
                cwd=tex_path.parent,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
            )

            if result.returncode != 0:
                logger.error(f"pdflatex stderr: {result.stderr}")
                raise ReportGenerationError(f"pdflatex failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            raise ReportGenerationError("pdflatex compilation timed out")
        except FileNotFoundError:
            raise ReportGenerationError(
                "pdflatex not found - ensure LaTeX is installed in the container"
            )

    def generate_lowres_pdf(self, output_path: Path, input_pdf: Path) -> Path:
        """
        Generate low-resolution version of PDF for faster downloads.

        Args:
            output_path: Path to save low-res PDF
            input_pdf: Path to full-resolution PDF

        Returns:
            Path to low-res PDF
        """
        logger.info(f"Generating low-resolution PDF: {output_path}")

        try:
            # Use Ghostscript to compress PDF
            result = subprocess.run(
                [
                    "gs",
                    "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.4",
                    "-dPDFSETTINGS=/screen",
                    "-dNOPAUSE",
                    "-dQUIET",
                    "-dBATCH",
                    f"-sOutputFile={output_path}",
                    str(input_pdf),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"Ghostscript failed: {result.stderr}")
                # Fallback: copy original
                import shutil

                shutil.copy(input_pdf, output_path)

            logger.info(f"Low-res PDF: {output_path.stat().st_size / 1_000_000:.1f} MB")
            return output_path

        except Exception as e:
            logger.warning(f"Failed to create low-res PDF: {e}, using original")
            import shutil

            shutil.copy(input_pdf, output_path)
            return output_path

    def _add_appendix(self, doc: pl.Document) -> None:
        """Add appendix with flight data and ODM statistics."""
        if not self.odm_stats or not self.odm_stats_dir:
            logger.info("Skipping appendix - ODM stats not available")
            return

        logger.info("Adding appendix with ODM statistics")

        doc.append(NoEscape(r"\newpage"))
        doc.append(NoEscape(r"\appendix"))

        # Appendix A: Drone and Flight Information
        doc.append(NoEscape(r"\section{Drone and Flight Information}"))

        # Add topview if available
        topview_path = self.odm_stats_dir / "topview.png"
        if topview_path.exists():
            # Copy topview to images directory for LaTeX access
            import shutil
            dest_path = self.images_dir / "topview.png"
            shutil.copy(topview_path, dest_path)

            doc.append(NoEscape(r"\newcommand{\rotatedimage}[1]{\includegraphics[angle=90, width=0.8\textwidth]{#1}}"))
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.append(NoEscape(r"\rotatedimage{report_images/topview.png}"))
                fig.add_caption("Drone Flight Path")

        doc.append("Drone Flight Information.")

        # Processing Statistics Table
        if "processing_statistics" in self.odm_stats:
            self._add_processing_stats_table(doc, self.odm_stats["processing_statistics"])

        # Appendix B: Orthophoto Data
        doc.append(NoEscape(r"\section{Orthophoto Data}"))

        # Matchgraph
        self._add_odm_image(doc, "matchgraph.png", "Feature Matching Graph")

        # GPS Errors Table
        if "gps_errors" in self.odm_stats:
            self._add_gps_errors_table(doc, self.odm_stats["gps_errors"])

        # Overlap diagram
        self._add_odm_image(doc, "overlap.png", "Image Overlap Diagram")

        # Reconstruction Statistics Table
        if "reconstruction_statistics" in self.odm_stats:
            self._add_reconstruction_stats_table(doc, self.odm_stats["reconstruction_statistics"])

        # Residual histogram
        self._add_odm_image(doc, "residual_histogram.png", "Model Residual Histogram")

        # Features Statistics Table
        if "features_statistics" in self.odm_stats:
            self._add_features_stats_table(doc, self.odm_stats["features_statistics"])

    def _add_odm_image(self, doc: pl.Document, filename: str, caption: str) -> None:
        """Add ODM visualization image to document."""
        image_path = self.odm_stats_dir / filename
        if image_path.exists():
            # Copy to images directory
            import shutil
            dest_path = self.images_dir / filename
            shutil.copy(image_path, dest_path)

            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(f"report_images/{filename}", width=NoEscape(r"0.8\textwidth"))
                fig.add_caption(caption)
        else:
            logger.warning(f"ODM image not found: {filename}")

    def _add_processing_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add processing statistics table."""
        if "steps_times" not in stats:
            return

        times = stats["steps_times"]
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(r"Processing Step & Time \\"))
        doc.append(NoEscape(r"\midrule"))

        for step, time_val in times.items():
            # Convert time to readable format if it's a number
            if isinstance(time_val, (int, float)):
                time_str = f"{time_val:.2f}s"
            else:
                time_str = str(time_val)
            doc.append(NoEscape(f"{step} & {time_str} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{Processing Statistics}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

    def _add_gps_errors_table(self, doc: pl.Document, gps_errors: dict) -> None:
        """Add GPS errors table."""
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(r"GPS Error Metric & Value \\"))
        doc.append(NoEscape(r"\midrule"))

        if "mean" in gps_errors:
            for axis in ["x", "y", "z"]:
                if axis in gps_errors["mean"]:
                    val = gps_errors["mean"][axis]
                    doc.append(NoEscape(f"Mean {axis.upper()} & {val:.4f} \\\\"))

        if "std" in gps_errors:
            for axis in ["x", "y", "z"]:
                if axis in gps_errors["std"]:
                    val = gps_errors["std"][axis]
                    doc.append(NoEscape(f"STD {axis.upper()} & {val:.4f} \\\\"))

        if "error" in gps_errors:
            for axis in ["x", "y", "z"]:
                if axis in gps_errors["error"]:
                    val = gps_errors["error"][axis]
                    doc.append(NoEscape(f"Error {axis.upper()} & {val:.4f} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{GPS Errors}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

    def _add_reconstruction_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add reconstruction statistics table."""
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(r"Reconstruction Metric & Value \\"))
        doc.append(NoEscape(r"\midrule"))

        metrics = [
            ("components", "Components"),
            ("has_gps", "Has GPS"),
            ("initial_points_count", "Initial Points"),
            ("reconstructed_points_count", "Reconstructed Points"),
        ]

        for key, label in metrics:
            if key in stats:
                val = stats[key]
                doc.append(NoEscape(f"{label} & {val} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{Reconstruction Statistics}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

    def _add_features_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add features statistics table."""
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(r"Feature Metric & Value \\"))
        doc.append(NoEscape(r"\midrule"))

        if "detected_features" in stats:
            detected = stats["detected_features"]
            for metric in ["min", "max", "mean", "median"]:
                if metric in detected:
                    val = detected[metric]
                    doc.append(NoEscape(f"Detected Features - {metric.capitalize()} & {val:.0f} \\\\"))

        if "reconstructed_features" in stats:
            reconstructed = stats["reconstructed_features"]
            for metric in ["min", "max", "mean", "median"]:
                if metric in reconstructed:
                    val = reconstructed[metric]
                    doc.append(NoEscape(f"Reconstructed Features - {metric.capitalize()} & {val:.0f} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{Feature Statistics}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

