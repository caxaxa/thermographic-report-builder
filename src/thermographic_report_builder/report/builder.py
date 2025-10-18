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
    ):
        """
        Initialize report builder.

        Args:
            panel_grid: Dictionary of panels with defects
            images_dir: Directory containing report images
            config: Report configuration
        """
        self.panel_grid = panel_grid
        self.images_dir = images_dir
        self.config = config

        # Count defects
        self.panels_with_defects = [p for p in panel_grid.values() if p.has_defects]
        self.total_defects = sum(p.defect_count for p in panel_grid.values())

        logger.info(
            f"Initialized report builder: {len(panel_grid)} panels, "
            f"{len(self.panels_with_defects)} with defects, {self.total_defects} total defects"
        )

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

        # Geometry
        doc.preamble.append(NoEscape(r"\setlength{\headsep}{3cm}"))
        doc.preamble.append(NoEscape(r"\setlength{\footskip}{1cm}"))
        doc.preamble.append(NoEscape(r"\geometry{top=4cm}"))

        # Header and footer
        logo_path = str(self.images_dir / "aisol_logo.png").replace("\\", "/")
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
        """Add title page."""
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

            # Orthophoto overview
            ortho_path = str(self.images_dir / "ortho.png").replace("\\", "/")
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(ortho_path, width=NoEscape(r"0.9\textwidth"))
                fig.add_caption("Ortofoto da área inspecionada")

            # Layer map
            layer_path = str(self.images_dir / "layer_img.pdf").replace("\\", "/")
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

                # Three images: layer map, crop, and raw drone image
                layer_img = self.images_dir / f"{defect_type}_({panel.panel_id})_layer.pdf"
                crop_img = self.images_dir / f"{defect_type}_({panel.panel_id})_cropped.jpg"
                drone_img = self.images_dir / f"{defect_type}_({panel.panel_id}).jpg"

                with doc.create(pl.Figure(position="h!")) as fig:
                    fig.append(NoEscape(r"\centering"))

                    # Layer map
                    if layer_img.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Localização]{\includegraphics[width=0.31\linewidth]{"
                                + str(layer_img).replace("\\", "/")
                                + r"}}"
                            )
                        )
                        fig.append(NoEscape(r"\hfill"))

                    # Crop
                    if crop_img.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Detalhe]{\includegraphics[width=0.31\linewidth]{"
                                + str(crop_img).replace("\\", "/")
                                + r"}}"
                            )
                        )
                        fig.append(NoEscape(r"\hfill"))

                    # Drone image
                    if drone_img.exists():
                        fig.append(
                            NoEscape(
                                r"\subfloat[Imagem Térmica]{\includegraphics[width=0.31\linewidth]{"
                                + str(drone_img).replace("\\", "/")
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
