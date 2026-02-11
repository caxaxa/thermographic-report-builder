"""Main report builder orchestrating PDF generation."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import re
from datetime import datetime
import subprocess

import pylatex as pl
from pylatex.utils import NoEscape, bold, escape_latex

from ..models.defect import Panel
from ..models.report import ReportConfig
from ..models.annotation_manifest import AnnotationManifest
from ..config.constants import (
    DEFECT_COLORS,
    get_defect_labels,
    get_report_abstract,
    get_report_intro,
    get_area_overview_text,
    get_report_text,
    get_title_page_text,
    get_defect_details_text,
    get_summary_table_text,
    get_thermal_legend_text,
    get_appendix_text,
    get_preview_text,
)
from ..utils.logger import get_logger
from ..utils.exceptions import ReportGenerationError
from ..processing.gps_matcher import DefectMatch

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
        defect_matches: Optional[Dict[str, DefectMatch]] = None,
        annotation_manifest: Optional[AnnotationManifest] = None,
        preview: bool = False,
    ):
        """
        Initialize report builder.

        Args:
            panel_grid: Dictionary of panels with defects
            images_dir: Directory containing report images
            config: Report configuration
            odm_stats: ODM statistics dictionary (optional)
            odm_stats_dir: Directory containing ODM stats images (optional)
            defect_matches: Dictionary of defect matches with temperature data (optional)
            annotation_manifest: Annotation manifest with temperature data for thermal images
            preview: If True, generate a preview PDF with watermark and suppressed content
        """
        self.preview = preview
        self.panel_grid = panel_grid
        self.images_dir = images_dir
        self.config = config
        self.odm_stats = odm_stats
        self.odm_stats_dir = odm_stats_dir
        self.defect_matches = defect_matches or {}
        self.annotation_manifest = annotation_manifest

        # Count defects
        self.panels_with_defects = [p for p in panel_grid.values() if p.has_defects]
        self.total_defects = sum(p.defect_count for p in panel_grid.values())
        self.total_hotspots = sum(len(p.hotspots) for p in panel_grid.values())
        self.total_faulty_diodes = sum(len(p.faulty_diodes) for p in panel_grid.values())
        self.total_offline = sum(len(p.offline_panels) for p in panel_grid.values())

        # Count temperature data
        self.temp_data_count = sum(
            1 for m in self.defect_matches.values()
            if m.temperature is not None
        )

        logger.info(
            f"Initialized report builder: {len(panel_grid)} panels, "
            f"{len(self.panels_with_defects)} with defects, {self.total_defects} total defects, "
            f"ODM stats: {'available' if odm_stats else 'not available'}, "
            f"Temperature data: {self.temp_data_count} defects"
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

    def _panel_sort_key(self, panel: Panel) -> Tuple[int, int, str]:
        panel_id = panel.panel_id or ""
        # Handle both "1-8A" and "Painel 1-8A" formats
        match = re.match(r"^(?:Painel\s+)?(\d+)-(\d+)([A-Za-z]*)$", panel_id)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            suffix = match.group(3) or ""
            return (first, second, suffix)

        suffix = panel.double_suffix or (panel.tracker if panel.tracker != "A" else "")
        if panel.is_horizontal:
            return (panel.row, panel.column, suffix)
        return (panel.column, panel.tracker_column, suffix)

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

        # Section 1: Introduction
        self._add_introduction_section(doc)

        # Section 2: Client data
        self._add_client_section(doc)

        # Section 3: Area overview
        self._add_area_overview(doc)

        # Section 4: Defect summary table
        self._add_defect_summary_table(doc)

        # Section 5-7: Defect details by type
        self._add_defect_details_by_type(doc)

        # Appendix: Flight data and ODM statistics (if available)
        self._add_appendix(doc)

        return doc

    def _add_preamble(self, doc: pl.Document) -> None:
        """Add LaTeX preamble with packages and configuration."""
        doc.preamble.append(pl.Command("usepackage", options="utf8", arguments="inputenc"))
        # Set babel language based on config
        babel_lang = "english" if self.config.language in ("en-US", "en") else "brazil"
        doc.preamble.append(pl.Command("usepackage", options=babel_lang, arguments="babel"))
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
        doc.packages.append(pl.Package("colortbl"))  # For colored table cells
        doc.packages.append(pl.Package("array"))  # For column styling
        doc.preamble.append(NoEscape(r"\usetikzlibrary{calc}"))

        # Geometry
        doc.preamble.append(NoEscape(r"\setlength{\headsep}{3cm}"))
        doc.preamble.append(NoEscape(r"\setlength{\footskip}{1cm}"))
        doc.preamble.append(NoEscape(r"\geometry{top=4cm}"))

        # Watermark for preview mode
        if self.preview:
            doc.packages.append(pl.Package("background"))
            doc.preamble.append(NoEscape(
                r"\backgroundsetup{"
                r"scale=1,"
                r"color=black,"
                r"opacity=0.08,"
                r"angle=0,"
                r"position=current page.center,"
                r"contents={\includegraphics[width=0.5\paperwidth]{report_images/aisol_logo.png}}"
                r"}"
            ))

        # Header and footer
        # Use relative path for LaTeX compilation in different container
        logo_path = "report_images/aisol_logo.png"
        doc.preamble.append(
            NoEscape(
                r"\fancyhead[L]{{\includegraphics[width=0.1\paperwidth]{" + logo_path + r"}}}"
            )
        )
        report_title = get_report_text("report_title", self.config.language)
        footer_text = get_report_text("footer_text", self.config.language)
        doc.preamble.append(NoEscape(r"\fancyhead[R]{" + report_title + r"}"))
        doc.preamble.append(
            NoEscape(r"\fancyfoot[C]{" + footer_text + r"}")
        )

    def _add_title_page(self, doc: pl.Document) -> None:
        """Add fancy TikZ cover page."""
        doc.append(NoEscape(r"\thispagestyle{empty}"))

        # Get localized texts
        report_title = get_report_text("report_title", self.config.language)
        subtitle_key = "report_subtitle_preview" if self.preview else "report_subtitle"
        report_subtitle = get_title_page_text(subtitle_key, self.config.language)
        developed_by = get_title_page_text("developed_by", self.config.language)

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
{\fontsize{25}{30} \selectfont \bf """ + escape_latex(self.config.area_name) + r""" \\[10pt]}
}
node[
midway,
right=0.25cm,
text width=6cm,
align=left,
RoyalBlue]
{
{\fontsize{72}{86.4} \selectfont 2026}
};

% Title
\node[align=center] at ($(current page.center)+(0,-5)$)
{
{\fontsize{60}{72} \selectfont {{""" + report_title + r"""}}} \\[1cm]
{\fontsize{16}{19.2} \selectfont \textcolor{RoyalBlue}{ \bf """ + report_subtitle + r"""}}\\[3pt]};
\node[align=center] at ($(current page.center)+(0,-9.5)$)
{ """ + developed_by + r"""};
\node[align=center] at ($(current page.center)+(0,-11)$)
{\includegraphics[width=0.2\paperwidth]{report_images/aisol_logo.png}};

\end{tikzpicture}%
"""))
        doc.append(NoEscape(r"\newpage"))

        # Second title page with details - get localized texts
        thermal_report = get_title_page_text("thermal_inspection_report", self.config.language)
        tech_responsible = get_title_page_text("technical_responsible", self.config.language)
        date_label = get_title_page_text("date", self.config.language)
        location_label = get_title_page_text("location", self.config.language)
        address_label = get_title_page_text("address", self.config.language)
        software_label = get_title_page_text("software", self.config.language)
        software_desc = get_title_page_text("software_desc", self.config.language)
        copyright_text = get_title_page_text("copyright", self.config.language)
        rights_text = get_title_page_text("rights_reserved", self.config.language)
        isbn_label = get_title_page_text("isbn", self.config.language)
        isbn_value = get_title_page_text("isbn_value", self.config.language)

        doc.append(NoEscape(r"\thispagestyle{empty}"))
        doc.append(NoEscape(r"\vspace*{0.4cm}"))
        doc.append(NoEscape(r"\rule{\linewidth}{0.5pt}"))
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"{\large\bfseries " + thermal_report + r"}\\"))
        doc.append(NoEscape(r"\vspace*{0.5cm}"))
        doc.append(NoEscape(f"\\textbf{{{tech_responsible}}} {escape_latex(self.config.engineer_name)}\\\\"))
        doc.append(NoEscape(f"\\textbf{{CREA:}} {escape_latex(self.config.crea_number)}\\\\"))
        doc.append(NoEscape(f"\\textbf{{{date_label}}} {datetime.now().strftime('%B %Y')}\\\\"))
        doc.append(NoEscape(f"\\textbf{{{location_label}}} {escape_latex(self.config.location)}\\\\"))
        doc.append(NoEscape(f"\\textbf{{{address_label}}} {escape_latex(self.config.address)}\\\\"))
        doc.append(NoEscape(f"\\textbf{{{software_label}}} {software_desc}\\\\"))
        doc.append(NoEscape(r"\end{center}"))
        doc.append(NoEscape(r"\rule{\linewidth}{0.5pt}"))
        doc.append(NoEscape(r"\vfill"))
        doc.append(NoEscape(r"\noindent\textbf{" + copyright_text + r"}\\"))
        doc.append(NoEscape(rights_text + r"\\"))
        doc.append(NoEscape(r"\vspace*{0.2cm}"))
        doc.append(NoEscape(r"\noindent\textbf{" + isbn_label + r"} " + isbn_value + r"\\"))
        doc.append(NoEscape(r"\newpage"))

    def _add_abstract(self, doc: pl.Document) -> None:
        """Add abstract page."""
        doc.append(NoEscape(r"\newpage"))
        doc.append(NoEscape(r"\begin{abstract}"))
        doc.append(NoEscape(get_report_abstract(self.config.language)))
        doc.append(NoEscape(r"\end{abstract}"))

    def _add_introduction_section(self, doc: pl.Document) -> None:
        """Add introduction section with methodology and report structure."""
        intro_title = get_report_text("introduction_title", self.config.language)
        doc.append(NoEscape(r"\section{" + intro_title + r"}"))
        doc.append(NoEscape(get_report_intro(self.config.language)))

        # Add drone intro text if ODM stats are available
        if self.odm_stats:
            if self.config.language in ("en-US", "en"):
                drone_intro = (
                    "The final section covers the \\textbf{Drone Flight and Orthophoto Construction}, "
                    "documenting drone flight parameters including altitude, speed, and trajectory, "
                    "as well as details of the infrared sensors employed."
                )
            else:
                drone_intro = (
                    "A última seção aborda o \\textbf{Voo do Drone e a Construção da Imagem Ortorretificada}, "
                    "onde são documentados os parâmetros do voo do drone, incluindo altitude, velocidade e trajetória, "
                    "além dos detalhes dos sensores infravermelhos empregados."
                )
            doc.append(NoEscape(drone_intro))

    def _add_client_section(self, doc: pl.Document) -> None:
        """Add client data section."""
        section_title = get_report_text("client_data_title", self.config.language)
        client_label = get_report_text("client_name", self.config.language)
        area_label = "Area" if self.config.language in ("en-US", "en") else "Área"
        location_label = get_report_text("location", self.config.language)

        with doc.create(pl.Section(section_title)):
            doc.append(bold(f"{client_label}: "))
            doc.append(f"{escape_latex(self.config.client_name)}\n\n")
            doc.append(bold(f"{area_label}: "))
            doc.append(f"{escape_latex(self.config.area_name)}\n\n")
            doc.append(bold(f"{location_label}: "))
            doc.append(f"{escape_latex(self.config.location)}\n\n")

    def _add_area_overview(self, doc: pl.Document) -> None:
        """Add area overview section with orthophoto images."""
        section_title = get_report_text("area_overview_title", self.config.language)
        defect_labels = get_defect_labels(self.config.language)

        with doc.create(pl.Section(section_title)):
            doc.append(NoEscape(get_area_overview_text(self.config.language)))

            # Summary table of total defects before detailed breakdown
            summary_rows = [
                (defect_labels.get("hotspots", "Hotspots"), self.total_hotspots),
                (defect_labels.get("faulty_diodes", "Faulty Bypass Diodes"), self.total_faulty_diodes),
                (defect_labels.get("offline_panels", "Offline Panels"), self.total_offline),
            ]

            summary_caption = get_report_text("defect_summary_title", self.config.language)
            defect_type_header = get_report_text("defect_type", self.config.language)
            quantity_header = get_report_text("quantity", self.config.language)

            with doc.create(pl.Table(position="h!")) as table:
                table.add_caption(summary_caption)
                table.append(NoEscape(r"\centering"))
                with doc.create(pl.Tabular("lc")) as tabular:
                    tabular.append(NoEscape(r"\toprule"))
                    tabular.add_row([defect_type_header, quantity_header], escape=False)
                    tabular.append(NoEscape(r"\midrule"))
                    for label, count in summary_rows:
                        tabular.add_row([label, str(count)], escape=False)
                    tabular.append(NoEscape(r"\bottomrule"))

            doc.append(NoEscape(r"\FloatBarrier"))

            # Orthophoto overview (use relative path)
            # Use both width and height constraints with keepaspectratio to prevent overflow
            # Available text height after header/footer is approximately 0.65\textheight
            ortho_path = "report_images/ortho.png"
            with doc.create(pl.Figure(position="h!")) as fig:
                # Constrain to 0.85 textwidth and 0.55 textheight, keeping aspect ratio
                fig.append(NoEscape(
                    r"\centering\includegraphics[width=0.85\textwidth,height=0.55\textheight,keepaspectratio]{"
                    + ortho_path + r"}"
                ))
                ortho_caption = "Orthophoto of the inspected area" if self.config.language in ("en-US", "en") else "Ortofoto da área inspecionada"
                fig.add_caption(ortho_caption)

            # Layer map (use relative path)
            layer_path = "report_images/layer_img.pdf"
            with doc.create(pl.Figure(position="h!")) as fig:
                # Same constraints for layer map
                fig.append(NoEscape(
                    r"\centering\includegraphics[width=0.85\textwidth,height=0.55\textheight,keepaspectratio]{"
                    + layer_path + r"}"
                ))
                layer_caption = "Map of trackers and detected defects" if self.config.language in ("en-US", "en") else "Mapa de rastreadores e defeitos detectados"
                fig.add_caption(layer_caption)

    def _add_defect_summary_table(self, doc: pl.Document) -> None:
        """Add summary table of all detected defects."""
        if not self.panels_with_defects:
            return

        if self.preview:
            suppressed_text = get_preview_text("table_suppressed", self.config.language)
            doc.append(NoEscape(r"\vspace{1cm}"))
            doc.append(NoEscape(r"\begin{center}"))
            doc.append(NoEscape(r"{\large\textit{" + suppressed_text + r"}}"))
            doc.append(NoEscape(r"\end{center}"))
            doc.append(NoEscape(r"\vspace{1cm}"))
            return

        # Get localized defect labels
        hotspots_label = get_defect_details_text("hotspots_title", self.config.language)
        faulty_diodes_label = get_defect_details_text("faulty_diodes_title", self.config.language)
        offline_panels_label = get_defect_details_text("offline_panels_title", self.config.language)

        # Flatten panel defects into list of (defect_type_order, panel_key, defect_label, panel_id, coords)
        # Order: by defect class, then by panel identifier order
        defect_rows = []
        defect_type_order = {
            "hotspots": (0, hotspots_label),
            "faulty_diodes": (1, faulty_diodes_label),
            "offline_panels": (2, offline_panels_label),
        }

        for panel in self.panels_with_defects:
            panel_key = self._panel_sort_key(panel)
            for defect_type_attr, (order, defect_label) in defect_type_order.items():
                defects = getattr(panel, defect_type_attr, [])
                for defect in defects:
                    coords = defect.panel_centroid_geospatial
                    coords_str = f"({coords.latitude:.6f}, {coords.longitude:.6f})"
                    defect_rows.append((order, panel_key, defect_label, panel.panel_id, coords_str))

        # Sort by: defect class (order), then panel order
        defect_rows.sort(key=lambda x: (x[0], x[1]))

        # Extract display fields (defect_label, panel_id, coords_str)
        defect_rows = [(r[2], r[3], r[4]) for r in defect_rows]

        # Get localized table texts
        caption_main = get_summary_table_text("defect_location_title", self.config.language)
        caption_cont = get_summary_table_text("defect_location_cont", self.config.language)
        problem_type = get_summary_table_text("problem_type", self.config.language)
        panel_location = get_summary_table_text("panel_location", self.config.language)
        coordinates = get_summary_table_text("coordinates", self.config.language)

        # Create tables with max 35 rows each (for pagination)
        rows_per_table = 35
        total_rows = len(defect_rows)

        for batch_idx in range(0, total_rows, rows_per_table):
            batch = defect_rows[batch_idx:batch_idx + rows_per_table]

            with doc.create(pl.Table(position="h!")) as table:
                caption = caption_main if batch_idx == 0 else caption_cont
                table.add_caption(caption)
                table.append(NoEscape(r"\centering"))

                with doc.create(pl.Tabular("lcl")) as tabular:
                    tabular.append(NoEscape(r"\toprule"))
                    tabular.add_row([problem_type, panel_location, coordinates], escape=False)
                    tabular.append(NoEscape(r"\midrule"))

                    for defect_label, panel_id, coords_str in batch:
                        tabular.add_row([defect_label, panel_id, coords_str])

                    tabular.append(NoEscape(r"\bottomrule"))

            doc.append(NoEscape(r"\FloatBarrier"))

    def _add_defect_details_by_type(self, doc: pl.Document) -> None:
        """Add three dedicated sections for each defect type."""
        # Get localized texts for each defect type
        defect_types = [
            ("hotspots",
             get_defect_details_text("hotspots_title", self.config.language),
             get_defect_details_text("hotspots_intro", self.config.language),
             get_defect_details_text("hotspots_panel_text", self.config.language),
             get_defect_details_text("hotspots_none", self.config.language)),
            ("faulty_diodes",
             get_defect_details_text("faulty_diodes_title", self.config.language),
             get_defect_details_text("faulty_diodes_intro", self.config.language),
             get_defect_details_text("faulty_diodes_panel_text", self.config.language),
             get_defect_details_text("faulty_diodes_none", self.config.language)),
            ("offline_panels",
             get_defect_details_text("offline_panels_title", self.config.language),
             get_defect_details_text("offline_panels_intro", self.config.language),
             get_defect_details_text("offline_panels_panel_text", self.config.language),
             get_defect_details_text("offline_panels_none", self.config.language)),
        ]

        for defect_attr, section_title, intro_text, panel_text_template, no_defects_text in defect_types:
            doc.append(NoEscape(r"\newpage"))
            doc.append(NoEscape(r"\section{" + section_title + "}"))

            # Get panels with this defect type
            panels_with_type = [p for p in self.panels_with_defects if getattr(p, defect_attr, [])]

            if panels_with_type:
                doc.append(intro_text + "\n\n")

                sorted_panels = sorted(panels_with_type, key=self._panel_sort_key)

                if self.preview:
                    # Show only the first panel, suppress the rest
                    self._add_panel_defect_by_type(doc, sorted_panels[0], defect_attr, panel_text_template)
                    if len(sorted_panels) > 1:
                        suppressed_text = get_preview_text("images_suppressed", self.config.language)
                        doc.append(NoEscape(r"\vspace{1cm}"))
                        doc.append(NoEscape(r"\begin{center}"))
                        doc.append(NoEscape(r"{\large\textit{" + suppressed_text + r"}}"))
                        doc.append(NoEscape(r"\end{center}"))
                        doc.append(NoEscape(r"\vspace{1cm}"))
                else:
                    for panel in sorted_panels:
                        self._add_panel_defect_by_type(doc, panel, defect_attr, panel_text_template)
            else:
                # No defects of this type
                doc.append(no_defects_text + "\n")

    def _add_panel_defect_by_type(self, doc: pl.Document, panel: Panel, defect_type: str, text_template: str) -> None:
        """Add subsubsection for a single panel with specific defect type."""
        # Get localized texts
        panel_subsection = get_defect_details_text("panel_subsection", self.config.language)
        doc.append(NoEscape(r"\subsubsection{" + panel_subsection.format(panel_id=panel.panel_id) + "}"))

        # Detailed caption
        col, row = panel.column, panel.row
        overall_caption_template = get_defect_details_text("panel_images_caption", self.config.language)
        overall_caption = overall_caption_template.format(row=row, col=col)

        # Get localized subfigure captions
        context_map = get_defect_details_text("context_map", self.config.language)
        problem_location = get_defect_details_text("problem_location", self.config.language)
        drone_image = get_defect_details_text("drone_image", self.config.language)

        # Build image paths
        # Note: GPS matcher uses defect_type without underscores (e.g., "faultydiodes")
        # but minimap and crop use underscores (e.g., "faulty_diodes")
        defect_type_no_underscore = defect_type.replace("_", "")

        minimap_img_path = self.images_dir / f"{defect_type}_({panel.panel_id})_minimap.pdf"
        crop_img_path = self.images_dir / f"{defect_type}_({panel.panel_id})_cropped.jpg"
        drone_img_path = self.images_dir / f"{defect_type_no_underscore}_({panel.panel_id}).jpg"
        annotated_img_path = self.images_dir / f"{defect_type_no_underscore}_({panel.panel_id})_annotated.jpg"

        # Relative paths for LaTeX
        minimap_img = f"report_images/{defect_type}_({panel.panel_id})_minimap.pdf"
        crop_img = f"report_images/{defect_type}_({panel.panel_id})_cropped.jpg"
        drone_img = f"report_images/{defect_type_no_underscore}_({panel.panel_id}).jpg"
        annotated_img = f"report_images/{defect_type_no_underscore}_({panel.panel_id})_annotated.jpg"

        # Create figure with subfloats - use 0.30 width to avoid overlapping captions
        with doc.create(pl.Figure(position="h!")) as fig:
            fig.append(NoEscape(r"\centering"))

            if minimap_img_path.exists():
                fig.append(NoEscape(r"\subfloat[" + context_map + r"]{\includegraphics[width=0.30\linewidth]{" + minimap_img + r"}}"))
                fig.append(NoEscape(r"\hfill"))

            if crop_img_path.exists():
                fig.append(NoEscape(r"\subfloat[" + problem_location + r"]{\includegraphics[width=0.30\linewidth]{" + crop_img + r"}}"))
                fig.append(NoEscape(r"\hfill"))

            if drone_img_path.exists():
                fig.append(NoEscape(r"\subfloat[" + drone_image + r"]{\includegraphics[width=0.30\linewidth]{" + drone_img + r"}}"))

            fig.append(NoEscape(r"\caption{" + overall_caption + r"}"))

        doc.append(NoEscape(r"\FloatBarrier"))
        doc.append(NoEscape(r"\vspace{0.5cm}"))  # Add spacing to prevent legend overlap

        # Add annotated thermal image(s) with temperature data if available
        # Check for multiple defect images first (defeitoN_annotated.jpg pattern)
        # then fall back to single annotated image
        annotated_images_found = []

        # Check for indexed defect images (defeito1, defeito2, etc.)
        for i in range(1, 20):  # Support up to 20 defects per panel
            indexed_img_path = self.images_dir / f"{defect_type_no_underscore}_({panel.panel_id})_defeito{i}_annotated.jpg"
            indexed_img_rel = f"report_images/{defect_type_no_underscore}_({panel.panel_id})_defeito{i}_annotated.jpg"
            if indexed_img_path.exists():
                annotated_images_found.append((indexed_img_path, indexed_img_rel, i))
            else:
                break  # Stop when we don't find the next index

        # If no indexed images found, check for single annotated image
        if not annotated_images_found and annotated_img_path.exists():
            annotated_images_found.append((annotated_img_path, annotated_img, None))

        # Get localized thermal analysis captions
        thermal_analysis = get_defect_details_text("thermal_analysis", self.config.language)
        thermal_analysis_defect = get_defect_details_text("thermal_analysis_defect", self.config.language)

        # Add all found annotated images with temperature legend tables
        for img_path, img_rel, defect_num in annotated_images_found:
            doc.append(NoEscape(r"\vspace{0.3cm}"))
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(img_rel, width=NoEscape(r"0.65\textwidth"))
                if defect_num is not None:
                    fig.add_caption(thermal_analysis_defect.format(panel_id=panel.panel_id, defect_num=defect_num))
                else:
                    fig.add_caption(thermal_analysis.format(panel_id=panel.panel_id))
            doc.append(NoEscape(r"\FloatBarrier"))

            # Add temperature legend table from annotation manifest
            self._add_thermal_legend_table(doc, panel.panel_id, defect_type_no_underscore, defect_num)

        # Add descriptive text
        panel_text = text_template.format(panel_id=panel.panel_id)
        doc.append(panel_text + "\n\n")

    def _add_thermal_legend_table(
        self,
        doc: pl.Document,
        panel_id: str,
        defect_type: str,
        defect_num: Optional[int],
    ) -> None:
        """
        Add a beautiful LaTeX legend table with temperature data below thermal image.

        Args:
            doc: LaTeX document
            panel_id: Panel identifier (e.g., "A-01")
            defect_type: Defect type without underscores (e.g., "hotspots")
            defect_num: Defect number (1-based) or None for single defect
        """
        if not self.annotation_manifest:
            return

        # Find matching annotation entry
        if defect_num is not None:
            defect_id = f"{defect_type}_({panel_id})_defeito{defect_num}"
        else:
            defect_id = f"{defect_type}_({panel_id})_defeito1"

        entry = None
        for ann in self.annotation_manifest.annotations:
            if ann.defect_id == defect_id:
                entry = ann
                break

        if not entry:
            logger.debug(f"No annotation entry found for {defect_id}")
            return

        # Severity color mapping for LaTeX
        severity_colors = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "green!60!black",
            "MINIMAL": "green",
        }
        severity_color = severity_colors.get(entry.severity, "black")

        # Get localized severity text
        severity_key = f"severity_{entry.severity.lower()}"
        severity_text = get_thermal_legend_text(severity_key, self.config.language)

        # Get localized table labels
        param_label = get_thermal_legend_text("parameter", self.config.language)
        value_label = get_thermal_legend_text("value", self.config.language)
        hotpoint_label = get_thermal_legend_text("hotpoint", self.config.language)
        reference_label = get_thermal_legend_text("reference", self.config.language)
        delta_t_label = get_thermal_legend_text("delta_t", self.config.language)
        classification_label = get_thermal_legend_text("classification", self.config.language)

        # Create academic-style legend table (matching Table 1 format)
        doc.append(NoEscape(r"\vspace{0.2cm}"))
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\small"))
        doc.append(NoEscape(r"\begin{tabular}{lc}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(r"\textbf{" + param_label + r"} & \textbf{" + value_label + r"} \\"))
        doc.append(NoEscape(r"\midrule"))
        doc.append(NoEscape(f"{hotpoint_label} & {entry.hot_point.temp:.1f}°C \\\\"))
        doc.append(NoEscape(f"{reference_label} & {entry.cold_point.temp:.1f}°C \\\\"))
        doc.append(NoEscape(f"{delta_t_label} & {entry.delta_t:.1f}°C \\\\"))
        doc.append(NoEscape(f"{classification_label} & \\textcolor{{{severity_color}}}{{\\textbf{{{severity_text}}}}} \\\\"))
        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\end{center}"))
        doc.append(NoEscape(r"\vspace{0.3cm}"))

    def _add_defect_details(self, doc: pl.Document) -> None:
        """Add detailed section for each panel with defects."""
        details_title = get_report_text("defect_details_title", self.config.language)
        total_panels_text = get_summary_table_text("total_panels_with_defects", self.config.language)
        with doc.create(pl.Section(details_title)):
            doc.append(
                total_panels_text.format(count=len(self.panels_with_defects)) + "\n\n"
            )

            for panel in sorted(self.panels_with_defects, key=self._panel_sort_key):
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

    def _add_appendix(self, doc: pl.Document) -> None:
        """Add appendix with flight data and orthophoto information."""
        logger.info("Adding appendix with flight and orthophoto information")

        doc.append(NoEscape(r"\newpage"))
        doc.append(NoEscape(r"\appendix"))

        if self.preview:
            suppressed_text = get_preview_text("appendix_suppressed", self.config.language)
            flight_info_title = get_appendix_text("flight_info_title", self.config.language)
            ortho_data_title = get_appendix_text("orthophoto_data_title", self.config.language)

            # Appendix A placeholder
            doc.append(NoEscape(r"\section{" + flight_info_title + "}"))
            doc.append(NoEscape(r"\vspace{1cm}"))
            doc.append(NoEscape(r"\begin{center}"))
            doc.append(NoEscape(r"{\large\textit{" + suppressed_text + r"}}"))
            doc.append(NoEscape(r"\end{center}"))

            # Appendix B placeholder
            doc.append(NoEscape(r"\section{" + ortho_data_title + "}"))
            doc.append(NoEscape(r"\vspace{1cm}"))
            doc.append(NoEscape(r"\begin{center}"))
            doc.append(NoEscape(r"{\large\textit{" + suppressed_text + r"}}"))
            doc.append(NoEscape(r"\end{center}"))
            return

        # Get localized texts
        flight_info_title = get_appendix_text("flight_info_title", self.config.language)
        flight_path_caption = get_appendix_text("flight_path_caption", self.config.language)
        dashboard_caption = get_appendix_text("dashboard_caption", self.config.language)
        flight_unavailable = get_appendix_text("flight_data_unavailable", self.config.language)
        flight_error = get_appendix_text("flight_viz_error", self.config.language)

        # Appendix A: Drone and Flight Information
        doc.append(NoEscape(r"\section{" + flight_info_title + "}"))

        # Try to generate flight visualizations from reconstruction.json
        try:
            from ..flight_viz.integration import generate_flight_appendix

            # images_dir = work_dir / "report_images", so images_dir.parent = work_dir
            work_dir = self.images_dir.parent

            # Check if reconstruction.json exists (downloaded in main.py STEP 7)
            reconstruction_path = work_dir / "reconstruction.json"
            if not reconstruction_path.exists():
                # Fallback to opensfm subdirectory (legacy location)
                reconstruction_path = work_dir / "opensfm" / "reconstruction.json"

            if not reconstruction_path.exists():
                raise FileNotFoundError(f"reconstruction.json not found at {reconstruction_path}")

            # Find orthophoto for background overlay (prefer raw ODM output)
            orthophoto_candidates = [
                work_dir / "odm_orthophoto_original.tif",
                work_dir / "odm_orthophoto.tif",
                work_dir / "orthophoto.tif",
            ]
            orthophoto_path = next(
                (path for path in orthophoto_candidates if path.exists()),
                None,
            )

            # Generate flight visualizations
            viz_paths, flight_stats = generate_flight_appendix(
                work_dir=work_dir,
                output_dir=self.images_dir,
                reconstruction_path=reconstruction_path,
                orthophoto_path=orthophoto_path,
                project_name=self.config.area_name,
            )

            # Add flight path visualization
            if "flight_path_static" in viz_paths:
                with doc.create(pl.Figure(position="h!")) as fig:
                    fig.add_image(
                        str(viz_paths["flight_path_static"].relative_to(self.images_dir.parent)),
                        width=NoEscape(r"0.75\textwidth")
                    )
                    fig.add_caption(flight_path_caption)

            # Add flight statistics table
            if flight_stats:
                self._add_flight_stats_table(doc, flight_stats)

            # Add combined dashboard
            if "dashboard" in viz_paths:
                with doc.create(pl.Figure(position="h!")) as fig:
                    fig.add_image(
                        str(viz_paths["dashboard"].relative_to(self.images_dir.parent)),
                        width=NoEscape(r"0.95\textwidth")
                    )
                    fig.add_caption(dashboard_caption)

        except FileNotFoundError as e:
            logger.warning(f"reconstruction.json not found, skipping flight visualizations: {e}")
            doc.append(flight_unavailable)
        except Exception as e:
            logger.error(f"Failed to generate flight visualizations: {e}", exc_info=True)
            doc.append(flight_error)

        # Legacy ODM stats (if available)
        if self.odm_stats and self.odm_stats_dir:
            # Processing Statistics Table
            if "processing_statistics" in self.odm_stats:
                self._add_processing_stats_table(doc, self.odm_stats["processing_statistics"])

        # Get localized texts for orthophoto section
        ortho_data_title = get_appendix_text("orthophoto_data_title", self.config.language)
        matchgraph_caption = get_appendix_text("matchgraph_caption", self.config.language)
        overlap_caption = get_appendix_text("overlap_caption", self.config.language)
        residual_caption = get_appendix_text("residual_histogram_caption", self.config.language)

        # Appendix B: Orthophoto Data
        doc.append(NoEscape(r"\section{" + ortho_data_title + "}"))

        # Matchgraph - Feature correspondence graph
        self._add_odm_image(doc, "matchgraph.png", matchgraph_caption)

        # Overlap diagram
        self._add_odm_image(doc, "overlap.png", overlap_caption)

        # Reconstruction Statistics Table
        if self.odm_stats and "reconstruction_statistics" in self.odm_stats:
            self._add_reconstruction_stats_table(doc, self.odm_stats["reconstruction_statistics"])

        # Residual histogram
        self._add_odm_image(doc, "residual_histogram.png", residual_caption)

        # Features Statistics Table
        if self.odm_stats and "features_statistics" in self.odm_stats:
            self._add_features_stats_table(doc, self.odm_stats["features_statistics"])

    def _add_odm_image(self, doc: pl.Document, filename: str, caption: str) -> None:
        """Add ODM visualization image to document."""
        image_path = self.odm_stats_dir / filename
        if image_path.exists():
            # Copy to images directory
            import shutil
            dest_path = self.images_dir / filename
            shutil.copy(image_path, dest_path)

            # Use smaller width for overlap image to fit page
            img_width = r"0.6\textwidth" if filename == "overlap.png" else r"0.8\textwidth"
            with doc.create(pl.Figure(position="h!")) as fig:
                fig.add_image(f"report_images/{filename}", width=NoEscape(img_width))
                fig.add_caption(caption)
        else:
            logger.warning(f"ODM image not found: {filename}")

    def _add_processing_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add processing statistics table."""
        if "steps_times" not in stats:
            return

        # Get localized texts
        processing_step = get_appendix_text("processing_step", self.config.language)
        time_label = get_appendix_text("time", self.config.language)
        caption = get_appendix_text("processing_stats_caption", self.config.language)

        times = stats["steps_times"]
        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(f"{processing_step} & {time_label} \\\\"))
        doc.append(NoEscape(r"\midrule"))

        for step, time_val in times.items():
            # Convert time to readable format if it's a number
            if isinstance(time_val, (int, float)):
                time_str = f"{time_val:.2f}s"
            else:
                time_str = str(time_val)
            # Escape LaTeX special characters in step names (underscores, etc.)
            step_escaped = escape_latex(str(step))
            time_escaped = escape_latex(str(time_str))
            doc.append(NoEscape(f"{step_escaped} & {time_escaped} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{" + caption + r"}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

    def _add_flight_stats_table(self, doc: pl.Document, flight_stats: dict) -> None:
        """Add flight statistics table."""
        # Get localized texts
        flight_metric = get_appendix_text("flight_metric", self.config.language)
        value_label = get_thermal_legend_text("value", self.config.language)
        flight_stats_caption = get_appendix_text("flight_stats_caption", self.config.language)
        total_images = get_appendix_text("total_images", self.config.language)
        flight_duration = get_appendix_text("flight_duration", self.config.language)
        total_distance = get_appendix_text("total_distance", self.config.language)
        min_altitude = get_appendix_text("min_altitude", self.config.language)
        max_altitude = get_appendix_text("max_altitude", self.config.language)
        mean_altitude = get_appendix_text("mean_altitude", self.config.language)
        coverage_area = get_appendix_text("coverage_area", self.config.language)
        mean_speed = get_appendix_text("mean_speed", self.config.language)
        mean_gsd = get_appendix_text("mean_gsd", self.config.language)

        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(f"{flight_metric} & {value_label} \\\\"))
        doc.append(NoEscape(r"\midrule"))

        # Total images
        if "total_images" in flight_stats:
            doc.append(NoEscape(f"{total_images} & {flight_stats['total_images']} \\\\"))

        # Flight duration
        if "flight_duration_min" in flight_stats and flight_stats["flight_duration_min"] != "N/A":
            doc.append(NoEscape(f"{flight_duration} & {flight_stats['flight_duration_min']} min \\\\"))

        # Total distance
        if "total_distance_km" in flight_stats:
            doc.append(NoEscape(f"{total_distance} & {flight_stats['total_distance_km']} km \\\\"))

        doc.append(NoEscape(r"\midrule"))

        # Altitude stats
        if "min_altitude_m" in flight_stats:
            doc.append(NoEscape(f"{min_altitude} & {flight_stats['min_altitude_m']} m \\\\"))
        if "max_altitude_m" in flight_stats:
            doc.append(NoEscape(f"{max_altitude} & {flight_stats['max_altitude_m']} m \\\\"))
        if "mean_altitude_m" in flight_stats:
            doc.append(NoEscape(f"{mean_altitude} & {flight_stats['mean_altitude_m']} m \\\\"))

        doc.append(NoEscape(r"\midrule"))

        # Coverage area
        if "coverage_area_ha" in flight_stats:
            doc.append(NoEscape(f"{coverage_area} & {flight_stats['coverage_area_ha']} ha \\\\"))

        # Speed stats (optional)
        if "mean_speed_ms" in flight_stats:
            doc.append(NoEscape(r"\midrule"))
            doc.append(NoEscape(f"{mean_speed} & {flight_stats['mean_speed_ms']} m/s \\\\"))
        elif "mean_speed_kmh" in flight_stats:
            doc.append(NoEscape(r"\midrule"))
            doc.append(NoEscape(f"{mean_speed} & {flight_stats['mean_speed_kmh']} km/h \\\\"))

        # GSD stats (optional)
        if "mean_gsd_cm" in flight_stats:
            doc.append(NoEscape(r"\midrule"))
            doc.append(NoEscape(f"{mean_gsd} & {flight_stats['mean_gsd_cm']} cm/px \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{" + flight_stats_caption + r"}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

        # GPS error table (componentwise - ODM format)
        if "mean_gps_error_x_m" in flight_stats:
            # Get localized GPS error texts
            gps_error_metric = get_appendix_text("gps_error_metric", self.config.language)
            gps_error_caption = get_appendix_text("gps_error_caption", self.config.language)
            mean_x = get_appendix_text("mean_x", self.config.language)
            mean_y = get_appendix_text("mean_y", self.config.language)
            mean_z = get_appendix_text("mean_z", self.config.language)
            std_x = get_appendix_text("std_x", self.config.language)
            std_y = get_appendix_text("std_y", self.config.language)
            std_z = get_appendix_text("std_z", self.config.language)
            error_3d_mean = get_appendix_text("error_3d_mean", self.config.language)
            error_3d_max = get_appendix_text("error_3d_max", self.config.language)

            doc.append(NoEscape(r"\vspace{1em}"))
            doc.append(NoEscape(r"\begin{center}"))
            doc.append(NoEscape(r"\begin{table}[H]"))
            doc.append(NoEscape(r"\centering"))
            doc.append(NoEscape(r"\begin{tabular}{ll}"))
            doc.append(NoEscape(r"\toprule"))
            doc.append(NoEscape(f"{gps_error_metric} & {value_label} \\\\"))
            doc.append(NoEscape(r"\midrule"))
            doc.append(NoEscape(f"{mean_x} & {flight_stats['mean_gps_error_x_m']} m \\\\"))
            doc.append(NoEscape(f"{mean_y} & {flight_stats['mean_gps_error_y_m']} m \\\\"))
            doc.append(NoEscape(f"{mean_z} & {flight_stats['mean_gps_error_z_m']} m \\\\"))
            doc.append(NoEscape(r"\midrule"))
            doc.append(NoEscape(f"{std_x} & {flight_stats['std_gps_error_x_m']} m \\\\"))
            doc.append(NoEscape(f"{std_y} & {flight_stats['std_gps_error_y_m']} m \\\\"))
            doc.append(NoEscape(f"{std_z} & {flight_stats['std_gps_error_z_m']} m \\\\"))
            doc.append(NoEscape(r"\midrule"))
            # Also show 3D errors for reference
            if "mean_gps_error_m" in flight_stats:
                doc.append(NoEscape(f"{error_3d_mean} & {flight_stats['mean_gps_error_m']} m \\\\"))
                if "max_gps_error_m" in flight_stats:
                    doc.append(NoEscape(f"{error_3d_max} & {flight_stats['max_gps_error_m']} m \\\\"))
            doc.append(NoEscape(r"\bottomrule"))
            doc.append(NoEscape(r"\end{tabular}"))
            doc.append(NoEscape(r"\caption{" + gps_error_caption + r"}"))
            doc.append(NoEscape(r"\end{table}"))
            doc.append(NoEscape(r"\end{center}"))

    def _add_reconstruction_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add reconstruction statistics table."""
        # Get localized texts
        recon_metric = get_appendix_text("reconstruction_metric", self.config.language)
        value_label = get_thermal_legend_text("value", self.config.language)
        recon_caption = get_appendix_text("reconstruction_stats_caption", self.config.language)
        components_label = get_appendix_text("components", self.config.language)
        has_gps_label = get_appendix_text("has_gps", self.config.language)
        initial_points_label = get_appendix_text("initial_points", self.config.language)
        recon_points_label = get_appendix_text("reconstructed_points", self.config.language)

        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(f"{recon_metric} & {value_label} \\\\"))
        doc.append(NoEscape(r"\midrule"))

        metrics = [
            ("components", components_label),
            ("has_gps", has_gps_label),
            ("initial_points_count", initial_points_label),
            ("reconstructed_points_count", recon_points_label),
        ]

        for key, label in metrics:
            if key in stats:
                val = stats[key]
                doc.append(NoEscape(f"{label} & {val} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{" + recon_caption + r"}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))

    def _add_features_stats_table(self, doc: pl.Document, stats: dict) -> None:
        """Add features statistics table."""
        # Get localized texts
        features_metric = get_appendix_text("features_metric", self.config.language)
        value_label = get_thermal_legend_text("value", self.config.language)
        features_caption = get_appendix_text("features_stats_caption", self.config.language)
        detected_features = get_appendix_text("detected_features", self.config.language)
        recon_features = get_appendix_text("reconstructed_features", self.config.language)
        minimum = get_appendix_text("minimum", self.config.language)
        maximum = get_appendix_text("maximum", self.config.language)
        mean = get_appendix_text("mean", self.config.language)
        median = get_appendix_text("median", self.config.language)

        doc.append(NoEscape(r"\begin{center}"))
        doc.append(NoEscape(r"\begin{table}[h!]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\begin{tabular}{lr}"))
        doc.append(NoEscape(r"\toprule"))
        doc.append(NoEscape(f"{features_metric} & {value_label} \\\\"))
        doc.append(NoEscape(r"\midrule"))

        metric_labels = {"min": minimum, "max": maximum, "mean": mean, "median": median}

        if "detected_features" in stats:
            detected = stats["detected_features"]
            for metric in ["min", "max", "mean", "median"]:
                if metric in detected:
                    val = detected[metric]
                    label = metric_labels[metric]
                    doc.append(NoEscape(f"{detected_features} - {label} & {val:.0f} \\\\"))

        if "reconstructed_features" in stats:
            reconstructed = stats["reconstructed_features"]
            for metric in ["min", "max", "mean", "median"]:
                if metric in reconstructed:
                    val = reconstructed[metric]
                    label = metric_labels[metric]
                    doc.append(NoEscape(f"{recon_features} - {label} & {val:.0f} \\\\"))

        doc.append(NoEscape(r"\bottomrule"))
        doc.append(NoEscape(r"\end{tabular}"))
        doc.append(NoEscape(r"\caption{" + features_caption + r"}"))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\end{center}"))
