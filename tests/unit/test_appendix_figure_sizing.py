"""Appendix figures must be bounded in BOTH dimensions.

2026-08-29: the "Trajetória de Voo" figure ran off the page on Bento's plant.
The flight-path PNG is as tall as the site is long (91 m x 389 m => 4.2:1), so
`width=0.75\\textwidth` alone renders it ~3x taller than the text block. Width
constrains a wide image; only a height cap constrains a tall one, and
keepaspectratio lets whichever limit binds first win.
"""

import re

import pylatex as pl
from pylatex.utils import NoEscape

from thermographic_report_builder.report import builder as builder_module

FIGURE_SRC = re.compile(
    r"add_image\(\s*str\(viz_paths\[\"(flight_path_static|dashboard)\"\]"
    r".*?width=NoEscape\(r\"([^\"]+)\"\)",
    re.DOTALL,
)


def _appendix_image_options() -> dict[str, str]:
    source = open(builder_module.__file__, encoding="utf-8").read()
    return {m.group(1): m.group(2) for m in FIGURE_SRC.finditer(source)}


def test_both_appendix_figures_are_height_capped():
    opts = _appendix_image_options()
    assert set(opts) == {"flight_path_static", "dashboard"}, opts
    for name, value in opts.items():
        assert "height=" in value, f"{name} has no height cap: {value}"
        assert "keepaspectratio" in value, f"{name} would be distorted: {value}"


def test_height_cap_leaves_room_for_the_caption():
    """A full \\textheight image plus a caption still overflows."""
    for name, value in _appendix_image_options().items():
        frac = float(re.search(r"height=([\d.]+)\\textheight", value).group(1))
        assert 0 < frac <= 0.8, f"{name} height {frac} leaves no room for a caption"


def test_options_survive_pylatex_untouched():
    """pylatex prefixes 'width=' and must not escape the rest of the key list."""
    fig = pl.Figure(position="h!")
    fig.add_image(
        "report_images/flight_path_static.png",
        width=NoEscape(r"0.75\textwidth,height=0.62\textheight,keepaspectratio"),
    )
    out = fig.dumps()
    assert (
        r"\includegraphics[width=0.75\textwidth,height=0.62\textheight,"
        r"keepaspectratio]{report_images/flight_path_static.png}" in out
    ), out


def test_a_tall_image_is_bound_by_height_not_width():
    """Sanity-check the geometry claim on Bento's 4.2:1 aspect."""
    aspect = 389.0 / 91.0  # height / width of the plant
    textwidth_in, textheight_in = 6.3, 8.9  # ~A4 with default margins
    unbounded_h = 0.75 * textwidth_in * aspect
    assert unbounded_h > textheight_in, "premise wrong: width alone would fit"
    bounded_h = min(unbounded_h, 0.62 * textheight_in)
    assert bounded_h <= 0.62 * textheight_in
