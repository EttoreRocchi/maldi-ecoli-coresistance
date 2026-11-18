"""Shared publication style for every figure of the paper (main text and supplement).

Rules (apply to all figures):
- Theme: pubfig's Nature theme (font size 7 pt, tick labels and legends 6 pt, 0.6 pt axes, ticks out, no grid), set in
  Palatino, the face of the manuscript body, through TeX Gyre Pagella (converted once to TrueType, see _register_pagella);
  mathematics in the same face; panel letters 9 pt bold; legend text in ink, never in the series colour (the pubfig
  theme's linecolor default is overridden).
- Text on filled cells (heatmaps) is white or ink, whichever contrasts more with the cell (text_on).
- No titles inside figures: captions carry them. Panels carry only a bold capital letter (panel_label), as ASM journals use.
- Colours: the Nature (NPG) palette from pubfig, assigned by entity and fixed across all figures (ROLE, DRUG, LEARNER, SITE
  below). Every set of hues that shares a panel was checked with the dataviz validator (OKLab, Machado 2009 simulation,
  all pairs): drugs navy, red, teal pass CVD (worst 11.6, deutan) and the normal-vision floor; methods use
  near-black for binary relevance because grey or dark grey cannot be told from the NPG navy or teal. Controls that recede
  (bagged ensembles) are neutral greys distinguished by marker shape. Variants of one entity use the same hue with a
  different line style or a hollow marker (probability passing = chain hue, dashed or hollow), never a new hue.
- Thin marks: 1.2 pt lines, 0.8 pt error bars with small caps, markers about 3.5 to 4.5 pt; no top or right spines.
- Legends: frameless, outside the data, one per figure when panels share series; direct labels only where they replace
  a legend or name an outlier.
- Sequential magnitude = one hue light to dark (SEQ_CMAP, NPG navy); signed differences = navy, grey, red (DIV_CMAP)
  centred at 0.
- Size: figures are designed on a 7.2 / 5.0 / 3.5 in grid and save() scales them to the ASM print size (ASM_SPEC, two
  columns 174 mm, 600 dpi raster) before placing the panel letters, so that text keeps its point size.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubfig as pf
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# Figures use Palatino, the face of the manuscript body, through TeX Gyre Pagella (GUST Font License). Its OpenType files
# have CFF outlines, which matplotlib embeds in PDF with a font-type mismatch, so each face is converted once to TrueType
# outlines in a user cache (the fonts are not redistributed with the code).
PAGELLA_OTF = os.environ.get("PAPER_PAGELLA_OTF", "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-{}.otf")
FONT_CACHE = Path(os.environ.get("PAPER_FONT_CACHE", Path.home() / ".cache" / "maldi-ecoli-fonts"))


def _otf_to_ttf(src, dst):
    """Convert an OpenType CFF font to TrueType (quadratic) outlines with fontTools."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(str(src)); order = font.getGlyphOrder(); gs = font.getGlyphSet()
    glyf = newTable("glyf"); glyf.glyphOrder = order; glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs); gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True)); glyf[name] = pen.glyph()
    font["glyf"] = glyf; font["loca"] = newTable("loca")
    maxp = newTable("maxp"); maxp.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 0)
    maxp.maxZones = 1; font["maxp"] = maxp
    font["head"].glyphDataFormat = 0; font["head"].indexToLocFormat = 0
    font["post"].formatType = 2.0; font["post"].extraNames = []; font["post"].mapping = {}; font["post"].glyphOrder = order
    for t in ("CFF ", "VORG"):
        if t in font: del font[t]
    font.sfntVersion = "\000\001\000\000"; dst.parent.mkdir(parents=True, exist_ok=True); font.save(str(dst))


def _register_pagella():
    for style in ("regular", "bold", "italic", "bolditalic"):
        dst = FONT_CACHE / f"texgyrepagella-{style}.ttf"; src = Path(PAGELLA_OTF.format(style))
        try:
            if not dst.exists() and src.exists(): _otf_to_ttf(src, dst)
            if dst.exists(): font_manager.fontManager.addfont(str(dst))
        except Exception: pass


_register_pagella()
_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("TeX Gyre Pagella", "Palatino Linotype", "Palatino", "URW Palladio L", "P052") if f in _AVAILABLE), "DejaVu Serif")

ASM_SPEC = pf.FigureSpec(name="asm", font_family=FONT, single_column_mm=84.0, double_column_mm=174.0, default_raster_dpi=600)
pf.register_figure_spec("asm", ASM_SPEC)
DESIGN_TO_ASM = (ASM_SPEC.double_column_mm / 25.4) / 7.2      # design grid (7.2 in double column) to ASM print width

NPG = pf.get_palette("nature")   # #E64B35 red, #4DBBD5 light blue, #00A087 teal, #3C5488 navy, #F39B7F salmon, #8491B4 grey blue, #91D1C2 mint, #DC0000 deep red, #7E6148 brown, #B09C85 beige
NAVY, RED, TEAL, LIGHTBLUE, DEEPRED, BROWN = NPG[3], NPG[0], NPG[2], NPG[1], NPG[7], NPG[8]
INK = "#111111"        # primary text, axes, binary relevance
INK2 = "#4a4a4a"       # secondary text
MUTED = "#8c8c8c"      # muted text, bagged controls
LIGHT = "#bdbdbd"      # lighter neutral (second bagged control, reference bands)
GRID = "#ececec"
SURFACE = "#ffffff"

ROLE = {               # methods and upstream-input variants (fixed hues, validated together where they share a panel)
    "br": INK,                     # binary relevance (reference)
    "bagged_boot": MUTED,          # bagged BR, bootstrap rows (recessive control, triangle down)
    "bagged_sub": LIGHT,           # bagged BR, random halves of the bins (recessive control, triangle up)
    "ensemble": NAVY,              # chain ensemble, realized hard predictions
    "proba": NAVY,                 # probability passing: same hue, dashed line or hollow marker
    "best": RED,                   # best chain ordering (Fig. 2d and 2e; for CTX this is the CIP-first ordering)
    "cip_alone": RED,              # ordering in which the drug receives the CIP prediction alone (Fig. 2a to 2c)
    "oracle": DEEPRED,             # true upstream labels (ceiling), drawn as a line
    "permuted": BROWN,             # permuted upstream labels (no information)
    "stacking": TEAL,              # cross-fitted stacking
}
DRUG = {"CIP": NAVY, "CTX": RED, "CRO": RED, "CAZ": TEAL}   # all pairs pass CVD (worst 11.6) and the normal-vision floor
LEARNER = {"logistic": NAVY, "catboost": RED, "tabpfn": TEAL}
SITE_ORDER = ["s_orsola", "driams_b", "driams_c", "driams_d", "driams_a"]   # by cohort size
SITE = {"s_orsola": "#a9b6d8", "driams_b": "#7f90c0", "driams_c": "#5a6ea6", "driams_d": NAVY, "driams_a": "#23335a"}  # ordinal navy ramp by n
SITE_ACCENT = {"s_orsola": NAVY, "driams_b": DEEPRED}   # two target sites compared directly (CVD dE 17.8, normal 31.2)
SITE_MARKER = {"s_orsola": "o", "driams_b": "s", "driams_c": "^", "driams_d": "D", "driams_a": "v"}
SITE_NAME = {"s_orsola": "Sant’Orsola", "local": "Sant’Orsola", "driams_a": "DRIAMS-A", "driams_b": "DRIAMS-B", "driams_c": "DRIAMS-C", "driams_d": "DRIAMS-D"}
LEARNER_NAME = {"logistic": "Logistic regression", "catboost": "CatBoost", "tabpfn": "TabPFN"}
ROLE_NAME = {"br": "Binary relevance", "bagged_boot": "Bagged BR (bootstrap)", "bagged_sub": "Bagged BR (bins)", "ensemble": "Chain ensemble",
             "proba": "Chain, probability passing", "best": "Best chain", "oracle": "Oracle (true upstream label)", "permuted": "Permuted upstream label", "stacking": "Stacking"}

SEQ_CMAP = LinearSegmentedColormap.from_list("paper_navy", ["#f4f6fa", "#d3d9ea", "#a3b0d2", "#6f80b3", NAVY, "#1f2d4f"])
DIV_CMAP = LinearSegmentedColormap.from_list("paper_div", [NAVY, "#9aa6cb", "#f0efec", "#f0a495", RED])

RC = {
    **pf.get_theme("nature").rc_params(),
    "font.family": "serif", "font.serif": [FONT, "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "mathtext.cal": FONT, "mathtext.sf": FONT,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "legend.labelcolor": INK, "grid.color": GRID, "errorbar.capsize": 1.8, "lines.markersize": 4,
    "figure.dpi": 150, "savefig.dpi": ASM_SPEC.default_raster_dpi, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
}
WIDTH = {"single": 3.5, "onehalf": 5.0, "double": 7.2}   # design grid; save() scales by DESIGN_TO_ASM


def apply():
    matplotlib.rcParams.update(RC)


def figure(width="double", height=None, nrows=1, ncols=1, **kw):
    """Figure with a journal column width; height defaults to a 0.62 ratio per panel row."""
    apply(); w = WIDTH.get(width, width); h = height or (0.62 * w / max(ncols, 1) * nrows + 0.3)
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def panel_label(ax, letter, dx=-0.02, dy=1.0):
    """Bold capital panel letter (ASM style) at the top-left corner of the panel's tight bounding box (tick labels and axis
    labels included), placed by save() after the layout is final so that it never collides with a tick label."""
    ax._paper_panel_letter = letter.upper()


def place_panel_letters(fig, pad_pt=4.0):
    """Draw the stored panel letters at the top-left of each panel's tight bounding box (figure coordinates)."""
    axes = [ax for ax in fig.axes if getattr(ax, "_paper_panel_letter", None)]
    if not axes: return []
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); fw, fh = fig.get_size_inches()
    tops = {}
    for ax in axes:
        bb = ax.get_tightbbox(r)
        if bb is not None: tops[ax] = (inv.transform((bb.x0, bb.y1)), inv.transform(ax.get_window_extent(r).get_points())[1][1])
    letters = []
    for ax, ((x0, y1), frame_top) in tops.items():
        # letters of panels whose frames share a top edge (one row) sit at the same height: the highest tight top of the row
        row = [t[0][1] for t in tops.values() if abs(t[1] - frame_top) < 0.02]
        letters.append(fig.text(x0, max(row) + pad_pt / 72.0 / fh, ax._paper_panel_letter, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK))
        ax._paper_panel_letter = None
    return letters


def lift_figure_legends(fig, letters, pad_pt=4.0):
    """Move a figure-level legend up just enough to clear any panel letter it overlaps (legends anchored by legend_above)."""
    if not fig.legends or not letters: return
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); pad = pad_pt * fig.dpi / 72.0
    boxes = [t.get_window_extent(r) for t in letters]
    for leg in fig.legends:
        lb = leg.get_window_extent(r)
        shift = max([bb.y1 + pad - lb.y0 for bb in boxes if bb.x1 > lb.x0 and bb.x0 < lb.x1 and bb.y1 + pad > lb.y0 and bb.y0 < lb.y1] + [0.0])
        if shift > 0:
            a = leg.get_bbox_to_anchor().get_points()[0]
            leg.set_bbox_to_anchor(tuple(inv.transform((a[0], a[1] + shift))), transform=fig.transFigure)


def despine(ax, grid=None):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(True, axis=grid, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)


def legend_above(fig_or_ax, handles=None, labels=None, ncol=None, y=1.0, **kw):
    """One frameless legend in a single row above the plotting area (figure-level when a Figure is passed)."""
    import matplotlib.figure as mf
    if isinstance(fig_or_ax, mf.FigureBase):
        if handles is None: handles, labels = fig_or_ax.axes[0].get_legend_handles_labels()
        return fig_or_ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles), frameon=False, **kw)
    if handles is None: handles, labels = fig_or_ax.get_legend_handles_labels()
    return fig_or_ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, y), ncol=ncol or len(handles), frameon=False, **kw)


def ci_point(ax, x, y, lo=None, hi=None, color=None, marker="o", hollow=False, label=None, zorder=3, ms=4.5, **kw):
    """Point estimate with a thin interval (lo, hi are absolute bounds)."""
    color = INK if color is None else color
    if lo is not None and hi is not None:
        import numpy as _np
        yv, lov, hiv = (_np.atleast_1d(_np.asarray(v, float)) for v in (y, lo, hi))
        ax.errorbar(_np.atleast_1d(x), yv, yerr=_np.vstack([_np.clip(yv - lov, 0, None), _np.clip(hiv - yv, 0, None)]),
                    fmt="none", ecolor=color, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=zorder - 1)
    return ax.plot(x, y, linestyle="none", marker=marker, ms=ms, color=color, mfc=SURFACE if hollow else color, mec=color, mew=1.0, label=label, zorder=zorder, **kw)


def hline(ax, y=0.0, color=None, lw=0.7, ls="-", **kw):
    ax.axhline(y, color=INK2 if color is None else color, lw=lw, ls=ls, zorder=1, **kw)


def _rel_luminance(c):
    from matplotlib.colors import to_rgb
    r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(fill):
    """White or ink for text on a filled cell, whichever has the larger WCAG contrast ratio with the fill."""
    lf = _rel_luminance(fill)
    ratio = lambda c: (max(lf, _rel_luminance(c)) + 0.05) / (min(lf, _rel_luminance(c)) + 0.05)
    return SURFACE if ratio(SURFACE) > ratio(INK) else INK


def _typographic_names(fig):
    """One spelling of the site name in every text element (typographic apostrophe)."""
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if "Sant'Orsola" in s: t.set_text(s.replace("Sant'Orsola", "Sant\u2019Orsola"))


def save(fig, stem, formats=("png", "pdf")):
    """Save <stem>.png (600 dpi) and <stem>.pdf at ASM print size; stem may be a Path without suffix.

    The figure is scaled from the design grid to ASM_SPEC (two columns 174 mm) before the panel letters are placed, so
    that text keeps its point size. pubfig's save_figure is not used for writing because it re-runs tight_layout, which
    would undo the hand-placed legends and panel letters of the multi-panel figures."""
    stem = Path(str(stem)).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches(); fig.set_size_inches(w * DESIGN_TO_ASM, h * DESIGN_TO_ASM, forward=True)
    _typographic_names(fig)
    lift_figure_legends(fig, place_panel_letters(fig))
    for ext in formats: fig.savefig(f"{stem}.{ext}", dpi=ASM_SPEC.default_raster_dpi, bbox_inches="tight", pad_inches=0.02, facecolor=SURFACE)
    plt.close(fig); return stem
