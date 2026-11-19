# Matplotlib style of OpenSurfaces' documentation figures: serif text and math, boxed axes with inward ticks,
# no grid, no titles, axis labels "description, symbol", italic panel labels (a), (b), lines with filled circle
# markers, dashed lines in the ink color for references (black on the light theme), legends without a frame. Colors
# by the data's job (QUAL qualitative, SEQ sequential, DIV diverging, FIELD fields, MESH the panel classes of a mesh).
#   import sys; sys.path.insert(0, <this folder>); import figstyle as fs; fs.apply()
# Docs mode (FIGSTYLE_DOCS=light or dark): a transparent background, black ink for the light theme and light gray
# for the dark one; `save` then writes docs/src/assets/<name>-light.png or <name>-dark.png (the site's
# CSS shows the one matching the theme).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DOCS = os.environ.get("FIGSTYLE_DOCS", "")       # "", "light" or "dark"
INK = "#d4d4d4" if DOCS == "dark" else "black"
GRAY = "0.5" if DOCS == "dark" else "0.55"
# Color by the data's job (one rule each); every role has its own steps for the dark docs theme:
#   qualitative (identity: cases, methods, mesh classes): QUAL, in this fixed order, never cycled — jewel tones
#     indigo, rose, teal, amber, violet, lime, sky, pink (checked: light on
#     white, worst adjacent CVD ΔE 10.2, normal-vision 24.3, all ≥ 3:1 contrast; dark on #1f2424, CVD 8.3,
#     normal-vision 21.8, all ≥ 3:1; the first three slots separate in every pairing in both modes);
#   sequential (an ordered parameter, a magnitude): SEQ, one hue (indigo) with monotone lightness, from the
#     strongest contrast with the background to the weakest; seq(n) samples it;
#   diverging (a signed quantity about a reference): DIV, teal → neutral → amber;
#   fields (a magnitude, or a signed field shown as one ordered scale): FIELD = DENSE, cmocean's "dense".
DARK = DOCS == "dark"
QUAL = (["#6875f5", "#f05470", "#14a394", "#c97a0a", "#9b6cf6", "#4d9e1a", "#2493d1", "#e7559c"] if DARK else
        ["#4f46e5", "#e11d48", "#0d9488", "#d97706", "#7c3aed", "#65a30d", "#0284c7", "#db2777"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#a5b4fc", "#818cf8", "#6366f1", "#4f46e5", "#4338ca"] if DARK else
                                        ["#26215f", "#3730a3", "#4f46e5", "#6366f1", "#8b97f9"])
DIV = LinearSegmentedColormap.from_list("div", ["#134e4a", "#14b8a6", "#2b3131" if DARK else "#f5f5f4",
                                                "#f59e0b", "#78350f"])
# cmocean "dense" (Thyng et al. 2016, MIT license), 32 samples of its 256-color table: light → dark, monotone lightness
DENSE = LinearSegmentedColormap.from_list("dense", [
    "#e6f1f1", "#d8eaec", "#cbe4e8", "#bcdce5", "#afd5e3", "#a3cee2", "#98c7e2", "#8dbfe2", "#84b7e3", "#7dafe4",
    "#78a7e4", "#759ee5", "#7394e4", "#748ae2", "#7581de", "#7777d9", "#786cd2", "#7963ca", "#795ac1", "#7951b7",
    "#7748ab", "#763fa0", "#733895", "#703088", "#6c297c", "#67226d", "#611d60", "#5a1853", "#531546", "#491238",
    "#40102e", "#360e24"])
FIELD = DENSE
ACCENT = QUAL[0]     # a single series
MESH = {"interior": QUAL[0], "graded": QUAL[2], "weighted": QUAL[1]}

LW = 1.4             # data lines
MS = 3.5             # markers
WIDTH = 6.5          # full figure width (in): draw at print size


def apply():
    """Set the rcParams."""
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 1.0, "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.grid": False,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "lines.linewidth": LW, "lines.markersize": MS,
        "legend.frameon": False, "legend.handlelength": 2.0, "legend.labelspacing": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100,
    })


def seq(n):
    """n colors along the sequential map, dark (first) to light (last)."""
    return [SEQ(x) for x in (np.linspace(0, 1, n) if n > 1 else [0.5])]


def panel(ax, label):
    """Italic panel label (a), (b), … above the top-left corner of the axes, clear of the top tick label."""
    ax.annotate(f"({label})", xy=(0, 1), xycoords="axes fraction", xytext=(-3, 7), textcoords="offset points",
                ha="right", va="bottom", fontstyle="italic", fontsize=10)


def line(ax, x, y, color, marker=True, **kw):
    """Data line with filled circle markers of the same color."""
    return ax.plot(x, y, "-o" if marker else "-", color=color, mfc=color, mec=color, **kw)


def save(fig, path):
    """Save as path.pdf and path.png (200 dpi); in docs mode as a transparent docs/src/assets/<name>-<mode>.png."""
    if DOCS:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "assets",
                           f"{os.path.basename(path)}-{DOCS}.png")
        fig.savefig(out, dpi=200, transparent=True)
        print(os.path.normpath(out))
        return
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    print(path + ".pdf")


def sparse(ax, x, y, n=18):
    """Thin a digitized curve to about n points evenly spaced along its arclength in axes units: the points of each
    arclength bin are averaged (which also smooths pixel noise). Call after the axis limits are set."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ok = (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))
    x, y = x[ok], y[ok]
    ds = np.hypot(np.diff(x) / (x1 - x0), np.diff(y) / (y1 - y0))
    cut = np.nonzero(ds > 0.03)[0] + 1            # gaps (columns dropped by the digitizer): no bin spans one
    ds[cut - 1] = 0
    s = np.r_[0, np.cumsum(ds)]
    k = np.minimum((s / s[-1] * n).astype(int), n - 1)
    seg = np.searchsorted(cut, np.arange(len(x)), side="right")
    big = np.bincount(seg)[seg] >= 3                 # isolated columns are overlap artifacts
    sel = []
    for i in range(n):                               # one point per bin, from its most populated segment
        m = (k == i) & big
        if m.any():
            sel.append(m & (seg == np.bincount(seg[m]).argmax()))
    return np.array([x[m].mean() for m in sel]), np.array([y[m].mean() for m in sel])


def ref_markers(ax, x, y, color, n=18, **kw):
    """Results of another reference (digitized or tabulated, not analytical): open circles, thinned by sparse()."""
    xs, ys = sparse(ax, x, y, n)
    kw = dict(dict(ms=3.5, mew=0.9, ls="none", marker="o", mfc="none", zorder=5), **kw)
    return ax.plot(xs, ys, color=color, mec=color, **kw)


def ref(ax, x, y, **kw):
    """A reference (exact value, error = tol, a slope): a dashed line in the ink color."""
    kw = dict(dict(color=INK, ls="--", lw=0.9, zorder=1), **kw)
    return ax.plot(x, y, **kw)
