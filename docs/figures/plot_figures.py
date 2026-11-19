# The figures of ProjectedSphericalHarmonics' documentation, from the CSV files of make_data.jl, in the light and
# dark renderings of the site (docs/src/assets/<name>-light.png, -dark.png):
#   for m in light dark; do FIGSTYLE_DOCS=$m ~/miniconda3/bin/python docs/figures/plot_figures.py; done
# (from the package root, a few seconds). Signed fields are colored by DIV, magnitudes by FIELD, cases and methods
# by the qualitative palette; exact values and reference slopes are dashed lines in the ink color.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter, LogLocator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
fs.apply()


def read(name):
    """A CSV of make_data.jl as a dict of columns (numbers where they parse)."""
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as f:
        head, *rows = [r for r in csv.reader(f) if r]
    cols = {}
    for j, h in enumerate(head):
        col = [r[j] for r in rows]
        try:
            cols[h] = np.array([float(x) for x in col])
        except ValueError:
            cols[h] = np.array(col)
    return cols


def series(d, key):
    """The distinct values of a column, in the order of first appearance."""
    _, idx = np.unique(d[key], return_index=True)
    return [d[key][i] for i in sorted(idx)]


def loglog(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())


def logy(ax, step=None):
    """A log y axis with a major tick every `step` decades (default: about five ticks)."""
    ax.set_yscale("log")
    if step is None:
        lo, hi = ax.get_ylim()
        step = max(1, int(np.ceil(np.log10(hi / lo) / 5)))
    ax.yaxis.set_major_locator(LogLocator(base=10 ** step, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=40) if step > 1 else
                               LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())


def below(ax, ncol=1, **kw):
    """The legend under the axes, clear of the data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=7.5, ncol=ncol, **kw)


def bare(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def polar(d, nr, *keys):
    """Columns of a CSV written theta-major on a polar grid of nr radii, reshaped to (n_theta, nr)."""
    return [d[k].reshape(-1, nr) for k in keys]


# ---- hero -----------------------------------------------------------------------------------------------------------
def hero():
    fig = plt.figure(figsize=(fs.WIDTH * 1.2, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.0], wspace=0.1)
    ax0, ax2 = (fig.add_subplot(gs[0, i]) for i in range(2))
    th = np.linspace(0, 2 * np.pi, 400)
    # (a) a projected spherical harmonic and the nodes of disk(16)
    d = read("hero_ylm.csv")
    R, T, Y = polar(d, 121, "r", "theta", "y")
    v = np.abs(Y).max()
    ax0.pcolormesh(R * np.cos(T), R * np.sin(T), Y, cmap=fs.FIELD, vmin=-v, vmax=v, shading="gouraud", rasterized=True)
    n = read("hero_nodes.csv")
    ax0.plot(n["x"], n["y"], "o", color=fs.INK, ms=0.9, mew=0, alpha=0.6)
    ax0.plot(np.cos(th), np.sin(th), color=fs.INK, lw=0.8)
    ax0.set_xlim(-1.03, 1.03); ax0.set_ylim(-1.03, 1.03); bare(ax0)
    ax0.set_xlabel(r"$\mathrm{Re}\, y_9^3$ and the nodes, $M = 16$")
    fs.panel(ax0, "a")
    # (b) a suspension of ellipses: |f|ω in the bodies, streamlines of the flow (width by speed) around them
    fl = read("hero_flow.csv")
    xs, ys = np.unique(fl["x"]), np.unique(fl["y"])
    U = fl["ux"].reshape(len(ys), len(xs)); V = fl["uy"].reshape(len(ys), len(xs))
    sp = np.hypot(U, V)
    smax = np.nanpercentile(sp, 99)
    lw = 0.25 + 1.1 * np.clip(np.nan_to_num(sp) / smax, 0, 1)
    ax2.streamplot(xs, ys, np.ma.masked_invalid(U), np.ma.masked_invalid(V), color=fs.GRAY, linewidth=lw,
                   density=1.6, arrowsize=0.55, broken_streamlines=True)
    bd = read("hero_bodies.csv")
    norm = Normalize(0, bd["fw"].max())
    mo = read("hero_motion.csv")
    for i in np.unique(bd["body"]):
        m = bd["body"] == i
        X, Yb, Fw = polar({k: bd[k][m] for k in bd}, 31, "x", "y", "fw")
        ax2.pcolormesh(X, Yb, Fw, cmap=fs.FIELD, norm=norm, shading="gouraud", rasterized=True, zorder=3)
        ax2.plot(X[:, -1], Yb[:, -1], color=fs.INK, lw=0.7, zorder=4)
        j = int(i) - 1      # the applied force, from the center
        ax2.annotate("", xy=(mo["xc"][j] + 0.75 * mo["Fx"][j], mo["yc"][j] + 0.75 * mo["Fy"][j]),
                     xytext=(mo["xc"][j], mo["yc"][j]), zorder=5,
                     arrowprops=dict(arrowstyle="-|>", color=fs.QUAL[3], lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax2.set_xlim(bd["x"].min() - 0.5, bd["x"].max() + 0.5); ax2.set_ylim(bd["y"].min() - 0.4, bd["y"].max() + 0.4)
    bare(ax2)
    ax2.set_xlabel(r"8 ellipses under applied forces (arrows): $|f|\,w$ and the flow")
    fs.panel(ax2, "b")
    fs.save(fig, "hero")


# ---- convergence ----------------------------------------------------------------------------------------------------
def label(name):
    return (name.replace("a/b = ", "$a/b = ").replace(", drag", ", drag $16/3$").replace(", torque", ", torque $16/3$")
            + ("$" if "a/b" in name else ""))


def by_case(ax, d, x, key, order=None, styles=None):
    for k, name in enumerate(order or series(d, key)):
        m = (d[key] == name) & (d["error"] > 0)
        st = (styles or {}).get(name, {})
        c = fs.QUAL[k]
        if st.get("open"):
            ax.plot(d[x][m], d["error"][m], "--o", color=c, mfc="none", mec=c, label=label(name))
        else:
            fs.line(ax, d[x][m], d["error"][m], c, label=label(name))


def convergence():
    fig, axs = plt.subplots(1, 4, figsize=(fs.WIDTH * 1.35, 3.0))
    # (a) Laplace: 𝒮⁻¹[1] on ellipses, pointwise
    d = read("conv_laplace.csv")
    by_case(axs[0], d, "M", "case")
    logy(axs[0]); axs[0].set_xticks([8, 16, 24, 32, 40, 48])
    axs[0].set_xlabel(r"resolution, $M$"); axs[0].set_ylabel(r"relative error, $w_e\,\mathcal{S}^{-1}[1]$")
    below(axs[0]); fs.panel(axs[0], "a")
    # (b) Stokes: drag on a translating plate (and the disk's drag and torque)
    d = read("conv_stokes.csv")
    by_case(axs[1], d, "M", "case", styles={"disk, torque": {"open": True}})
    logy(axs[1]); axs[1].set_xticks([8, 16, 24, 32, 40, 48])
    axs[1].set_xlabel(r"resolution, $M$"); axs[1].set_ylabel("relative error, drag")
    below(axs[1], ncol=2, columnspacing=1.0, handlelength=1.6); fs.panel(axs[1], "b")
    # (c) Poisson
    d = read("conv_poisson.csv")
    m = d["error"] > 0
    fs.line(axs[2], d["M"][m], d["error"][m], fs.QUAL[0], label=r"$u = e^x \sin y + \cos(2x + y)$")
    logy(axs[2]); axs[2].set_xticks([4, 8, 16, 24, 32])
    axs[2].set_xlabel(r"resolution, $M$"); axs[2].set_ylabel(r"max relative error, $u$")
    below(axs[2]); fs.panel(axs[2], "c")
    # (d) close evaluation: cross velocity against the edge gap
    d = read("conv_close.csv")
    for k, case in enumerate(series(d, "case")):
        m = d["case"] == case
        c = fs.QUAL[k]
        nm = case.replace("a/b = 4", "$a/b = 4$")
        fs.line(axs[3], d["gap"][m], d["closed_form"][m], c, label=f"{nm}, closed form")
        axs[3].plot(d["gap"][m], d["node_quadrature"][m], "--o", color=c, mfc="none", mec=c, label=f"{nm}, nodes only")
    loglog(axs[3])
    axs[3].set_xlabel(r"edge gap, $d$"); axs[3].set_ylabel("relative error, cross velocity")
    below(axs[3]); fs.panel(axs[3], "d")
    fig.tight_layout(w_pad=1.2)
    fs.save(fig, "convergence")


# ---- suspension timings ---------------------------------------------------------------------------------------------
def suspension():
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 3.1))
    # (a) one cross-body sum
    d = read("susp_sum.csv")
    for k, meth in enumerate(["Ewald apply", "Ewald plan", "FMM3D", "direct"]):
        m = d["method"] == meth
        lab = {"Ewald apply": "Ewald + NUFFT, apply", "Ewald plan": "Ewald + NUFFT, plan",
               "FMM3D": "FMM3D", "direct": "direct sum"}[meth]
        if meth == "Ewald plan":
            axs[0].plot(d["N"][m], d["time"][m], "--o", color=fs.QUAL[0], mfc="none", mec=fs.QUAL[0], label=lab)
        else:
            fs.line(axs[0], d["N"][m], d["time"][m], fs.QUAL[{"Ewald apply": 0, "FMM3D": 1, "direct": 2}[meth]], label=lab)
    N = np.array([16, 256.0])
    a = d["time"][(d["method"] == "Ewald apply") & (d["N"] == 256)][0]
    fs.ref(axs[0], N, 2.5 * a * N / 256, label=r"$\propto N$")
    loglog(axs[0]); axs[0].set_xticks([2, 8, 32, 128]); axs[0].set_xticklabels(["2", "8", "32", "128"])
    axs[0].set_xlabel(r"bodies, $N$ (561 nodes each)"); axs[0].set_ylabel("time of one sum (s)")
    below(axs[0], ncol=2, columnspacing=1.0); fs.panel(axs[0], "a")
    # (b) one mobility solve
    d = read("susp_solve.csv")
    for k, meth in enumerate(["Ewald", "FMM3D"]):
        m = d["method"] == meth
        fs.line(axs[1], d["N"][m], d["time"][m], fs.QUAL[k], label=("Ewald + NUFFT" if k == 0 else "FMM3D"))
    m = d["method"] == "FMM3D"
    for n, t, it in zip(d["N"][m], np.maximum(d["time"][m], d["time"][d["method"] == "Ewald"]), d["iterations"][m]):
        axs[1].annotate(f"{int(it)}", xy=(n, t), xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=7, color=fs.GRAY)
    b = d["time"][(d["method"] == "Ewald") & (d["N"] == 256)][0]
    fs.ref(axs[1], N, 1.6 * b * N / 256, label=r"$\propto N$")
    loglog(axs[1]); axs[1].set_xticks([2, 8, 32, 128]); axs[1].set_xticklabels(["2", "8", "32", "128"])
    lo, hi = axs[1].get_ylim(); axs[1].set_ylim(lo, hi * 2)
    axs[1].set_xlabel(r"bodies, $N$"); axs[1].set_ylabel("time of one mobility solve (s)")
    below(axs[1], ncol=2, columnspacing=1.0); fs.panel(axs[1], "b")
    fig.tight_layout(w_pad=2.0)
    fs.save(fig, "suspension")


hero()
convergence()
suspension()
