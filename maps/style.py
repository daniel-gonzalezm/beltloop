"""Common style of the paper figures.

Widths follow the usual two-column journal sizes (single column 85 mm, full width 174 mm);
check them against the author guidelines of the chosen journal before submission (phase 7).
Fonts are STIX (bundled with matplotlib, Times-like, so the figures reproduce on any machine);
line colours are the Okabe-Ito colour-blind safe set; maps use perceptually uniform colour maps.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 85 * MM
DOUBLE = 174 * MM

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    "hatch.linewidth": 0.5,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,          # TrueType in the PDF (editable text, accepted by publishers)
}

OUT = Path(__file__).parent / "figures" / "paper"


def apply():
    plt.rcParams.update(RC)


def panel_label(ax, text, **kw):
    """Panel label above the axes, flush left, e.g. '(a)' or '(a) B1'."""
    ax.set_title(text, loc="left", fontsize=8, pad=3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    print("figure:", OUT / f"{name}.pdf")
