"""Shared IEEE-conference figure style and data locations for the ICRA2027 paper.

All figures produced by this package use one font, one palette and one set of
line weights so that the experimental section reads as a single system.
Nothing here modifies recorded experiment data.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "IEEE_ICRA2027"          # recorded analysis products
SUBMISSION = HERE.parent / "IEEE_ICRA2027_SUBMIT"  # manuscript and its figures
OUTPUT = HERE.parent.parent / "experiment_output"

ANALYSIS = PAPER / "paper_analysis"
PROJECTED = PAPER / "projected_analysis"
CONFIRMATION = PAPER / "confirmation_analysis"
M3C2 = PAPER / "m3c2_analysis"
REAL = PAPER / "real_analysis"
STATIONARY = OUTPUT / "stationary_speed_sweep/batch_20260901_012253"
EVACUATION = OUTPUT / "evacuation_paper_run/closed_loop/icra2027_20260911_235806"

FIGURES = HERE / "figures"
TABLES = HERE / "tables"
PAPER_FIGURES = SUBMISSION / "figures"

TARGETS = ("m01", "m02", "m03", "m05", "m06")
ALL_TARGETS = ("m01", "m02", "m03", "m04", "m05", "m06")
SPEEDS = (1, 3, 5, 10)
METHODS = ("m_detector", "spatiotemporal_normals", "dynablox", "m3c2", "alert")
METHOD_LABELS = dict(zip(METHODS, ("M-detector", "ST normals", "Dynablox", "M3C2", "Ours")))

# Colour-blind safe qualitative palette (Okabe and Ito) with one accent for the
# proposed method. Annotation text is always bold black, so colour only carries
# identity in legends and in the marks themselves.
OURS = "#6A3D9A"
TRUTH = "#000000"
SURFACE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#6A3D9A", "#A6761D")
# Deformation speed is ordinal, so it uses a single sequential ramp.
SPEED_COLORS = ("#C3AAD9", "#9B78C0", "#7A4FA6", "#4B2678")
REFERENCE = "#4D4D4D"
MODE_COLORS = dict(u="#C9CDD2", a="#0072B2", c="#6A3D9A")
PHASE_FILL = ("#FFFFFF", "#EFEFF1", "#F7F3EC")
GRID = "#DCDFE3"
ANNOTATE = dict(color="#000000", fontweight="bold", fontsize=6.4)

COLUMN_WIDTH = 3.5      # inches, IEEE single column
DOUBLE_WIDTH = 7.16     # inches, IEEE double column


def use_style():
    """Register the bundled Times faces and set rcParams once per process."""
    for face in sorted((PAPER / ".local_fonts").glob("*.TTF")):
        fontManager.addfont(str(face))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.35,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.borderpad": 0.1,
        "legend.labelspacing": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
    })


def trim_axes(ax, grid_axis="y"):
    """Remove the top and right spines and apply the shared grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.35, zorder=0)
        ax.set_axisbelow(True)


def phase_bands(ax, baseline=10.0, motion=50.0, end=60.0):
    """Shade the static, motion and hold phases of a stationary trial."""
    ax.axvspan(0, baseline, color=PHASE_FILL[1], lw=0, zorder=0)
    ax.axvspan(motion, end, color=PHASE_FILL[2], lw=0, zorder=0)
    for edge in (baseline, motion):
        ax.axvline(edge, color="#9AA0A6", lw=0.5, ls=(0, (3, 2)), zorder=1)


def panel_tag(ax, text, loc="left", pad=3):
    ax.set_title(text, loc=loc, pad=pad)


def digest(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in dict.fromkeys(map(str, paths))}


def save(fig, name, provenance, tight=True):
    """Write PDF and PNG into the analysis folder and record provenance.

    Set tight to False when the layout deliberately reserves empty canvas that
    a tight bounding box would crop away.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    box = "tight" if tight else None
    fig.savefig(pdf, bbox_inches=box)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches=box)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches=box)
    plt.close(fig)
    record = dict(provenance)
    record["output_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    (FIGURES / f"{name}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {pdf}")
    return record
