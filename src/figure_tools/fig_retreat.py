"""Restyle the retreat demonstration figures without touching their geometry.

The two recorded panels are produced by the existing evacuation renderer. This
script re-runs that renderer with the shared palette of this package and turns
every in-plot annotation into bold black text, then writes the result into the
analysis folder. Map values, trajectories and evidence curves are unchanged and
the renderer's own geometry checks still run.
"""

import sys
from pathlib import Path

import paper_style as ps

sys.path.insert(0, str(ps.PAPER))
sys.path.insert(0, str(ps.PAPER.parent / "analyze_script"))
import render_figures as rf  # noqa: E402

OUTPUTS = ("fig_evacuation_map.pdf", "fig_evacuation_response.pdf")


def restyle(fig):
    """Annotations in bold black, and one white-on-fill label moved beside its bar."""
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_color() in ("white", "#ffffff", "#FFFFFF"):
                # The retention duration sat inside a dark bar. Lift it above the
                # bar so that it can also be read as bold black.
                x, y = text.get_position()
                text.set_position((x, y + .42))
                text.set_va("bottom")
            text.set_color("#000000")
            text.set_fontweight("bold")


def install():
    original = rf.save_paper_figure
    records = {}

    def patched(fig, name, panels=None):
        restyle(fig)
        ps.FIGURES.mkdir(parents=True, exist_ok=True)
        target = ps.FIGURES / name
        fig.canvas.draw()
        fig.savefig(target)
        fig.savefig(target.with_suffix(".png"))
        fig.savefig(target.with_suffix(".svg"))
        records[name] = dict(width_inches=float(fig.get_figwidth()),
                             height_inches=float(fig.get_figheight()),
                             panels=panels or len(fig.axes),
                             geometry_sha256=rf.geometry_digest(fig))
        rf.plt.close(fig)
        print(f"wrote {target}")
        return records[name]

    rf.save_paper_figure = patched
    return original, records


def main():
    ps.use_style()
    rf.COLORS["alert"] = ps.OURS
    rf.OUTBOUND, rf.RETURN, rf.WARNING = ps.SURFACE_COLORS[0], ps.SURFACE_COLORS[2], "#D55E00"
    original, records = install()
    try:
        results, inputs = rf.evacuation_figures()
    finally:
        rf.save_paper_figure = original
    for name in OUTPUTS:
        record = dict(records[name])
        record["inputs"] = inputs
        (ps.FIGURES / f"{Path(name).stem}.json").write_text(
            __import__("json").dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
