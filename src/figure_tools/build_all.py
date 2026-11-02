"""Rebuild every table and figure of the experimental section and install them.

Usage
    python3 build_all.py            rebuild everything and copy into the paper
    python3 build_all.py --keep     rebuild only, do not touch the paper folder

Nothing in experiment_output or in the recorded analysis products is modified.
"""

import shutil
import sys

import paper_style as ps

import make_tables
import fig_detection
import fig_micro_accuracy
import fig_real_response
import fig_retreat

# analysis name -> name used by the manuscript
FIGURE_NAMES = {"fig_detection": "fig3_detection",
                "fig_micro_accuracy": "fig4_displacement_error",
                "fig_real_warning": "fig5_real_sequence",
                "fig_evacuation_map": "fig6a_retreat_maps",
                "fig_evacuation_response": "fig6b_retreat_response"}


def main():
    ps.use_style()
    make_tables.main()
    for module in (fig_detection, fig_micro_accuracy, fig_real_response, fig_retreat):
        module.main()
    if "--keep" in sys.argv:
        return
    ps.PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for name, target in FIGURE_NAMES.items():
        for suffix in (".pdf", ".svg"):
            shutil.copy(ps.FIGURES / f"{name}{suffix}", ps.PAPER_FIGURES / f"{target}{suffix}")
        print(f"installed {target}.pdf and {target}.svg")



if __name__ == "__main__":
    main()
