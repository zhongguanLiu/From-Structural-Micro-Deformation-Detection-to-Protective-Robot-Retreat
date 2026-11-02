"""Figure: displacement error in the micro regime, per surface and per speed.

Samples whose projected ground-truth displacement lies between 5 and 20 mm are
collected for every monitored surface. Filled markers give our vector error in
the observable direction, one marker per surface with the spread over the ten
runs. Open markers give the signed normal-distance error that M3C2 reports for
the same surface, so each method is scored in the direction it produces.
"""

import numpy as np
import pandas as pd

import paper_style as ps

OFFSETS = np.linspace(-0.26, 0.26, len(ps.TARGETS))


def load():
    runs = pd.read_csv(ps.ANALYSIS / "accuracy_runs.csv")
    runs = runs[(runs.method == "alert") & runs.model_name.isin(ps.TARGETS)]
    m3c2 = pd.read_csv(ps.M3C2 / "accuracy_summary.csv")
    m3c2 = m3c2[m3c2.model_name.isin(ps.TARGETS)]
    return runs, m3c2


def build():
    runs, m3c2 = load()
    fig, ax = ps.plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.35))
    fig.subplots_adjust(left=.135, right=.985, bottom=.185, top=.86)
    ps.trim_axes(ax)
    record = []

    for index, speed in enumerate(ps.SPEEDS):
        for offset, color, target in zip(OFFSETS, ps.SURFACE_COLORS, ps.TARGETS):
            block = runs[(runs.speed_mm_s == speed) & (runs.model_name == target)]
            values = np.array([v for v in block.conditional_micro_rmse_mm if pd.notna(v)])
            x = index + offset
            if values.size:
                ax.vlines(x, values.min(), values.max(), color=color, lw=.7, alpha=.55, zorder=2)
                ax.plot(x, np.sqrt((values ** 2).mean()), marker="o", ms=3.1, color=color,
                        mec="none", ls="", zorder=3,
                        label=target.upper() if index == 0 else None)
            native = m3c2[(m3c2.speed_mm_s == float(speed)) & (m3c2.model_name == target)]
            value = float(native.micro_rmse_mm_mean.iloc[0])
            ax.plot(x, value, marker="o", ms=3.4, mfc="none", mec="#4D4D4D", mew=.7,
                    ls="", zorder=3, label="M3C2" if index == 0 and target == ps.TARGETS[0] else None)
            record.append(dict(speed=speed, target=target,
                               ours=float(np.sqrt((values ** 2).mean())) if values.size else None,
                               runs=int(values.size), m3c2=value))

    ax.set_yscale("log")
    ax.set(xlim=(-.5, 3.5), xticks=range(len(ps.SPEEDS)),
           xticklabels=[f"{s}" for s in ps.SPEEDS],
           xlabel="Deformation speed (mm/s)", ylim=(.2, 60),
           ylabel="Displacement RMSE (mm)")
    ax.set_yticks((0.3, 1, 3, 10, 30))
    ax.set_yticklabels(("0.3", "1", "3", "10", "30"))
    ax.minorticks_off()
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(t.upper()) for t in ps.TARGETS] + [labels.index("M3C2")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", ncol=6, columnspacing=.55, handletextpad=.3,
              handlelength=.7, borderaxespad=.2, bbox_to_anchor=(-.015, 1.19), fontsize=6.4)
    return fig, record


def main():
    ps.use_style()
    fig, record = build()
    ps.save(fig, "fig_micro_accuracy", dict(
        targets=list(ps.TARGETS), speeds=list(ps.SPEEDS), values=record,
        inputs=ps.digest([ps.ANALYSIS / "accuracy_runs.csv", ps.M3C2 / "accuracy_summary.csv"])))


if __name__ == "__main__":
    main()
