"""IEEE-sized evidence figures; statistics are computed by the analysis entry point."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np

METHODS = ("m_detector", "spatiotemporal_normals", "dynablox", "alert")
LABELS = dict(zip(METHODS, ("M-detector", "ST normals", "Dynablox", "ALERT")))
COLORS = dict(zip(METHODS, ("#4377A5", "#C87832", "#298976", "#762A83")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D")))
STYLES = dict(zip(METHODS, (":", "--", "-.", "-")))
SPEEDS = (1., 3., 5., 10.)
MODELS = ("m01", "m02", "m03", "m04", "m05", "m06")
FIGURES = {
    "fig_deformation_scale": (2, 2, "首次确认真实参考点平移分布；漏检保留在50事件分母中。"),
    "fig_response_persistence": (2, 2, "单帧响应与三/五帧确认分开；ALERT单列原生确认。"),
}
for speed in SPEEDS:
    FIGURES["fig_response_timeline_{:g}mmps".format(speed)] = (
        2, 2, "{:g} mm/s全程响应：每个目标十条运行轨道，实际输出刻痕与首次确认标记。".format(speed))
    FIGURES["fig_displacement_tracking_{:g}mmps".format(speed)] = (
        2, 3, "{:g} mm/s六目标：全部十次运行、真值、运行间四分位带及每时段有效运行数。".format(speed))
FIGURES["fig_computation_cost"] = (1, 2, "各速度逐次检测均值与P95，保留基线计算效率优势。")


def configure():
    font = "DejaVu Serif"
    for candidate in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            findfont(FontProperties(family=candidate), fallback_to_default=False)
            font = candidate
            break
        except ValueError:
            pass
    plt.rcParams.update({
        "font.family": font, "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.linewidth": .6, "axes.edgecolor": "#424242",
        "text.color": "#20262D", "axes.labelcolor": "#20262D", "axes.facecolor": "white",
        "figure.facecolor": "white", "figure.dpi": 120,
        "grid.color": "#DDE2E5", "grid.linewidth": .45, "grid.alpha": .7,
        "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": None})
    return font


def panel_grid(rows=2, columns=2, height=4.5):
    fig, axes = plt.subplots(rows, columns, figsize=(7.16, height), squeeze=False)
    fig.subplots_adjust(left=.095, right=.97, bottom=.13, top=.85,
                        wspace=.32 if columns == 2 else .30, hspace=.62)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=(0, (2, 3)))
    return fig, axes.ravel()


def title(ax, index, label):
    ax.set_title("({}) {}".format(chr(97 + index), label), loc="left", pad=8)


def method_legend(fig):
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls=STYLES[m],
                         lw=1.5, ms=4, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .99),
               ncol=4, frameon=False, handlelength=1.8, columnspacing=1.0)


def phase_background(ax):
    for a, b, color in ((0,10,"#DFE6EC"), (10,50,"#F9EDD7"), (50,60,"#E5EEE9")):
        ax.axvspan(a, b, color=color, alpha=.45, lw=0, zorder=0)
    for x in (10, 50):
        ax.axvline(x, color="#9A9A9A", lw=.55, ls="--", zorder=1)
    ax.set(xlim=(0,60), xticks=(0,10,30,50,60))


def deformation_scale(curves, summary):
    fig, axes = panel_grid(height=4.6)
    positive = [r["translation_mm"] for r in curves if r["confirmed_fraction"] > 0]
    xmin = min(.8, min(positive)*.8) if positive else .8
    xmax = max(500, max(r["translation_mm"] for r in curves)*1.1)
    for i, (ax, speed) in enumerate(zip(axes, SPEEDS)):
        for j, method in enumerate(METHODS):
            rows = [r for r in curves if r["method"] == method and r["speed_mm_s"] == speed]
            x = [max(xmin, r["translation_mm"]) for r in rows]
            y = [100*r["confirmed_fraction"] for r in rows]
            ax.fill_between(x, [100*r["ci95_low"] for r in rows],
                            [100*r["ci95_high"] for r in rows], step="post",
                            color=COLORS[method], alpha=.12, lw=0)
            ax.step(x, y, where="post", color=COLORS[method], ls=STYLES[method], lw=1.7)
            s = next(r for r in summary if r["method"] == method and r["speed_mm_s"] == speed)
            # Marker positions distinguish coincident zeros without shifting any values.
            ax.plot(xmin*(1.04+j*.04), 0, marker=MARKERS[method], ms=3.5, color=COLORS[method],
                    markeredgecolor="white", markeredgewidth=.3)
            ax.text(.85, .64-j*.145, "{}  {}/50".format(LABELS[method], s["confirmed_count"]),
                    transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
                    color=COLORS[method], bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=.7))
            if method == "alert" and s["translation_median_mm"] is not None:
                ax.annotate("Median {:.2f} mm".format(s["translation_median_mm"]),
                            xy=(s["translation_median_mm"], s["confirmed_count"]),
                            xytext=(.02,.84), textcoords="axes fraction", fontsize=8.5,
                            color=COLORS[method], arrowprops=dict(arrowstyle="-", lw=.7, color=COLORS[method]))
        ax.set(xscale="log", xlim=(xmin,xmax), ylim=(-5,105), yticks=(0,25,50,75,100),
               xlabel="Recorded translation (mm; log)")
        ax.set_xticks((1,5,20,100,400))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        if i % 2 == 0:
            ax.set_ylabel("Confirmed events / all events (%)")
        title(ax, i, "{:g} mm/s".format(speed))
    method_legend(fig)
    return fig


def response_persistence(rows, summary):
    fig, axes = panel_grid(height=4.4)
    fig.subplots_adjust(left=.15, wspace=.56)
    for i, (ax, speed) in enumerate(zip(axes, SPEEDS)):
        ax.grid(False)
        ax.grid(axis="x", ls=":")
        for j, method in enumerate(METHODS):
            y = 3-j
            ax.axhspan(y-.39, y+.39, color=COLORS[method], alpha=.045, lw=0)
            if method == "alert":
                r = next(r for r in summary if r["method"] == method and r["speed_mm_s"] == speed)
                x = 100*r["primary_recall"]
                ax.plot(x, y, "D", color=COLORS[method], ms=5)
                label = "Native: {:g}%".format(x)
            else:
                values = [100*next(r["primary_recall"] for r in rows if r["method"] == method
                                  and r["speed_mm_s"] == speed and r["consecutive_frames"] == n)
                          for n in (1,3,5)]
                ax.plot(values, [y+.10,y,y-.10], color=COLORS[method], lw=1.1)
                for x, dy, marker, face in zip(values, (.10,0,-.10), ("o","o","|"),
                                              ("white",COLORS[method],COLORS[method])):
                    ax.plot(x, y+dy, marker=marker, ms=5, color=COLORS[method],
                            markerfacecolor=face, markeredgewidth=1)
                label = " / ".join("{:g}".format(v) for v in values) + "%"
            ax.text(.97, y+.22, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", color=COLORS[method], fontsize=8)
        ax.set(xlim=(-3,105), ylim=(-.55,3.60), xticks=(0,25,50,75,100),
               yticks=(3,2,1,0), yticklabels=[LABELS[m] for m in METHODS], xlabel="Target-event response (%)")
        title(ax, i, "{:g} mm/s".format(speed))
    fig.text(.52, .965, "Open: 1 frame    Filled: 3 frames    Tick: 5 frames    Diamond: native ALERT",
             ha="center", va="top", fontsize=8.5)
    return fig


def response_timeline(response, events, speed):
    fig, axes = panel_grid(height=5.0)
    fig.subplots_adjust(left=.095, bottom=.11, top=.81, hspace=.55)
    for i, (ax, method) in enumerate(zip(axes, METHODS)):
        phase_background(ax)
        ax.grid(False)
        rows = [r for r in response if r["method"] == method and r["speed_mm_s"] == speed]
        if not rows:
            ax.text(.5,.5,"No target-associated output",transform=ax.transAxes,
                    ha="center",color="#62666A",fontsize=8)
        for j, model in enumerate(MODELS):
            center = 5-j
            ax.axhline(center-.46, color="white", lw=1.5)
            selected = [r for r in rows if r["model_name"] == model]
            ax.scatter([r["experiment_elapsed_s"] for r in selected],
                       [center+(r["repeat"]-5.5)*.071 for r in selected],
                       marker="|", s=4, color=COLORS[method], alpha=.62, linewidths=.45, rasterized=True)
            confirmed = [r for r in events if r["method"] == method and r["speed_mm_s"] == speed
                         and r["model_name"] == model and r["detected"]]
            ax.scatter([r["motion_onset_experiment_s"]+r["confirmation_latency_s"] for r in confirmed],
                       [center+(r["repeat"]-5.5)*.071 for r in confirmed], marker="D", s=9,
                       facecolor="white", edgecolor=COLORS[method], linewidth=.7, zorder=4)
            ax.text(59, center+.38, "{}/10".format(len(confirmed)), ha="right", va="center",
                    fontsize=7.5, color="#3B4045", bbox=dict(facecolor="white", edgecolor="none", pad=.2))
        ax.set(ylim=(-.5,5.6), yticks=range(6), yticklabels=list(reversed([m.upper() for m in MODELS])),
               xlabel="Experiment time (s)")
        title(ax, i, LABELS[method])
    fig.text(.5, .985, "{:g} mm/s  |  Each target: 10 run lanes; diamond = first confirmation".format(speed),
             ha="center", va="top", fontsize=9)
    fig.text(.5, .942, "Ticks: baseline dynamic points; ALERT formal estimates. M04 auxiliary.",
             ha="center", va="top", fontsize=8.5)
    fig.text(.5, .902, "Command schedule: 0–10 s static; 10–50 s moving; 50–60 s hold",
             ha="center", va="top", fontsize=8.5)
    return fig


def column_quantiles(matrix):
    result = np.full((3,matrix.shape[1]), np.nan)
    for k in range(matrix.shape[1]):
        values = matrix[:,k][np.isfinite(matrix[:,k])]
        if len(values):
            result[:,k] = np.percentile(values, (25,50,75))
    return result


def displacement_tracking(rows, accuracy, events, speed):
    fig, axes = panel_grid(2,3,height=5.15)
    fig.subplots_adjust(left=.085, right=.975, top=.805, bottom=.115, hspace=.50, wspace=.30)
    t = np.arange(120)*.5+.25
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        selected = [r for r in rows if r["speed_mm_s"] == speed and r["model_name"] == model]
        estimate = np.full((10,120), np.nan)
        truth, matched = np.full_like(estimate,np.nan), np.full_like(estimate,np.nan)
        for r in selected:
            k = int(r["experiment_elapsed_s"]*2)
            estimate[r["repeat"]-1,k] = r["estimate_mm"] if r["estimate_mm"] is not None else np.nan
            truth[r["repeat"]-1,k] = r["truth_translation_mm"]
            matched[r["repeat"]-1,k] = r["matched_truth_mm"] if r["matched_truth_mm"] is not None else np.nan
        phase_background(ax)
        for values in estimate:
            ax.plot(t, values, color=COLORS["alert"], lw=.45, alpha=.20)
        q1, median, q3 = column_quantiles(estimate)
        ax.fill_between(t, q1, q3, color=COLORS["alert"], alpha=.18, lw=0)
        ax.plot(t, column_quantiles(truth)[1], color="#242C32", lw=1.2, ls="--")
        ax.plot(t, column_quantiles(matched)[1], color="#B46928", lw=.9, ls=":")
        ax.plot(t, median, color=COLORS["alert"], lw=1.5)
        limit = max(float(np.nanmax(truth)), float(np.nanmax(estimate)) if np.isfinite(estimate).any() else 0,
                    float(np.nanmax(matched)) if np.isfinite(matched).any() else 0, 1.)
        ax.set_ylim(-.18*limit, 1.08*limit)
        coverage = np.isfinite(estimate).sum(axis=0)/10
        ax.fill_between(t, -.15*limit, -.15*limit+.10*limit*coverage,
                        color=COLORS["alert"], alpha=.55, step="mid", lw=0)
        ax.axhline(0, color="#C4C8CB", lw=.5)
        ax.text(.02, .005, "n/10", transform=ax.transAxes, fontsize=7, va="bottom", color="#62666A")
        detected = sum(r["detected"] for r in events if r["method"] == "alert"
                       and r["speed_mm_s"] == speed and r["model_name"] == model)
        title(ax, i, "{}  {}/10{}".format(model.upper(), detected, " (aux.)" if model == "m04" else ""))
        rmses = [r["magnitude_rmse_mm"] for r in accuracy if r["speed_mm_s"] == speed
                 and r["model_name"] == model and r["magnitude_rmse_mm"] is not None]
        if rmses:
            ax.text(.03,.92,"RMSE {:.2f} mm".format(np.mean(rmses)),transform=ax.transAxes,
                    fontsize=8,color="#555555",va="top")
        if not np.isfinite(estimate).any():
            ax.text(.5,.5,"No formal estimate",transform=ax.transAxes,ha="center",color=COLORS["alert"],fontsize=8)
        if i % 3 == 0:
            ax.set_ylabel("Displacement (mm)")
        ax.set_xlabel("Experiment time (s)")
        ax.tick_params(labelsize=8)
    handles = [plt.Line2D([],[],color="#242C32",ls="--",label="Gazebo translation"),
               plt.Line2D([],[],color="#B46928",ls=":",label="Matched-point truth"),
               plt.Line2D([],[],color=COLORS["alert"],label="ALERT median / IQR")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.985),ncol=3,frameon=False,fontsize=8.5)
    fig.text(.5,.878,"{:g} mm/s | Thin curves: all runs; lower strip: available runs (0–10). No gap filling.".format(speed),
             ha="center",fontsize=8)
    return fig


def computation_cost(runs):
    fig, axes = panel_grid(1,2,3.35)
    fig.subplots_adjust(bottom=.18,top=.82,left=.10,wspace=.32)
    for i, (ax,key,name) in enumerate(zip(axes,("runtime_mean_ms","runtime_p95_ms"),("Run mean","Within-run P95"))):
        for si,speed in enumerate(SPEEDS):
            for mi,method in enumerate(METHODS):
                values = [r[key] for r in runs if r["method"] == method and r["speed_mm_s"] == speed]
                y = 3-si+(mi-1.5)*.16
                ax.plot([min(values),max(values)],[y,y],color=COLORS[method],lw=1)
                ax.scatter(values,y+np.linspace(-.035,.035,len(values)),s=8,color=COLORS[method],alpha=.5)
                ax.plot(np.median(values),y,marker=MARKERS[method],color=COLORS[method],ms=4,
                        markeredgecolor="white",markeredgewidth=.35)
        ax.set(xscale="log",xlim=(min(12,min(r[key] for r in runs)*.8),max(1100,max(r[key] for r in runs)*1.15)),
               ylim=(-.5,3.5),yticks=(3,2,1,0),yticklabels=("1","3","5","10"),
               xlabel="Detector time (ms; log)",ylabel="Speed (mm/s)")
        ax.set_xticks((20,100,600))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        title(ax,i,name)
    method_legend(fig)
    return fig


def render_all(output, summary, events, runs, sensitivity_rows, evidence):
    font = configure()
    curves, physical, response, tracking, accuracy = evidence
    jobs = [("fig_deformation_scale",deformation_scale,(curves,physical)),
            ("fig_response_persistence",response_persistence,(sensitivity_rows,summary))]
    for speed in SPEEDS:
        jobs.extend([("fig_response_timeline_{:g}mmps".format(speed),response_timeline,(response,events,speed)),
                     ("fig_displacement_tracking_{:g}mmps".format(speed),displacement_tracking,(tracking,accuracy,events,speed))])
    jobs.append(("fig_computation_cost",computation_cost,(runs,)))
    checks = []
    with PdfPages(output / "FIGURE_GALLERY.pdf") as gallery:
        for name,plot,data in jobs:
            fig = plot(*data)
            nr,nc,_ = FIGURES[name]
            if len(fig.axes) != nr*nc or any(not ax.axison or not ax.has_data() for ax in fig.axes):
                raise ValueError("empty or incomplete panel grid: " + name)
            cells = {(ax.get_subplotspec().rowspan.start,ax.get_subplotspec().colspan.start) for ax in fig.axes}
            if cells != {(r,c) for r in range(nr) for c in range(nc)}:
                raise ValueError("panels do not fill grid: " + name)
            fig.canvas.draw()
            outside = []
            for text in fig.findobj(plt.Text):
                if text.get_visible() and text.get_text():
                    b = text.get_window_extent(fig.canvas.get_renderer())
                    if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                        outside.append(text.get_text())
            if outside:
                raise ValueError("text exceeds fixed canvas: {} {}".format(name,outside))
            fig.savefig(output / (name+".pdf"))
            fig.savefig(output / (name+".png"))
            gallery.savefig(fig)
            checks.append({"figure":name,"rows":nr,"columns":nc,"data_panels":len(fig.axes),
                           "empty_panels":0,"width_inches":float(fig.get_figwidth()),
                           "height_inches":float(fig.get_figheight()),"font":font,"text_within_canvas":True})
            plt.close(fig)
    return checks
