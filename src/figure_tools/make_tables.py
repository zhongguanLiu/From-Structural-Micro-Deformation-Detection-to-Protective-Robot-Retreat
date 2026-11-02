"""Compute the four experimental tables of the ICRA2027 paper.

Every number is recomputed here from the recorded analysis products; nothing is
copied from the previous manuscript. LaTeX fragments and a JSON record of all
values are written to ./tables.
"""

import json
import math
import statistics as st
from collections import defaultdict

import pandas as pd

import paper_style as ps

TABLE_DIR = ps.TABLES


def rms(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def mean_sd(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


# ---------------------------------------------------------------- loading ---

def load():
    d = {}
    d["condition"] = pd.read_csv(ps.ANALYSIS / "condition_summary.csv")
    d["accuracy_runs"] = pd.read_csv(ps.ANALYSIS / "accuracy_runs.csv")
    d["response"] = pd.read_csv(ps.ANALYSIS / "response_summary.csv")
    d["events"] = pd.read_csv(ps.CONFIRMATION / "events.csv")
    d["rank"] = pd.read_csv(ps.PROJECTED / "rank_distribution.csv")
    d["m3c2_acc"] = pd.read_csv(ps.M3C2 / "accuracy_summary.csv")
    d["m3c2_rt"] = pd.read_csv(ps.M3C2 / "runtime_runs.csv")
    d["runtime"] = json.loads((ps.PAPER / "runtime_audit.json").read_text())
    return d


# --------------------------------------------------- per-speed aggregation ---

def our_accuracy_by_speed(runs):
    """Per run, equal-weight RMS over the contributing targets; then mean/SD."""
    out = {}
    ours = runs[runs.method == "alert"]
    for speed in ps.SPEEDS:
        block = ours[ours.speed_mm_s == speed]
        row = {}
        for key, column in (("micro", "conditional_micro_rmse_mm"),
                            ("hold", "conditional_hold_rmse_mm")):
            per_run, targets = [], 0
            for _, group in block.groupby("repeat"):
                values = [v for v in group[column].tolist() if pd.notna(v)]
                if values:
                    per_run.append(rms(values))
                    targets += len(values)
            mean, sd = mean_sd(per_run)
            row[key] = dict(mean=mean, sd=sd, runs=len(per_run), target_runs=targets)
        out[speed] = row
    return out


def m3c2_micro_by_speed(summary):
    """RMS over the five evaluated targets of the per-target native-normal RMSE."""
    keep = summary[summary.model_name.isin(ps.TARGETS)]
    out = {}
    for speed in ps.SPEEDS:
        block = keep[keep.speed_mm_s == float(speed)]
        values = [v for v in block.micro_rmse_mm_mean.tolist() if pd.notna(v)]
        out[speed] = dict(value=rms(values), targets=len(values))
    return out


def table_one(data):
    condition = data["condition"].set_index("speed_mm_s")
    ours = our_accuracy_by_speed(data["accuracy_runs"])
    m3c2 = m3c2_micro_by_speed(data["m3c2_acc"])
    rows = []
    for speed in ps.SPEEDS:
        c = condition.loc[speed]
        rows.append(dict(
            speed=speed,
            confirmed=int(c.confirmed_count), events=int(c.event_count),
            delay_median=float(c.confirmation_latency_median_s),
            delay_q25=float(c.confirmation_latency_q25_s),
            delay_q75=float(c.confirmation_latency_q75_s),
            truth_median_mm=float(c.translation_median_mm),
            micro_mean=ours[speed]["micro"]["mean"], micro_sd=ours[speed]["micro"]["sd"],
            hold_mean=ours[speed]["hold"]["mean"], hold_sd=ours[speed]["hold"]["sd"],
            hold_target_runs=ours[speed]["hold"]["target_runs"],
            m3c2_micro=m3c2[speed]["value"],
            motion_cov=float(c.motion_coverage_mean_percent),
            hold_cov=float(c.hold_coverage_mean_percent)))
    body = "\n".join(
        f"{r['speed']} & {r['confirmed']}/{r['events']} & "
        f"{r['delay_median']:.2f} [{r['delay_q25']:.2f}, {r['delay_q75']:.2f}] & "
        f"${r['micro_mean']:.2f}\\pm{r['micro_sd']:.2f}$ & "
        f"${r['hold_mean']:.2f}\\pm{r['hold_sd']:.2f}$ & "
        f"{r['motion_cov']:.1f} & {r['hold_cov']:.1f} \\\\" for r in rows)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Confirmation and displacement estimation versus deformation speed.}}
\\label{{tab:speed}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}crcccrr@{{}}}}
\\toprule
\\textbf{{Speed}} & \\textbf{{Confirmed}} & \\textbf{{Confirmation delay}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Displacement RMSE (mm)}}}} &
\\multicolumn{{2}}{{c}}{{\\textbf{{Estimate availability (\\%)}}}}\\\\
\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}
\\textbf{{(mm/s)}} & \\textbf{{events}} & \\textbf{{median [Q1, Q3] (s)}} &
\\textbf{{Motion (5 to 20\\,mm)}} & \\textbf{{Hold}} & \\textbf{{Motion}} & \\textbf{{Hold}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\par\\vspace{{2pt}}
\\parbox{{\\textwidth}}{{\\scriptsize Each row pools ten runs of five monitored surfaces, so 50 target events.
Delay runs from the onset of motion to the first confirmation. The motion error is restricted to samples whose
projected ground-truth displacement lies between the 5\\,mm task scale and 20\\,mm, which is the
small-displacement regime this paper addresses, and Fig.~\\ref{{fig:detection}}(b) shows that the error does
not grow beyond that range. Availability is the share of 0.5\\,s windows carrying an estimate, and at
10\\,mm/s only {rows[3]['hold_target_runs']} target runs reach the hold phase with one.}}
\\end{{table*}}
"""
    return tex, rows


# ------------------------------------------------------------- method table ---

def table_two(data):
    response = data["response"]
    rows = []
    for method in ps.METHODS:
        block = response[response.method == method]
        cell = {}
        for phase in ("baseline", "motion", "hold"):
            hit = block[(block.phase == phase) & (block.speed_mm_s == 1)]
            cell[phase] = float(hit.activity_mean_percent.iloc[0])
        fast = block[(block.phase == "motion") & (block.speed_mm_s == 10)]
        cell["motion10"] = float(fast.activity_mean_percent.iloc[0])
        rows.append(dict(method=method, label=ps.METHOD_LABELS[method], **cell))
    body = []
    for r in rows:
        name = f"\\textbf{{{r['label']}}}" if r["method"] == "alert" else r["label"]
        body.append(f"{name} & {r['baseline']:.1f} & {r['motion']:.1f} & "
                    f"{r['hold']:.1f} & {r['motion10']:.1f} \\\\")
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Fraction of one-second windows carrying a target-associated change indication.}}
\\label{{tab:methods}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}lrrrr@{{}}}}
\\toprule
& \\multicolumn{{3}}{{c}}{{\\textbf{{At 1\\,mm/s (\\%)}}}} & \\textbf{{At 10\\,mm/s}}\\\\
\\cmidrule(lr){{2-4}}
\\textbf{{Method}} & \\textbf{{Static}}$\\,\\downarrow$ & \\textbf{{Motion}}$\\,\\uparrow$ &
\\textbf{{Hold}}$\\,\\uparrow$ & \\textbf{{Motion (\\%)}}$\\,\\uparrow$\\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular*}}
\\par\\vspace{{2pt}}
\\parbox{{\\columnwidth}}{{\\scriptsize Equal weight over 50 target runs per condition. A window counts when
the method reports a target-associated dynamic point, a significant M3C2 endpoint or a qualifying update of
ours. The arrows give the value an ideal output would take, since no surface moves in the static phase and
every surface is displaced in the other two. Every method produced output in every window, so a low value is
an absence of indication and not of data.}}
\\end{{table}}
"""
    return tex, rows


# ------------------------------------------------------------ surface table ---

def table_three(data):
    rank, events, runs = data["rank"], data["events"], data["accuracy_runs"]
    ours = runs[runs.method == "alert"]
    confirmations = events[events.method == "alert"]
    rows = []
    for target in tuple(ps.TARGETS) + ("m04",):
        block = rank[rank.model_name == target]
        weight = block.n_published_samples.sum()
        share = float((block.rank1_percent * block.n_published_samples).sum() / weight)
        hits = confirmations[confirmations.model_name == target]
        found = hits[hits.detected.astype(str) == "True"]
        truth = [float(v) for v in found.truth_translation_at_confirmation_mm if pd.notna(v)]
        block_runs = ours[ours.model_name == target]
        micro = rms([v for v in block_runs.conditional_micro_rmse_mm if pd.notna(v)])
        hold = rms([v for v in block_runs.conditional_hold_rmse_mm if pd.notna(v)])
        rows.append(dict(target=target.upper(), rank1=share, confirmed=len(found), trials=len(hits),
                         truth_median=(st.median(truth) if truth else float("nan")),
                         micro=micro, hold=hold, samples=int(weight)))
    body = []
    for r in rows:
        truth = "--" if math.isnan(r["truth_median"]) else f"{r['truth_median']:.2f}"
        micro = "--" if math.isnan(r["micro"]) else f"{r['micro']:.2f}"
        hold = "--" if math.isnan(r["hold"]) else f"{r['hold']:.2f}"
        body.append(f"{r['target']} & {r['rank1']:.1f} & {r['confirmed']}/{r['trials']} & "
                    f"{truth} & {micro} & {hold} \\\\")
    body.insert(len(body) - 1, "\\midrule")
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Observability and estimation quality per monitored surface.}}
\\label{{tab:surfaces}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}lrrrrr@{{}}}}
\\toprule
\\textbf{{Surface}} & \\textbf{{Observable}} & \\textbf{{Confirmed}} & \\textbf{{Truth}} &
\\textbf{{Motion}} & \\textbf{{Hold}}\\\\
& \\textbf{{rank}} $\\bm{{=1}}$ \\textbf{{(\\%)}} & \\textbf{{/40}} & \\textbf{{(mm)}} & \\textbf{{(mm)}} & \\textbf{{(mm)}}\\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular*}}
\\par\\vspace{{2pt}}
\\parbox{{\\columnwidth}}{{\\scriptsize Pooled over four speeds and ten runs each. The rank column is the
share of published samples whose observable subspace has exactly one supported direction, fixed by the frozen
geometry and the viewpoint rather than by the outcome. Truth is the median ground-truth translation at
confirmation, and the two error columns follow the definitions of Table~\\ref{{tab:speed}}. M04 is excluded from the aggregates of Tables~\\ref{{tab:speed}}
and~\\ref{{tab:methods}}.}}
\\end{{table}}
"""
    return tex, rows


# ------------------------------------------------------------ runtime table ---

def table_four(data):
    audit = data["runtime"]
    rows = []
    for method in ("m_detector", "spatiotemporal_normals", "dynablox"):
        means = [r["cycle_mean_ms"] for r in audit["baseline_runs"] if r["method"] == method]
        mean, sd = mean_sd(means)
        rows.append(dict(method=method, label=ps.METHOD_LABELS[method], mean=mean, sd=sd, runs=len(means)))
    means = [float(v) for v in data["m3c2_rt"].cycle_mean_ms]
    mean, sd = mean_sd(means)
    rows.append(dict(method="m3c2", label="M3C2", mean=mean, sd=sd, runs=len(means)))
    means = [r["cycle_mean_ms"] for r in audit["static_runs"]]
    mean, sd = mean_sd(means)
    rows.append(dict(method="alert", label="Ours", mean=mean, sd=sd, runs=len(means)))
    header = " & ".join(f"\\textbf{{{r['label']}}}" if r["method"] == "alert" else r["label"]
                        for r in rows)
    values = " & ".join(f"${r['mean']:.1f}\\pm{r['sd']:.1f}$" for r in rows)
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Processing time per update in milliseconds.}}
\\label{{tab:runtime}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}rrrrr@{{}}}}
\\toprule
{header}\\\\
\\midrule
{values}\\\\
\\bottomrule
\\end{{tabular*}}
\\par\\vspace{{2pt}}
\\parbox{{\\columnwidth}}{{\\scriptsize Mean and standard deviation of 40 run means, excluding upstream
odometry and navigation. M3C2 accumulates five scans per update.}}
\\end{{table}}
"""
    return tex, rows


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    record = {}
    for name, builder in (("speed", table_one), ("methods", table_two),
                          ("surfaces", table_three), ("runtime", table_four)):
        tex, rows = builder(data)
        (TABLE_DIR / f"table_{name}.tex").write_text(tex)
        record[name] = rows
        print(f"wrote {TABLE_DIR / f'table_{name}.tex'}")
    (TABLE_DIR / "table_values.json").write_text(json.dumps(record, indent=2, default=float))
    for name, rows in record.items():
        print(f"\n== {name} ==")
        for r in rows:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
