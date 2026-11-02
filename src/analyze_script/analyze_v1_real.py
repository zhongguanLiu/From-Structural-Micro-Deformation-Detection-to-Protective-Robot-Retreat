#!/usr/bin/env python3
"""Real-data response timeline, using the recorded RViz arrow gates unchanged."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def stamp(value):
    return value["secs"] + value["nsecs"] * 1e-9


def records(run, name):
    with gzip.open(run / "measurements" / (name + ".jsonl.gz"), "rt") as stream:
        for line in stream:
            yield json.loads(line)


def recording_window(run, bag, times):
    """Use the executed bag interval, never silently crop out-of-window estimates."""
    path = run / "metadata/execution.json"
    execution = json.loads(path.read_text()) if path.exists() else {}
    start = float(execution.get("playback_start_s", 0.))
    end = (start + float(execution["playback_duration_s"]) if "playback_duration_s" in execution
           else float(bag["end"])-float(bag["start"]) if bag else times[-1])
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Invalid executed playback window")
    if not np.isfinite(times).all() or min(times) < start-1e-6 or max(times) > end+1e-6:
        raise ValueError("Recorded estimates fall outside the executed playback window")
    return start, end


def sample_response(sample, config):
    """Mirror BuildV1DeformationMarkers eligibility, but retain physical mm."""
    probability = float(sample["active_probability_lower"])
    threshold = config["minimum_arrow_active_probability"]
    displacement = np.linalg.norm([float(sample["displacement_mean"][k]) for k in ("x", "y", "z")])
    if threshold > 0 and (not np.isfinite(probability) or probability < threshold or probability > 1):
        return float(displacement)*1000, False
    eligible = (np.isfinite(displacement) and displacement > 0 and
                config["displacement_display_scale"]*displacement >= config["minimum_display_arrow_m"])
    return float(displacement)*1000, bool(eligible)


def frame_summary(samples, config):
    amplitudes, units = [], set()
    for sample in samples:
        amplitude, eligible = sample_response(sample, config)
        if eligible:
            amplitudes.append(amplitude)
            units.add((sample["gdu_id"]["high"], sample["gdu_id"]["low"]))
    quantiles = np.percentile(amplitudes, [25,50,75,100]) if amplitudes else [np.nan]*4
    return dict(zip(("p25_mm", "median_mm", "p75_mm", "maximum_mm"), map(float,quantiles)),
                responding_queries=len(amplitudes), responding_units=len(units),
                published_queries=len(samples))


def anchor_key(sample):
    # Physical queries lack surfel_id; the publisher uses exactly the frozen
    # surfel position. Match within session/epoch by GDU and exact position.
    position=tuple(float(sample["position"][k]) for k in ("x","y","z"))
    if not np.isfinite(position).all():
        raise ValueError("Nonfinite anchor position")
    return (sample["gdu_id"]["high"],sample["gdu_id"]["low"],*position)


def reference_index(support):
    lookup, ids={},set()
    for anchor in support:
        key=anchor_key(anchor)
        sid=(anchor["surfel_id"]["high"],anchor["surfel_id"]["low"])
        if key in lookup or sid in ids:
            raise ValueError("Ambiguous or duplicate reference anchor")
        lookup[key]=sid;ids.add(sid)
    return lookup


def anchor_counts(samples, lookup, config):
    estimated, responding=set(),set()
    for sample in samples:
        key=anchor_key(sample)
        if key not in lookup:
            raise ValueError("Displacement query is not a registered reference anchor")
        sid=lookup[key]
        if sid in estimated:
            raise ValueError("Duplicate displacement query for one anchor")
        estimated.add(sid)
        if sample_response(sample,config)[1]: responding.add(sid)
    return dict(reference_anchors=len(lookup),anchors_with_estimates=len(estimated),
                anchors_without_estimates=len(lookup)-len(estimated),responding_anchors=len(responding))


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k:"" if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()})


def response_intervals(rows, max_gap):
    """Runs of positive sampled responses, not proof of continuous physical motion."""
    intervals, current, previous = [], None, None
    for row in rows:
        disconnected = previous is not None and (
            row["time_s"]-previous["time_s"] > max_gap or
            (row["session"],row["epoch"]) != (previous["session"],previous["epoch"]))
        if current is not None and (disconnected or not row["responding_units"]):
            intervals.append(current); current=None
        if row["responding_units"]:
            if current is None:
                current=dict(first_sample_s=row["time_s"],last_sample_s=row["time_s"],
                    samples=0,session=row["session"],epoch=row["epoch"])
            current["last_sample_s"]=row["time_s"]
            current["samples"]+=1
        previous=row
    if current is not None: intervals.append(current)
    return intervals


def analyze(run):
    manifest = json.loads((run/"metadata/manifest.json").read_text())
    if not manifest["complete"] or not manifest["closed_cleanly"]:
        raise ValueError("Incomplete recording: inspect metadata/manifest.json")
    parameters = json.loads((run/"metadata/parameters.json").read_text())
    config = {k:float(parameters["/deform_monitor_v1_rviz"][k]) for k in
              ("minimum_arrow_active_probability", "displacement_display_scale", "minimum_display_arrow_m")}
    if not all(np.isfinite(list(config.values()))) or not 0<=config["minimum_arrow_active_probability"]<=1 or config["displacement_display_scale"]<=0 or config["minimum_display_arrow_m"]<0:
        raise ValueError("Invalid or unsupported recorded display parameters")
    info_path = run/"metadata/bag_info.yaml"
    bag = yaml.safe_load(info_path.read_text()) if info_path.exists() else None
    odometry = list(records(run,"odometry"))
    t0 = float(bag["start"]) if bag else stamp(odometry[0]["message"]["header"]["stamp"])
    rows, selected, epochs = [], [], []
    references={}
    for reference in records(run,"references"):
        common=reference["common"]
        epoch=(common["session_id"],common["reference_epoch"])
        if epoch in references: raise ValueError("Duplicate reference epoch")
        references[epoch]=reference_index(reference["reference_support"])
    for record in records(run,"snapshots"):
        common = record["common"]
        time = stamp(common["header"]["stamp"])-t0
        epoch = (common["session_id"], common["reference_epoch"])
        epochs.append(epoch)
        row = dict(time_s=time, session=epoch[0], epoch=epoch[1], **frame_summary(record["physical_deformation"],config))
        if epoch not in references: raise ValueError("Missing reference epoch")
        row.update(anchor_counts(record["physical_deformation"],references[epoch],config))
        row["formal_detected_queries"] = sum(e["detection_allowed"] and e["detection_admission"]==0 for e in record["detection_evidence"])
        rows.append(row)
        for sample in record["physical_deformation"]:
            amplitude, eligible = sample_response(sample,config)
            if eligible:
                selected.append(dict(time_s=time, session=epoch[0], epoch=epoch[1],
                    surfel_id="{:016x}{:016x}".format(*references[epoch][anchor_key(sample)]),
                    gdu_id=f'{sample["gdu_id"]["high"]:016x}{sample["gdu_id"]["low"]:016x}',
                    x_m=sample["position"]["x"],y_m=sample["position"]["y"],z_m=sample["position"]["z"],
                    displacement_mm=amplitude,active_probability_lower=sample["active_probability_lower"],
                    validity=sample["validity"],detection_admission=sample["detection_admission"]))
    if len(rows)<2 or any(b["time_s"]<=a["time_s"] for a,b in zip(rows,rows[1:])):
        raise ValueError("Need at least two time-ordered snapshots")
    start, end = recording_window(run, bag, [r["time_s"] for r in rows])
    output, figures = run/"analysis", run/"figures"
    output.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    csv_write(output/"response_timeline.csv",rows)
    if selected: csv_write(output/"response_samples.csv",selected)
    time = np.array([r["time_s"] for r in rows])
    gap = np.r_[False,np.diff(time)>2*np.median(np.diff(time))]
    gap |= np.array([False]+[a!=b for a,b in zip(epochs,epochs[1:])])
    intervals=response_intervals(rows,2*np.median(np.diff(time)))
    for interval in intervals:
        positive_rows=[r for r in rows if interval["first_sample_s"]<=r["time_s"]<=interval["last_sample_s"]
                       and (r["session"],r["epoch"])==(interval["session"],interval["epoch"])]
        groups=[{q["gdu_id"] for q in selected if q["time_s"]==r["time_s"] and
                 (q["session"],q["epoch"])==(r["session"],r["epoch"])} for r in positive_rows]
        interval["units_present_at_every_sample"]=len(set.intersection(*groups))
    if intervals: csv_write(output/"response_intervals.csv",intervals)
    # Insert a NaN separator, retaining both endpoint measurements of every gap.
    def series(key):
        x,y = [],[]
        for i,row in enumerate(rows):
            if gap[i]: x.append(np.nan);y.append(np.nan)
            x.append(row["time_s"]);y.append(row[key])
        return np.array(x),np.array(y)
    from matplotlib.font_manager import FontProperties, findfont, fontManager
    for path in (Path(__file__).resolve().parents[1]/"IEEE_ICRA2027/.local_fonts").glob("Times*.TTF"):
        fontManager.addfont(str(path))
    findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman",
        "font.size":8,"axes.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "axes.spines.top":False,"axes.spines.right":False})
    fig,(amplitude,count)=plt.subplots(2,1,sharex=True,figsize=(3.5,3.6),
        gridspec_kw={"height_ratios":[1.5,1]},layout="constrained")
    blue,red = "#276A92","#B84B4A"
    x,maximum=series("maximum_mm")
    _,median=series("median_mm");_,q25=series("p25_mm");_,q75=series("p75_mm")
    amplitude.fill_between(x,q25,q75,color=blue,alpha=.18,linewidth=0,label="Interquartile range")
    amplitude.plot(x,maximum,color=red,lw=1.25,label="Maximum")
    amplitude.plot(x,median,color=blue,lw=1.4,label="Median")
    amplitude.set(ylabel="Displacement (mm)",ylim=(0,None))
    amplitude.set_title("(a) Reported displacement",loc="left",fontsize=8)
    handles,labels=amplitude.get_legend_handles_labels()
    amplitude.legend([handles[i] for i in (1,2,0)],[labels[i] for i in (1,2,0)],
        loc="upper left",fontsize=6.5,frameon=False,handlelength=1.4)
    x,number=series("responding_anchors")
    _,total=series("reference_anchors")
    count.fill_between(x,number,step="post",color=blue,alpha=.15,linewidth=0)
    count.step(x,total,where="post",color="#737D86",linestyle="--",lw=1.2,label="All reference anchors")
    count.step(x,number,where="post",color=blue,lw=1.15,label="Anchors with responses")
    count.set_yscale("symlog",linthresh=10,linscale=1,base=10)
    upper=max(10,np.nanmax(total))*2
    count.set(ylabel="Number of anchors",xlabel="Time (s)",ylim=(0,upper))
    ticks=[0,10]+[10**i for i in range(2,8) if 10**i<=upper]
    count.set_yticks(ticks);count.set_yticklabels([f"{v:,}" for v in ticks])
    count.minorticks_off()
    count.set_title("(b) Reference anchors and deformation responses",loc="left",fontsize=7.5)
    count.legend(loc="upper left",bbox_to_anchor=(.24,.80),fontsize=6.3,frameon=False,
        handlelength=1.6,borderaxespad=0,ncol=1)
    count.annotate(f"{int(total[-1]):,}",(x[-1],total[-1]),xytext=(-4,4),
        textcoords="offset points",ha="right",fontsize=6.5,color="#737D86")
    count.text(.98,.04,"Linear 0–10; log above 10",transform=count.transAxes,
        ha="right",va="bottom",fontsize=6,color="#58616A")
    for axis in (amplitude,count):
        axis.set_xlim(start,end)
        axis.grid(axis="y",color="#DADDE1",linewidth=.45)
        axis.set_axisbelow(True)
        axis.axvspan(start,time[0],color="#ECEFF2",zorder=0)
        axis.axvline(time[0],color="#929BA4",lw=.7,linestyle=":")
        axis.tick_params(direction="out",length=3,width=.7)
    amplitude.text((start+time[0])/2,.48,"Initialization",transform=amplitude.get_xaxis_transform(),
        ha="center",va="center",rotation=90,color="#737D86",fontsize=7)
    if selected:
        peak=max(selected,key=lambda s:s["displacement_mm"])
        amplitude.annotate(f'{peak["displacement_mm"]:.1f} mm',
            (peak["time_s"],peak["displacement_mm"]),xytext=(-8,8),textcoords="offset points",
            ha="right",fontsize=7,color=red)
        amplitude.set_ylim(0,peak["displacement_mm"]*1.17)
    for suffix in ("pdf","png","svg"):
        fig.savefig(figures/f"fig_real_response.{suffix}",dpi=600)
    plt.close(fig)
    statuses=[r["message"] for r in records(run,"status")]
    processing=[s["last_attempt_processing_time_ms"] for s in statuses if s["last_attempt_published"]]
    summary=dict(source_run=str(run), playback_window_s=[start,end],
        time_origin="bag start" if bag else "first odometry stamp",time_zero_s=t0,
        snapshots=len(rows),first_estimate_s=time[0],last_estimate_s=time[-1],
        snapshots_with_display_response=sum(r["responding_units"]>0 for r in rows),
        zero_response_times_s=[r["time_s"] for r in rows if not r["responding_units"]],
        sampled_response_intervals=intervals,
        first_display_response_s=next((r["time_s"] for r in rows if r["responding_queries"]),None),
        maximum_responding_units=max(r["responding_units"] for r in rows),
        maximum_responding_anchors=max(r["responding_anchors"] for r in rows),
        reference_anchors_by_epoch=[dict(session=k[0],epoch=k[1],count=len(v)) for k,v in references.items()],
        anchor_count_axis="Shared symlog: linear 0–10, logarithmic above 10. Counts are not normalized.",
        recorded_display_gates=config,ground_truth_available=False,
        formal_detected_queries=sum(r["formal_detected_queries"] for r in rows),
        processing_median_ms=float(np.median(processing)),processing_p95_ms=float(np.percentile(processing,95)))
    sources=[run/"measurements/snapshots.jsonl.gz",run/"measurements/references.jsonl.gz",run/"measurements/status.jsonl.gz",
             run/"metadata/parameters.json",Path(__file__)]
    sources += [p for p in (run/"metadata/execution.json",run/"metadata/bag_info.yaml",run/"metadata/manifest.json") if p.exists()]
    summary["sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    caption=("Real-data response timeline. (a) Median, interquartile range, and maximum physical "
        "displacement magnitude among query estimates passing the recorded arrow-display gates. "
        "These are instantaneous population statistics, not one object's trajectory; the contributing "
        "queries can change over time. (b) Total registered reference anchors and the subset with "
        "displacement estimates passing the same display gates. Queries are matched to frozen anchors "
        "by reference epoch, spatial-unit identity, and reference position, and counted by unique surfel ID. "
        "An anchor without an estimate is not evidence of an unchanged surface. Both curves use one "
        "count axis, linear from 0 to 10 and logarithmic above 10. Shading denotes "
        "initialization, including initial data acquisition, reference "
        "construction, and estimator initialization; its boundary is the first estimate's input timestamp, "
        "not the wall-clock completion of a service call. Empty response sets "
        "produce zero counts but no displacement statistic. No independent displacement ground truth "
        "or physical-event timestamps are available; display responses are not calibrated confirmations.")
    (output/"FIGURE_CAPTION.md").write_text(caption+"\n",encoding="utf-8")
    (output/"REPORT.md").write_text(
        "# 真实实验响应时序\n\n"
        f"来源：{run}。分析执行的回放区间 {start:g}–{end:g} s；不裁剪越界估计、不修改检测结果。\n\n"
        "## 如何读图\n\n"
        "上图：满足本次箭头显示规则的估计位移，蓝线是中位数，浅蓝是中间 50% 的范围，红线是最大值。"
        "每一时刻统计当时全部满足条件的查询，不把曲线说成同一物体的连续轨迹。"
        "下图灰色虚线为全部登记参考锚点，蓝色曲线为其中满足原显示门槛的锚点。"
        "两条线都按同一参考周期的 surfel_id 去重计数；位移查询通过 GDU 与冻结位置精确对应，"
        "不把插值查询点数混作锚点数，不把物体数或区域数混入统计。"
        "为同时看清总量与少量响应，共用一个 0–10 线性、10 以上对数的纵轴；"
        "刻度都是实际锚点个数，不归一化，不使用双轴。\n\n"
        f"本次每帧参考锚点数范围 {min(r['reference_anchors'] for r in rows)}–{max(r['reference_anchors'] for r in rows)}，"
        f"具有位移查询的锚点数范围 {min(r['anchors_with_estimates'] for r in rows)}–{max(r['anchors_with_estimates'] for r in rows)}，"
        f"响应锚点最多 {max(r['responding_anchors'] for r in rows)} 个。"
        "只有参考、没有当前估计的锚点不能解释为未发生形变；图中总数指冻结参考覆盖范围，不是全场景。\n\n"
        f"本次显示阈值为 A+C 下界 ≥ {config['minimum_arrow_active_probability']:g}、物理位移 ≥ "
        f"{1000*config['minimum_display_arrow_m']/config['displacement_display_scale']:g} mm。"
        "直接复现 rviz_adapter.cpp 的箭头门槛，没有额外人工选区或后验挑阈值。"
        "统计使用未放大、未按显示长度截断的物理位移。与当前 RViz 一样，不额外剔除 validity 标记；"
        "标记保存在 response_samples.csv，显示响应不等于正式确认。\n\n"
        f"记录 {len(rows)} 次检测输出，覆盖 {time[0]:.2f}–{time[-1]:.2f} s；"
        f"最多 {summary['maximum_responding_units']} 个单元产生显示响应。"
        f"正式 DETECTED 条目 {summary['formal_detected_queries']}。"
        "初始灰段标为 Initialization，包含初始数据采集、参考构建和估计器初始化，不能解释为已观测到静止。"
        "灰段边界仍为首份估计的输入时刻，不冒充服务执行完成时刻；没有响应的快照在上图断开、下图计零。"
        "大于中位采样间隔两倍的缺口和参考周期变更处断线，不作平滑。\n\n"
        "## 响应是否持续\n\n"
        f"{sum(r['responding_units']>0 for r in rows)}/{len(rows)} 次快照有显示响应。"
        "response_intervals.csv 列出连续采样有响应的区间，并统计每个区间始终出现的同一 GDU 数量。"
        "没有显示响应只说明当时没有查询满足显示门槛，不证明实际形变消失；"
        "持续有某个单元响应，不等于同一物体持续形变。\n\n"
        "## 手动标注\n\n"
        "fig_real_response.svg 保留矢量曲线与可编辑文字，可直接添加实际事件竖线和文字。"
        "自动的 First response 箭头已移除，以免与人为核对的首次真实形变混淆；"
        "算法首个显示响应仍保存在 summary.json。PDF/PNG/SVG 中所有横坐标都是同一个时间轴，"
        f"本次 0 s 对应传感器/回放时刻 {t0:.6f} s，不是节点启动时间。"
        "若事件来自视频，需先对齐视频与 bag 时间，不能直接用打开播放器后的墙钟秒数。\n\n"
        "## 数据与适用范围\n\n"
        "response_timeline.csv 保存每个快照统计，response_samples.csv 保存全部入图查询的位置、"
        "surfel_id、GDU、位移、门槛概率和有效性，原始 measurements 保留所有未入图结果。"
        "summary.json 保存参数及源数据哈希。FIGURE_CAPTION.md 为英文图注草稿，未写入论文。\n\n"
        "现有数据满足这张响应时序图，不满足真实位移精度、运动起点或预警提前量分析；"
        "没有把原 ALERT 论文中的坍塌事件时刻迁移到本次回放，也没有凭曲线虚构动作阶段。"
        "本次数据不能说明最初未处理时段的检测表现，重复绘图或重复回放不能补出独立真值。\n",
        encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    analyze(parser.parse_args().run.resolve())
