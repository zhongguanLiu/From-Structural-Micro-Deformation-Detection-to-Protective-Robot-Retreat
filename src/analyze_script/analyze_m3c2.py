#!/usr/bin/env python3
"""M3C2 source-time detection and signed-normal accuracy; no detector feedback."""
import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'IEEE_ICRA2027'))
import confirmation_audit as confirmation
import analyze_compare_methods as common

MODELS = confirmation.MODELS


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def geometry(cores, manifest, initial):
    x, y, z, yaw = manifest['camera_init_origin_gazebo_xyzyaw']
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = cores[['x', 'y', 'z']].to_numpy() @ rotation.T + [x, y, z]
    normals = cores[['nx', 'ny', 'nz']].to_numpy() @ rotation.T
    models = {m['model_name']: m for m in initial['models']}
    contains = np.column_stack([np.all(
        (points >= np.array(models[m]['aabb_min']) - 1e-12) &
        (points <= np.array(models[m]['aabb_max']) + 1e-12), axis=1) for m in MODELS])
    owners = np.where(contains.sum(axis=1) == 1, contains.argmax(axis=1), -1)
    return owners, normals


def run_metrics(directory, record):
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    check = json.loads((directory / 'm3c2/record_check.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['method'] == 'm3c2'
    for relative, expected in check['record_sha256'].items():
        assert sha256(directory / relative) == expected, relative
    origin, truths, truth_path = confirmation.truth_data(directory, 'm3c2')
    events = confirmation.baseline_events(record, directory, origin, truths, truth_path)
    with gzip.open(truth_path, 'rt') as f:
        initial = json.loads(next(f))
        for line in f:
            assert all(abs(m['rotation_angle']) < 1e-8 for m in json.loads(line)['models']), \
                'This accuracy adapter requires the recorded pure-translation experiment'
    cores = pd.read_csv(directory / 'm3c2/reference_cores.csv.gz')
    frames = pd.read_csv(directory / 'm3c2/frames.csv.gz')
    assert np.array_equal(cores.core_id, np.arange(len(cores)))
    assert np.array_equal(frames.frame_index, np.arange(len(frames)))
    owners, normals = geometry(cores, manifest, initial)
    stamps = frames.source_stamp_s.to_numpy()
    elapsed = stamps - origin
    inside = (elapsed >= 0) & (elapsed < 60)
    truth_covered = inside.copy()
    for times, _ in truths.values():
        truth_covered &= (stamps >= times[0]) & (stamps <= times[-1])
    translations = np.zeros((len(frames), 6, 3))
    for mi, model in enumerate(MODELS):
        times, vectors = truths[model]
        for axis in range(3):
            translations[:, mi, axis] = np.interp(stamps, times, vectors[:, axis])
    # Streaming aggregation keeps all original core records on disk, not in RAM.
    sums = np.zeros((len(frames) * 6, 11))
    processed = 0
    for chunk in pd.read_csv(directory / 'm3c2/measurements.csv.gz', chunksize=250000):
        fi, ci = chunk.frame_index.to_numpy(), chunk.core_id.to_numpy()
        sequence = np.arange(processed, processed + len(chunk))
        assert np.array_equal(fi * len(cores) + ci, sequence)
        processed += len(chunk)
        owner = owners[ci]
        select = (owner >= 0) & truth_covered[fi]
        f, c, m = fi[select], ci[select], owner[select]
        ids = f * 6 + m
        values = chunk.loc[select]
        valid = values.valid.to_numpy().astype(bool)
        d = values.signed_distance_m.to_numpy() * 1000
        gt = np.einsum('ij,ij->i', normals[c], translations[f, m]) * 1000
        error = d - gt
        micro = valid & (np.abs(gt) >= 5) & (np.abs(gt) <= 20)
        arrays = [np.ones(len(ids)), valid, values.significant.to_numpy(),
                  np.where(valid, error**2, 0), np.where(valid, np.abs(error), 0),
                  np.where(valid, np.abs(d), 0), np.where(valid, np.abs(gt), 0),
                  micro, np.where(micro, error**2, 0),
                  np.where(valid, d, 0), np.where(valid, gt, 0)]
        for col, weights in enumerate(arrays):
            sums[:, col] += np.bincount(ids, weights=weights, minlength=len(sums))
    assert processed == check['samples'] == len(frames) * len(cores)
    series, metrics = [], []
    detection_frames = common.load_frames(directory / 'detections/detection_frames.csv.gz')
    baseline = [f for f in detection_frames if 0 <= f['source_stamp_s'] - origin < 10]
    for mi, model in enumerate(MODELS):
        a = sums[mi::6]
        event = events[mi]
        assert event['model_name'] == model
        valid = a[:, 1] > 0
        for index in np.flatnonzero(inside):
            r = a[index]
            series.append(dict(speed_mm_s=float(record['speed_mm_s']), repeat=int(record['repeat']),
                model_name=model, source_stamp_s=stamps[index], elapsed_s=elapsed[index],
                bin_start_elapsed_s=math.floor(elapsed[index] * 2) / 2,
                core_count=int(r[0]), valid_count=int(r[1]), significant_count=int(r[2]),
                estimate_abs_mm=r[5]/r[1] if r[1] else None,
                truth_abs_mm=r[6]/r[1] if r[1] else None,
                signed_estimate_mm=r[9]/r[1] if r[1] else None,
                signed_truth_mm=r[10]/r[1] if r[1] else None,
                rmse_mm=math.sqrt(r[3]/r[1]) if r[1] else None))
        row = dict(speed_mm_s=float(record['speed_mm_s']), repeat=int(record['repeat']),
                   model_name=model, reference_cores=int((owners == mi).sum()),
                   response=event['detected_1_frame'], confirmed=event['detected'],
                   source_directory=str(directory))
        for phase, mask, denominator in (
                ('all', inside, 120), ('motion', inside & (elapsed >= 10) & (elapsed < 50), 80),
                ('hold', inside & (elapsed >= 50), 20)):
            v = a[mask].sum(axis=0)
            row[phase + '_valid_samples'] = int(v[1])
            row[phase + '_rmse_mm'] = math.sqrt(v[3]/v[1]) if v[1] else None
            row[phase + '_mae_mm'] = v[4]/v[1] if v[1] else None
            bins = set(np.floor(elapsed[mask & valid] * 2).astype(int))
            row[phase + '_covered_bins'] = len(bins)
            row[phase + '_possible_bins'] = denominator
        v = a[inside].sum(axis=0)
        row['micro_valid_samples'] = int(v[7])
        row['micro_rmse_mm'] = math.sqrt(v[8]/v[7]) if v[7] else None
        confirmed_stamp = event['confirmation_source_stamp_s']
        eligible = inside & (stamps >= confirmed_stamp) if confirmed_stamp is not None else inside & False
        v = a[eligible].sum(axis=0)
        row['confirmed_micro_valid_samples'] = int(v[7])
        row['confirmed_micro_rmse_mm'] = math.sqrt(v[8]/v[7]) if v[7] else None
        _, false_confirmation = common.confirmation(
            [dict(f, point_count=f['counts'][model]) for f in baseline], 1, 3, .35)
        row['baseline_confirmed'] = false_confirmation is not None
        row['baseline_positive_frames'] = sum(f['counts'][model] > 0 for f in baseline)
        metrics.append(row)
    selected = frames[inside]
    runtime = dict(method='m3c2', speed_mm_s=float(record['speed_mm_s']), repeat=int(record['repeat']),
        runtime_source=str(directory / 'm3c2/frames.csv.gz'), cycle_count=len(selected),
        cycle_mean_ms=float(selected.processing_time_ms.mean()),
        cycle_p95_ms=float(selected.processing_time_ms.quantile(.95)),
        input_point_count_mean=float(selected.input_points.mean()),
        method_mean_ms=float(selected.method_time_ms.mean()),
        recording_mean_ms=float(selected.recording_time_ms.mean()),
        fusion_window_span_mean_s=float((selected.window_last_source_s-selected.window_first_source_s).mean()),
        source_interval_mean_s=float(np.diff(selected.source_stamp_s).mean()),
        truth_unavailable_frame_count=int((inside & ~truth_covered).sum()),
        other_positive_frames=sum(f['counts']['OTHER'] > 0 for f in detection_frames
                                  if 0 <= f['source_stamp_s'] - origin < 60))
    return events, metrics, series, runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--paper', action='store_true', help='publish audited compact CSVs to the manuscript')
    args = parser.parse_args()
    batch = args.batch.resolve()
    ledger = common.read_csv(batch / 'batch_status.csv')
    assert len(ledger) == 40 and all(r['status'] == 'SUCCESS' and r['method'] == 'm3c2' for r in ledger)
    assert {(float(r['speed_mm_s']), int(r['repeat'])) for r in ledger} == {
        (s, r) for s in (1, 3, 5, 10) for r in range(1, 11)}
    output = batch / 'analysis'
    output.mkdir(exist_ok=True)
    events, metrics, series, runtimes, sources = [], [], [], [], []
    configurations = set()
    for record in ledger:
        directory = batch / 'runs' / record['scenario_id']
        run_manifest = json.loads((directory / 'run_manifest.json').read_text())
        with gzip.open(directory / 'configuration/ros_parameters.json.gz','rt') as f:
            parameters = json.load(f)['m3c2_baseline']
        parameters.pop('output_directory')
        configurations.add(json.dumps([run_manifest['world_file_sha256'],
            run_manifest['motion_config_file_sha256'],run_manifest['camera_init_origin_gazebo_xyzyaw'],
            parameters],sort_keys=True))
        assert len(configurations) == 1, 'M3C2 experiment configuration changed within the batch'
        e, m, s, r = run_metrics(directory, record)
        events.extend(e); metrics.extend(m); series.extend(s); runtimes.append(r)
        for relative in ('run_manifest.json', 'm3c2/record_check.json',
                         'truth/motion_truth.jsonl.gz', 'detections/detection_frames.csv.gz',
                         'configuration/ros_parameters.json.gz'):
            path = directory / relative
            sources.append(dict(path=str(path), sha256=sha256(path)))
        print('ANALYZED', record['scenario_id'], flush=True)
    summaries, curves = confirmation.statistics_rows(events, methods=('m3c2',), labels={'m3c2':'M3C2'})
    accuracy = pd.DataFrame(metrics).groupby(['speed_mm_s','model_name'])
    accuracy_summary = []
    for (speed, model), group in accuracy:
        row = dict(speed_mm_s=speed, model_name=model, n_runs=len(group))
        for key in ('micro_rmse_mm', 'confirmed_micro_rmse_mm', 'all_rmse_mm', 'hold_rmse_mm'):
            values = group[key].dropna()
            row.update({key+'_n':len(values), key+'_mean':float(values.mean()) if len(values) else None,
                        key+'_sd':float(values.std()) if len(values)>1 else None})
        for phase in ('motion', 'hold'):
            row[phase+'_coverage_percent'] = 100*group[phase+'_covered_bins'].sum()/group[phase+'_possible_bins'].sum()
        row['baseline_confirmed_runs'] = int(group.baseline_confirmed.sum())
        accuracy_summary.append(row)
    tables = dict(events=events, summary=summaries, curves=curves, accuracy_runs=metrics,
                  accuracy_summary=accuracy_summary, tracking_frames=series, runtime_runs=runtimes)
    for name, rows in tables.items():
        common.write_csv(output / (name+'.csv'), rows)
    manifest = dict(run_count=40, event_count=240, batch=str(batch), sources=sources,
        script_sha256=sha256(Path(__file__)),
        helper_sha256={str(Path(p)):sha256(Path(p)) for p in (confirmation.__file__, common.__file__)},
        error_policy='signed distance minus same-normal projected recorded translation; unique initial AABB, no padding; all valid samples, with post-confirmation cohort separately reported',
        time_policy='measurement source timestamp; truth interpolated without extrapolation; trailing fusion-window endpoints retained',
        output_sha256={name+'.csv':sha256(output/(name+'.csv')) for name in tables})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    text = ['# M3C2 results', '', '40 runs; 1, 3, 5, 10 mm/s, ten runs each. Six targets per run.', '',
            '| Speed (mm/s) | Initial responses /60 | Confirmations /60 | Median delay (s) | W40 (s) |',
            '|---:|---:|---:|---:|---:|']
    for r in summaries:
        delay = '--' if r['confirmation_latency_median_s'] is None else f"{r['confirmation_latency_median_s']:.2f}"
        text.append(f"| {r['speed_mm_s']:g} | {r['confirmed_count_1_frame']} | {r['confirmed_count']} | {delay} | {r['restricted_mean_wait_s']:.2f} |")
    text += ['', 'Accuracy uses signed normal displacement, not full translation magnitude. Invalid, nonsignificant and off-target outputs are retained in raw records. See accuracy_summary.csv for per-target run-level errors and independent output coverage. Three-frame confirmation follows the existing comparison policy, including its 0.35 s maximum source-time gap.', '',
             'Tracking plots average absolute normal displacements over the valid, uniquely attributable cores of each frame. Truth uses exactly the same cores and normals, not the object-centre translation magnitude. Thin lines are individual runs; thick lines and shading show available-run medians and IQRs of half-second bin means. Missing bins remain gaps. Changes in the valid-core subset can change projected truth even during the nominal hold.']
    text += ['', '## Normal-distance accuracy', '',
             'Errors below are means of per-run RMSEs, in mm; n counts contributing runs. Full-range errors include all valid measurements in 0–60 s. Coverage counts half-second bins containing a valid, uniquely attributable measurement, with all ten runs in the denominator.', '',
             '| Speed | Target | 5–20 mm RMSE | n | Full-range RMSE | Motion coverage (%) | Hold coverage (%) |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for r in accuracy_summary:
        micro = '--' if r['micro_rmse_mm_mean'] is None else f"{r['micro_rmse_mm_mean']:.2f}"
        full = '--' if r['all_rmse_mm_mean'] is None else f"{r['all_rmse_mm_mean']:.2f}"
        text.append(f"| {r['speed_mm_s']:g} | {r['model_name'].upper()} | {micro} | {r['micro_rmse_mm_n']} | {full} | {r['motion_coverage_percent']:.1f} | {r['hold_coverage_percent']:.1f} |")
    text += ['', '## Output cadence and baseline responses', '',
             'Timing entries average the ten per-run means at each speed. Baseline responses count target–run events with at least one positive frame while the recorded objects are stationary; they are not a calibrated long-duration false-alarm rate.', '',
             '| Speed | Processing (ms) | Recording (ms) | Source interval (s) | Fusion span (s) | Baseline response /60 | Baseline confirmation /60 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for speed in (1,3,5,10):
        group = pd.DataFrame([r for r in runtimes if r['speed_mm_s'] == speed])
        targets = [r for r in metrics if r['speed_mm_s'] == speed]
        text.append(f"| {speed} | {group.cycle_mean_ms.mean():.2f} | {group.recording_mean_ms.mean():.2f} | {group.source_interval_mean_s.mean():.3f} | {group.fusion_window_span_mean_s.mean():.3f} | {sum(r['baseline_positive_frames']>0 for r in targets)} | {sum(r['baseline_confirmed'] for r in targets)} |")
    text += ['', 'Raw core records include background, invalid support and nonsignificant measurements. Only the independent accuracy attribution restricts cores to unique target geometry. No detection misses are discarded or retried. manifest.json links source records, integrity checks and derived CSV hashes. The per-speed tracking figures are fig_m3c2_tracking_*mmps.pdf/png; per-run errors, sample counts and stage timings remain in the CSV files.']
    (output / 'REPORT.md').write_text('\n'.join(text)+'\n')
    plot_results(output)
    if args.paper:
        destination = SOURCE / 'IEEE_ICRA2027/m3c2_analysis'
        destination.mkdir(exist_ok=True)
        names = ['events', 'summary', 'curves', 'accuracy_summary', 'accuracy_runs', 'runtime_runs']
        for name in names:
            shutil.copy2(output / (name+'.csv'), destination / (name+'.csv'))
        shutil.copy2(output / 'manifest.json', destination / 'manifest.json')
    print('M3C2 ANALYSIS COMPLETE', output)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import compare_figures as style
    style.configure()
    frame = pd.read_csv(output / 'tracking_frames.csv')
    for speed in (1, 3, 5, 10):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1), sharex=True)
        for ax, model in zip(axes.flat, MODELS):
            selected = frame[(frame.speed_mm_s == speed) & (frame.model_name == model)]
            binned = selected.groupby(['repeat','bin_start_elapsed_s'])[['estimate_abs_mm','truth_abs_mm']].mean()
            binned = binned.reindex(pd.MultiIndex.from_product(
                [range(1,11),np.arange(0,60,.5)],names=['repeat','bin_start_elapsed_s']))
            for _, run in binned.groupby(level=0):
                ax.plot(run.index.get_level_values(1)+.25, run.estimate_abs_mm, color='#765C45', alpha=.15, lw=.5)
            for key, color, label in [('estimate_abs_mm','#765C45','M3C2'),('truth_abs_mm','#30383F','Projected truth')]:
                q = binned[key].groupby(level=1).quantile([.25,.5,.75]).unstack()
                x = q.index.to_numpy()+.25
                ax.plot(x,q[.5],color=color,lw=1.2,label=label,ls='-' if key.startswith('estimate') else '--')
                ax.fill_between(x,q[.25],q[.75],color=color,alpha=.12,lw=0)
            ax.axvspan(10,50,color='#E9EFF2',zorder=-1)
            ax.set_title(model.upper(),loc='left',fontsize=9)
            ax.set_xlim(0,60)
            ax.spines[['right','top']].set_visible(False)
            ax.grid(axis='y',ls=':',lw=.4)
        for ax in axes[1]: ax.set_xlabel('Experiment time (s)')
        for ax in axes[:,0]: ax.set_ylabel('Normal displacement\nmagnitude (mm)')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.subplots_adjust(left=.09,right=.99,bottom=.13,top=.87,wspace=.36,hspace=.42)
        fig.savefig(output / f'fig_m3c2_tracking_{speed}mmps.pdf')
        fig.savefig(output / f'fig_m3c2_tracking_{speed}mmps.png',dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
