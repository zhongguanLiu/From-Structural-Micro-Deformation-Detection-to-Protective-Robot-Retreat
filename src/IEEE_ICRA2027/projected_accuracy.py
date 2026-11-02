#!/usr/bin/env python3
"""Offline observable-point accuracy from the recorded stationary experiments.

Writes only paper-local CSV/JSON. No command integration, detector modification,
downstream carried estimates, uncertainty calibration, or missing-value filling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


PAPER = Path(__file__).resolve().parent
BATCH = PAPER.parent.parent / "experiment_output/stationary_speed_sweep/batch_20260901_012253"
MODELS = ["m%02d" % number for number in range(1, 7)]
XYZ = list("xyz")
POSE_POSITION = ["position_%s_m" % axis for axis in XYZ]
POSE_QUATERNION = ["orientation_%s" % axis for axis in "xyzw"]
MAP_POSITION = ["position_map_%s_m" % axis for axis in XYZ]
WORLD_POSITION = ["position_world_%s_m" % axis for axis in XYZ]
ESTIMATE = ["estimated_d%s_m" % axis for axis in XYZ]
SOURCE_KEY = ["reference_epoch", "snapshot_id", "tile_key", "sample_index"]
REFERENCE_KEY = ["reference_epoch", "gdu_id"] + MAP_POSITION
GDU_FRAME = ["reference_epoch", "accepted_frame_index", "gdu_id"]
POSE_TOLERANCE = 1e-8  # Absolute m/rad; exceeds the truth JSON's decimal rounding.
MAX_TRUTH_GAP = 0.101  # Recorded 10 Hz truth cadence, with 1 ms clock allowance.
INPUTS = ["run_manifest.json", "deformation/algorithm_displacement.csv",
          "deformation/reference_support.csv", "deformation/detection_evidence.csv",
          "truth/m01_m06_motion_truth.csv", "analysis/detection_track_inventory.csv"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrices(series):
    result = np.stack([np.fromstring(value, sep="|") for value in series]).reshape(-1, 3, 3)
    assert np.isfinite(result).all(), "Nonfinite recorded matrix"
    return result


def projector_ranks(projectors):
    assert np.max(np.linalg.norm(projectors - projectors.transpose(0, 2, 1), axis=(1, 2))) < 1e-8
    assert np.max(np.linalg.norm(projectors @ projectors - projectors, axis=(1, 2))) < 1e-8
    rank = np.rint(np.trace(projectors, axis1=1, axis2=2)).astype(int)
    assert np.all((rank >= 0) & (rank <= 3))
    return rank


def initial_owners(points_world, truths):
    first = {model: truth["frame"].iloc[0] for model, truth in truths.items()}
    containment = np.column_stack([np.all(
        (points_world >= first[model][["world_aabb_min_%s_m" % axis for axis in XYZ]].to_numpy(dtype=float) - 1e-12)
        & (points_world <= first[model][["world_aabb_max_%s_m" % axis for axis in XYZ]].to_numpy(dtype=float) + 1e-12), axis=1)
        for model in MODELS])
    counts = containment.sum(axis=1)
    owners = np.where(counts == 1, np.asarray(MODELS)[containment.argmax(axis=1)], "")
    return owners, counts


def same_point_displacement(points_world, initial_position, initial_rotation,
                            positions, rotations, map_rotation):
    local = (points_world - initial_position) @ initial_rotation
    moved = np.einsum("nij,nj->ni", rotations, local) + positions
    return (moved - points_world) @ map_rotation.T


def prepare_truth(frame):
    """Keep real endpoint poses; flag unresolved interpolation, not large errors."""
    frame = frame.sort_values("truth_sim_time_s", kind="stable")
    for _, repeated in frame[frame.duplicated("truth_sim_time_s", False)].groupby("truth_sim_time_s"):
        assert np.ptp(repeated[POSE_POSITION + POSE_QUATERNION].to_numpy(), axis=0).max() < POSE_TOLERANCE
    frame = frame.drop_duplicates("truth_sim_time_s", keep="last").reset_index(drop=True)
    times = frame.truth_sim_time_s.to_numpy()
    positions = frame[POSE_POSITION].to_numpy()
    rotations = Rotation.from_quat(frame[POSE_QUATERNION].to_numpy())
    elapsed = frame.experiment_elapsed_s.to_numpy()
    assert np.all(np.diff(times) > 0)
    assert np.ptp(times - elapsed) < 1e-7, "Experiment/source clocks are not affine"
    waiting = frame.motion_state.eq("WAITING").to_numpy()
    assert waiting.any() and np.ptp(positions[waiting], axis=0).max() <= POSE_TOLERANCE
    assert (rotations[waiting][0].inv() * rotations[waiting]).magnitude().max() <= POSE_TOLERANCE
    dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    da = (rotations[:-1].inv() * rotations[1:]).magnitude()
    changed = (dp > POSE_TOLERANCE) | (da > POSE_TOLERANCE)
    progress = frame.model_motion_elapsed_s.to_numpy()
    static_progress = np.abs(np.diff(progress)) <= 1e-10
    # No motion is prescribed between these equal-progress updates. A changed
    # pose is retained, but its sub-100 ms transition trajectory is unrecorded.
    transitions = changed & static_progress
    gaps = np.diff(times) > MAX_TRUTH_GAP
    hold_rows = np.flatnonzero(elapsed >= 50 - 1e-8)
    hold_start = 50.0
    if len(hold_rows):
        changed_hold = np.flatnonzero(changed & (elapsed[1:] > 50 + 1e-8))
        if len(changed_hold):
            hold_start = float(elapsed[changed_hold[-1] + 1])
    hold_available = bool(len(hold_rows) and elapsed[-1] >= 60 - 1e-8
                          and 60 - hold_start >= .1 - 1e-8
                          and not np.any(gaps & (elapsed[1:] > hold_start)))
    audit = {"initial_truth_time_s": float(times[0]), "clock_offset_s": float(times[0] - elapsed[0]),
             "hold_start_elapsed_s": hold_start if hold_available else None,
             "stable_hold_duration_s": 60 - hold_start if hold_available else None,
             "truth_gap_count": int(gaps.sum()),
             "unresolved_static_transition_intervals": [
                 {"before_elapsed_s": float(elapsed[i]), "after_elapsed_s": float(elapsed[i + 1]),
                  "translation_step_m": float(dp[i]), "rotation_step_rad": float(da[i])}
                 for i in np.flatnonzero(transitions)]}
    return {"frame": frame, "times": times, "positions": positions, "rotations": rotations,
            "excluded_intervals": transitions | gaps, "transitions": transitions,
            "audit": audit, "hold_start": hold_start if hold_available else np.nan}


def interpolate_truth(truth, stamps):
    times = truth["times"]
    index = np.searchsorted(times, stamps, side="left")
    endpoint = np.minimum(index, len(times) - 1)
    exact = np.abs(times[endpoint] - stamps) <= 1e-9
    inside = (stamps >= times[0] - 1e-9) & (stamps <= times[-1] + 1e-9)
    lower = np.clip(index - 1, 0, len(times) - 2)
    transition = inside & ~exact & truth["transitions"][lower]
    excluded = inside & ~exact & truth["excluded_intervals"][lower]
    valid = inside & ~excluded
    clipped = np.clip(stamps[valid], times[0], times[-1])
    positions = np.column_stack([np.interp(clipped, times, truth["positions"][:, axis]) for axis in range(3)])
    rotations = (Slerp(times, truth["rotations"])(clipped).as_matrix()
                 if len(clipped) else np.empty((0, 3, 3)))
    return valid, inside, transition, positions, rotations


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan


def bin_rows(frame, metadata, model, state=False):
    """Explicit empty half-second bins; no interpolation of absent estimates."""
    frame = frame[(frame.elapsed_s >= 0) & (frame.elapsed_s < 60)].copy()
    frame["bin"] = np.floor(frame.elapsed_s * 2 + 1e-9).astype(int)
    output = []
    groups = dict(tuple(frame.groupby("bin")))
    for index in range(120):
        row = dict(metadata, model_name=model, bin_start_elapsed_s=index / 2,
                   bin_end_elapsed_s=(index + 1) / 2)
        group = groups.get(index, frame.iloc[:0])
        row.update(n_gdu_frames=len(group[GDU_FRAME].drop_duplicates()),
                   n_source_frames=group.accepted_frame_index.nunique(),
                   n_gdus=group.gdu_id.nunique())
        if state:
            row.update(n_probability_rows=len(group), **{
                column + "_mean": group[column].mean() for column in ("p_u", "p_a", "p_c")})
        else:
            row.update(n_samples=len(group), estimate_projected_norm_mean_mm=group.estimate_norm_mm.mean(),
                       truth_projected_norm_mean_mm=group.truth_norm_mm.mean(),
                       vector_error_mean_mm=group.vector_error_mm.mean(),
                       vector_rmse_mm=rms(group.vector_error_mm),
                       projected_magnitude_bias_mm=group.magnitude_bias_mm.mean(),
                       covariance_trace_sigma_mean_mm=group.sigma_mm.mean(),
                       n_accuracy_eligible_samples=int(group.accuracy_eligible.sum()))
        output.append(row)
    return output


def independent_bin_rows(owned, truths, metadata):
    """Geometry-only published population; neither tracks nor confirmation select it."""
    output = []
    for model in MODELS:
        truth = truths[model]
        selected = owned[owned.geometry_model_name.eq(model)].copy()
        selected["elapsed_s"] = selected.message_time_s - truth["audit"]["clock_offset_s"]
        selected = selected[selected.elapsed_s.between(0, 60, inclusive="left")].copy()
        selected["local_rank"] = (projector_ranks(matrices(selected.observable_projector_row_major))
                                  if len(selected) else pd.Series(dtype=int))
        stamps, inverse = np.unique(selected.message_time_s.to_numpy(), return_inverse=True)
        truth_valid = interpolate_truth(truth, stamps)[0][inverse]
        selected["evaluable"] = (truth_valid & selected.local_rank.gt(0)
                                   & np.isfinite(selected[ESTIMATE]).all(axis=1))
        selected["truth_valid"] = truth_valid
        selected["bin"] = np.floor(selected.elapsed_s * 2 + 1e-9).astype(int)
        groups = dict(tuple(selected.groupby("bin")))
        for index in range(120):
            group = groups.get(index, selected.iloc[:0])
            evaluable = group[group.evaluable]
            counts = {"rank%d_n_samples" % rank: int(group.local_rank.eq(rank).sum()) for rank in range(4)}
            assert sum(counts.values()) == len(group)
            output.append(dict(metadata, model_name=model, bin_start_elapsed_s=index / 2,
                               bin_end_elapsed_s=(index + 1) / 2,
                               evaluable_available=bool(len(evaluable)),
                               n_published_samples=len(group), n_truth_valid_samples=int(group.truth_valid.sum()),
                               n_evaluable_samples=len(evaluable),
                               n_published_source_frames=group.accepted_frame_index.nunique(),
                               n_evaluable_source_frames=evaluable.accepted_frame_index.nunique(), **counts))
    return output


def analyze_run(run, batch):
    directory = batch / "runs" / run.scenario_id
    metadata = {"run_id": int(run.run_id), "run_name": run.scenario_id,
                "speed_mm_s": float(run.speed_mm_s), "repeat": int(run.repeat)}
    manifest = json.loads((directory / "run_manifest.json").read_text())
    truth_frame = pd.read_csv(directory / INPUTS[4])
    truths = {model: prepare_truth(group) for model, group in truth_frame.groupby("model_name")}
    assert set(truths) == set(MODELS)
    track = pd.read_csv(directory / INPUTS[5])
    track = track[~track.gdu_id.duplicated(False) & track.association_status.eq("MATCHED")].copy()
    track = track[track.model_name.isin(MODELS)]
    track["confirmation_s"] = pd.to_numeric(track.confirmation_experiment_elapsed_s, errors="coerce")
    native_confirmed = track.confirmed & track.confirmation_time_available & track.confirmation_s.notna()
    track["confirmed_cohort"] = native_confirmed & track.confirmation_s.between(10, 50, inclusive="left")
    legacy_excluded = track[native_confirmed & track.evaluation_category.ne("TRUE_POSITIVE")][
        ["gdu_id", "model_name", "confirmation_s", "evaluation_category", "confirmed_cohort"]].to_dict("records")
    track = track[["gdu_id", "model_name", "confirmation_s", "confirmed_cohort"]]
    support = pd.read_csv(directory / INPUTS[2], usecols=["reference_epoch", "gdu_id", "surfel_id",
                                                        "message_time_s"] + POSE_POSITION)
    support = support.rename(columns=dict(zip(POSE_POSITION, MAP_POSITION)))
    assert not support[REFERENCE_KEY].duplicated().any(), "Ambiguous frozen coordinate identity"
    columns = SOURCE_KEY + ["accepted_frame_index", "gdu_id", "message_time_s"] + MAP_POSITION + WORLD_POSITION + ESTIMATE
    columns += ["observable_projector_row_major", "displacement_covariance_row_major", "validity", "detection_admission"]
    raw = pd.read_csv(directory / INPUTS[1], usecols=columns)
    assert not raw[SOURCE_KEY].duplicated().any(), "Duplicate published physical sample"
    joined = raw.merge(support[REFERENCE_KEY + ["surfel_id"]], on=REFERENCE_KEY, how="left", validate="many_to_one")
    assert joined.surfel_id.notna().all(), "Missing exact frozen-reference correspondence"
    audit = dict(metadata, raw_sample_count=len(raw), reference_sample_count=len(support),
                 reference_epoch_count=int(support.reference_epoch.nunique()),
                 duplicate_sample_keys=0, missing_reference_joins=0,
                 reference_first_message_time_s=float(support.message_time_s.min()),
                 raw_validity_counts={str(k): int(v) for k, v in raw.validity.value_counts().items()},
                 raw_admission_counts={str(k): int(v) for k, v in raw.detection_admission.value_counts().items()},
                 native_confirmations_with_legacy_excluded_labels=legacy_excluded)
    anchor = manifest["initial_map_from_world_se2"]
    rotation = Rotation.from_euler("z", anchor[2]).as_matrix()
    translation = np.array([anchor[0], anchor[1], manifest["initial_z_map_from_world_m"]])
    expected_world = (joined[MAP_POSITION].to_numpy() - translation) @ rotation
    assert np.allclose(expected_world, joined[WORLD_POSITION].to_numpy(), atol=1e-10, rtol=0), "Stored coordinate anchor mismatch"
    joined["geometry_model_name"], membership_counts = initial_owners(joined[WORLD_POSITION].to_numpy(), truths)
    audit["raw_initial_aabb_membership_counts"] = {
        "unique": int((membership_counts == 1).sum()), "ambiguous": int((membership_counts > 1).sum()),
        "outside": int((membership_counts == 0).sum())}
    independent = independent_bin_rows(joined, truths, metadata)
    a = joined.merge(track, on="gdu_id", how="inner", validate="many_to_one")
    audit["unique_matched_track_sample_count"] = len(a)
    a = a[a.confirmed_cohort].copy()
    audit["confirmed_matched_track_sample_count"] = len(a)
    del raw, joined
    a["unique_initial_membership"] = a.geometry_model_name.eq(a.model_name)
    metrics, bins, state_bins, model_audits = [], [], [], {}
    evidence = pd.read_csv(directory / INPUTS[3], usecols=GDU_FRAME + ["message_time_s", "p_u", "p_a", "p_c"])
    for field in ("p_u", "p_a", "p_c"):
        assert evidence.groupby(GDU_FRAME)[field].nunique().max() == 1
    audit["raw_probability_rows"] = len(evidence)
    evidence = evidence.drop_duplicates(GDU_FRAME).merge(track[track.confirmed_cohort], on="gdu_id", validate="many_to_one")
    for model in MODELS:
        truth = truths[model]
        selected = a[a.model_name.eq(model)]
        counts = {"confirmed_track_samples": len(selected),
                  "excluded_initial_membership_samples": int((~selected.unique_initial_membership).sum())}
        selected = selected[selected.unique_initial_membership].copy()
        stamps = selected.message_time_s.to_numpy()
        valid, inside, transition, positions, rotations = interpolate_truth(truth, stamps)
        counts.update(excluded_outside_truth_samples=int((~inside).sum()),
                      excluded_transition_interpolation_samples=int(transition.sum()),
                      excluded_truth_gap_samples=int((inside & ~valid & ~transition).sum()),
                      excluded_transition_source_frames=selected.loc[transition, "accepted_frame_index"].nunique())
        selected = selected.loc[valid].copy()
        if len(selected):
            p = matrices(selected.observable_projector_row_major)
            rank = projector_ranks(p)
            d = same_point_displacement(selected[WORLD_POSITION].to_numpy(), truth["positions"][0],
                                        truth["rotations"][0].as_matrix(), positions, rotations, rotation)
            projected_truth = np.einsum("nij,nj->ni", p, d)
            projected_estimate = np.einsum("nij,nj->ni", p, selected[ESTIMATE].to_numpy())
            covariance = matrices(selected.displacement_covariance_row_major)
            selected["truth_norm_mm"] = 1000 * np.linalg.norm(projected_truth, axis=1)
            selected["estimate_norm_mm"] = 1000 * np.linalg.norm(projected_estimate, axis=1)
            selected["vector_error_mm"] = 1000 * np.linalg.norm(projected_estimate - projected_truth, axis=1)
            selected["magnitude_bias_mm"] = selected.estimate_norm_mm - selected.truth_norm_mm
            selected["sigma_mm"] = 1000 * np.sqrt(np.maximum(0, np.trace(p @ covariance @ p, axis1=1, axis2=2)))
            counts["excluded_zero_rank_samples"] = int((rank == 0).sum())
            selected = selected.loc[rank > 0].copy()
        else:
            counts["excluded_zero_rank_samples"] = 0
            for field in ("truth_norm_mm", "estimate_norm_mm", "vector_error_mm", "magnitude_bias_mm", "sigma_mm"):
                selected[field] = pd.Series(dtype=float)
        selected["elapsed_s"] = selected.message_time_s - truth["audit"]["clock_offset_s"]
        in_interval = selected.elapsed_s.between(0, 60, inclusive="both")
        counts["excluded_outside_experiment_samples"] = int((~in_interval).sum())
        selected = selected[in_interval].copy()
        selected["accuracy_eligible"] = (selected.elapsed_s >= selected.confirmation_s) & (selected.truth_norm_mm >= 5)
        eligible = selected[selected.accuracy_eligible]
        micro = eligible[eligible.truth_norm_mm <= 20]
        hold = eligible[eligible.elapsed_s >= truth["hold_start"]]
        counts.update(excluded_before_confirmation_samples=int((selected.elapsed_s < selected.confirmation_s).sum()),
                      excluded_below_5mm_after_confirmation_samples=int(((selected.elapsed_s >= selected.confirmation_s) & (selected.truth_norm_mm < 5)).sum()),
                      current_projected_samples=len(selected), accuracy_sample_count=len(eligible))
        assert (sum(value for key, value in counts.items()
                    if key.startswith("excluded_") and key.endswith("_samples"))
                + len(eligible)) == counts["confirmed_track_samples"], "Nonconserving filter audit"
        metrics.append(dict(metadata, model_name=model,
                            n_confirmed_gdus=int(track[track.model_name.eq(model) & track.confirmed_cohort].gdu_id.nunique()),
                            n_samples=len(eligible), n_source_frames=eligible.accepted_frame_index.nunique(),
                            n_gdu_frames=len(eligible[GDU_FRAME].drop_duplicates()),
                            first_accuracy_sample_elapsed_s=eligible.elapsed_s.min(),
                            last_accuracy_sample_elapsed_s=eligible.elapsed_s.max(),
                            vector_rmse_mm=rms(eligible.vector_error_mm),
                            projected_magnitude_bias_mm=eligible.magnitude_bias_mm.mean(),
                            projected_magnitude_rmse_mm=rms(eligible.magnitude_bias_mm),
                            stable_hold_start_elapsed_s=truth["hold_start"],
                            stable_hold_duration_s=truth["audit"]["stable_hold_duration_s"],
                            hold_n_samples=len(hold), hold_n_source_frames=hold.accepted_frame_index.nunique(),
                            first_hold_accuracy_sample_elapsed_s=hold.elapsed_s.min(),
                            last_hold_accuracy_sample_elapsed_s=hold.elapsed_s.max(),
                            hold_vector_rmse_mm=rms(hold.vector_error_mm),
                            hold_projected_magnitude_bias_mm=hold.magnitude_bias_mm.mean(),
                            n_inconclusive_accuracy_samples=int(eligible.validity.eq(1).sum()), **counts,
                            micro_vector_rmse_mm=rms(micro.vector_error_mm),
                            micro_n_samples=len(micro), micro_n_source_frames=micro.accepted_frame_index.nunique(),
                            micro_n_gdu_frames=len(micro[GDU_FRAME].drop_duplicates())))
        bins.extend(bin_rows(selected, metadata, model))
        supported_gdus = set(a[a.model_name.eq(model) & a.unique_initial_membership].gdu_id)
        state = evidence[evidence.model_name.eq(model) & evidence.gdu_id.isin(supported_gdus)].copy()
        state["elapsed_s"] = state.message_time_s - truth["audit"]["clock_offset_s"]
        state_bins.extend(bin_rows(state, metadata, model, state=True))
        model_audits[model] = dict(truth["audit"], **counts)
    audit["models"] = model_audits
    audit["inputs_sha256"] = {str(directory / name): sha256(directory / name) for name in INPUTS}
    return metrics, bins, state_bins, audit, independent


def self_check():
    q0 = Rotation.identity().as_matrix()
    point = np.array([[1., 0., 0.]])
    rotated = Rotation.from_euler("z", [90], degrees=True).as_matrix()
    d = same_point_displacement(point, np.zeros(3), q0, np.zeros((1, 3)), rotated, q0)
    assert np.allclose(d, [[-1., 1., 0.]])  # Rotation must move an off-origin point.
    p = np.diag([1., 0., 0.])
    assert np.allclose(d @ p, [[-1., 0., 0.]])
    rows = []
    for t, x in ((0., 0.), (.1, 0.), (.2, 1.)):
        rows.append(dict(zip(POSE_POSITION + POSE_QUATERNION, [x, 0, 0, 0, 0, 0, 1]),
                         truth_sim_time_s=t, experiment_elapsed_s=t,
                         model_motion_elapsed_s=0., motion_state="WAITING" if t < .2 else "HOLDING"))
    truth = prepare_truth(pd.DataFrame(rows))
    valid, _, transition, positions, _ = interpolate_truth(truth, np.array([-.1, .05, .15, .2, .3]))
    assert valid.tolist() == [False, True, False, True, False]
    assert transition.tolist() == [False, False, True, False, False]
    assert np.allclose(positions[:, 0], [0, 1])  # Jump endpoint retained, gap never filled.
    assert interpolate_truth(truth, np.array([]))[4].shape == (0, 3, 3)
    assert projector_ranks(np.stack([np.diag([1.] * rank + [0.] * (3 - rank))
                                     for rank in range(4)])).tolist() == [0, 1, 2, 3]
    owned = pd.DataFrame([dict(geometry_model_name="m01", message_time_s=stamp,
                               accepted_frame_index=index, observable_projector_row_major=projector,
                               **dict(zip(ESTIMATE, [0., 0., 0.])))
                          for index, (stamp, projector) in enumerate([
                              (.05, "1|0|0|0|0|0|0|0|0"), (.15, "0|0|0|0|0|0|0|0|0")])])
    diagnostic = independent_bin_rows(owned, dict.fromkeys(MODELS, truth), {})
    assert len(diagnostic) == 720 and diagnostic[0]["n_published_samples"] == 2
    assert diagnostic[0]["rank0_n_samples"] == diagnostic[0]["rank1_n_samples"] == 1
    assert diagnostic[0]["n_evaluable_samples"] == 1 and diagnostic[0]["evaluable_available"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "projected_analysis")
    parser.add_argument("--run", help="One exact scenario_id for a reproducible pilot")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_test:
        print("PASS: observable SE(3) point displacement, no extrapolation, jump endpoints")
        return
    output = args.output_dir.resolve()
    assert PAPER in output.parents, "Outputs must stay below the paper directory"
    ledger = pd.read_csv(args.batch / "matrix.csv")
    if args.run:
        ledger = ledger[ledger.scenario_id.eq(args.run)]
    assert len(ledger) == (1 if args.run else 40)
    metrics, tracking, states, audits, independent = [], [], [], [], []
    for index, run in enumerate(ledger.itertuples(index=False), 1):
        result = analyze_run(run, args.batch)
        for destination, values in zip((metrics, tracking, states), result[:3]):
            destination.extend(values)
        audits.append(result[3])
        independent.extend(result[4])
        print("%d/%d %s: %d eligible samples" % (index, len(ledger), run.scenario_id,
                                                sum(row["n_samples"] for row in result[0])), flush=True)
    metric_frame = pd.DataFrame(metrics)
    summary = metric_frame.groupby(["speed_mm_s", "model_name"]).agg(
        n_runs=("run_id", "size"), n_valid_runs=("vector_rmse_mm", "count"),
        vector_rmse_mean_mm=("vector_rmse_mm", "mean"), vector_rmse_sd_mm=("vector_rmse_mm", "std"),
        projected_magnitude_bias_mean_mm=("projected_magnitude_bias_mm", "mean"),
        n_samples=("n_samples", "sum"), n_source_frames=("n_source_frames", "sum"),
        n_gdu_frames=("n_gdu_frames", "sum"),
        hold_n_valid_runs=("hold_vector_rmse_mm", "count"), hold_n_samples=("hold_n_samples", "sum"),
        hold_vector_rmse_mean_mm=("hold_vector_rmse_mm", "mean"),
        hold_vector_rmse_sd_mm=("hold_vector_rmse_mm", "std"),
        stable_hold_start_min_s=("stable_hold_start_elapsed_s", "min"),
        stable_hold_start_max_s=("stable_hold_start_elapsed_s", "max"),
        micro_vector_rmse_mean_mm=("micro_vector_rmse_mm", "mean"),
        micro_vector_rmse_sd_mm=("micro_vector_rmse_mm", "std"),
        micro_n_valid_runs=("micro_vector_rmse_mm", "count"),
        micro_n_samples=("micro_n_samples", "sum"),
        micro_n_source_frames=("micro_n_source_frames", "sum"),
        micro_n_gdu_frames=("micro_n_gdu_frames", "sum")).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    tracking_frame = pd.DataFrame(tracking)
    hold_bins = tracking_frame[tracking_frame.bin_start_elapsed_s.between(50, 60, inclusive="left")].copy()
    hold_bins["available"] = hold_bins.n_samples > 0
    hold_bins["accuracy_available"] = hold_bins.n_accuracy_eligible_samples > 0
    coverage_runs = hold_bins.groupby(["run_id", "run_name", "speed_mm_s", "repeat", "model_name"]).agg(
        possible_nominal_hold_bins=("available", "size"), valid_nominal_hold_bins=("available", "sum"),
        accuracy_valid_nominal_hold_bins=("accuracy_available", "sum")).reset_index()
    assert coverage_runs.possible_nominal_hold_bins.eq(20).all()
    coverage_runs["nominal_hold_coverage_percent"] = 100 * coverage_runs.valid_nominal_hold_bins / 20
    coverage_runs = coverage_runs.merge(metric_frame[["run_id", "model_name",
        "first_accuracy_sample_elapsed_s", "last_accuracy_sample_elapsed_s",
        "first_hold_accuracy_sample_elapsed_s", "last_hold_accuracy_sample_elapsed_s"]],
        on=["run_id", "model_name"], validate="one_to_one")
    coverage = coverage_runs.groupby(["speed_mm_s", "model_name"]).agg(
        n_runs=("run_id", "size"), possible_nominal_hold_bins=("possible_nominal_hold_bins", "sum"),
        valid_nominal_hold_bins=("valid_nominal_hold_bins", "sum"),
        accuracy_valid_nominal_hold_bins=("accuracy_valid_nominal_hold_bins", "sum"),
        n_runs_with_nominal_hold_output=("valid_nominal_hold_bins", lambda values: int((values > 0).sum()))).reset_index()
    coverage["nominal_hold_coverage_percent"] = 100 * coverage.valid_nominal_hold_bins / coverage.possible_nominal_hold_bins
    independent_frame = pd.DataFrame(independent)
    run_model = ["run_id", "run_name", "speed_mm_s", "repeat", "model_name"]
    independent_windows = independent_frame[independent_frame.bin_start_elapsed_s >= 10].copy()
    independent_windows["phase"] = np.where(independent_windows.bin_start_elapsed_s < 50, "motion", "hold")
    independent_windows["published_available"] = independent_windows.n_published_samples > 0
    independent_coverage_runs = independent_windows.groupby(run_model + ["phase"]).agg(
        possible_bins=("evaluable_available", "size"), evaluable_bins=("evaluable_available", "sum"),
        published_bins=("published_available", "sum"), n_evaluable_samples=("n_evaluable_samples", "sum"),
        n_published_samples=("n_published_samples", "sum")).reset_index()
    assert independent_coverage_runs.possible_bins.eq(independent_coverage_runs.phase.map({"motion": 80, "hold": 20})).all()
    independent_coverage_runs["coverage_percent"] = 100 * independent_coverage_runs.evaluable_bins / independent_coverage_runs.possible_bins
    independent_coverage = independent_coverage_runs.groupby(["speed_mm_s", "model_name", "phase"]).agg(
        n_runs=("run_id", "size"), possible_bins=("possible_bins", "sum"), evaluable_bins=("evaluable_bins", "sum"),
        published_bins=("published_bins", "sum"), n_evaluable_samples=("n_evaluable_samples", "sum"),
        n_published_samples=("n_published_samples", "sum"),
        n_runs_with_output=("evaluable_bins", lambda values: int((values > 0).sum()))).reset_index()
    independent_coverage["coverage_percent"] = 100 * independent_coverage.evaluable_bins / independent_coverage.possible_bins
    rank_columns = ["rank%d_n_samples" % rank for rank in range(4)]
    rank_runs = independent_frame.groupby(run_model)[["n_published_samples"] + rank_columns].sum().reset_index()
    rank_summary = rank_runs.groupby(["speed_mm_s", "model_name"])[["n_published_samples"] + rank_columns].sum().reset_index()
    for frame in (rank_runs, rank_summary):
        assert frame[rank_columns].sum(axis=1).eq(frame.n_published_samples).all()
        for rank in range(4):
            frame["rank%d_percent" % rank] = 100 * frame["rank%d_n_samples" % rank] / frame.n_published_samples.replace(0, np.nan)
    products = {"run_metrics.csv": metric_frame, "summary.csv": summary,
                "tracking_bins.csv": tracking_frame, "state_bins.csv": pd.DataFrame(states),
                "coverage.csv": coverage, "coverage_runs.csv": coverage_runs,
                "independent_bins.csv": independent_frame,
                "independent_coverage_runs.csv": independent_coverage_runs,
                "independent_coverage.csv": independent_coverage,
                "rank_distribution_runs.csv": rank_runs, "rank_distribution.csv": rank_summary}
    for name, frame in products.items():
        frame.to_csv(output / name, index=False, float_format="%.12g", na_rep="")
    manifest = {
        "script_sha256": sha256(Path(__file__)), "batch": str(args.batch.resolve()),
        "matrix_sha256": sha256(args.batch / "matrix.csv"), "run_count": len(ledger),
        "formula": "e_i=P_i*dhat_i-P_i*R_RW*(Q_t*Q_0^T*(x_i^W-p_0)+p_t-x_i^W)",
        "units": "CSV length/error/sigma: mm; timestamps/durations: s; speed: mm/s; probabilities: dimensionless",
        "accuracy_population": "Unique MATCHED GDU whose first native confirmation occurs in the motion window 10<=elapsed<50s, exact frozen-coordinate join, sole initial AABB owner agrees with track, source timestamp within truth, local P rank>0, after this GDU's recorded confirmation through elapsed 60s, projected point truth norm>=5mm",
        "estimator": "Raw current physical_deformation samples, projected by their own recorded P; no downstream evolution carry",
        "weighting": "Within each run: equal eligible physical-sample weight. Summary: unweighted mean and sample SD of per-run RMSE, never pooled across runs. Bins: equal available sample weight; U/A/C bins: equal deduplicated GDU-frame weight.",
        "curve_population": "Retrospectively confirmed MATCHED cohort at elapsed 0<=t<60, including preconfirmation and truth<5mm; accuracy_eligible counts distinguish the table population. Missing bins stay empty.",
        "state_population": "Same confirmed cohort, with at least one sole-owner frozen sample; p_U/A/C deduplicated per reference epoch/source frame/GDU. Descriptive posterior curves, not true-state accuracy.",
        "coverage": "Nominal hold [50,60): nonempty 0.5s bins with raw current rank>0 samples and usable matched point truth, divided by 20 bins per run (200 per speed/model). No carry or gap filling. accuracy_valid bins additionally require postconfirmation and truth>=5mm.",
        "micro_accuracy": "Additional subset of the unchanged accuracy population with 5<=projected frozen-point truth norm<=20mm. Empty subsets have no RMSE; mean/SD aggregate only contributing run-level values and report micro_n_valid_runs and sample/frame counts.",
        "independent_coverage": "All published raw physical samples uniquely inside one initial truth AABB, with exact frozen-coordinate identity, regardless of track MATCHED or native confirmation. A bin is evaluable if at least one finite current estimate has local P rank>0 and timestamp-aligned truth. Every run/model has 120 explicit half-second bins; motion [10,50) denominator=80 per run, hold [50,60) denominator=20 per run. No displacement floor is applied.",
        "rank_distribution": "Sample-weighted rank0/1/2/3 counts among all published raw physical samples with unique initial AABB ownership and elapsed 0<=t<60; no confirmation, MATCHED-track, truth-floor, or rank-positive filter. The denominator excludes unpublished geometry and surfaces without physical field samples; it is not full-scene observability or detection success. Per-bin rank counts allow phase-specific summaries.",
        "truth_interpolation": "Source message_time_s to truth_sim_time_s: linear position and normalized shortest-arc quaternion SLERP; no extrapolation. Exact endpoints of pose changes are retained.",
        "transition_policy": "Exclude only interpolation strictly inside a changed-SE3 interval with unchanged model_motion_elapsed_s, because the zero-progress update has no recorded within-interval trajectory. Keep actual endpoint poses and all subsequent errors. Also exclude gaps>0.101s. No filtering uses estimate error.",
        "pose_change_tolerance_m_and_rad": POSE_TOLERANCE, "maximum_truth_gap_s": MAX_TRUTH_GAP,
        "stable_hold": "Last recorded SE3 change after elapsed 50s through an unchanged observed suffix reaching 60s, at least one 0.1s truth interval; same postconfirmation/rank/truth-floor filters as accuracy. Actual transition poses are retained.",
        "limitations": ["Fixed startup SE(2)+z world/map anchor assumes aligned roll/pitch; it is not independently calibrated full SE(3).",
                        "Reference first publication precedes truth recording; baseline uses the first WAITING pose, with all recorded WAITING poses checked constant. Individual reference-exposure truth poses were not recorded.",
                        "Unique AABB and independent-frame track relation are geometric attribution, not recorded material-point IDs; centroid noise and residual map/world alignment remain.",
                        "INCONCLUSIVE research outputs are intentionally included. These errors and covariance-trace scales do not establish probability/covariance calibration.",
                        "Surfel samples within GDU/frame/run are correlated. No independence assumption, confidence interval, cross-sample mean covariance, or zero-error substitution is made.",
                        "The confirmed cohort is selected retrospectively; this conditional measurement analysis does not replace detection success and missing-output denominators.",
                        "M04 native confirmations inside [10,50)s are retained despite the legacy FALSE_POSITIVE_EXCLUDED_MODEL label. One 5mm/s r04 track first confirmed at elapsed 7.408s is excluded by the first-native-timestamp window, with its original confirmation preserved in this audit. No error-based or target-identity exclusion is used.",
                        "The accuracy cohort uses the recorded native first-confirmation timestamp and does not require every confirming frame to occur after motion onset; therefore it differs from a stricter three-fresh-post-onset-frame detection-event audit.",
                        "Independent coverage removes confirmation/track survival selection, but remains conditional on published frozen field samples and unique AABB geometry attribution; one evaluable point does not establish coverage of the whole physical object."],
        "totals": {column: int(metric_frame[column].sum()) for column in metric_frame.columns
                   if column.startswith("excluded_") or column in ("n_samples", "hold_n_samples", "confirmed_track_samples", "current_projected_samples", "n_inconclusive_accuracy_samples")},
        "micro_totals": {"n_samples": int(metric_frame.micro_n_samples.sum()),
                         "n_contributing_model_runs": int(metric_frame.micro_vector_rmse_mm.notna().sum())},
        "independent_totals": {"n_published_assigned_samples_0_60": int(rank_runs.n_published_samples.sum()),
                               **{column: int(rank_runs[column].sum()) for column in rank_columns}},
        "runs": audits,
        "outputs_sha256": {name: sha256(output / name) for name in products},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
