#!/usr/bin/env python3
"""Check M3C2 recording integrity, never gate success on detection or accuracy."""
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream)


def check(run):
    cores = list(rows(run / "m3c2/reference_cores.csv.gz"))
    frames = list(rows(run / "m3c2/frames.csv.gz"))
    assert cores and frames, "Missing core or frame records"
    assert [int(r["core_id"]) for r in cores] == list(range(len(cores)))
    assert [int(r["frame_index"]) for r in frames] == list(range(len(frames)))
    with gzip.open(run / "truth/motion_truth.jsonl.gz", "rt") as stream:
        truth = [json.loads(line) for line in stream]
    origin = float(truth[0]["sim_time"]) - float(truth[0]["experiment_elapsed"])
    assert max(float(c["reference_last_source_s"]) for c in cores) < origin
    previous = -1
    for frame in frames:
        stamp = float(frame["source_stamp_s"])
        assert stamp > previous
        assert float(frame["window_last_source_s"]) == stamp
        assert float(frame["window_first_source_s"]) <= stamp
        assert int(frame["window_frames"]) == 5
        assert int(frame["core_count"]) == len(cores)
        for key in ("processing_time_ms", "method_time_ms", "fusion_time_ms", "recording_time_ms"):
            assert math.isfinite(float(frame[key])) and float(frame[key]) >= 0
        previous = stamp
    assert float(frames[-1]["source_stamp_s"]) >= origin + 59.5
    count = 0
    valid_counts = np.zeros(len(frames))
    significant_counts = np.zeros(len(frames))
    support = np.array([int(c['reference_support']) for c in cores])
    for chunk in pd.read_csv(run / 'm3c2/measurements.csv.gz', chunksize=250000):
        fi, ci = chunk.frame_index.to_numpy(), chunk.core_id.to_numpy()
        sequence = np.arange(count, count + len(chunk))
        assert np.array_equal(fi, sequence // len(cores))
        assert np.array_equal(ci, sequence % len(cores))
        assert fi.max() < len(frames)
        v, s = chunk.valid.to_numpy(), chunk.significant.to_numpy()
        assert np.isin(v,[0,1]).all() and np.isin(s,[0,1]).all()
        valid = v.astype(bool)
        d, lod = chunk.signed_distance_m.to_numpy(), chunk.lod95_m.to_numpy()
        assert np.isfinite(d[valid]).all() and np.isfinite(lod[valid]).all() and (lod[valid] >= 0).all()
        assert np.array_equal(s[valid], np.abs(d[valid]) > lod[valid])
        assert (chunk.query_support.to_numpy()[valid] >= 5).all() and (support[ci[valid]] >= 5).all()
        assert np.isnan(d[~valid]).all() and np.isnan(lod[~valid]).all() and not s[~valid].any()
        valid_counts += np.bincount(fi,weights=v,minlength=len(frames))
        significant_counts += np.bincount(fi,weights=s,minlength=len(frames))
        count += len(chunk)
    assert np.array_equal(valid_counts,[int(f['valid_count']) for f in frames])
    assert np.array_equal(significant_counts,[int(f['significant_count']) for f in frames])
    assert count == len(cores) * len(frames), "Truncated M3C2 sample stream"
    package = Path(__file__).resolve().parents[1]
    sources = [p for folder in ("src", "include", "config", "launch", "scripts")
               for p in (package / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    sources += [package / "CMakeLists.txt", package / "package.xml"]
    binary = package.parents[1] / "devel/lib/m3c2_baseline/m3c2_node"
    sources.append(binary)
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    result = {"cores": len(cores), "frames": len(frames), "samples": count,
              "reference_frozen_before_experiment": True,
              "detection_outcome_used_as_success_gate": False,
              "source_sha256": {str(p): digest(p) for p in sources},
              "record_sha256": {str(p.relative_to(run)): digest(p)
                                for p in (run / "m3c2").glob("*.csv.gz")}}
    (run / "m3c2/record_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print("M3C2_RECORD_CHECK PASS", len(frames), "frames", count, "samples")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
