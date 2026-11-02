#!/usr/bin/env python3
"""Shared, ROS-independent I/O for probabilistic risk-map datasets."""
import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 1
LAYERS = ("p_deformation", "p_novel_occupancy", "decision_cost",
          "log_bayes_factor_last", "visibility_probability")

def load_run(run_dir):
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported risk-map schema")
    frames = sorted((root / "frames").glob("frame_*.npz"))
    return manifest, frames

def load_frame(path):
    data = np.load(path, allow_pickle=False)
    if int(data["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("frame schema does not match reader")
    return data

def require_comparable(manifest, frame):
    if abs(float(frame["voxel_size"]) - float(manifest["voxel_size"])) > 1e-9:
        raise ValueError("voxel size changed within one run")
    if str(frame["model_version"]) != manifest["model_version"]:
        raise ValueError("model version changed within one run")

def finite_layer(frame, layer):
    if layer not in frame.files:
        raise KeyError(layer)
    values = frame[layer]
    mask = np.isfinite(values)
    return frame["center_R"][mask], values[mask], mask
