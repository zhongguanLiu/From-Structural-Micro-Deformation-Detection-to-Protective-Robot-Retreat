#!/usr/bin/env python3
"""Validate schema and semantic invariants without interpreting calibration."""
import argparse, json
import numpy as np
from risk_map_io import load_run, load_frame, require_comparable

def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");a=ap.parse_args()
    manifest,paths=load_run(a.run);errors=[];last_stamp=-1.0;last_epoch=-1
    required=("voxel_index","center_R","p_deformation","p_novel_occupancy","unknown_mask",
      "knowledge_state","decision_cost","posterior_log_odds_def","posterior_log_odds_novel",
      "log_bayes_factor_last","visibility_probability","principal_direction","direction_confidence",
      "event_semantics","region_id")
    for path in paths:
      try:
        f=load_frame(path);require_comparable(manifest,f)
        for name in required:
          if name not in f.files:raise ValueError("missing "+name)
        n=len(f["voxel_index"])
        if any(len(f[x])!=n for x in required):raise ValueError("array length mismatch")
        if len(np.unique(f["voxel_index"],axis=0))!=n:raise ValueError("duplicate voxel key")
        stamp=float(f["stamp_sec"])+1e-9*float(f["stamp_nsec"]);epoch=int(f["reference_epoch"])
        if stamp<last_stamp:raise ValueError("non-monotonic timestamp")
        if epoch<last_epoch:raise ValueError("reference epoch regressed")
        last_stamp,last_epoch=stamp,epoch
        for name in ("p_deformation","p_novel_occupancy"):
          x=f[name];finite=np.isfinite(x)
          if np.any((x[finite]<0)|(x[finite]>1)):raise ValueError(name+" outside [0,1]")
        if np.any((f["decision_cost"]<0)|(f["decision_cost"]>1)):raise ValueError("cost outside [0,1]")
        if np.any(~np.isfinite(f["center_R"])):raise ValueError("non-finite center")
        expected=(f["voxel_index"].astype(float)+.5)*float(f["voxel_size"])+f["grid_origin"]
        if not np.allclose(expected,f["center_R"],atol=2e-5):raise ValueError("index/center mismatch")
        if np.any((f["unknown_mask"]!=0)&(f["knowledge_state"]!=0)):raise ValueError("UNKNOWN mismatch")
      except Exception as exc:errors.append(f"{path.name}: {exc}")
    report={"schema_version":1,"frame_count":len(paths),"valid":not errors,"errors":errors}
    print(json.dumps(report,indent=2));raise SystemExit(0 if not errors else 1)
if __name__=="__main__":main()
