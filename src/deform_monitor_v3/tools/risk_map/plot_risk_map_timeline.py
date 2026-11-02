#!/usr/bin/env python3
import argparse,numpy as np,matplotlib.pyplot as plt
from risk_map_io import load_run,load_frame
def main():
 ap=argparse.ArgumentParser();ap.add_argument("run");ap.add_argument("--voxel",nargs=3,type=int,required=True);ap.add_argument("--output");a=ap.parse_args();m,fs=load_run(a.run);times=[];series={x:[] for x in ["p_deformation","p_novel_occupancy","decision_cost","log_bayes_factor_last"]};key=np.asarray(a.voxel)
 for path in fs:
  f=load_frame(path);hit=np.flatnonzero(np.all(f["voxel_index"]==key,axis=1));times.append(float(f["stamp_sec"])+1e-9*float(f["stamp_nsec"]));
  for name in series:series[name].append(float(f[name][hit[0]]) if hit.size else np.nan)
 times=np.asarray(times);times-=times[0];
 for n,v in series.items():plt.plot(times,v,label=n)
 plt.xlabel("time [s]");plt.legend();plt.grid();plt.savefig(a.output,dpi=180,bbox_inches="tight") if a.output else plt.show()
if __name__=="__main__":main()
