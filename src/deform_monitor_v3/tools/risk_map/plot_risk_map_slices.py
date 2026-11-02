#!/usr/bin/env python3
import argparse,numpy as np,matplotlib.pyplot as plt
from risk_map_io import load_run,load_frame
def main():
 ap=argparse.ArgumentParser();ap.add_argument("run");ap.add_argument("--frame",type=int,default=-1);ap.add_argument("--layer",default="decision_cost");ap.add_argument("--output");a=ap.parse_args();m,fs=load_run(a.run);f=load_frame(fs[a.frame]);idx=f["voxel_index"];val=f[a.layer];fig,axes=plt.subplots(1,3,figsize=(14,4));pairs=[(0,1,2,"XY"),(0,2,1,"XZ"),(1,2,0,"YZ")]
 for ax,(u,v,w,title) in zip(axes,pairs):
  plane=int(np.median(idx[:,w]));mask=(idx[:,w]==plane)&np.isfinite(val);sc=ax.scatter(idx[mask,u],idx[mask,v],c=val[mask],marker="s",s=18,cmap="inferno",vmin=0,vmax=1);ax.set_title(f"{title}, index={plane}");ax.set_aspect("equal")
 fig.colorbar(sc,ax=axes.tolist(),label=a.layer);fig.savefig(a.output,dpi=180,bbox_inches="tight") if a.output else plt.show()
if __name__=="__main__":main()
