#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from risk_map_io import load_run,load_frame,require_comparable,finite_layer
def main():
 ap=argparse.ArgumentParser();ap.add_argument("run");ap.add_argument("--frame",type=int,default=-1);ap.add_argument("--layer",default="decision_cost");ap.add_argument("--threshold",type=float,default=0);ap.add_argument("--output");a=ap.parse_args();m,fs=load_run(a.run);f=load_frame(fs[a.frame]);require_comparable(m,f);p,v,_=finite_layer(f,a.layer);keep=v>=a.threshold
 fig=plt.figure();ax=fig.add_subplot(projection="3d");s=ax.scatter(p[keep,0],p[keep,1],p[keep,2],c=v[keep],s=8,cmap="inferno",vmin=0,vmax=1,alpha=.75);fig.colorbar(s,label=a.layer);ax.set(xlabel="x [m]",ylabel="y [m]",zlabel="z [m]",title=f"{a.layer} t={int(f['stamp_sec'])}.{int(f['stamp_nsec']):09d}");fig.savefig(a.output,dpi=180,bbox_inches="tight") if a.output else plt.show()
if __name__=="__main__":main()
