#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
from risk_map_io import load_run
def main():
 ap=argparse.ArgumentParser();ap.add_argument("run");ap.add_argument("--layer",default="decision_cost");ap.add_argument("--output",required=True);ap.add_argument("--fps",type=int,default=5);a=ap.parse_args();_,frames=load_run(a.run);out=Path(a.output);tmp=out.parent/(out.stem+"_frames");tmp.mkdir(parents=True,exist_ok=True);script=Path(__file__).with_name("plot_risk_map_3d.py")
 for i,_ in enumerate(frames):subprocess.check_call([sys.executable,str(script),a.run,"--frame",str(i),"--layer",a.layer,"--output",str(tmp/("%06d.png"%i))])
 subprocess.check_call(["ffmpeg","-y","-framerate",str(a.fps),"-i",str(tmp/"%06d.png"),"-pix_fmt","yuv420p",str(out)])
if __name__=="__main__":main()
