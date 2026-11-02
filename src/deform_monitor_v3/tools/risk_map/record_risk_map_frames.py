#!/usr/bin/env python3
"""Asynchronously record /deform/probabilistic_risk/map as sparse NPZ frames."""
import argparse, json, queue, threading, time
from pathlib import Path
import numpy as np

def message_to_arrays(msg):
    v = msg.voxels
    f3 = lambda name: np.asarray([[getattr(x, name).x, getattr(x, name).y, getattr(x, name).z] for x in v], np.float32)
    f1 = lambda name, dtype=np.float32: np.asarray([getattr(x, name) for x in v], dtype)
    return dict(schema_version=np.int32(msg.schema_version), frame_index=np.uint64(msg.frame_index),
      stamp_sec=np.uint32(msg.header.stamp.secs), stamp_nsec=np.uint32(msg.header.stamp.nsecs),
      world_frame=np.asarray(msg.header.frame_id), reference_epoch=np.uint32(msg.reference_epoch),
      grid_origin=np.asarray([msg.grid_origin.x,msg.grid_origin.y,msg.grid_origin.z],np.float64),
      voxel_size=np.float64(msg.voxel_size), dropped_frame_count=np.uint64(msg.dropped_frame_count),
      model_version=np.asarray(msg.model_version), calibration_version=np.asarray(msg.calibration_version),
      probabilities_calibrated=np.uint8(msg.probabilities_calibrated),
      approximate_joint_model=np.uint8(msg.approximate_joint_model),
      voxel_index=np.asarray([[x.ix,x.iy,x.iz] for x in v],np.int32), center_R=f3("center"),
      p_deformation=f1("p_deformation"), p_novel_occupancy=f1("p_novel_occupancy"),
      unknown_mask=np.asarray([x.knowledge_state==x.KNOWLEDGE_UNKNOWN for x in v],np.uint8),
      knowledge_state=f1("knowledge_state",np.uint8), decision_cost=f1("decision_cost"),
      posterior_log_odds_def=f1("deformation_log_odds"), posterior_log_odds_novel=f1("novel_log_odds"),
      log_bayes_factor_last=f1("log_bayes_factor_last"), visibility_probability=f1("visibility_probability"),
      principal_direction=f3("principal_direction"), direction_confidence=f1("direction_confidence"),
      event_semantics=f1("event_semantics",np.uint8), region_id=f1("region_id",np.int32))

class Recorder:
    def __init__(self, root, capacity):
        self.root=Path(root);(self.root/"frames").mkdir(parents=True,exist_ok=True)
        self.q=queue.Queue(capacity);self.dropped=0;self.closed=False
        self.worker=threading.Thread(target=self._work,daemon=True);self.worker.start()
    def submit(self,msg):
        try:self.q.put_nowait(message_to_arrays(msg))
        except queue.Full:self.dropped+=1
    def _work(self):
        while not self.closed or not self.q.empty():
            try:d=self.q.get(timeout=.2)
            except queue.Empty:continue
            np.savez_compressed(self.root/"frames"/("frame_%06d.npz"%int(d["frame_index"])),**d);self.q.task_done()
    def stop(self):self.closed=True;self.worker.join()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--topic",default="/deform/probabilistic_risk/map");ap.add_argument("--queue",type=int,default=8);args=ap.parse_args()
    import rospy
    from deform_monitor_v3.msg import ProbabilisticRiskMap
    rec=Recorder(args.output,args.queue);first={}
    def cb(msg):
        if not first:
            first.update(schema_version=1,run_id=Path(args.output).name,world_frame=msg.header.frame_id,
              start_time=time.time(),voxel_size=msg.voxel_size,grid_origin=[msg.grid_origin.x,msg.grid_origin.y,msg.grid_origin.z],
              default_values={"probability":"NaN","unknown_mask":1,"decision_cost":0},model_version=msg.model_version,
              calibration_version=msg.calibration_version,probabilities_calibrated=msg.probabilities_calibrated,
              approximate_joint_model=msg.approximate_joint_model,color_scales={"probability":[0,1],"decision_cost":[0,1],"llr":[-10,10]})
            (Path(args.output)/"manifest.json").write_text(json.dumps(first,indent=2))
        rec.submit(msg)
    rospy.init_node("risk_map_frame_recorder");rospy.Subscriber(args.topic,ProbabilisticRiskMap,cb,queue_size=1)
    rospy.on_shutdown(rec.stop);rospy.spin()
if __name__=="__main__":main()
