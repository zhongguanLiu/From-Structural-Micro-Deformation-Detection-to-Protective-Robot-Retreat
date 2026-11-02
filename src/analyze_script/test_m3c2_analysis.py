#!/usr/bin/env python3
"""Small geometric and temporal checks independent of the experiment outcomes."""
import math
import numpy as np
import pandas as pd
from analyze_m3c2 import geometry, MODELS, common

cores = pd.DataFrame(dict(x=[1.,2.], y=[0.,0.], z=[0.,0.],
                         nx=[1.,1.], ny=[0.,0.], nz=[0.,0.]))
models = [dict(model_name=m, aabb_min=[20+i,20,20], aabb_max=[21+i,21,21])
          for i,m in enumerate(MODELS)]
models[0].update(aabb_min=[-.1,.9,-.1],aabb_max=[.1,1.1,.1])
owners,normals = geometry(cores,dict(camera_init_origin_gazebo_xyzyaw=[0,0,0,math.pi/2]),dict(models=models))
assert list(owners) == [0,-1]
assert np.allclose(normals,[[0,1,0],[0,1,0]])
assert math.isclose(float(normals[0] @ np.array([.1,.005,0])),.005)
models[1].update(aabb_min=models[0]['aabb_min'],aabb_max=models[0]['aabb_max'])
assert geometry(cores,dict(camera_init_origin_gazebo_xyzyaw=[0,0,0,math.pi/2]),dict(models=models))[0][0] == -1
frames = [dict(source_stamp_s=t,point_count=1) for t in (10.,10.1,10.2)]
assert common.confirmation(frames,1,3,.35)[1] is frames[-1]
assert common.confirmation([dict(source_stamp_s=t,point_count=1) for t in (10.,10.5,11.)],1,3,.35)[1] is None
print('PASS: normal projection, unique ownership, missing ownership and source-time confirmation')
