"""Bounded sensor/proprioception records for pre-release placement research."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .rgbd import save_rgbd_packet


def json_arrays(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class PlacementProbeRecorder:
    def __init__(self,root,spec,*,interval=20,limit=32):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.interval,self.limit=int(interval),int(limit)
        if min(self.interval,self.limit)<1:raise ValueError("Invalid recording budget")
        self.count=0
        (self.root/"protocol.json").write_text(json.dumps(dict(schema="rgbd_placement_probe_v1",
            geometry_spec=spec,interval=self.interval,limit=self.limit,object_truth=False,
            state_dim=8,training_ready=False,
            timing="observation and robot FK before action application"),default=json_arrays,indent=2))

    def add(self,step,frame,observation,goal,*,tool_position,tool_rotation,pad_midpoint,jaw_gap,opening,
            calibrations=None):
        if frame is None or goal is None or step<100 or (step-1)%self.interval or self.count>=self.limit:return
        root=self.root/f"step_{step:04d}";root.mkdir()
        save_rgbd_packet(root/"rgbd.npz",frame)
        fields={k:np.asarray(v) for k,v in observation.items() if k.startswith("observation.")}
        fields['observation.state']=np.asarray(observation['observation.state'])[:8]
        np.savez_compressed(root/"policy_observation.npz",**fields)
        metadata=dict(step=step,timestamp=frame.timestamp,goal=asdict(goal),
            tool_position=tool_position,tool_rotation=tool_rotation,pad_midpoint=pad_midpoint,
            jaw_gap_m=jaw_gap,opening=opening,camera_calibrations=calibrations)
        (root/"robot_and_goal.json").write_text(json.dumps(metadata,default=json_arrays,indent=2))
        self.count+=1
