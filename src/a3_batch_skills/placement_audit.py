"""Optional privileged labels AFTER control decisions, never fed back to control."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from .motion import average_rotation


def symmetric_angle(a,b):
    # A rectangular cookie's unoriented geometry has proper axis-sign symmetries.
    return min(float(np.rad2deg(np.arccos(np.clip((np.trace(a.T@b@np.diag(sign))-1)/2,-1,1))))
               for sign in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)))


class PlacementTruthAudit:
    def __init__(self,path):
        self.stream=open(path,'x');self.ids=None;self.batch=None

    def write_after_decision(self,env,goal,pose,timestamp):
        if goal is None:return
        if goal.index!=self.batch:self.batch=goal.index;self.ids=None
        bodies=np.asarray(env._cookie_bodies,dtype=int)
        positions=env.data.xpos[bodies].copy()
        rotations=env.data.xmat[bodies].reshape(-1,3,3).copy()
        row=dict(timestamp=float(timestamp),batch=goal.index,
                 timing='after_decision_before_action_application',used_for_control=False,
                 visual_pose_available=pose is not None)
        if self.ids is None and pose is not None:
            distance=np.linalg.norm(positions[:,None]-pose.positions[None],axis=2)
            a,b=linear_sum_assignment(distance)
            self.ids=a[np.argsort(b)]
            row['initial_association_max_error_m']=float(distance[a,b].max())
        if self.ids is not None:
            actual=positions[self.ids];actual_rotation=average_rotation(rotations[self.ids])
            box_position=env.data.xpos[env._target_bin_body].copy()
            box_rotation=env.data.xmat[env._target_bin_body].reshape(3,3).copy()
            desired=box_position+box_rotation@goal.target_positions_box.mean(0)
            offset=(actual.mean(0)-desired)@box_rotation
            row.update(audit_cookie_ids=self.ids.tolist(),true_center=actual.mean(0).tolist(),
                true_xy_error_m=float(np.linalg.norm(offset[:2])),true_height_above_goal_m=float(offset[2]),
                true_angle_mod_symmetry_deg=symmetric_angle(box_rotation,actual_rotation))
            if pose is not None:
                row.update(visual_center_error_m=float(np.linalg.norm(pose.center-actual.mean(0))),
                    visual_angle_error_mod_symmetry_deg=symmetric_angle(actual_rotation,pose.rotation))
        self.stream.write(json.dumps(row)+'\n');self.stream.flush()

    def close(self):self.stream.close()
