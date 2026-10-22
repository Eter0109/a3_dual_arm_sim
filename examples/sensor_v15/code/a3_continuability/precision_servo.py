"""Bounded local IK with tolerances appropriate for millimetre motions.

The pinned main adapter uses gtol=1e-4. With a short/singular lever arm this
can satisfy the gradient test before a submillimetre Cartesian step is taken.
This robot-only solver changes no simulator source, observations or scoring.
"""
import numpy as np


def precision_joint_target(model,work,qids,site,limits,initial_q,target_position,
                           target_quaternion,*,max_joint_step=.025):
    import mujoco
    from scipy.optimize import least_squares
    initial=np.asarray(initial_q,dtype=float)
    limits=np.asarray(limits,dtype=float)
    lower=np.maximum(limits[:,0]+1e-4,initial-max_joint_step)
    upper=np.minimum(limits[:,1]-1e-4,initial+max_joint_step)
    if np.any(upper<=lower):raise ValueError('Invalid bounded precision IK interval')
    seed=np.clip(initial,lower,upper)
    def residual(q):
        work.qpos[list(qids)]=q
        mujoco.mj_kinematics(model,work)
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,work.site_xmat[site])
        rotation=np.empty(3);mujoco.mju_subQuat(rotation,target_quaternion,quat)
        return np.r_[work.site_xpos[site]-target_position,.12*rotation,.001*(q-seed)]
    result=least_squares(residual,seed,bounds=(lower,upper),max_nfev=35,
        ftol=1e-7,xtol=1e-7,gtol=1e-8)
    if not np.isfinite(result.x).all():raise ValueError('Nonfinite precision IK result')
    return result.x.copy()
