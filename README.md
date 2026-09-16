# A3 Dual-Arm MuJoCo Sandbox

An independent, policy-neutral MuJoCo environment for the fourteen-axis A3 dual-arm robot.
It supports code policies, keyboard teleoperation, canonical action replay, force feedback, and
LeRobot v3 dataset collection. It does not modify or depend on `vla_ur5e_sim`.

## What is modeled

- Seven URDF joints per arm, using the source link geometry, mass/inertia, limits, velocity, and
  effort values.
- Robotiq 2F-85 grippers using vendored robosuite visual meshes and an 85 mm, single-opening
  simplified parallel-jaw collision model. Each gripper exposes two fingertip touch signals,
  actuator force, and a wrist six-axis force/torque signal.
- Front, left-wrist, and right-wrist `256x256` RGB cameras.
- A central mast matching the photographed overhead mounting, a tabletop, deterministic reset,
  position control, rate limiting, and an emergency stop.
- Two separate scenes: the original three-object sandbox and a video-inspired cookie packing task
  with a `4x20` source bin, a `2x5` target box, and eighty upright thin cookie proxies.

The two source files `L_LAST_S.STL` and `R_LAST_S.STL` are invalid header-only files and are not
used. The arm STL files are visual geometry; conservative capsules are used for collision.

This is a **functional sandbox**, not a calibrated digital twin. The supplied robot package has
no motor transmission model, identified damping/friction, calibrated zero pose, controller gains,
camera calibration, or gripper calibration. The 2F-85 appearance comes from robosuite's MIT-licensed
meshes, but its linkage dynamics are intentionally reduced to synchronized parallel jaw travel.
Defaults are centralized in `configs/default.yaml` so they can later be replaced with measured parameters. Robot self-collision is also omitted in v1;
robot-to-table and robot-to-object contacts remain active.

## Install and inspect

Python 3.10+ and MuJoCo 3.3+ are supported. LeRobot is optional unless recording, replaying,
or training.

```bash

cd a3_dual_arm_sim/
python -m pip install -e ".[dataset,dev]"
# Add the training extra when using SmolVLA:
python -m pip install -e ".[train,dev]"

a3-sim inspect --write-xml models/a3_generated.xml
MUJOCO_GL=egl a3-sim smoke --steps 1000

# Compile and headlessly stabilize the cookie-transfer variant.
a3-sim inspect --scene cookie_transfer
MUJOCO_GL=egl a3-sim smoke --scene cookie_transfer --steps 1000
```

Use `MUJOCO_GL=egl` for headless execution. Do not set it when your platform requires a different
interactive OpenGL backend.

The editable install includes Pillow, Matplotlib, and OpenCV for camera inspection and future live
viewers. Save a labeled snapshot of all three policy cameras, or add `--show` to open it interactively:

```bash
MUJOCO_GL=egl python examples/preview_cameras.py
MUJOCO_GL=egl python examples/preview_cameras.py --show
```

The output directory contains the three original `256x256` frames plus `all_cameras.png`.

## Rendering modes

Use viewer-only mode while checking teleoperation. It opens a dedicated A3 control panel beside
MuJoCo's display-only Human Viewer, skips all three Policy RGB render passes, and does not record a
dataset:

```bash
unset MUJOCO_GL
a3-sim teleop --scene cookie_transfer --no-camera-render --steps 1000
```

The observation keys and shapes remain unchanged in this mode, but the three images are black. This
keeps state-only policies and the runner contract stable while avoiding the expensive offscreen GL
work. `--no-camera-render` is available on both `run` and `teleop`.

Use headless capture/policy mode when images are required. Leave camera rendering enabled (the
default), keep the Human Viewer off, and select EGL before the process starts:

```bash
MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --policy your_package.your_policy:make_policy \
  --record outputs/datasets/a3_cookie_policy \
  --repo-id local/a3-cookie-policy --steps 1000
```

The CLI rejects `--no-camera-render` together with `--record` so an accidental debug launch cannot
write a dataset containing black camera streams. Manual keyboard collection is the necessary
exception to the viewer-off rule: `teleop --record ...` keeps both the Human Viewer and Policy RGB
on because keyboard input comes from the viewer.

On the current Python 3.13/MuJoCo/GLFW combination, a process that has owned both the interactive
viewer and offscreen camera renderer can otherwise segfault during interpreter shutdown even after
both contexts were explicitly closed. Interactive CLI commands therefore flush/close all project
state and bypass only that faulty native-library finalization step; their real exit status is still
preserved. This workaround is local to the CLI and can be removed after the native stack is upgraded.

## Action and observation contract

The canonical `joint_position` action is a physical 16-vector:

```text
[L_q1..L_q7, L_gripper, R_q1..R_q7, R_gripper]
```

Arm values are radians and grippers are normalized opening in `[0, 1]`. The optional normalized
`cartesian_delta` adapter accepts:

```text
[L_dx..L_dRz, L_gripper, R_dx..R_dRz, R_gripper]
```

Cartesian values are in `[-1, 1]`; gripper `-1` is closed and `+1` is open. Damped least-squares
IK maps them to the canonical joint targets. Every recorder stores the post-IK, safety-limited
16-vector so scripted, teleoperated, and learned episodes can be mixed.

Each observation contains:

| Key | Shape | Meaning |
| --- | ---: | --- |
| `observation.images.front` | `256x256x3` | front RGB |
| `observation.images.left_wrist` | `256x256x3` | left wrist RGB |
| `observation.images.right_wrist` | `256x256x3` | right wrist RGB |
| `observation.state` | `16` | logical arm joints and gripper openings |
| `observation.velocity` | `16` | matching velocities |
| `observation.eef_pose` | `14` | two positions and quaternions |
| `observation.force` | `18` | 12-D wrist wrench, four touch, two gripper forces |
| `time`, `safety_stop` | scalar | simulator time and safety state |

## Replaceable policies and collection

A plugin is a zero-argument factory returning an object with `action_mode`, `reset(context)`,
`act(observation, task)`, and `close()`:

```bash
# Included code controller
MUJOCO_GL=egl a3-sim run \
  --policy a3_dual_arm_sim.policy:make_sine_policy \
  --task "exercise both shoulders" --steps 200

# Collect a new LeRobot v3 episode (existing non-empty roots are protected)
MUJOCO_GL=egl a3-sim run \
  --policy a3_dual_arm_sim.examples.custom_policy:make_policy \
  --record datasets/a3_scripted --repo-id local/a3-scripted --steps 200

a3-sim replay --root datasets/a3_scripted --repo-id local/a3-scripted --episode 0 --render
```

The adapter boundary is also used by the included SmolVLA wrapper; another VLA can implement the
same four methods without changing the environment, runner, or recorder.

## Cookie transfer scene

`A3CookieTransferEnv` is a separate task variant based on the supplied deployment photograph and
4.8-second packing video. The A3 base is carried by a central dark mast, the arms start in a hanging
ready pose mirrored about the central stand. Eighty cookie proxies begin upright in four columns
of twenty. Each piece is 50 mm wide, 6.333 mm thick, and 25 mm high (one-third of the
previous thickness and half the previous height); mass scales by the same volume ratio to
5.83 g. Adjacent pieces have 0.4 mm clearance. The source cavity is approximately
207.2 × 140.27 mm, leaving 3 mm between the packed array and each wall;
the empty destination cavity is 100.4 × 33.27 mm with 2 × 5 slots.
Both boxes sit toward the right arm (world -y); the destination's two columns align
with source columns 0/1, with matching row pitch and grid alignment.
Source walls rise 30 mm above the inside floor; destination walls rise 25 mm,
matching cookie height. Their geometry includes an additional 6 mm below the floor top.
Box walls overlap at all four corners,
so there are no corner escape gaps. Both boxes use the same uniform light-gray material while the
cookie proxies alternate between two yellow-orange shades, making adjacent tightly packed pieces
visually separable without debug markers. Look down into the boxes using the rotatable
viewer to inspect the pieces behind the raised walls. The
source material provides no metric calibration, so stand, bin, camera, and cookie dimensions are
explicit approximations rather than claimed real-world measurements.

Success is not inferred from an arm waypoint. Exactly ten complete pieces must be upright and stable
inside the target box, with one piece assigned to each `2x5` slot and the packed array reaching all
four inner walls, while the other seventy remain in the source bin. This exact fill must persist below the linear/angular speed limits for 20 consecutive
control steps (one second). Counts, per-piece masks, slot occupancy, and the hold counter are reported
in `info`; they remain privileged task/evaluation state and are not added to the policy observation.

```bash
# Open the scene with a stationary policy for visual inspection.
unset MUJOCO_GL
a3-sim run --scene cookie_transfer \
  --policy a3_dual_arm_sim.policy:make_hold_policy --render --steps 1000

# Keyboard demonstration, saved through the same canonical 16-D recorder.
a3-sim teleop --scene cookie_transfer \
  --task "transfer exactly ten upright square cookie blocks into the 2x5 box" \
  --record outputs/datasets/a3_cookie_manual \
  --repo-id local/a3-cookie-manual --steps 1000

# Any code or VLA policy uses exactly the same scene/runner boundary.
MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --policy your_package.your_policy:make_policy \
  --task "transfer exactly ten upright square cookie blocks into the 2x5 box" \
  --record outputs/datasets/a3_cookie_policy \
  --repo-id local/a3-cookie-policy --steps 1000
```

`A3CookieTransferExpert` is a privileged, feedback-driven state machine rather than a VLA. It
replans one Cookie at a time, verifies grasp/lift/release/slot outcomes from simulator truth, and
retries failed phases. These privileged checks are Expert-only and are not added to policy
observations.

Run one visible demonstration or a headless evaluation report with:

```bash
python examples/run_cookie_transfer.py --render
python examples/evaluate_cookie_transfer.py --seeds 0-19 \
  --output artifacts/cookie_transfer_dev.json
python examples/evaluate_cookie_transfer.py --seeds 100-119 \
  --output artifacts/cookie_transfer_validation.json
```

The JSON report records success, verified grasp/lift, transfer drops, completed Cookies, failed
Cookie and phase, failure reason, maximum left-gripper touch force, steps, final source/target
counts, slot occupancy, and retries. With the current zero pose-noise configuration, different
seeds intentionally share the same layout; seed sweeps become meaningful after noise is enabled.

## Grasp expert and SmolVLA

`A3GraspEnv` randomizes a red cube in the reachable left-arm workspace. Its success check is
physical rather than waypoint-based: both fingers must contact the cube, the cube must be centered
inside the gripper and clear of the table, and it must remain at least 8 cm above its reset height
with low linear/angular velocity for 20 consecutive control steps (one second). `A3GraspExpert`
computes absolute joint waypoints with MuJoCo kinematics and then executes approach, two-finger
closure, lift, and stable-hold phases. Object pose is privileged only to the scripted expert and is
not part of the policy observation.

Collect only successful demonstrations (failed attempts are discarded):

```bash
MUJOCO_GL=egl a3-sim collect-grasp \
  --root outputs/datasets/a3_grasp_100 \
  --repo-id local/a3-grasp-100 --episodes 100 --seed 0
```

Every training launch first rejects non-v3 data, wrong camera/state/action shapes, inconsistent
episode metadata, and unsuccessful episodes. The A3 adapter changes the supplied SmolVLA base from
its original 10-D state, 7-D action, and two cameras to the A3 16-D state, 16-D action, and three
cameras. SmolVLA already pads state and action to 32 internally, so this does not change learned
weight shapes. The large base weights are symlinked into a generated runtime view instead of copied.

```bash
# Inspect the exact command and feature contract without starting training.
a3-sim train-smolvla \
  --root outputs/datasets/a3_grasp_100 --repo-id local/a3-grasp-100 \
  --output outputs/train/a3_grasp_smolvla --dry-run

# Formal GPU run. The default base checkpoint is the sibling reference repository's local asset;
# use --model /path/to/pretrained_model to select another compatible SmolVLA checkpoint.
a3-sim train-smolvla \
  --root outputs/datasets/a3_grasp_100 --repo-id local/a3-grasp-100 \
  --output outputs/train/a3_grasp_smolvla \
  --steps 20000 --batch-size 4 --device cuda
```

Run a saved checkpoint through the same `EpisodeRunner` policy interface:

```bash
export A3_SMOLVLA_CHECKPOINT="$PWD/outputs/train/a3_grasp_smolvla/checkpoints/last/pretrained_model"
export A3_SMOLVLA_DATASET_ROOT="$PWD/outputs/datasets/a3_grasp_100"
export A3_SMOLVLA_REPO_ID="local/a3-grasp-100"
export A3_SMOLVLA_DEVICE="cuda"
MUJOCO_GL=egl a3-sim run \
  --policy a3_dual_arm_sim.smolvla_policy:make_policy \
  --task "pick up the red cube" --steps 300
```

The one-step smoke proves loading, preprocessing, forward/backward, optimizer update, and checkpoint
serialization only. It is not evidence that the model learned the task. A meaningful run needs many
diverse successful demonstrations, held-out seeds, full training, and closed-loop success evaluation.
On the current host PyTorch reports no usable CUDA driver, so long CPU training is intentionally not
presented as the recommended workflow. TorchCodec may also warn on this installation; the recorder
and trainer explicitly use the available PyAV image path.

## Keyboard teleoperation

Idle Cartesian arms retain their last commanded joint positions, including while
recording or controlling only the other arm. The cookie scene uses ideal model-based
arm gravity compensation, routed through force-limited joints; payload tracking
error can still occur. This is a simulator approximation requiring real calibration.

The cookie scene starts with both tool axes pointing down, a 1.10 m base height,
and separated bins. The target bin is a free rigid body initially resting on the
table; it follows the right gripper only through contact, never through a weld or
kinematic attachment. The expert first approaches its near rim, verifies both
finger contacts, raises it 3 cm, and commands an 8-degree tilt. The left arm then
transfers cookies while the right arm holds its grasp. Missing support or unstable
released cookies are reported as failures, not counted as completed transfers.

Tune `cookie_transfer.base_height_m`, `deployment_home`,
`target_bin_world_position_m`, `box_lift_m`, `box_tilt_deg`, `right_grasp_pitch_deg`, and the two
`*_gripper_kp` values in `configs/default.yaml`. These gains, contact parameters,
box mass, and mount dimensions are prototype estimates. Changing mount height
requires solving a compatible home pose again. The old `target_bin_attach_*`
fields are retained for right wrist camera compatibility; they no longer attach
the box to the arm.

Cooperative expert preview (experimental, **not a verified ten-cookie expert**).
The new thin 80-cookie scene changes grasp clearances and box dimensions; the
existing expert's grasp waypoints still need retuning for this layout. Use the hold
policy or teleoperation to inspect the updated environment.

```bash
cd /home/eter/桌面/workspace/a3_dual_arm_sim
unset MUJOCO_GL
python examples/run_cookie_transfer.py \
  --config configs/cookie_cooperative.yaml --render --max-steps 6000
```

The viewer starts in free-camera mode at the front oblique viewpoint: left-drag
rotates, right-drag pans, and the wheel zooms. This does not change recorded policy
camera poses. The current contact-based expert still encounters interference in
the near rows. It stops on support loss, disturbed previously packed cookies, or
placement timeout, rather than sweeping back through the loaded box. Check final
`Success` and `Cookies in Target Bin`; a historical verified placement is not proof
that the cookie remains packed. Do not use failed attempts as successful training
demonstrations. The ideal gravity compensation and simplified Robotiq jaws are
simulation approximations, not real-hardware controller calibration.

Historical regression on the previous 30-cookie layout (`--max-steps 6000`) stopped at step 2412:
six historical verified placements, five cookies passing final containment checks,
and `success=false` after a packed cookie was disturbed. Right box support remained
verified (about 9.9 degrees tilt). The report is
`artifacts/cooperative_guarded_seed0.json`; this is a partial failure result, not
a ten-cookie success-rate benchmark.

```bash
a3-sim teleop --record datasets/a3_manual --repo-id local/a3-manual \
  --task "manual tabletop demonstration"
```

Teleoperation opens two windows. Keep keyboard focus on **A3 Teleoperation Control**; use the MuJoCo
Viewer only to watch the robot or adjust the viewing camera with the mouse. This prevents movement
keys from activating MuJoCo's built-in wireframe, joint, geometry-group, and pause shortcuts. The
control panel also provides press-and-hold buttons, arm selection, a speed slider, gripper controls,
recording status, normal/discard exits, and a red emergency-stop button.

- `1`, `2`, `3`: select left, right, or both arms
- `W/S`, `A/D`, `R/F`: world-frame `+X/-X`, `+Y/-Y`, `+Z/-Z` translation
- `I/K`, `J/L`, `U/O`: world-frame `+Rx/-Rx`, `+Ry/-Ry`, `+Rz/-Rz` rotation
- `[` / `]`: close / open gripper
- `P`: pause/resume recording; `Space`: emergency stop
- `Q`: save and quit; `X`: discard and quit

Keyboard and panel buttons keep moving while held and stop on release. The mouse keeps MuJoCo's
standard orbit, pan, and zoom behavior for inspecting the scene.

Episode metadata is stored in `a3_episode_metadata.jsonl`, including source controller, source
action mode, canonical stored action mode, seed, task, frame count, and optional success label.
