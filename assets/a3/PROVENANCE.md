# A3 model provenance

The source URDF and meshes were copied, without modifying the source directory, from:

`/home/eter/下载/A3_n_waiguan-urdf-20260409`

The package describes `lanxin-urdf-20260204` and contains fourteen revolute arm joints.
`L_LAST_S.STL` and `R_LAST_S.STL` were deliberately not vendored: each is an invalid,
header-only STL. The runtime model replaces those empty links with parameterized flange,
camera, force/torque, and gripper elements.

No motor transmission, joint friction identification, controller tuning, calibrated zero pose,
camera calibration, or production gripper model was present in the source package. Values for
those properties in this project are functional simulation defaults, not digital-twin claims.

## Robotiq 2F-85 gripper visuals

The five `robotiq_arg2f_85_*` STL files in `meshes/` were copied from the locally installed
robosuite 1.4.0 package at:

`/home/eter/miniforge3/lib/python3.13/site-packages/robosuite/models/assets/grippers/meshes/robotiq_85_gripper`

They retain robosuite's MIT license in `../ROBOSUITE_LICENSE.txt`. The visual shell uses these
meshes, while this project keeps a simplified 85 mm parallel-jaw collision and actuation model so
the public one-opening action, fingertip touch sensors, and force feedback remain stable.
