# V7: shared observation, execution-margin and terminal-settling rules

This experimental RGB-D geometric controller preserves the V6 architecture.
It does not load or modify pi0.5 weights. V6/V15 remain separately reproducible.

Runtime changes are triggered by measured evidence, never benchmark seed IDs.

1. When Park fitting worsens an already accurate nominal mount, retain that
   mount only after independent leave-one-view-out, horizontal-normal and the
   original absolute precision checks certify it. Wrong fits remain rejected.
2. If three actually captured views yield zero complete rims across every
   coarse focal candidate, physically change the left wrist view early. Only
   five reachable independent views are added, once, with all time charged to
   the original episode budget. A late insufficient-view solve can use the same
   bounded fallback. Hidden edges are not synthesized.
3. Retain the original default placement only when its full-path IK has ample
   accuracy margin; otherwise compare the existing bounded candidate poses.
   Original wall, floor, neighbouring-row and hard reachability limits remain.
4. After the last verified batch, hold the withdrawn open-gripper pose until
   current independent RGB-D/encoders remain stable for at least 0.5 seconds
   and six frames. Motion, missing evidence or acquisition gaps reset the
   window. Native success/score does not enter this rule.

Enable --handeye --table-height .75 --refine --scene-memory --placement-planning
through code/run_main_sensor_grid.py. The right arm remains at its initial
safe pose. Native cameras/scoring and 1500/3000 step budgets are retained.

Evaluation-only final_benchmark_info.json copies info already emitted by main.
It is never passed to the controller and does not change observations, actions,
rewards, termination flags or success predicates. Bounded telemetry also keeps
original image timestamps and measured left-arm state for future diagnostics.

Old failure-driven regression cohorts are development data. Unseen-grid results
must be reported separately; unit tests and short calibration prefixes do not
establish a 90% manipulation success rate.
