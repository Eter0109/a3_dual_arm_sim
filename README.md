# A3 dual-arm cookie manipulation

[简体中文](README.zh-CN.md) · [Architecture](docs/architecture.md) · [Data and troubleshooting](docs/data-workflow.md)

A MuJoCo environment for A3 manipulation, expert demonstrations, LeRobot action
policies, and a Planner–Executor–Verifier loop inspired by Agentic Robot. The
main learning task is **one target box, ten cookies, two batches of five from
the same source column**. The source column can be fixed or randomized.

This is a research implementation. An expert demonstration, a trained checkpoint,
an offline verifier score, and a successful learned-policy rollout are different
results. Downloaded base models need A3 adaptation and training; evaluate task
success explicitly rather than inferring it from training loss.

## 1. Components and task scope

The environment defines geometry, physics, cameras, and success conditions. The
controller supplies actions; MuJoCo simulates their effect. Loading a scene does
not itself specify how to manipulate its objects.

| Component | Role | Code |
| --- | --- | --- |
| Expert | Uses simulator poses, contacts, and robot state to generate physical actions | `experts/` |
| Planner | Turns the task and scene into supported skill requests; also offers an explicit fixed-plan baseline | `agents/planning.py` |
| Executor | Executes a skill using the expert or a trained LeRobot action policy | `agents/executors.py`, `policies/` |
| Verifier | Judges completion from subgoal and chronological images, then diagnoses incomplete behavior | `agents/verification.py` |
| Controller | Switches skills, clears stale action queues, schedules checks, retries, and safety stops | `agents/controller.py` |

The expert is a privileged rule-based controller, not a learned VLA. Visual
planning and verification use a local visual-language model or an HTTP backend.
The action model is independently replaceable.

```text
PICK_FIVE batch 1 → PLACE_FIVE target column 1
                 → PICK_FIVE batch 2 → PLACE_FIVE target column 2
```

There are two skill types, parameterized by source column, batch, and destination.
Examples are `Pick up five cookies from source column 1, batch 1.` and
`Place the five held cookies into target box column 1.` Four source columns do
not require sixteen separate models. Arbitrary batch sizes such as three then
seven, and a general two-box Agent plan, are outside this skill library.

## 2. Repository structure

```text
assets/                       required robot geometry and scene assets
configs/envs/                 scene, physics, camera and controller settings
configs/agents/               Planner, Verifier and Agent-loop settings
src/a3_dual_arm_sim/
  __main__.py                 python -m a3_dual_arm_sim entry point
  core/                       observation/action/skill contracts and resource paths
  envs/                       MuJoCo environments, scene generation and cameras
  experts/                    single-cookie, cross-column and same-column experts
  policies/                   policy interface, SmolVLA and LeRobot adapters
  agents/                     planning, execution, verification and coordination
  data/                       recording, collection, labels and dataset audits
  training/                   action-policy training and verifier LoRA
  evaluation/                 expert, policy and visual-verifier measurements
  teleop/                     manual-control panel and keyboard input
  cli/                        grouped command-line interfaces
tests/                        tests organized by the same responsibilities
docs/                         design and workflow documentation
```

`src/` is the Python source layout; `a3_dual_arm_sim/` is the importable package,
not a second project. Generated files belong in `datasets/` (data), `models/`
(downloaded weights), `outputs/` (trained checkpoints), `artifacts/` (diagnostics),
and `.runtime/` (caches and local jobs). New files in these roots are ignored by
Git. Required robot assets remain under `assets/`.

| Scene configuration | Purpose |
| --- | --- |
| `configs/envs/default.yaml` | Shared simulation/camera defaults |
| `configs/envs/cookie_same_column.yaml` | Same-column five-plus-five manipulation |
| `configs/envs/cookie_batch.yaml` | Batch manipulation and collection baseline |
| `configs/envs/cookie_cooperative.yaml` | Experimental cooperative transfer |
| `configs/envs/packed_elastic.yaml` | Alternative packed-contact configuration |

## 3. Installation on a new machine

Use a full Git checkout, including `assets/`, and run these examples from its
root. No specific username, disk drive, cloud provider, or Conda path is needed.
Python 3.12 is tested for the learning pipeline. Linux with CUDA-enabled PyTorch
is the tested training path; Windows supports desktop simulation, while learning
depends on the installed third-party packages.

```bash
git clone https://github.com/Eter0109/a3_dual_arm_sim.git
cd a3_dual_arm_sim
conda create -n a3sim python=3.12 pip -y
conda activate a3sim
python -m pip install --upgrade pip
python -m pip install -e .
```

Select the branch/tag containing this modular version if it is not yet the
repository default. If activation fails, run `conda init bash` or
`conda init powershell`, then reopen the terminal. Desktop teleoperation needs
Tk support in the Python environment, available through `conda install tk`.

Install optional features in the same environment:

| Feature | Command |
| --- | --- |
| LeRobot v3 collection | `python -m pip install -e ".[dataset]"` |
| SmolVLA training/execution | `python -m pip install -e ".[train]"` |
| π₀.₅ training dependencies | `python -m pip install -e ".[train-pi]"` |
| Local visual Agent and verifier LoRA | `python -m pip install -e ".[agent]"` |
| Tests and lint | `python -m pip install -e ".[dev]"` |

For SmolVLA plus the visual Agent, install `".[train,agent,dev]"`. The integrated
environment has been checked with LeRobot 0.5.1, Transformers 5.3.0, and PyTorch
2.10.0; this is a tested combination, not a lockfile. Install the PyTorch build
appropriate for your GPU/driver and check it inside the active environment:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pip check
a3-sim inspect environment
a3-sim --help
```

`python -m a3_dual_arm_sim` is equivalent to `a3-sim`. If the latter is missing,
check activation and rerun the editable install. Old `python examples/run_*.py`
commands are retired. Every subcommand accepts `--help`.

Multi-line examples use Bash continuation (`\`). In PowerShell, put the command
on one line or use PowerShell's backtick continuation instead.

## 4. Rendering, simulation, and manual control

For headless Linux with working EGL drivers:

```bash
export MUJOCO_GL=egl
a3-sim sim same-column --config configs/envs/cookie_same_column.yaml --max-steps 1000
a3-sim inspect cameras --config configs/envs/cookie_same_column.yaml --output artifacts/cameras
```

The camera command saves three PNGs and `all_cameras.png`. Headless RGB rendering
does not open a desktop window. For a local Linux desktop, run `unset MUJOCO_GL`;
for a Windows PowerShell desktop, run
`Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue`, then:

```bash
a3-sim sim same-column --config configs/envs/cookie_same_column.yaml --render
a3-sim sim control --scene cookie_transfer --config configs/envs/cookie_same_column.yaml --no-camera-render
```

The control panel receives keyboard input; the MuJoCo Viewer displays the scene.
Manual control records nothing unless `--record` is supplied. The
`--no-camera-render` option is for interactive debugging, not VLA execution or
RGB collection. An ordinary SSH terminal has no GUI display: use saved images
or configure an actual remote desktop before adding `--render`.

For a seeded expert benchmark:

```bash
a3-sim eval experts --policy same_column --profile diverse --episodes 20 --workers 4 --output artifacts/expert_eval.json
```

## 5. Collect action demonstrations

| Command | One successful full transfer becomes |
| --- | --- |
| `data actions --episodes N` | One whole-task LeRobot episode |
| `data skills --rollouts N` | Four skill-labeled LeRobot episodes |

Skill collection executes a continuous expert rollout and splits it only when
saving. The scene is not reset at skill boundaries. Failed complete-task attempts
are logged but excluded from the skill training set. A score of ten is not by
itself success: source-cookie count, release, settling, and safety also matter.

Collect 100 successful first-column transfers with scene randomization:

```bash
a3-sim data skills \
  --config configs/envs/cookie_same_column.yaml \
  --root datasets/skills_col1 --repo-id local/a3-skills-col1 \
  --rollouts 100 --max-steps 1000 --workers 4 \
  --profile diverse --source-column first --seed-start 1000
```

This requests **100 successful complete transfers / 400 skill episodes**, not
100 attempts or 100 images. The `local/...` ID is dataset metadata, not an upload
request. Use the same ID for collection, training, and inference.

For random source columns, use a different dataset root:

```bash
a3-sim data skills \
  --config configs/envs/cookie_same_column.yaml \
  --root datasets/skills_all_columns --repo-id local/a3-skills-all-columns \
  --rollouts 200 --max-steps 1000 --workers 4 \
  --profile diverse --source-column random --seed-start 3000
```

`first` selects the reference first-column expert; numeric `1`–`4` select the
column-aware expert; `random` samples across columns. Numeric/random selection
requires `diverse`. Fixing the source column does not disable the rest of the
randomization. Accepted per-column counts can differ because failed attempts
are discarded; inspect both attempts and accepted counts per column.

`diverse` varies box position, target yaw, cookie pose, camera position/FOV,
lighting, and colors. `baseline` uses smaller benchmark perturbations and is not
the Agent's `fixed` profile. Exact settings are in the [data guide](docs/data-workflow.md).

| Output | Meaning |
| --- | --- |
| `meta/`, `data/` | LeRobot v3 metadata and frame data; images need not be separate MP4 files |
| `a3_skill_segments.jsonl` | Parent seed, skill, column, batch, target and frame range |
| `attempts.jsonl` | Every completed attempt, including failures and diagnostics |
| `collection_summary.json` | Configuration and provenance; not a live progress counter |
| `seed_reservations.jsonl` | Seed bookkeeping for parallel/resumed collection |

Interrupt once to request a graceful pause, then let in-flight work close.
Resume with the same arguments plus `--resume`; `--rollouts` remains the final
total. Repeated expert failures stop the collector for inspection. Changing
scene or collection settings requires a new dataset root.

## 6. Train a skill-conditioned SmolVLA

Download a local base checkpoint:

```bash
a3-sim setup models --model lerobot/smolvla_base --output models/smolvla_base --endpoint https://huggingface.co
```

This downloads weights, not Python dependencies. A different network endpoint
can be selected explicitly, for example `--endpoint https://hf-mirror.com`.
SmolVLA may fetch its configured visual-language backbone and tokenizer on
first load. Set the shell's `HF_ENDPOINT` if these subsequent downloads must
use the same endpoint. Enable `HF_HUB_OFFLINE=1` only after all dependencies are
cached or supplied locally.

Audit the dataset and preview training:

```bash
a3-sim train skills \
  --root datasets/skills_col1 --repo-id local/a3-skills-col1 \
  --base-model models/smolvla_base --policy-type smolvla \
  --output outputs/train/smolvla_col1 \
  --steps 20000 --save-freq 20000 --batch-size 4 --num-workers 4 \
  --device cuda --validation-fraction 0.2 --seed 0 --dry-run
```

Replace `--dry-run` with `--run` to start training. With neither flag, it also
defaults to a dry run. The public collection command does not automatically
start training afterward.

The split groups all four segments by parent rollout. For 100 parents with a
0.2 holdout, 80 parents / 320 episodes train and 20 parents / 80 episodes are
reserved. This does not automatically calculate validation loss or run robot
evaluation. Normalization currently reuses full-dataset metadata statistics;
train-only statistics are not recomputed.

A single policy learns the different skill instructions. Mixed training does
not allocate exactly one quarter of the optimization steps to each column.
Using `models/smolvla_base` starts a fresh fine-tune rather than continuing a
previous A3 checkpoint.

```text
outputs/train/
  smolvla_col1_validation_manifest.json       parent split and provenance
  .smolvla_col1_smolvla_source/                adapted configuration/base-weight link
  smolvla_col1/checkpoints/
    020000/pretrained_model/                  final policy, config and processors
    last                                     latest-checkpoint pointer
```

In this example `--save-freq 20000` saves at the final step. A smaller value
retains intermediate checkpoints. The hidden prepared source is a runtime
dependency, not another trained model; retain it and its referenced base files.
Use the numbered checkpoint path when evaluating or copying a model.

## 7. Run the learned policy and inspect the motion

A fixed Planner and oracle Verifier help isolate the action policy. The policy
still sees cameras, state, and the skill instruction. `oracle` specifies the
completion judge; it does not generate the learned actions.

```bash
a3-sim agent run \
  --config configs/envs/cookie_same_column.yaml \
  --planner fixed --verifier oracle --executor lerobot \
  --policy-type smolvla --device cuda \
  --checkpoint outputs/train/smolvla_col1/checkpoints/020000/pretrained_model \
  --dataset-root datasets/skills_col1 --repo-id local/a3-skills-col1 \
  --profile diverse --source-column 1 --seed 20000 --max-steps 2400 \
  --output artifacts/evaluation/col1_seed20000.json \
  --verification-data artifacts/evaluation/col1_seed20000_frames
```

Choose seeds absent from the actual collected parent list. Change to
`--profile fixed` for a simpler fixed-layout check and label that result
separately. Add `--render` only on a configured desktop. Headless execution
still renders the policy's camera inputs.

`--verification-data` saves diagnostic windows, not action training data:

```text
artifacts/evaluation/col1_seed20000_frames/
  images/00000000/04_front.jpg
  images/00000000/04_left_wrist.jpg
  images/00000000/04_right_wrist.jpg
  images/00000001/04_front.jpg
  ...
  samples.jsonl
  truth_audit.jsonl
```

With the default five-frame window, `00`–`04` are chronological images within a
window; numbered folders are consecutive windows. Early windows may be shorter.
Windows overlap and are sampled observations, not full-frame-rate recordings.
Use a new output directory for each run. The [preview recipe](docs/data-workflow.md#preview-a-saved-rollout)
can turn these observations into a compact MP4.

The result JSON contains `success`, `completed_skills`, `cookies_in_target`,
`failure_reason`, and `events`. Task failure returns exit code 1 after saving
the result; an exception without a result is a different failure. For example,
`PLACE_FIVE batch=1 attempt=2` retries the first placement rather than beginning
the second batch.

Test multiple independent seeds using the same configuration. Change both seed
and output name, and omit `--verification-data` for metrics-only runs. Count full
task successes, failure stages, safety stops, and steps; one attractive frame is
not a task-level evaluation. See the [data guide](docs/data-workflow.md) for a
multi-seed script.

## 8. Run the visual Planner and Verifier

```bash
a3-sim setup models --model Qwen/Qwen3.5-4B --output models/Qwen3.5-4B --endpoint https://huggingface.co
a3-sim agent run --config configs/envs/cookie_same_column.yaml \
  --agent-config configs/agents/agentic_qwen35.yaml \
  --planner vlm --verifier vlm --executor expert \
  --profile diverse --source-column 1 --seed 20000 \
  --output artifacts/evaluation/visual_agent.json
```

This configuration uses a public base model, not an A3-tuned verifier. Its
Executor remains the simulator expert. To test the full learned stack, use
`--executor lerobot` and the policy/data arguments from section 7. `--task "..."`
accepts a user task within the supported skill library; keep the selected source
column consistent with it.

The YAML's `planner` and `verifier` are independent configurations.
`kind: local_qwen` loads local weights. `kind: http` uses `model`, `base_url`, and
`api_key_env` for a compatible external multimodal service. Set credentials in
the named environment variable; that backend sends task text and images to the
service. Local visual verification does not silently fall back to oracle truth.

## 9. Collect verifier data and fine-tune LoRA

Verifier examples contain the subgoal, ordered RGB images, and independently
computed `completed`, `continuing`, or `stuck` labels. They are different from
LeRobot action demonstrations. Training predicts `Yes`/`No`, then
`StillTrying`/`Stuck` for incomplete samples. Simulator truth supplies labels and
audits and never enters the visual model input.

This small first-column example demonstrates the full pipeline. Expand the seed
set and inspect class coverage before a larger run:

```bash
a3-sim data verifier \
  --root datasets/verifier/raw --config configs/envs/cookie_same_column.yaml \
  --seeds 100 101 102 103 104 105 106 107 108 109 --source-column 1 \
  --cases natural miss_grasp drop stuck --profile diverse --max-steps 1500 \
  --post-fault-steps 120 --skill-hold-steps 40 --terminal-hold-steps 80 \
  --window-frames 5 --frame-stride 10 --record-every 20 \
  --render-recorded-frames-only

a3-sim data prepare-verifier \
  --data-roots datasets/verifier/raw --output datasets/verifier/prepared \
  --holdout-seeds 108 109 --window-frames 5 --frame-stride 10

a3-sim train verifier \
  --data-root datasets/verifier/prepared --output outputs/train/verifier \
  --base-model models/Qwen3.5-4B --model-family qwen3_5 \
  --steps 2000 --gradient-accumulation 4 --learning-rate 5e-5 \
  --rank 8 --alpha 16 --dropout 0.05 --image-max-edge 256 \
  --max-sequence-length 4096 --save-every 200
```

Ten seeds × four cases means 40 attempts, potentially many windows per attempt.
Case names describe interventions, not their labels. If preparation lacks an
outcome class, collect more independent seeds into another raw root and provide
both with `--data-roots`. All windows sharing a parent seed stay in one split,
including those from different raw roots.

Raw collection saves three RGB views. The current verifier uses front and left
wrist, five frames per view, ten control steps apart: ten images spanning two
seconds at 20 Hz. `--render-recorded-frames-only` skips unnecessary RGB rendering
but keeps every physics step. Hold options collect stable completed scenes while
completion remains valid.

Training requires a BF16-capable CUDA GPU. The vision tower is frozen and
language-side LoRA parameters are trained. Outputs include `adapter/`,
`checkpoints/`, `progress.jsonl`, `training_audit.json`, and
`training_summary.json`. `--init-adapter` is a weight warm-start, not a full
optimizer-state resume; keep its training ancestry separate from holdout seeds.

`configs/agents/agentic_qwen35_lora.yaml` is a template for your locally trained
adapter. Set `verifier.adapter_path` to its directory; the documented example
uses `outputs/train/verifier/adapter`. Keep image resolution and window settings
consistent with training. The adapter must exist before using this configuration.

```bash
a3-sim eval verifier --data-root datasets/verifier/prepared \
  --parent-seeds 108 109 --all \
  --agent-config configs/agents/agentic_qwen35_lora.yaml \
  --output artifacts/evaluation/verifier_holdout.json

a3-sim agent run --config configs/envs/cookie_same_column.yaml \
  --agent-config configs/agents/agentic_qwen35_lora.yaml \
  --planner vlm --verifier vlm --executor expert \
  --profile diverse --source-column 1 --seed 20000 \
  --output artifacts/evaluation/verifier_online.json
```

The prepared `samples.jsonl` includes both splits, so select held-out parent
seeds explicitly for offline scoring. `--all`, `--limit`, and `--per-class-limit`
are alternative sampling modes. Offline window scores and full-box success are
different metrics. An online run using the expert does not validate a learned VLA.

## 10. Model switching and portability

The LeRobot executor supports `smolvla` and `pi05`. π₀.₅ has a training interface,
but its A3 task performance and hardware requirements need separate validation.
Install the matching extra and supply a compatible local base checkpoint. Use
`train skills --policy-type pi05 --dry-run` with the same required dataset,
base-model and output arguments to inspect its configuration first.

The training contract uses three RGB views and a 16-D state. Actions are 16-D
absolute targets: left seven joints + left gripper, then right seven joints +
right gripper. A stationary right arm remains part of this contract, and its
camera can still see moving objects. Removing a view or changing action
dimensions requires corresponding data, model, and inference changes. Optional
metric depth is available for diagnostics; it is not automatically fed to the
current SmolVLA or verifier.

When moving to another machine, copy the full `pretrained_model/` directory,
including processors, plus required backbone/tokenizer files and matching dataset
`meta/`. Copying only `model.safetensors` is insufficient. Check saved absolute
model/tokenizer paths and update or recreate them on the destination. A Git clone
does not include datasets or trained weights.

## 11. Development and sharing

```bash
python -m pytest -q
python -m ruff check src tests
git diff --check
```

Resource defaults resolve from the checkout. `A3_PROJECT_ROOT` can point to
another complete resource checkout; run the relative-path examples from its
root. Commit source, configs, tests, docs, and required assets. Store datasets,
weights and generated diagnostics separately, and do not commit credentials.

Before uploading reviewed changes to your own branch:

```bash
git branch --show-current
git status --short
# Add the intended source/config/doc changes, then review the staged diff.
git diff --cached --stat
git diff --cached --check
git commit -m "Document and organize A3 simulation and learning workflows"
git push -u origin HEAD
```

`HEAD` pushes the current branch; check its name first. Collection and training
commands do not upload to GitHub or create a pull request. See the
[data guide](docs/data-workflow.md) for performance, rendering, checkpoint,
and failed-rollout troubleshooting.
