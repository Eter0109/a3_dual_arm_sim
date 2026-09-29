# Data, evaluation and troubleshooting / 数据、评估与排错

[English README](../README.md) · [中文 README](../README.zh-CN.md)

This guide supplements the end-to-end commands in the READMEs. All paths are
relative to the project root and can be renamed consistently. The examples are
procedures, not experiment records or precomputed results.

本文补充 README 的完整命令。路径均相对项目根目录，可按实验统一修改；下面是
通用操作方法，不是已有实验的运行记录或成绩。

## Data contracts / 数据接口

| Product / 数据 | Contents / 内容 | Consumer / 用途 |
| --- | --- | --- |
| Whole-task actions / 整任务动作 | RGB, state, applied absolute joint targets, whole-task text | Whole-task policy baseline / 整任务策略 |
| Skill actions / 技能动作 | Same LeRobot v3 features, segmented by skill with parameterized instructions | Skill-conditioned action policy / 带技能条件的动作模型 |
| Verifier windows / 验证窗口 | Ordered images, subgoal, completion/diagnosis labels | Visual-verifier LoRA / 视觉验证器微调 |

The LeRobot recorder stores `observation.images.front`,
`observation.images.left_wrist`, `observation.images.right_wrist`,
`observation.state` (16), `observation.velocity` (16),
`observation.eef_pose` (14), `observation.force` (18), and `action` (16).
LeRobot stores task text through its task metadata. The skill-policy trainer
currently selects RGB plus state as inputs; recorded force, velocity, and end
effector pose are not automatically policy inputs.

动作记录使用 16 维绝对关节/夹爪目标，顺序为左臂七关节、左夹爪、右臂七关节、
右夹爪；关节角单位为弧度，夹爪沿用环境的开合动作约定。专家内部可能使用笛卡尔
控制，但不能把它内部的 14 维增量指令当成录制的 16 维动作。训练输入当前只选择
三路 RGB 和 state；被记录的力、速度和末端姿态不会自动送进模型。

The current RGB default is 256×256. The default front camera is at
`(0.38, -0.02, 1.30)` m, aimed at `(0.16, 0.24, 0.80)` m, with 42° vertical FOV.
Inspect the resolved environment and preview images before collecting. A scene
can inherit defaults from `envs/config.py` even if it omits them in its YAML.
Changing camera geometry changes the training distribution despite unchanged
tensor dimensions. Optional metric depth uses the environment's diagnostic API;
it is not a field in the present skill-policy training contract.

相机分辨率相同不代表数据分布相同。调整相机位置、视角或场景后，先检查三路画面，
并保留采集时的完整配置。不要把旧视角权重的表现直接当成新视角的表现。

## Randomization / 随机化

`DIVERSE_RANDOMIZATION` in `evaluation/benchmark.py` is shared by the skill
collector and the Agent's `diverse` reset. Parameter values are:

| Parameter | Value | 含义 |
| --- | --- | --- |
| `source_bin_noise_m` | 0.020 m | 源盒位置扰动 |
| `target_bin_noise_m` | 0.020 m | 目标盒位置扰动 |
| `target_bin_yaw_noise_rad` | 0.080 rad | 目标盒朝向扰动 |
| `cookie_noise_m` | 0.0003 m | 饼干位置扰动 |
| `cookie_yaw_noise_rad` | 0.015 rad | 饼干朝向扰动 |
| `camera_position_noise_m` | 0.009 m | 相机位置扰动 |
| `camera_fovy_noise_deg` | 2.0° | 垂直视场角扰动 |
| `light_noise_fraction` | 0.20 | 光照扰动参数 |
| `color_noise_fraction` | 0.15 | 颜色扰动参数 |

These are configuration parameters; scene constraints can limit final sampled
poses. `diverse` does not imply every possible randomization: for example,
friction, mass, texture, and latency are not varied by this dictionary.

这些数值是配置参数，最终采样还受环境约束。`diverse` 并不包含所有想象得到的变化，
例如这里没有自动随机摩擦、质量、纹理或延迟。

| Context | Profile | Meaning / 含义 |
| --- | --- | --- |
| `data skills`, `data actions`, `eval experts` | `baseline` | Smaller benchmark perturbations, not fully fixed / 较小扰动，并非完全固定 |
| Same commands / 同上 | `diverse` | Expanded shared randomization / 上表中的较大随机化 |
| `agent run`, `data verifier` | `fixed` | Disable cookie/box reset randomization / 关闭饼干与盒子重置随机化 |
| Same commands / 同上 | `diverse` | Shared diverse reset / 共享 diverse 设置 |

`--source-column first` in action collection chooses the reference expert;
numeric columns choose the column-aware expert. Both `first` and `1` refer to
physical source column 1, but do not select identical expert implementations.
`random` affects source selection and keeps the other profile settings.

`random` 不会固定盒子位置，也不意味着各列最终成功示范严格等量。补采应使用与原数据
不重叠的 seed 起点，并检查实际父 seed 列表。若使用 `--source-bin-x-min` 针对某类
布局补采，应记录这个过滤条件，不能把过滤后的表现当作完整分布成功率。

## Splits and learning / 划分与训练

All four skills from one parent rollout stay together. For verifier data, all
cases, skills, and overlapping windows sharing a parent seed stay together even
across raw directories. Avoid splitting individual frames or adjacent windows.
Reserve independent seeds for online evaluation, beyond the offline holdout.

技能四段共用一个父轨迹，验证器的同一父 seed 也可能生成多个故障尝试和大量重叠窗口。
逐帧随机拆分会泄漏近乎相同的画面。准备后的验证器 `samples.jsonl` 同时包含两侧，
评估时用 `--parent-seeds` 指定 `manifest.json` 中的 `holdout_seeds`。

The skill launcher writes a split manifest and trains only its selected episode
indices. It reserves validation data without running online evaluation, and
currently reuses the existing full-dataset normalization statistics. For a strict
train-only-statistics study, recomputation and provenance must be implemented
explicitly. The prepared SmolVLA source may link to base weights and use absolute
backbone/tokenizer paths; keep those references valid during training/deployment.

一份带不同指令的技能数据集会训练一个共同模型。开始某个技能时，执行器设置该技能
语言并重置模型状态；技能切换需要清空上一技能缓存的动作。训练完成后应在闭环中
测试语言、图像、归一化、动作顺序和控制频率是否一致。

## Multi-seed policy evaluation / 多 seed 策略评估

Save the following as a local script, for example `.runtime/evaluate_policy.py`,
and run `python .runtime/evaluate_policy.py` from the checkout root. Update the
model/data paths and repo ID together. Use a fresh report directory each run.
The code continues after ordinary task failures but stops on missing results.

把下列代码保存成本地脚本并从项目根目录运行。它会统计完整任务成功，普通任务失败
不会中断整组评估；若程序异常而未产生 JSON，则停止，避免误当作正常测试结果。

```python
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

report_dir = Path("artifacts/evaluation/col1_suite")
report_dir.mkdir(parents=True, exist_ok=False)
rows = []
for seed in range(20000, 20010):
    output = report_dir / f"seed_{seed}.json"
    command = [
        sys.executable, "-m", "a3_dual_arm_sim", "agent", "run",
        "--config", "configs/envs/cookie_same_column.yaml",
        "--planner", "fixed", "--verifier", "oracle", "--executor", "lerobot",
        "--policy-type", "smolvla", "--device", "cuda",
        "--checkpoint", "outputs/train/smolvla_col1/checkpoints/020000/pretrained_model",
        "--dataset-root", "datasets/skills_col1", "--repo-id", "local/a3-skills-col1",
        "--profile", "diverse", "--source-column", "1",
        "--seed", str(seed), "--max-steps", "2400", "--output", str(output),
    ]
    process = subprocess.run(command, check=False)
    if process.returncode not in (0, 1) or not output.is_file():
        raise RuntimeError(f"Evaluation crashed at seed {seed}; inspect its output")
    rows.append(json.loads(output.read_text(encoding="utf-8")))

successes = sum(row["success"] for row in rows)
summary = {
    "episodes": len(rows), "successes": successes,
    "success_rate": successes / len(rows),
    "failure_reasons": dict(Counter(
        row["failure_reason"] for row in rows if not row["success"]
    )),
}
(report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
```

Do not compare models tested with different profiles, camera settings, verifier
types, or retry budgets without stating those differences. Compare expert and
learned execution on the same scene/seeds, then evaluate the visual verifier as
a separate change. Ten seeds are a preliminary check, not a precise population
success-rate estimate.

对照实验应固定场景、seed、随机化、验证器和重试上限。先比较专家与学习策略，
再检查替换视觉验证器带来的影响。少量 seed 适合初查，不能证明所有布局都可靠。

## Preview a saved rollout

The Agent's `--verification-data` records overlapping temporal windows. This
recipe deduplicates by simulation step and creates a sampled front-view MP4 using
the already required OpenCV package. Save it as `.runtime/preview_rollout.py`,
adjust `root`, and run it from the project root. Use only a single Agent rollout
directory: combining multiple seeds with the same step numbers would mix scenes.

下面将同一轮 Agent 运行的窗口按控制步去重，生成正视图采样回放。它不是原始全帧率
视频；默认每十个控制步一帧、控制频率 20 Hz，所以预览帧率设为 2。换相机时修改
`camera`，改变采样间隔后相应调整帧率。需要支持 MP4 的播放器，某些编辑器的内置
预览不支持该编码时，可下载后用外部播放器打开。

```python
import json
from pathlib import Path
import cv2

root = Path("artifacts/evaluation/col1_seed20000_frames")
camera = "front"  # front, left_wrist, right_wrist
frames = {}
for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines():
    sample = json.loads(line)
    for frame in sample["input"]["frames"]:
        frames[frame["step"]] = root / frame["images"][camera]
if not frames:
    raise RuntimeError("No saved frames")
ordered = sorted(frames.items())
first = cv2.imread(str(ordered[0][1]))
if first is None:
    raise RuntimeError("Cannot read the first frame")
height, width = first.shape[:2]
output = root / f"preview_{camera}.mp4"
if output.exists():
    raise FileExistsError(output)
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
if not writer.isOpened():
    raise RuntimeError("MP4 encoder unavailable; inspect the saved JPEGs instead")
try:
    for step, path in ordered:
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"Invalid frame: {path}")
        cv2.putText(frame, f"step {step}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
finally:
    writer.release()
print(output)
```

## Troubleshooting / 常见问题

| Symptom / 现象 | Explanation and next check / 原因与检查 |
| --- | --- |
| `conda activate` fails; `a3-sim` missing | Initialize the shell, reopen it, activate the intended environment, reinstall editable, or use `python -m a3_dual_arm_sim`. / 先初始化终端并检查当前 Python，再检查项目安装。 |
| No viewer over SSH / SSH 没有窗口 | Headless EGL renders camera images without a desktop. `--render` needs a display server; use saved windows or a configured desktop. / 有相机输出不等于有桌面窗口。 |
| Wireframe-looking scene / 显示像线框 | Check the Viewer's rendering toggles and focus; keyboard teleop belongs to the control panel. / 检查显示选项，避免在 Viewer 里按遥控快捷键。 |
| `torchcodec` falls back to `pyav` | This concerns video decoding, not MuJoCo's renderer or whether CUDA inference is enabled. / 解码后端提示不代表退回 CPU 渲染。 |
| Low GPU utilization during collection / 采集时 GPU 占用低 | Simulation/contact solving and some preprocessing are CPU work. RGB rendering, encoding and I/O can each become bottlenecks. Try bounded `--workers` increases while monitoring memory and storage. / 并行环境才利用更多 CPU 核心，不是所有步骤都在 GPU 上。 |
| Low GPU utilization during training / 训练时 GPU 占用低 | Inspect data-loading time versus update time. Tune `--num-workers`; the fast-index wrapper already avoids decoding images just to build the frame index. / 先分辨读数与计算耗时，再调整加载进程。 |
| `score=10` but `success=false` | Check remaining source cookies, release, stability, and safety reason. / 十块进盒也可能同时碰倒或移出其他饼干，按最终成功条件判断。 |
| Same placement keeps repeating / 一直重复放置 | Read `batch`, `attempt`, verifier reason, and actual cookie state. Reissuing a placement skill cannot by itself recover a cookie dropped outside the box. / 重试次数变化不代表开始下一批。 |
| Low loss but poor manipulation / loss 低却抓不好 | Check RGB/normalization/joint ordering/absolute-action semantics/skill text/chunk reset/timing; compare saved expert actions with policy predictions using `a3-sim inspect actions --help`. This is a one-step diagnostic, not online success. / 先检查训练与推理的一致性。 |
| CUDA out of memory / 显存不足 | Lower action-training batch size; reduce verifier image size or sequence size consistently; avoid loading competing models. Checkpoint/model settings must still match inference. / 减小批量或视觉 token 开销，同时保持输入约定一致。 |
| Windows `WinError 1314` when saving `last` | The trainer may have written a numbered checkpoint before failing to create a symlink. Inspect that checkpoint and training state; use a symlink-capable environment or OS configuration. / 报错后的进程不是仍在训练，不要只看目录存在就认定保存完整。 |
| Offline model loading fails / 离线加载失败 | Base policy weights alone may not include the referenced backbone/tokenizer. Inspect `config.json` and saved processors and finish downloading dependencies before offline mode. / 同时检查迁移后失效的绝对路径。 |
| Reusing output roots fails / 输出目录已存在 | Recorders/trainers guard against unsafe overwrite/append. Use a fresh root, or the collector's explicit compatible `--resume`. / 不要直接删除仍被其他任务引用的目录。 |

MuJoCo physics, policy-camera rendering, PyTorch training/inference, and video
decoding are separate workloads. A GPU process appearing in a device monitor is
not proof that every stage is GPU-bound. Monitor runtime per rollout, per-column
success, data-loader time, disk capacity, and checkpoint progress alongside GPU
utilization.

物理、渲染、训练/推理、解码各有不同瓶颈。加速之前先检查每条轨迹耗时、各列成功率、
数据加载时间和磁盘空间；更快地产生失败示范不会提升训练集质量。
