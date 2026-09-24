# A3 Dual-Arm MuJoCo Sandbox

[中文说明](#中文说明) · [English](#english)

面向 A3 十四轴双臂机器人的 MuJoCo 仿真、规则 Expert、随机化评估、LeRobot v3
数据采集与 SmolVLA 训练项目。

MuJoCo simulation, scripted experts, randomized evaluation, LeRobot v3 data
collection, and SmolVLA training for the 14-axis A3 dual-arm robot.

---

## 中文说明

### 当前主干是什么

当前 `main` 是一个**单目标盒 5＋5 饼干装盒任务**：

- 大源盒内有 4 × 20，共 80 块倒角饼干。
- 左臂每次夹取 5 块，分两次向 2 × 5 小盒放入 10 块。
- 默认使用同列 Expert：先搬 0–4，再搬 5–9。
- 也保留跨列 Expert：先搬 0–4，再搬 20–24。
- 右臂在当前批量基线中停在待命位。

Expert 会读取 MuJoCo 真值位置、姿态和接触信息，所以它是闭环规则控制器，**不是
VLA，也不是通过相机识别目标**。它用于验证环境、生成成功示范，并为学习策略提供基线。

当前主干已经接通以下完整链路：

1. 规则 Expert 运行与随机化评估。
2. 三路 RGB 相机和机器人状态采集。
3. 只保存成功 episode 的 LeRobot v3 数据集。
4. 数据格式与成功标记审计。
5. 使用本地 SmolVLA 基座进行微调。
6. 通过统一 Policy 接口加载 checkpoint 并闭环运行或评估。

> 当前 `main` 不包含此前实验性的双盒接力文件。双盒并不是当前主干任务。

### 系统架构

```text
规则 Expert / 键盘遥控 / SmolVLA / 自定义 Policy
                         │
                         ▼
        统一 Policy 接口：reset / act / close
                         │
          ┌──────────────┴──────────────┐
          │                             │
  14-D Cartesian delta          16-D joint target
          │                             │
          └──────► IK 与安全限幅 ◄─────┘
                         │
                         ▼
                 MuJoCo A3 环境
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      固定 observation            LeRobot v3 Recorder
```

核心数据约定：

| 数据 | 形状 | 内容 |
| --- | ---: | --- |
| `observation.images.front` | 256 × 256 × 3 | 固定前视 RGB |
| `observation.images.left_wrist` | 256 × 256 × 3 | 左腕 RGB |
| `observation.images.right_wrist` | 256 × 256 × 3 | 右腕 RGB |
| `observation.state` | 16 | 左 7 关节＋左夹爪＋右 7 关节＋右夹爪 |
| `observation.velocity` | 16 | 对应速度 |
| `observation.eef_pose` | 14 | 双臂末端位置与四元数 |
| `observation.force` | 18 | 双腕力/力矩、四个指尖触觉和夹爪力 |
| `action` | 16 | 实际送入仿真的绝对关节/夹爪目标 |

Expert 内部产生 14 维末端增量动作，但 Recorder 保存经过 IK、关节限制和速率限制后的
16 维实际动作。训练和部署使用同一套 16 维动作定义。

### Windows 安装

项目当前位于：

```text
D:\download\a3_dual_arm_sim
```

推荐使用 Python 3.10 或 3.11。PowerShell：

```powershell
cd D:\download\a3_dual_arm_sim
conda create -n a3sim python=3.10 -y
conda activate a3sim
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

按需要安装额外能力：

```powershell
# 仅录制、读取和回放 LeRobot 数据集
python -m pip install -e ".[dataset,dev]"

# SmolVLA 训练与推理；已经包含数据集依赖
python -m pip install -e ".[train,dev]"
```

原生 Windows 不需要设置 `MUJOCO_GL`。Linux/WSL 无窗口渲染时可使用
`MUJOCO_GL=egl`。

### 1. 基础检查与相机预览

```powershell
# 编译模型并打印关节、执行器和传感器信息
a3-sim inspect --scene cookie_transfer

# 无窗口稳定性检查
a3-sim smoke --scene cookie_transfer --config configs\cookie_batch.yaml --steps 200

# 保存三路策略相机图片，并打开合成预览
python examples\preview_cameras.py --scene cookie_transfer --config configs\cookie_batch.yaml --output outputs\camera_preview --show
```

相机参数仍是原型估计值，不是标定结果。

### 2. 运行单盒 5＋5 Expert

同列版是默认方案：

```powershell
python -u examples\run_cookie_batch.py --expert same_column --render --debug --max-steps 1000
```

跨列版：

```powershell
python -u examples\run_cookie_batch.py --expert cross_column --render --debug --max-steps 1000
```

四列随机（由 `--seed` 决定）或指定第 3 列观察动作：

```powershell
python -u examples\run_cookie_batch.py --expert varied_column --seed 0 --render --max-steps 1000
python -u examples\run_cookie_batch.py --expert varied_column --source-column 3 --render --max-steps 1000
```

这些命令只运行 Expert 并输出评估结果，**不会录制训练数据**。无窗口运行时删除
`--render`，通常更快。单次成功需要最终目标盒中恰好有 10 块已释放、直立、稳定且
被盒体包含，源盒剩余 70 块。

### 3. 随机化评估

```powershell
python -u examples\benchmark_cookie_batch.py --compare --episodes 20 --seed-start 0 --workers 4 --output artifacts\benchmark_results_local.json
```

`20/20` 表示**同一个单盒任务独立 reset 并运行 20 个 episode，20 次全部成功**，
不是一个场景中有 20 个盒子。比较模式会分别为同列和跨列 Expert 各运行 20 次。

默认评估包含：

- 源盒和目标盒位置各自最多约 ±1 cm 扰动。
- 目标盒小角度偏航扰动。
- 每块饼干 0.3 mm 位置扰动和 0.015 rad 偏航扰动。
- 每轮按进入目标盒的数量计 0–10 分，同时记录成功率、步数、耗时和失败原因。

仓库中的
[`artifacts/benchmark_results.json`](artifacts/benchmark_results.json)
报告 seeds 0–19 上：

| Expert | 成功 episode | 平均步数 | 平均耗时 |
| --- | ---: | ---: | ---: |
| same-column | 20/20 | 429.6 | 63.13 s |
| cross-column | 20/20 | 425.9 | 60.83 s |

这是规则 Expert 的单盒结果，不是已训练 VLA 的成功率。

### 4. 采集单盒训练数据

安装 `dataset` 或 `train` 依赖后运行：

```powershell
python -u examples\collect_cookie_benchmark.py --episodes 100
```

默认输出：

```text
datasets/a3_single_box_same_column_100
repo-id: local/a3-single-box-same-column-100
```

采集器会：

- 使用随机化后的同列 Expert。
- 渲染并保存三路策略相机。
- 记录 16-D 状态、速度、14-D 末端位姿和 18-D 力反馈。
- 保存经过 IK 与安全限制后的 16-D `joint_position` 动作。
- 将失败尝试写入 `attempts.jsonl`，但不放入训练数据集。
- 连续失败 5 次时停止，要求先检查环境。
- 完成后自动审计 LeRobot v3 schema、相机尺寸、动作维度和成功标记。

按一次 Ctrl+C 会在当前 episode 完成后干净停止。继续未完成的数据集：

```powershell
python -u examples\collect_cookie_benchmark.py --episodes 100 --resume
```

需要更丰富的新数据时，用独立的多样化采集配置（不会改写上面的 100 组数据）：

```powershell
python -u examples\collect_cookie_benchmark.py --profile diverse --episodes 100
# 中途停止后继续同一批数据
python -u examples\collect_cookie_benchmark.py --profile diverse --episodes 100 --resume
```

它写入 `datasets/a3_single_box_diverse`（repo-id 为
`local/a3-single-box-diverse`）。每个 seed 会改变大小盒位置、小盒角度、饼干
初始姿态、三路相机位置/视角、灯光和颜色；此命令仍抓第一列。失败尝试
只进 `attempts.jsonl`，不会混入训练数据。`collection_summary.json` 记录配置，恢复
采集时会检查配置一致性。正在运行的训练仍使用启动时指定的旧数据集；要利用新数据，
需要另行启动训练或续训。

盒位随机化上限为每个平面轴 ±2 cm，小盒偏航 ±0.08 rad；饼干之间缝隙很小，
所以单块饼干的扰动保持在已验证的较小范围。若 Expert 判断随机到的盒位不可达，
采集器会将该 seed 记为失败尝试后继续，而不会保存半成品 episode。

如果要让 Expert 从四列中轮流选择来源列，使用独立的多列数据集：

```powershell
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column random --episodes 100
# 中途停止后继续
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column random --episodes 100 --resume
# 或只指定第 3 列（1–4）
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column 3 --episodes 20
```

随机模式按 seed 每四次覆盖四列，独立写入
`datasets/a3_single_box_diverse_all_columns`；指定列模式写入单独的 `..._column_3`
目录。每个 episode 的语言任务会标明来源列。多列模式与上面的 `--profile diverse`
使用完全相同的盒位、饼干、相机、光照和颜色扰动范围，只额外随机化来源列。
这样不会影响原来的单列多样化数据或正在训练的模型；失败尝试仍不会保存为示范。
每次仍是两批各五块，尚不支持 3+7 等可变批量。

采集前可先用相同的随机化参数做 20 组成功率检查（不会录制数据）：

```powershell
python -u examples\benchmark_cookie_batch.py --policy varied_column --profile diverse --episodes 20 --seed-start 0 --workers 4 --output artifacts\cookie_diverse_all_columns_seed0_19.json
```

也可以用键盘遥控录制人工数据：

```powershell
a3-sim teleop --scene cookie_transfer --config configs\cookie_batch.yaml --record datasets\a3_manual --repo-id local/a3-manual --videos
```

`--no-camera-render` 只适合不录制的交互调试；程序会拒绝将它与 `--record`
同时使用，避免保存黑色图像。

### 5. 训练 SmolVLA

训练代码已经实现，但需要你提供本地 SmolVLA 基座目录。该目录至少应包含：

```text
config.json
model.safetensors
```

基座配置必须是 `type: smolvla`，并支持至少 16 维状态和 16 维动作。独立克隆本项目时，
默认的相邻 `vla_ur5e_sim/assets/policy/base/pretrained_model` 通常不存在，因此建议
始终显式传入 `--model`。训练进程使用离线模式，基座依赖的 VLM/tokenizer 也必须已在
本地或 Hugging Face 缓存中。

先审计数据并查看实际训练命令：

```powershell
a3-sim train-smolvla --root datasets\a3_single_box_same_column_100 --repo-id local/a3-single-box-same-column-100 --model D:\models\smolvla\pretrained_model --output outputs\train\a3_cookie_smolvla --steps 20000 --batch-size 4 --device cuda --dry-run
```

确认无误后删除 `--dry-run`：

```powershell
a3-sim train-smolvla --root datasets\a3_single_box_same_column_100 --repo-id local/a3-single-box-same-column-100 --model D:\models\smolvla\pretrained_model --output outputs\train\a3_cookie_smolvla --steps 20000 --batch-size 4 --device cuda
```

注意：

- `--output` 指向的目录必须尚不存在，防止覆盖训练结果。
- `--device cpu` 可以用于兼容性检查，但完整训练通常应使用 CUDA GPU。
- Windows 无符号链接权限时，程序会依次尝试硬链接和复制权重。
- 一步 smoke/dry-run 只验证加载、数据、前后向和保存链路，不证明模型学会任务。

### 6. 加载 checkpoint 并评估

PowerShell 中配置策略：

```powershell
$env:A3_SMOLVLA_CHECKPOINT = "D:\download\a3_dual_arm_sim\outputs\train\a3_cookie_smolvla\checkpoints\<step>\pretrained_model"
$env:A3_SMOLVLA_DATASET_ROOT = "D:\download\a3_dual_arm_sim\datasets\a3_single_box_same_column_100"
$env:A3_SMOLVLA_REPO_ID = "local/a3-single-box-same-column-100"
$env:A3_SMOLVLA_DEVICE = "cuda"
```

查看单次闭环运行：

```powershell
a3-sim run --scene cookie_transfer --config configs\cookie_batch.yaml --policy a3_dual_arm_sim.smolvla_policy:make_policy --task "transfer 10 cookies into target box" --steps 1000 --render
```

先用训练过的 seed 做单次闭环排查，再用未参与训练的 seed 检查泛化：

```powershell
python -u examples\benchmark_cookie_batch.py --policy a3_dual_arm_sim.smolvla_policy:make_policy --episodes 1 --seed-start 0 --max-steps 600 --workers 1 --output artifacts\smolvla_train_seed_diagnostic.json
python -u examples\benchmark_cookie_batch.py --policy a3_dual_arm_sim.smolvla_policy:make_policy --episodes 20 --seed-start 200 --workers 1 --output artifacts\smolvla_validation.json
```

评估外部策略时，基准程序会自动生成三路 policy RGB；规则 Expert 则默认关相机以加速。
终端和 JSON 报告里的 `Policy RGB Cameras` / `policy_rgb_cameras` 可以核对这一点。
`--render` 只控制供人观看的窗口，与模型相机画面是两回事。GPU 策略建议
`--workers 1`，避免多个进程重复加载模型。建议先确认训练 seed 闭环成功，再花时间跑
20 个未见过的 seed；训练 loss 下降不等于装盒成功。SmolVLA 的推理采样也会随
episode seed 重置，便于复测。报告里的 `min_cookies_in_source` 表示过程中源盒最少
剩几块；它和最后的 `cookies_in_source` 不同，可识别“拿出后又掉回去”的情况。

如需把问题拆开，可先比较保存的专家帧和模型的即时动作（左臂/右臂分别计误差）：

```powershell
python -u examples\diagnose_smolvla_actions.py --checkpoint outputs\train\a3_cookie_smolvla\checkpoints\020000\pretrained_model --dataset-root datasets\a3_single_box_same_column_100 --repo-id local/a3-single-box-same-column-100 --device cuda --indices 0 50 100 300
```

这只测“给定正确专家画面时下一步动作有多准”；**不能替代**上面的闭环成功率。

### 遥控按键

遥控会打开 MuJoCo Viewer 和独立控制面板。键盘焦点应放在 **A3 Teleoperation
Control** 面板上：

| 按键 | 功能 |
| --- | --- |
| `1 / 2 / 3` | 选择左臂 / 右臂 / 双臂 |
| `W/S`, `A/D`, `R/F` | 世界坐标 XYZ 平移 |
| `I/K`, `J/L`, `U/O` | 世界坐标 Rx/Ry/Rz 旋转 |
| `[` / `]` | 夹爪闭合 / 张开 |
| `P` | 暂停或继续录制 |
| `Space` | 紧急停止 |
| `Q` | 保存并退出 |
| `X` | 丢弃并退出 |

仅调试、不渲染三路策略 RGB：

```powershell
a3-sim teleop --scene cookie_transfer --config configs\cookie_batch.yaml --no-camera-render
```

### 主要文件

| 路径 | 用途 |
| --- | --- |
| [`configs/default.yaml`](configs/default.yaml) | 通用仿真、相机和默认任务参数 |
| [`configs/cookie_batch.yaml`](configs/cookie_batch.yaml) | 当前单盒 5＋5 基线配置 |
| [`examples/run_cookie_batch.py`](examples/run_cookie_batch.py) | 同列/跨列 Expert 单轮运行 |
| [`examples/benchmark_cookie_batch.py`](examples/benchmark_cookie_batch.py) | 随机化多 episode 评估 |
| [`examples/collect_cookie_benchmark.py`](examples/collect_cookie_benchmark.py) | 成功示范采集与恢复 |
| [`src/a3_dual_arm_sim/env.py`](src/a3_dual_arm_sim/env.py) | 双臂环境、动作转换、观测和安全限制 |
| [`src/a3_dual_arm_sim/cookie_transfer.py`](src/a3_dual_arm_sim/cookie_transfer.py) | 饼干场景、随机化和任务判定 |
| [`src/a3_dual_arm_sim/batch_expert.py`](src/a3_dual_arm_sim/batch_expert.py) | 跨列批量 Expert |
| [`src/a3_dual_arm_sim/same_column_batch_expert.py`](src/a3_dual_arm_sim/same_column_batch_expert.py) | 同列批量 Expert |
| [`src/a3_dual_arm_sim/benchmark.py`](src/a3_dual_arm_sim/benchmark.py) | Policy/Expert 统一评估 |
| [`src/a3_dual_arm_sim/recording.py`](src/a3_dual_arm_sim/recording.py) | LeRobot v3 数据写入 |
| [`src/a3_dual_arm_sim/training.py`](src/a3_dual_arm_sim/training.py) | 数据审计和 SmolVLA 训练启动 |
| [`src/a3_dual_arm_sim/smolvla_policy.py`](src/a3_dual_arm_sim/smolvla_policy.py) | checkpoint 推理适配器 |
| [`src/a3_dual_arm_sim/cli.py`](src/a3_dual_arm_sim/cli.py) | `a3-sim` 命令行入口 |

### 测试

```powershell
python -m pytest -q
```

仅检查当前主任务：

```powershell
python -m pytest -q tests\test_cookie_batch.py tests\test_cookie_same_column.py tests\test_benchmark.py tests\test_training.py
```

### 已知边界

- 机器人、相机、摩擦、控制增益和盒子尺寸仍是原型估计，不是实机标定。
- 夹爪使用简化的同步平行指动力学；机器人自碰撞尚未纳入。
- 规则 Expert 使用 privileged state，不能当作视觉策略。
- 已提交的 20/20 报告属于 Expert，不属于 SmolVLA。
- 仓库提供采集、训练和推理代码，但不附带 100-episode 数据集或训练完成的 checkpoint。
- 当前主干是单盒任务；双盒接力不在 `main`。
- 从仿真迁移到实机仍需要相机外参、关节零位、执行器、摩擦和安全限制标定。

---

## English

### Project status

The current `main` branch implements a **single-target-bin 5+5 Cookie packing
task**. The source bin contains 80 beveled Cookie proxies in a 4 × 20 layout.
The left arm grasps five at a time and fills one 2 × 5 target bin in two
transfers.

Two scripted experts share the same environment:

- `same_column` (default): IDs 0–4, then 5–9.
- `cross_column`: IDs 0–4, then 20–24.

The experts use MuJoCo ground truth for closed-loop control. They are not VLA
policies. The repository does provide an end-to-end pipeline for randomized
expert evaluation, successful LeRobot v3 demonstration collection, dataset
auditing, SmolVLA fine-tuning, checkpoint loading, and closed-loop evaluation.

The experimental two-box relay is not present on the current `main` branch.

### Architecture and data contract

```text
scripted expert / teleoperation / SmolVLA / custom policy
                              │
                              ▼
                 common Policy protocol
                              │
              Cartesian delta or joint target
                              │
                              ▼
                 IK, limits, and rate limiting
                              │
                              ▼
                      MuJoCo environment
                              │
                 observation + applied action
                              │
                              ▼
                    LeRobot v3 recorder
```

The observation contract contains three 256 × 256 RGB streams, 16-D joint and
gripper state, 16-D velocity, 14-D dual-end-effector pose, and 18-D force
feedback. Scripted batch experts request 14-D Cartesian-delta actions. The
environment applies IK and safety limits, and the recorder stores the actual
16-D absolute joint/gripper target sent to MuJoCo. Training and inference
therefore use the same 16-D action definition.

### Setup on Windows

```powershell
cd D:\download\a3_dual_arm_sim
conda create -n a3sim python=3.10 -y
conda activate a3sim
python -m pip install --upgrade pip

# Simulation and tests
python -m pip install -e ".[dev]"

# Add recording support
python -m pip install -e ".[dataset,dev]"

# Add SmolVLA training and inference support
python -m pip install -e ".[train,dev]"
```

Do not set `MUJOCO_GL` on native Windows. For headless Linux/WSL rendering,
use `MUJOCO_GL=egl`.

### Quick start

```powershell
# Inspect and smoke-test the Cookie scene
a3-sim inspect --scene cookie_transfer
a3-sim smoke --scene cookie_transfer --config configs\cookie_batch.yaml --steps 200

# Preview the three policy cameras
python examples\preview_cameras.py --scene cookie_transfer --config configs\cookie_batch.yaml --show

# Run the default same-column expert
python -u examples\run_cookie_batch.py --expert same_column --render --debug --max-steps 1000

# Run the cross-column expert
python -u examples\run_cookie_batch.py --expert cross_column --render --debug --max-steps 1000

# Choose a source column by seed, or specify column 3 (1–4)
python -u examples\run_cookie_batch.py --expert varied_column --seed 0 --render --max-steps 1000
python -u examples\run_cookie_batch.py --expert varied_column --source-column 3 --render --max-steps 1000
```

The expert runners do not record training data.

### Randomized benchmark

```powershell
python -u examples\benchmark_cookie_batch.py --compare --episodes 20 --seed-start 0 --workers 4 --output artifacts\benchmark_results_local.json
```

`20/20` means twenty independent resets of the same single-bin task, not
twenty bins in one scene. The committed report records 20/20 successful
episodes for each scripted expert on seeds 0–19, averaging 429.6 steps for
same-column and 425.9 for cross-column. These are expert results, not SmolVLA
results.

### Collect a LeRobot v3 Cookie dataset

```powershell
python -u examples\collect_cookie_benchmark.py --episodes 100

# Continue a cleanly paused partial dataset
python -u examples\collect_cookie_benchmark.py --episodes 100 --resume
```

For a separate, more varied dataset, use:

```powershell
python -u examples\collect_cookie_benchmark.py --profile diverse --episodes 100
python -u examples\collect_cookie_benchmark.py --profile diverse --episodes 100 --resume
```

This writes to `datasets/a3_single_box_diverse` with repo ID
`local/a3-single-box-diverse`; it does not alter the baseline 100 episodes.
Source/target poses, cookie poses, cameras, lighting, and colors vary within
bounded ranges. This command still grasps the first column. Failed
attempts remain in `attempts.jsonl` only. Resume checks the saved configuration.
An already-running training job continues to use its original dataset until a
new training run is launched.
Box centers vary by up to ±2 cm per planar axis, and target yaw by ±0.08 rad.
Per-cookie jitter remains small because the source stack is tightly packed.
Unreachable randomized layouts are logged as failed attempts, not saved demos.

To collect from all four source columns in a separate dataset, or target one
specific column, use:

```powershell
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column random --episodes 100
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column random --episodes 100 --resume
python -u examples\collect_cookie_benchmark.py --profile diverse --source-column 3 --episodes 20
```

The random mode covers all four columns per four seeds and writes to
`datasets/a3_single_box_diverse_all_columns`; a fixed column uses its own
`..._column_3` dataset. Each episode records the selected source column in
its language task. Multi-column mode uses exactly the same box, Cookie, camera,
lighting, and color randomization ranges as `--profile diverse`, and also varies
the source column. Only strict
successes are saved. Batch sizes remain five plus five; 3+7 is not yet supported.

To check success rate on 20 seeds with the same randomization before collecting
data (without recording episodes):

```powershell
python -u examples\benchmark_cookie_batch.py --policy varied_column --profile diverse --episodes 20 --seed-start 0 --workers 4 --output artifacts\cookie_diverse_all_columns_seed0_19.json
```

The default output is `datasets/a3_single_box_same_column_100`, with repo ID
`local/a3-single-box-same-column-100`. Only successful episodes enter the
dataset; all attempts are logged. The collector saves three 256 × 256 RGB
cameras, 16-D state and velocity, 14-D end-effector pose, 18-D force feedback,
the language task, and the final applied 16-D joint action.

### Train SmolVLA

Provide a local compatible SmolVLA base checkpoint. A standalone clone normally
does not contain the CLI's default sibling checkpoint, so pass `--model`
explicitly. The base model and its VLM/tokenizer dependencies must already be
available locally or in the Hugging Face cache because training launches in
offline mode.

```powershell
# Audit the dataset and print the exact training command
a3-sim train-smolvla --root datasets\a3_single_box_same_column_100 --repo-id local/a3-single-box-same-column-100 --model D:\models\smolvla\pretrained_model --output outputs\train\a3_cookie_smolvla --steps 20000 --batch-size 4 --device cuda --dry-run

# Remove --dry-run to start training.
```

The output directory must not already exist. CPU is supported for compatibility
checks but is generally impractical for full training. On Windows, the runtime
checkpoint view falls back from a symbolic link to a hard link or file copy.

### Run and evaluate a trained checkpoint

```powershell
$env:A3_SMOLVLA_CHECKPOINT = "D:\path\to\checkpoint\pretrained_model"
$env:A3_SMOLVLA_DATASET_ROOT = "D:\download\a3_dual_arm_sim\datasets\a3_single_box_same_column_100"
$env:A3_SMOLVLA_REPO_ID = "local/a3-single-box-same-column-100"
$env:A3_SMOLVLA_DEVICE = "cuda"

a3-sim run --scene cookie_transfer --config configs\cookie_batch.yaml --policy a3_dual_arm_sim.smolvla_policy:make_policy --task "transfer 10 cookies into target box" --steps 1000 --render

python -u examples\benchmark_cookie_batch.py --policy a3_dual_arm_sim.smolvla_policy:make_policy --episodes 1 --seed-start 0 --max-steps 600 --workers 1 --output artifacts\smolvla_train_seed_diagnostic.json
python -u examples\benchmark_cookie_batch.py --policy a3_dual_arm_sim.smolvla_policy:make_policy --episodes 20 --seed-start 200 --workers 1 --output artifacts\smolvla_validation.json
```

External policies automatically receive all three rendered policy RGB images;
scripted experts leave camera rendering off for speed. Check `Policy RGB Cameras`
in the terminal or `policy_rgb_cameras` in the JSON report. `--render` only opens
the human viewer and does not control policy cameras. Start with a training seed
to test closed-loop imitation, then evaluate on held-out seeds. Keep `--workers 1`
for a GPU policy to avoid loading multiple model copies. Low training loss alone
does not demonstrate successful placement.
SmolVLA inference sampling is seeded at each episode reset, so repeating a
benchmark seed is comparable. `min_cookies_in_source` records the lowest source
count reached during an episode and can reveal a temporary extraction that
later falls back into the box.

For a teacher-forced action check on saved expert frames (separate left/right
arm errors), run:

```powershell
python -u examples\diagnose_smolvla_actions.py --checkpoint outputs\train\a3_cookie_smolvla\checkpoints\020000\pretrained_model --dataset-root datasets\a3_single_box_same_column_100 --repo-id local/a3-single-box-same-column-100 --device cuda --indices 0 50 100 300
```

This diagnostic does not replace closed-loop evaluation.

### Teleoperation and manual recording

```powershell
# Responsive viewer-only debugging; policy RGB is disabled
a3-sim teleop --scene cookie_transfer --config configs\cookie_batch.yaml --no-camera-render

# Manual LeRobot recording with all three policy cameras
a3-sim teleop --scene cookie_transfer --config configs\cookie_batch.yaml --record datasets\a3_manual --repo-id local/a3-manual --videos
```

Keep keyboard focus on the separate **A3 Teleoperation Control** panel. Select
the left, right, or both arms with `1/2/3`; translate with `W/S`, `A/D`,
`R/F`; rotate with `I/K`, `J/L`, `U/O`; and close/open the gripper with
`[`/`]`. `Space` is the emergency stop, `Q` saves and exits, and `X`
discards the episode. Recording cannot be combined with
`--no-camera-render`.

### Key files

- [`examples/run_cookie_batch.py`](examples/run_cookie_batch.py): one expert rollout.
- [`examples/benchmark_cookie_batch.py`](examples/benchmark_cookie_batch.py): randomized evaluation.
- [`examples/collect_cookie_benchmark.py`](examples/collect_cookie_benchmark.py): successful demonstration collection.
- [`src/a3_dual_arm_sim/env.py`](src/a3_dual_arm_sim/env.py): actions, observations, safety, and rendering.
- [`src/a3_dual_arm_sim/benchmark.py`](src/a3_dual_arm_sim/benchmark.py): shared expert/policy evaluator.
- [`src/a3_dual_arm_sim/recording.py`](src/a3_dual_arm_sim/recording.py): LeRobot v3 writer.
- [`src/a3_dual_arm_sim/training.py`](src/a3_dual_arm_sim/training.py): dataset audit and SmolVLA launcher.
- [`src/a3_dual_arm_sim/smolvla_policy.py`](src/a3_dual_arm_sim/smolvla_policy.py): checkpoint inference adapter.

### Validation and limitations

```powershell
python -m pytest -q
```

- Geometry, dynamics, controller gains, and camera poses are prototype values,
  not hardware calibration.
- The scripted experts use privileged simulator state.
- The repository does not ship a 100-episode dataset, a SmolVLA base model, or a
  trained task checkpoint.
- A smoke run proves software compatibility, not that the model learned the task.
- Sim-to-real use still requires robot, gripper, camera, contact, and safety
  calibration.

## License and asset provenance

See [`assets/a3/PROVENANCE.md`](assets/a3/PROVENANCE.md) for source robot assets.
The Robotiq visual meshes retain the license documented in
[`assets/a3/ROBOSUITE_LICENSE.txt`](assets/a3/ROBOSUITE_LICENSE.txt).
