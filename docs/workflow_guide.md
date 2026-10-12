# 采集、训练、评测与自定义策略

本文面向源码安装。所有命令在仓库根目录执行，先激活同一个 Python 环境。推荐 Python 3.12，安装实际需要的依赖：

```bash
python -m pip install -e '.[train,dev]'
python -m a3_dual_arm_sim.cli --help
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

仅使用仿真和专家时可安装 `'.[dev]'`；录制 LeRobot 数据需要 `'.[dataset]'`。训练需要可用的 Torch/CUDA；`nvidia-smi` 与 Python 的 `torch.cuda.is_available()` 都应检查。Linux 无头相机渲染在命令前加 `MUJOCO_GL=egl`；桌面窗口使用 `env -u MUJOCO_GL`。Windows 终端不使用这种环境变量前缀，可先设置/清除环境变量再运行 Python。

## 1. 任务与实验配置

| 任务 | 目标 | 专家 | 采集随机化配置 | 默认专家采集步数上限 |
| --- | --- | --- | --- | ---: |
| `2x5` | 源 80 块搬出 10 块，目标 2 列各 5 块 | `same_column` | `configs/randomization.yaml` | 1600 |
| `2x10` | 源 80 块搬出 20 块，目标 2 列各 10 块 | `variable_grasp` | `configs/randomization_2x10.yaml` | 3200 |

统一评测中 `--policy expert` 自动选择对应专家。`cross_column` 仅适用于 2×5。两种任务的成功规则分别实现，不应只换几何 YAML 而保留另一任务的计分器。

配置字段 `profile: basic|medium|advanced` 控制扰动强度，`source_column: 1|2|3|4|random` 控制来源列。2×10 还包含 `first_grasp: 0..9|random`，生成 `[n,10-n,n,10-n]` 抓取计划。三者独立；列号按源布局 X 从小到大排序。随机化具体范围见 [随机化说明](randomization.md)。

建议复制 YAML 为每个实验保存一份配置。采集脚本通过 YAML 选强度/来源列，统一评测额外提供 `--profile`、`--source-column` 覆盖项，二者 CLI 参数不要混用。

## 2. 使用已有策略采数据

### 2.1 用内置专家采集可训练演示

先用下面命令各采少量成功演示，确认磁盘、三相机录制和专家执行正常：

```bash
# 2×5：默认 YAML 为 basic、第 1 列；按需要编辑或复制配置。
MUJOCO_GL=egl python examples/collect_cookie_benchmark.py \
  --randomization-config configs/randomization.yaml \
  --episodes 5 --max-attempts 15 --seed-start 6001 \
  --root datasets/cookie_2x5_trial --repo-id local/cookie-2x5-trial

# 2×10：数量、来源列和强度从独立 YAML 读取。
python examples/collect_cookie_2x10.py --dry-run
MUJOCO_GL=egl python examples/collect_cookie_2x10.py \
  --randomization-config configs/randomization_2x10.yaml \
  --episodes 5 --max-attempts 15 --seed-start 7001 \
  --root datasets/cookie_2x10_trial --repo-id local/cookie-2x10-trial
```

`--episodes` 是目标成功保存条数，`--max-attempts` 包含失败；达到尝试上限不保证已得到目标条数。连续 25 次失败会停止。默认录制视频；`--no-videos` 改为 parquet 内嵌图像，通常更占空间。

完整训练应增加采集量并检查任务覆盖，5 条只是流程检查。默认录制状态/动作为双臂 16D，保存的是动作前观测和环境实际应用的关节目标。采集产物包括：

- `meta/`、`data/`、`videos/`：LeRobot v3 数据和统计。
- `collection_summary.json`：配置、随机种子起点、动作约定等实验信息。
- `attempts.jsonl`：包括失败在内的尝试记录。
- `a3_episode_metadata.jsonl`：成功保存回合的指令及溯源信息。

Ctrl+C 会请求完成当前回合后退出；用原命令加 `--resume` 续采。必须保留配置、repo-id、随机种子起点和目标目录的一致性；配置变化使用新目录。续采时 `--episodes` 和 `--max-attempts` 是数据集累计目标/上限。

采集的成功筛选仍用完整任务规则；评测按放入数量计分，不会自动放宽训练数据的筛选规则。

### 2.2 审计与可选左臂转换

```bash
python examples/audit_smolvla_data.py \
  --root datasets/cookie_2x5_trial --repo-id local/cookie-2x5-trial \
  --output outputs/cookie_2x5_audit.json

# 可选：16D 转 8D。保留原数据，输出目录必须是新的。
python examples/prepare_left_arm_dataset.py \
  --source datasets/cookie_2x5_trial --output datasets/cookie_2x5_left
```

训练器支持 8D 或 16D；无需为了训练强制转换。转换后训练的 `--root` 和评测的 `--dataset-root` 都应指向转换结果。转换可能用符号链接引用原视频，勿删除原始数据。已有 8D 数据不要重复转换。

2×10 使用相同审计入口，替换数据路径和 repo-id；其审计还检查二十块的四次抓取计划和释放记录。

### 2.3 用已有模型或自定义策略录制调试轨迹

标准两个装盒采集 CLI 当前固定使用专家，**没有通用 `--policy` 选项**。任意插件可通过通用执行器录制单条轨迹，例如运行仓库提供的正弦动作示例：

```bash
MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --config configs/cookie_batch.yaml \
  --policy a3_dual_arm_sim.examples.custom_policy:make_policy \
  --task "transfer 10 cookies into target box" --steps 100 \
  --record datasets/plugin_debug --repo-id local/plugin-debug --videos
```

示例只演示接口，不会完成搬运。用已有 SmolVLA 替换插件：

```bash
A3_SMOLVLA_CHECKPOINT=outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model \
A3_SMOLVLA_DATASET_ROOT=datasets/a3_front_close_left_100 \
A3_SMOLVLA_REPO_ID=local/a3-front-close-left-100 \
A3_SMOLVLA_DEVICE=cuda MUJOCO_GL=egl \
a3-sim run --scene cookie_transfer --config configs/cookie_batch.yaml \
  --policy a3_dual_arm_sim.policies.smolvla:make_policy \
  --task "transfer 10 cookies into target box" --steps 500 \
  --record datasets/smolvla_debug --repo-id local/smolvla-debug --videos
```

上例检查点是本机已有训练产物的路径示例，仓库不附带权重；使用自己的检查点和对应训练数据。`--record` 应指向新目录。

通用执行器默认可以保存失败轨迹，不生成标准专家采集的完整 `collection_summary.json`；其输出不能直接作为已经通过训练审计的成功演示集。尤其 2×10 审计要求四次抓取的真值记录，任意神经策略不自动提供这些记录。若要从任意策略批量采集可训练数据，需要补齐专用采集、成功筛选及真实元数据，不能伪造摘要绕过审计。通用 `a3-sim run --scene cookie_transfer` 也不能代替 2×10 评测的任务调度。

## 3. 训练模型

### 3.1 SmolVLA：先冒烟，再完整训练

完整下载流程、可固定的版本与离线缓存见 [SmolVLA 手册](smolvla.md)。可用项目下载入口准备示例数据和基座：

```bash
a3-sim download-smolvla \
  --dataset-root datasets/a3_front_close_left_100 \
  --model-root models/smolvla_base \
  --manifest outputs/download_manifest.json
```

已有自采数据不需要重新下载示例数据。基座也可由训练器在线解析，或使用已经准备好的 `models/smolvla_base`。先审计自采数据，再做短训练：

```bash
a3-sim train-smolvla \
  --root datasets/cookie_2x5_trial --repo-id local/cookie-2x5-trial \
  --model models/smolvla_base --output outputs/cookie_2x5_smoke/model \
  --steps 20 --batch-size 1 --num-workers 0 --save-freq 20 --device cuda
```

通过后，用完整数据在新目录训练：

```bash
mkdir -p outputs/cookie_2x5_train
set -o pipefail
a3-sim train-smolvla \
  --root datasets/cookie_2x5_trial --repo-id local/cookie-2x5-trial \
  --model models/smolvla_base --output outputs/cookie_2x5_train/model \
  --steps 20000 --batch-size 8 --num-workers 2 --save-freq 2000 \
  --lr 0.00005 --seed 1000 --device cuda \
  2>&1 | tee outputs/cookie_2x5_train/train.log
```

这里数据路径沿用教程，正式训练应替换为足量且审计通过的数据。批大小按显存调整。`--dry-run` 可以输出训练命令，但仍会审计并生成适配文件；`--output` 指定的训练目录必须不存在。

2×10 使用自己的数据目录、repo-id 和新输出目录。训练器读取 `collection_summary.json` 的 `target_layout: 2x10` 后，为 SmolVLA 配置 128-词元文本长度。仅将旧十块模型放到 2×10 环境是迁移测试，不等于模型经过该任务训练；旧检查点的 tokenizer 设置不会由评测自动升级。

检查点通常位于 `model/checkpoints/<step>/pretrained_model`，保存配置、处理器和权重。精确评测应指定该目录，勿把基座误作微调模型；传实验运行目录时解析器选择最新有效步骤，并可能优先该步骤的 EMA 权重。继续训练需恢复训练状态，具体命令见 [恢复训练](smolvla.md#5-完整训练与恢复)，不要仅重复新训练命令或覆盖目录。

### 3.2 ACT

```bash
a3-sim train-act \
  --root datasets/cookie_2x5_trial --repo-id local/cookie-2x5-trial \
  --output outputs/cookie_act_smoke --steps 20 --batch-size 1 \
  --num-workers 0 --save-freq 20 --device cuda
```

完整训练增大步数并换新输出目录。ACT 检查点用 `--policy act:<路径>` 评测，数据目录必须匹配；ACT 不读取任务文本。骨干预训练及时间集成设置见 [ACT 手册](act.md)。

## 4. 统一仿真评测

统一入口是 `examples/benchmark_cookie_batch.py`，或模块 `python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli`。支持两个任务和专家、SmolVLA、ACT、`module:factory` 插件。

### 4.1 专家与模型数量评测

```bash
# 2×10 专家；换成 --task 2x5 即可评测十块专家。
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task 2x10 --policy expert --profile basic --source-column 1 \
  --first-grasp random --episodes 20 --seed-start 13000 \
  --max-steps 500 --workers 1 --output outputs/expert_2x10_basic_500.json
```

换成 `--task 2x5` 时删除 `--first-grasp`。下面以本机现有 40k 普通权重为例；其他机器必须替换检查点和训练数据：

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task 2x10 \
  --policy smolvla:outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model \
  --dataset-root datasets/a3_front_close_left_100 \
  --repo-id local/a3-front-close-left-100 --device cuda \
  --profile basic --source-column 1 --first-grasp random \
  --episodes 20 --seed-start 13000 --inference-seed 13000 \
  --max-steps 500 --n-action-steps 8 --num-steps 25 --workers 1 \
  --output outputs/smolvla_2x10_basic_500.json
```

改 `--profile medium|advanced` 切换难度，`--randomization-config` 可指定自己的 YAML。2×5 默认 `--prompt-style training` 使用原短指令 `transfer 10 cookies into target box`；若模型用新按列指令训练，应显式选 `--prompt-style task`。2×10 始终使用包含当前数量计划的原生长指令。

`--seed-start` 控制场景；`--inference-seed` 控制策略采样。统一 CLI 的单个 `--inference-seed` 会在每次重置重置为同一个值；网格脚本则逐回合设为该回合的场景随机种子，复现网格单条时两者都传对应随机种子。模型评测用单进程；专家支持 `--workers N`。

### 4.2 可视化与故障诊断

在桌面终端执行，沿用上面的模型/任务参数，增加 `--render` 并用 `--episodes 1 --workers 1`：

```bash
env -u MUJOCO_GL python examples/benchmark_cookie_batch.py \
  --task 2x10 --policy expert --profile basic --source-column 1 \
  --first-grasp 3 --episodes 1 --seed-start 13000 \
  --max-steps 500 --workers 1 --render
```

无桌面时保留 `MUJOCO_GL=egl` 并添加 `--diagnostic-dir outputs/<新的目录>`，会记录三路视频、动作和每步计数。可同时用 `--output` 保存报告。`raw_action`、`processed_action`、`applied_action` 分别用于区分模型输出、后处理和实际执行。

### 4.3 如何读成绩

主要成绩取 **回合结束时** `cookies_in_target`，不取中途最大值：

- 2×5：每次 0–10 块；均值 `mean_score`，总数 `total_score / (10N)`。
- 2×10：每次 0–20 块；均值 `mean_score`，总数 `total_score / (20N)`。

JSON 的 `summary` 和 `episodes` 保存汇总与逐次记录。完整任务 `success` 保留作诊断，不作为这组数量评测的主排名指标。计数采用任务环境的盒内、姿态和稳定等规则，肉眼看见接触目标盒不一定立即计入。

500 是控制步上限；20 Hz 下为最多 25 秒仿真时间，推理和渲染会使实际耗时更长。2×10 的四次抓取可能被预算截断，必须与同为 500 步的结果比较。更改步数、指令、来源列、权重、后处理或统计量后，使用新的结果文件，避免混合协议。

## 5. 六项 SmolVLA 网格测试

脚本：`src/a3_dual_arm_sim/workflows/evaluate_smolvla_grid_cli.py`。与单次评测共用 `make_cookie_benchmark(...).evaluate(...)`。

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.evaluate_smolvla_grid_cli \
  --checkpoint outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model \
  --dataset-root datasets/a3_front_close_left_100 --repo-id local/a3-front-close-left-100 \
  --max-steps 500 --episodes 20 --seed-start 13000 \
  --n-action-steps 8 --num-steps 25 --device cuda \
  --output outputs/smolvla_grid_500
```

默认六项为 `2x5/2x10 × basic/medium/advanced`；每项 20 次，总共 120 次。按随机种子一轮轮执行六项，固定来源第 1 列；2×10 首夹数量读取 `configs/randomization_2x10.yaml`。2×5 固定训练短指令，2×10 使用原生任务指令。当前脚本不自动切换不同任务的检查点；如果只在 2×5 训练，2×10 成绩应标为迁移测试。

首次运行输出目录必须不存在。先加 `--dry-run` 可打印配置、不加载策略；中断后用**完全相同的命令加 `--resume`**，跳过已写入的完整回合。Ctrl+C/SIGTERM 会标记停止；正在执行但未写完的回合不计入成绩，续跑会重做。配置、检查点、脚本或共享执行代码哈希变化会拒绝续跑，应换新目录。

| 文件 | 用途 |
| --- | --- |
| `protocol.json` | 实验参数、权重/数据统计/代码哈希及处理器配置 |
| `episodes.jsonl` | 每个完成回合的最终计数、场景与运行信息 |
| `summary.md` / `summary.json` | 六项均值、累计放入量、分布、进度与状态 |
| `current.json` | 最近开始的任务/档位/随机种子，不代表已完成 |
| `pid` | 本次进程 ID；需结合进程状态确认是否还在运行 |
| `error.json` | 若出现运行异常，保存错误；异常不当作零分加入统计 |

同一输出目录不要同时启动两个进程。不要把 `completed=0` 的 `pending` 当作零分；必须检查 `completed/planned`。普通权重和 EMA 权重分别跑，使用同一数据统计、随机种子和步数。按需重定向终端输出到独立 `.log` 文件；脚本不会自动生成训练视频。

## 6. 加入自己的策略

创建仓库根目录下的 `my_policy.py`，让 Python 可以导入。下面是可以运行的保持初始关节姿态模板，替换 `act()` 中的预测部分即可：

```python
import numpy as np

class MyPolicy:
    action_mode = "joint_position"

    def __init__(self):
        self.target = None

    def reset(self, context):
        self.target = None
        # 在这里清空模型动作队列/循环状态；context 提供 seed、task、action_mode。

    def act(self, observation, task=""):
        if self.target is None:
            self.target = np.asarray(observation["observation.state"], dtype=np.float64).copy()
        return self.target.copy()  # 用你的模型预测替换；返回有限的 16D 绝对关节目标。

    def close(self):
        pass


def make_policy():
    return MyPolicy()
```

工厂必须零参数；配置和检查点可以由构造器读取配置文件/环境变量。环境契约：

- 三相机输入为 `H×W×3 uint8 RGB`，键为 `observation.images.front/left_wrist/right_wrist`；状态为 16D。
- `joint_position` 输出为 `[L1..L7,L_gripper,R1..R7,R_gripper]`，关节 rad，夹爪 0 关闭、1 张开。保持右臂应复制它的重置状态，不补零。
- `cartesian_delta` 输出是 14D，两臂各 6 个归一化位姿增量和一个夹爪指令，范围 `[-1,1]`，详见 [数据契约](contracts.md)。
- 评测可调用可选的 `bind_env(env)`。规则专家可以读取仿真真值；若模型读取物体位置等真值，应明确标记，不再作为纯视觉策略比较。通用执行器不保证同样的 `bind_env` 调用，通用插件优先只依赖 observation。
- 自定义 8D 模型需要自行将输出补成环境要求的 16D；内置 SmolVLA/ACT 已做此适配。模型归一化/反归一化只做一次。

验证模板和你的插件：

```bash
# 桌面通用 执行器，或去掉 --render 用 EGL 无头运行。
env -u MUJOCO_GL a3-sim run --scene cookie_transfer \
  --config configs/cookie_batch.yaml --policy my_policy:make_policy --steps 20 --render

# 统一 评测，适用于两种任务；自定义工厂负责自己的推理配置。
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task 2x10 --policy my_policy:make_policy --profile basic \
  --source-column 1 --episodes 1 --seed-start 13000 \
  --max-steps 500 --workers 1 --output outputs/my_policy_2x10.json
```

不要给 `module:factory` 添加仅适用于内置模型适配器的 `--device`、`--dataset-root`、`--n-action-steps` 等覆盖项；在工厂内部设置。先确认重置清空历史、输出形状正确和相机键匹配，再测完整预算。对策略做修改后，用相同场景/推理随机种子做配对对照。

## 7. 场景、遥操作与回放

安装 wheel 的用户可在自己的工作目录运行 `a3-sim` 或 `python -m a3_dual_arm_sim.<模块>`；`examples/` 是源码仓库的便利入口。

### 7.1 场景配置


| 配置 | 用途 |
| --- | --- |
| `configs/default.yaml` | 基础场景与通用配置 |
| `configs/cookie_batch.yaml` | 单盒 5+5 批次装盒 |
| `configs/cookie_2x10.yaml` | 2×10 四次抓取装盒，共二十个 |
| `configs/randomization.yaml` | 2×5 采集的随机化强度与选列 |
| `configs/randomization_2x10.yaml` | 2×10 采集的随机化强度、选列与首夹数量 |
| `configs/cookie_same_column.yaml` | 同列批次专家场景 |
| `configs/cookie_cooperative.yaml` | 双臂协作实验场景 |
| `configs/packed_elastic.yaml` | 接触/摩擦实验配置 |

通过 `--config` 指定场景 YAML，`--randomization-config` 指定采集 YAML；省略时入口使用其内置默认。
装盒场景控制 20 Hz、物理 1000 Hz，源盒 80 块；2×5 目标盒容量十块，2×10 容量二十块。
具体几何和初始位置由配置定义。配置中的参数没有真实硬件标定保证。

```bash
a3-sim inspect --scene cookie_transfer
MUJOCO_GL=egl a3-sim smoke --scene cookie_transfer --steps 20
```

### 7.2 相机与窗口操作

窗口中用鼠标拖动调整视角、滚轮缩放。专家演示和评测不会自动录制训练数据；阶段截图只用于诊断。


```bash
MUJOCO_GL=egl python examples/preview_cameras.py --help
MUJOCO_GL=egl python examples/preview_cameras.py
env -u MUJOCO_GL a3-sim teleop --scene cookie_transfer --no-camera-render
```

遥操作通过独立控制面板接收键盘焦点，MuJoCo 窗口用于观察。`w/s`、`a/d`、`r/f` 控制位置，`i/k`、`j/l`、`u/o` 控制旋转；夹爪、左右臂选择、录制与停止可通过面板按钮操作。按住移动，松开停止。需要训练图像时启用相机渲染；`--no-camera-render` 不可用于录制带相机的数据。

### 7.3 人工录制与数据回放

```bash
# 录制目录必须是新目录；录制图像时不能关闭相机渲染。
env -u MUJOCO_GL a3-sim teleop --scene cookie_transfer \
  --record datasets/teleop16 --repo-id local/a3-teleop16

# 回放本指南第 2 节采集的数据。
MUJOCO_GL=egl a3-sim replay --root datasets/cookie_2x5_trial \
  --repo-id local/cookie-2x5-trial --episode 0
```

回放用于检查动作、录制时序和场景；不同物理配置下的接触结果不保证完全一致。人工数据应单独审计，不能自动视为通过专家采集筛选的成功演示。

### 7.4 常见运行问题


- 无窗口：检查显示服务、驱动和桌面权限；窗口命令取消 `MUJOCO_GL=egl`。
- 离屏相机失败：检查 EGL 支持；可按机器条件使用 `MUJOCO_GL=osmesa` 并安装相应系统库。
- 录制依赖缺失：安装 `.[train]`；遥操作控制面板需要 Python Tk 支持，Ubuntu 可安装 `python3-tk`。
- 命令返回任务失败：先检查报告的失败原因、最大步数和布局；不能把启动正常标为装盒成功。
- 数据或输出目录冲突：使用新目录，保留旧产物，不覆盖已完成实验运行。
