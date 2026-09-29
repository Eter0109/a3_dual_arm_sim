# A3 双臂饼干搬运

[English](README.md) · [架构说明](docs/architecture.md) · [数据详解与排错](docs/data-workflow.md)

本项目基于 MuJoCo，提供 A3 机器人仿真、专家示范采集、LeRobot 动作模型训练，
以及参考 Agentic Robot 思路实现的 Planner–Executor–Verifier 闭环。
主要学习任务是：**一个目标盒、十块饼干，从同一来源列分两次各抓五块并放入目标盒**。
来源列可以固定，也可以在采集时随机选择。

这是研究用途的实现。专家演示成功、生成训练权重、验证器离线判断正确、学习策略完成
整盒任务，分别需要验证。下载通用基座后，还需要 A3 数据适配与训练；不能仅凭训练
loss 下降就判断机械臂已经学会任务。

## 1. 环境、专家、Agent 和动作模型怎样配合

环境负责机器人、盒子、饼干、物理接触、相机和成功条件；控制器负责发出动作，
MuJoCo 模拟动作产生的物理结果。换一个场景配置，不等于已经为新场景写好了搬运策略。

| 部分 | 输入与职责 | 主要代码 |
| --- | --- | --- |
| 专家 Expert | 读取仿真物体位置、接触和机器人状态，用规则控制物理抓放 | `experts/` |
| 规划器 Planner | 根据用户任务和场景生成受支持的技能请求；也提供固定顺序基线 | `agents/planning.py` |
| 执行器 Executor | 按当前技能调用专家，或调用训练好的 LeRobot 动作策略 | `agents/executors.py`、`policies/` |
| 验证器 Verifier | 根据子目标和连续 RGB 画面判断是否完成，未完成时再判断是否卡住 | `agents/verification.py` |
| 闭环控制器 Controller | 切换技能、清除旧动作队列、安排验证、重试和安全停止 | `agents/controller.py` |

专家是能够访问仿真真值的规则控制器，不是学习出来的 VLA。视觉规划和验证可以调用
本地视觉语言模型，也可以配置 HTTP 后端。动作模型是独立组件，可以单独替换。

```text
第一次 PICK_FIVE → 放入目标第 1 列 PLACE_FIVE
                → 第二次 PICK_FIVE → 放入目标第 2 列 PLACE_FIVE
```

技能类型只有 `PICK_FIVE` 和 `PLACE_FIVE`，来源列、批次和目标列作为参数。
动作模型收到的语言例如：

```text
Pick up five cookies from source column 1, batch 1.
Place the five held cookies into target box column 1.
```

四个来源列不需要分别建立十六个独立模型。当前技能库仍限定每次五块，暂不支持任意
指定“三块加七块”，也不能把单盒 Agent 的能力直接等同于通用双盒协作。

## 2. 目录与配置

```text
assets/                       必需的机器人几何与场景资源
configs/envs/                 场景、物理、相机与控制参数
configs/agents/               规划器、验证器与 Agent 闭环参数
src/a3_dual_arm_sim/
  __main__.py                 python -m a3_dual_arm_sim 的入口
  core/                       共享观测、动作、技能定义和资源路径
  envs/                       MuJoCo 环境、场景生成与相机
  experts/                    单块、跨列、同列抓放专家
  policies/                   策略接口及 SmolVLA / LeRobot 适配器
  agents/                     规划、执行、验证与调度
  data/                       数据录制、采集、标注与检查
  training/                   动作模型训练与验证器 LoRA 微调
  evaluation/                 专家、策略和视觉验证器评估
  teleop/                     遥控面板与键盘输入
  cli/                        按功能分组的命令行入口
tests/                        按相同功能划分的测试
docs/                         架构与工作流程说明
```

外层 `src/` 表示 Python 源码布局，内层 `a3_dual_arm_sim/` 是可导入的包名，
并不是复制了两份项目。生成文件分别放在 `datasets/`（数据）、`models/`（下载权重）、
`outputs/`（训练结果）、`artifacts/`（诊断与评估）、`.runtime/`（缓存与本机任务）。
这些目录中的新文件不进入 Git；机器人运行必需的资源保存在 `assets/`。

| 配置 | 用途 |
| --- | --- |
| `configs/envs/default.yaml` | 共享仿真和相机默认设置 |
| `configs/envs/cookie_same_column.yaml` | 同列分两次搬五块 |
| `configs/envs/cookie_batch.yaml` | 批量搬运与采集基线 |
| `configs/envs/cookie_cooperative.yaml` | 实验性协作搬运 |
| `configs/envs/packed_elastic.yaml` | 另一种密集接触配置 |

## 3. 在新机器上安装

使用包含 `assets/` 的完整 Git 仓库，以下命令均从项目根目录运行。无需固定用户名、
磁盘、云平台或 Conda 安装位置。学习流程使用 Python 3.12 验证；GPU 训练主要在
Linux 与 CUDA 版 PyTorch 上验证。Windows 可运行桌面仿真，学习相关功能还取决于
第三方依赖对平台的支持。

```bash
git clone https://github.com/Eter0109/a3_dual_arm_sim.git
cd a3_dual_arm_sim
conda create -n a3sim python=3.12 pip -y
conda activate a3sim
python -m pip install --upgrade pip
python -m pip install -e .
```

如果模块化版本尚未位于默认分支，需要切换到包含本版本的分支或标签。
Conda 无法激活时，按终端类型运行 `conda init bash` 或 `conda init powershell`，
随后重新打开终端。桌面遥控面板还需要 Tk，可以在环境中运行 `conda install tk`。

在同一个环境内，按功能安装可选依赖：

| 功能 | 命令 |
| --- | --- |
| LeRobot v3 数据采集 | `python -m pip install -e ".[dataset]"` |
| SmolVLA 训练和推理 | `python -m pip install -e ".[train]"` |
| π₀.₅ 训练依赖 | `python -m pip install -e ".[train-pi]"` |
| 本地视觉 Agent 和验证器 LoRA | `python -m pip install -e ".[agent]"` |
| 测试与代码检查 | `python -m pip install -e ".[dev]"` |

需要 SmolVLA 和视觉 Agent 时，可合并安装 `".[train,agent,dev]"`。集成环境已检查
LeRobot 0.5.1、Transformers 5.3.0、PyTorch 2.10.0 的组合，但这不是完整依赖锁文件。
PyTorch 应选择与本机显卡和驱动匹配的版本，然后在当前环境内检查：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pip check
a3-sim inspect environment
a3-sim --help
```

`python -m a3_dual_arm_sim` 与 `a3-sim` 等价。找不到 `a3-sim` 时，先检查环境
激活和 editable 安装。旧版 `python examples/run_*.py` 入口已撤下；每个新子命令
都支持 `--help`。

下文多行命令使用 Bash 的 `\` 续行。在 PowerShell 中，请合并成一行，或换成
PowerShell 的反引号续行，不能直接把 Bash 续行符照搬进去。

## 4. 仿真显示、相机与遥控

Linux 无桌面机器在具备 EGL 驱动时使用：

```bash
export MUJOCO_GL=egl
a3-sim sim same-column --config configs/envs/cookie_same_column.yaml --max-steps 1000
a3-sim inspect cameras --config configs/envs/cookie_same_column.yaml --output artifacts/cameras
```

相机检查会生成三张 PNG 和 `all_cameras.png`。无界面渲染仍能输出相机图像，
只是不会弹出桌面窗口。Linux 桌面运行 `unset MUJOCO_GL`；Windows PowerShell
运行 `Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue`，然后可以执行：

```bash
a3-sim sim same-column --config configs/envs/cookie_same_column.yaml --render
a3-sim sim control --scene cookie_transfer --config configs/envs/cookie_same_column.yaml --no-camera-render
```

键盘焦点应放在控制面板，MuJoCo Viewer 负责显示。遥控默认不录数据，只有显式加
`--record` 才录制。`--no-camera-render` 用于交互调试，不能用于依赖 RGB 的 VLA
推理或数据采集。普通 SSH 终端没有图形显示服务；无桌面机器使用保存画面，或先配置
真正的远程桌面再加 `--render`。

统计专家在一组 seed 上的表现：

```bash
a3-sim eval experts --policy same_column --profile diverse --episodes 20 --workers 4 --output artifacts/expert_eval.json
```

## 5. 采集动作示范

| 命令 | 一次成功完整搬运保存成什么 |
| --- | --- |
| `data actions --episodes N` | 一条整任务 LeRobot episode |
| `data skills --rollouts N` | 四条带技能语言标签的 LeRobot episode |

技能采集时，专家连续完成整盒任务，保存时才切分四段，不会在技能边界重新初始化环境。
只有完整任务成功后才保存到技能训练集，失败尝试仅记日志。分数达到十也不一定成功，
还需要满足源盒剩余数量、释放、稳定和安全等条件。

固定来源为第 1 列，同时保留场景随机化，采集 100 条成功完整搬运：

```bash
a3-sim data skills \
  --config configs/envs/cookie_same_column.yaml \
  --root datasets/skills_col1 --repo-id local/a3-skills-col1 \
  --rollouts 100 --max-steps 1000 --workers 4 \
  --profile diverse --source-column first --seed-start 1000
```

这里的 100 指 **100 次成功整盒搬运，即 400 个技能 episode**，不是 100 次尝试，
也不是 100 张图片。`local/a3-skills-col1` 是数据集标识，不会触发上传；采集、训练、
推理时使用同一个标识。

如果要随机四列，另建数据集：

```bash
a3-sim data skills \
  --config configs/envs/cookie_same_column.yaml \
  --root datasets/skills_all_columns --repo-id local/a3-skills-all-columns \
  --rollouts 200 --max-steps 1000 --workers 4 \
  --profile diverse --source-column random --seed-start 3000
```

`first` 使用参考版第一列专家；数字 `1`–`4` 使用能够选择列的专家；`random`
在来源列之间采样。数字和随机列要求 `diverse`。固定列不会固定盒子位置，也不会
关闭这个 profile 的其他随机化。失败尝试会被丢弃，因此最终保存的各列数量可能不同，
训练前应同时检查各列的尝试成功率和成功示范数。

`diverse` 包含盒子位置、目标盒朝向、饼干位姿、相机位置/视场角、光照和颜色变化。
`baseline` 是较小的基准扰动，不等同于 Agent 的 `fixed`。具体数值见
[数据详解](docs/data-workflow.md)。

| 文件或目录 | 内容 |
| --- | --- |
| `meta/`、`data/` | LeRobot v3 元数据与帧数据；图像不一定另存为 MP4 |
| `a3_skill_segments.jsonl` | 父 seed、技能、列、批次、目标位置与帧范围 |
| `attempts.jsonl` | 每次完成的尝试，包括失败与诊断 |
| `collection_summary.json` | 采集配置和来源信息，不是实时进度计数 |
| `seed_reservations.jsonl` | 并行采集和恢复使用的 seed 记录 |

暂停时按一次中断，等正在运行的任务保存并关闭。恢复时保持参数一致，增加
`--resume`；`--rollouts` 仍填写最终总目标。连续多次专家失败时采集器会停止，
应先查看原因。修改场景或采集设置时，使用新的数据目录。

## 6. 训练带技能语言条件的 SmolVLA

先下载本地基座：

```bash
a3-sim setup models --model lerobot/smolvla_base --output models/smolvla_base --endpoint https://huggingface.co
```

该命令只下载权重，不安装 Python 依赖。网络需要时可以明确改成
`--endpoint https://hf-mirror.com`。首次加载 SmolVLA 还可能下载其配置中的视觉语言
骨干和 tokenizer；如果这些后续下载也要走同一端点，需要在终端设置 `HF_ENDPOINT`。
所有相关模型和 tokenizer 都已在本地后，才开启 `HF_HUB_OFFLINE=1`。

先检查数据并预览训练配置：

```bash
a3-sim train skills \
  --root datasets/skills_col1 --repo-id local/a3-skills-col1 \
  --base-model models/smolvla_base --policy-type smolvla \
  --output outputs/train/smolvla_col1 \
  --steps 20000 --save-freq 20000 --batch-size 4 --num-workers 4 \
  --device cuda --validation-fraction 0.2 --seed 0 --dry-run
```

把 `--dry-run` 换成 `--run` 才正式训练。两个参数都不写时，也是只检查并打印命令。
公开 CLI 中采集和训练是独立步骤，采集结束不会自动启动训练。

数据按父轨迹划分，同一次搬运切出的四段始终在同一侧。100 条父轨迹、留出比例 0.2
对应训练 80 条父轨迹 / 320 个技能 episode，留出 20 条 / 80 个 episode。
这里只预留数据，不会自动计算验证 loss 或运行机器人评估。归一化目前沿用完整数据集
元数据中的统计量，没有重新计算仅训练集统计量。

一个模型共同学习数据中的不同技能指令。混合训练 20,000 步不等于严格给每列分配
5,000 步。使用 `models/smolvla_base` 是从基座新建微调，不是从以前的 A3 权重续训。

```text
outputs/train/
  smolvla_col1_validation_manifest.json       父轨迹划分与训练来源
  .smolvla_col1_smolvla_source/                适配后的配置与基座权重链接
  smolvla_col1/checkpoints/
    020000/pretrained_model/                  最终策略、配置、预处理和后处理
    last                                     最新检查点指针
```

这里 `--save-freq 20000` 表示在最终步保存；需要中间检查点时可以减小数值。
隐藏的适配目录是运行依赖，不是多训练了一个模型，不要随手删除它或所引用的基座文件。
评估和复制时优先使用明确的数字检查点目录。

## 7. 实际运行学习策略并查看动作

先使用固定 Planner 和真值 Verifier，以检查动作模型本身。动作模型仍接收相机、
机器人状态和技能语言；`oracle` 只说明完成条件由谁判断，不会替模型生成抓放动作。

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

评估 seed 必须与实际采集到的父 seed 列表分离。想先测试更简单的固定布局，可把
`--profile diverse` 改成 `--profile fixed`，并单独标明该测试条件。具备图形桌面时
加 `--render`；无桌面运行不要加，但策略仍然能够读取相机图像。

`--verification-data` 保存诊断画面窗口，不会往动作训练集追加示范：

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

默认每个窗口五帧，`00`–`04` 是窗口内部的时间顺序，编号文件夹代表连续的窗口；
最初未填满的窗口可能不足五帧。相邻窗口会重叠，这些是采样观察，不是逐仿真帧视频。
每次运行换一个新画面目录。需要连贯预览时，参考[生成回放视频](docs/data-workflow.md#preview-a-saved-rollout)。

结果 JSON 中重点看 `success`、`completed_skills`、`cookies_in_target`、
`failure_reason` 和 `events`。任务失败时会先保存结果，再返回退出码 1；这与异常
崩溃、完全没生成结果不同。例如 `PLACE_FIVE batch=1 attempt=2` 表示第一批放置
进入第二次尝试，不是已经开始第二批搬运。

用相同配置跑多个未见 seed，逐次修改 seed 和输出名。只统计指标时去掉
`--verification-data`，减少存图开销。应统计整盒成功率、失败阶段、安全停止和步数，
不能只看一张图或一个 seed。[数据详解](docs/data-workflow.md)提供批量评估脚本示例。

## 8. 接入视觉 Planner 和 Verifier

```bash
a3-sim setup models --model Qwen/Qwen3.5-4B --output models/Qwen3.5-4B --endpoint https://huggingface.co
a3-sim agent run --config configs/envs/cookie_same_column.yaml \
  --agent-config configs/agents/agentic_qwen35.yaml \
  --planner vlm --verifier vlm --executor expert \
  --profile diverse --source-column 1 --seed 20000 \
  --output artifacts/evaluation/visual_agent.json
```

该配置使用公开基座，不是 A3 专用微调验证器，示例中的动作执行器也仍是仿真专家。
测试完整学习系统时，换成 `--executor lerobot`，并补上第 7 节的模型和数据参数。
`--task "..."` 可传入受支持技能范围内的用户任务，同时保持来源列参数与任务一致。

YAML 中 `planner` 和 `verifier` 分别配置后端。`kind: local_qwen` 加载本地模型；
`kind: http` 使用 `model`、`base_url` 和 `api_key_env` 调用兼容的外部多模态服务。
密钥放在指定的环境变量中，该后端会将任务文本和图像发送到服务端。
本地视觉路径不会在判断错误时自动改用仿真真值。

## 9. 采集验证器数据并进行 LoRA 微调

验证器数据包含子目标、按时间排列的 RGB 图片，以及独立计算的“已完成、正常进行、
卡住”标签，与 LeRobot 动作示范不同。训练先回答 `Yes`/`No`，对于未完成样本
再训练 `StillTrying`/`Stuck`。仿真真值只用于标签和审计，不进入视觉模型输入。

下面是第一列的小规模完整流程示例。正式扩大训练前，需要增加独立 seed 并检查类别覆盖：

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

十个 seed × 四种情况是 40 次尝试，每次可能生成多个窗口。故障名称描述的是干预方式，
不直接决定标签。如果准备数据时缺少某种状态，应在新的 raw 目录补采独立 seed，
再用 `--data-roots` 同时传入多个目录。即使来自不同目录，相同父 seed 的窗口也必须
保留在同一个划分中。

原始采集保存三路相机，当前验证器取 front 和左腕两路：每路五帧，相隔十个控制步，
在 20 Hz 下覆盖约两秒，共十张图。`--render-recorded-frames-only` 只减少无用的 RGB
渲染，物理步仍完整执行。hold 参数只在完成条件仍成立时补充稳定终态图像。

训练需要支持 BF16 的 CUDA GPU，冻结视觉部分，仅更新语言侧 LoRA 参数。
结果包括 `adapter/`、`checkpoints/`、`progress.jsonl`、`training_audit.json` 和
`training_summary.json`。`--init-adapter` 是权重热启动，不是完整优化器状态恢复；
需要保留其数据来源，防止先前训练数据与新的留出集重叠。

`configs/agents/agentic_qwen35_lora.yaml` 是本地微调权重配置模板。
把 `verifier.adapter_path` 指向自己的 adapter；本示例使用
`outputs/train/verifier/adapter`。图像尺寸、时间窗口设置须与训练一致。
在实际训练产出权重前，不能直接用该配置运行。

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

prepared 的 `samples.jsonl` 含训练和留出两侧，离线评分时须显式选择留出父 seed。
`--all`、`--limit`、`--per-class-limit` 是互斥的抽样方式。离线窗口分数和整盒成功率
不是同一个指标；以专家执行的在线测试，也不能算作学习型 VLA 的成功率。

## 10. 更换动作模型与迁移机器

LeRobot 执行层支持 `smolvla` 和 `pi05`。π₀.₅ 已有训练入口，A3 任务效果与硬件需求
仍需单独验证。安装对应依赖并准备兼容的本地基座后，可使用
`train skills --policy-type pi05 --dry-run`，同时补齐数据、基座和输出目录参数，
先检查生成的训练配置。仅修改模型名称不能让未适配权重直接工作。

当前训练接口使用三路 RGB 和 16 维状态；动作是 16 维绝对目标，顺序为“左臂七关节、
左夹爪、右臂七关节、右夹爪”。右臂静止仍占这些维度，右腕相机也可能拍到运动物体。
去掉相机或缩减动作维度，需要同时修改数据、模型和推理接口。环境支持可选的米制
深度诊断，但当前 SmolVLA 与验证器不会自动把 depth 当作输入。

换机器部署时，复制完整 `pretrained_model/`（含预处理和后处理），以及所需的骨干、
tokenizer 和匹配数据集的 `meta/`。仅复制 `model.safetensors` 不够。
检查保存的模型和 tokenizer 路径；绝对路径需要在新机器存在，或一致地更新。
Git 仓库不包含数据集和训练好的权重。

## 11. 开发检查与上传

```bash
python -m pytest -q
python -m ruff check src tests
git diff --check
```

默认资源路径从仓库定位。非标准部署可用 `A3_PROJECT_ROOT` 指向另一个完整资源仓库；
文档中的相对路径命令仍请在项目根目录执行。提交源码、配置、测试、文档和必需资源；
数据、权重和生成诊断单独存放，不将凭据加入 Git。

向自己的开发分支上传前：

```bash
git branch --show-current
git status --short
# 添加本次需要提交的源码、配置和文档后，检查暂存区。
git diff --cached --stat
git diff --cached --check
git commit -m "Document and organize A3 simulation and learning workflows"
git push -u origin HEAD
```

`HEAD` 推送当前检出的分支，先确认分支名。采集和训练命令不会上传 GitHub 或自动
创建合并请求。采集速度、无窗口、依赖和模型执行失败的排查见
[数据详解与排错](docs/data-workflow.md)。
