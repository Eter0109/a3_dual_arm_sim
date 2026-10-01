# A3 双臂 MuJoCo 仿真

A3 机器人双臂仿真项目，提供规则专家、键盘遥操作、可替换策略接口、LeRobot v3 数据录制，以及 SmolVLA 微调、ACT 训练和闭环评估。机器人每臂 7 个关节和一个夹爪控制量。

| 装盒任务 | 抓取计划 | 场景配置 | 采集配置 | 采集入口 |
| --- | --- | --- | --- | --- |
| 2×5，搬运 10 个 | `[5,5]` | `configs/cookie_batch.yaml` | `configs/randomization.yaml` | `examples/collect_cookie_benchmark.py` |
| 2×10，搬运 20 个 | `[n,10-n,n,10-n]` | `configs/cookie_2x10.yaml` | `configs/randomization_2x10.yaml` | `examples/collect_cookie_2x10.py` |

规则专家读取仿真位置与接触真值；SmolVLA 使用相机、关节状态和任务文本；ACT 使用相机和关节状态。仿真几何、动力学、相机及夹爪参数是功能性默认值，未完成真实硬件标定。本项目不承诺模型的任务成功率或真实机器人部署效果。

## 安装与快速运行

推荐 Linux、Python 3.12；本次实际验证环境见 [发布检查](docs/release_checklist.md)。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
a3-sim --help
a3-sim inspect
MUJOCO_GL=egl a3-sim smoke --steps 20
```

图形窗口需要桌面显示服务；无头相机渲染使用 EGL 和支持该后端的驱动：

```bash
env -u MUJOCO_GL python examples/run_cookie_batch.py --expert same_column --render
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy same_column --episodes 1 --seed-start 1000 --workers 1 \
  --max-steps 6000 --output outputs/expert/result.json
```

训练和数据功能按需安装 `python -m pip install -e '.[train]'`。默认基座为
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)，数据为
[`Eter0109/a3-front-close-left-100`](https://huggingface.co/datasets/Eter0109/a3-front-close-left-100)。
仓库不附带数据、模型权重或训练结果；首次使用需要下载。完整命令见训练手册。

## 文档导航

- [项目架构](docs/architecture.md)：模块、数据流、资源打包与策略扩展。
- [操作手册](docs/operations.md)：场景、专家、遥操作、相机、采集和回放。
- [2×5 分级随机化与专家采集](docs/randomization.md)：两批各五个，配置强度、来源列和外观，采集与续采。
- [2×10 专家演示与数据采集](docs/cookie_2x10.md)：配置首夹数量，运行四次抓取、图形演示与采集。
- [动作与数据契约](docs/contracts.md)：维度、时序、统计量与成功判定。
- [SmolVLA 训练与测试全流程](docs/smolvla.md)：下载、审计、训练、部署、评估和故障排查。
- [ACT 训练与评估](docs/act.md)：安装、短训练、部署、离线误差与闭环评估。
- [入口与实验管理](docs/experiments.md)：正式入口、实验工具和结果追踪。
- [发布检查与限制](docs/release_checklist.md)：验证记录与待解决的发布阻碍。

## 2×5 数据采集

在仓库根目录、已安装训练与数据依赖的环境中执行：

```bash
MUJOCO_GL=egl python examples/collect_cookie_benchmark.py \
  --randomization-config configs/randomization.yaml \
  --episodes 5 --max-attempts 15 \
  --root datasets/cookie_advanced_random_trial \
  --repo-id local/cookie-advanced-random-trial
```

先在 `configs/randomization.yaml` 中设置 `profile: advanced` 和
`source_column: random`。`profile` 选择 basic/medium/advanced，`source_column`
独立选择 1/2/3/4/random；所有幅度与颜色也在该文件中调整。
旧的 `--profile` 和 `--source-column` 命令参数已移除。基础档保持原有小范围布局变化，
中级增加轻微相机和光照变化，高级进一步增大变化并从四种饼干颜色中每轮选择一种。
三档共用专家主体，并针对不同列调整控制策略，仍采集单盒、两批各五块的整任务
LeRobot v3 数据，不使用 Planner/Verifier，也不切分 skill。

`--episodes` 是目标成功保存条数，`--max-attempts` 是包括失败在内的总尝试上限。
随机列均衡的是尝试分配，不保证最终成功数据各列等量。每组设置使用独立数据目录；
不要直接续采旧版缺少随机化元数据的数据集。详细参数、恢复采集和评估方式见
[随机化说明](docs/randomization.md)。上述 EGL 写法适用于 Linux 无头服务器；
Windows 使用 PowerShell 时省略 `MUJOCO_GL=egl` 前缀，并将多行命令合成一行。

## 2×10 专家演示与数据采集

在 `configs/randomization_2x10.yaml` 设置：

```yaml
profile: basic        # basic / medium / advanced
source_column: 1      # 1 / 2 / 3 / 4 / random
first_grasp: random   # 0–9 / random
```

三个选项分别控制环境扰动强度、来源列和首夹数量。`first_grasp: 3` 执行
`[3,7,3,7]`，前两次填满目标第 1 列，后两次填满第 2 列；`0` 跳过第 1、3 次。
数量 `random` 每个 episode 抽一次，首夹与第三夹共用该数量。
来源列 `random` 按原有的四列均衡规则抽样，与数量和强度独立。

在仓库根目录、已激活的 Python 环境中执行：

```bash
# 查看计划与实际任务指令。
python examples/collect_cookie_2x10.py --dry-run

# 在桌面环境中观看专家抓取。
env -u MUJOCO_GL python examples/run_cookie_2x10.py --render

# 采集完整回合。
MUJOCO_GL=egl python examples/collect_cookie_2x10.py \
  --randomization-config configs/randomization_2x10.yaml \
  --episodes 5 --max-attempts 15 \
  --root datasets/cookie_2x10_trial --repo-id local/cookie-2x10-trial
```

默认读取 2×10 采集配置。复制配置后可通过 `--randomization-config` 指定实验文件。
续采使用相同配置、数据目录、repo-id 和 seed 起点，加 `--resume`。
成功要求目标两列各十个、共二十个，源盒保留六十个。详细用法见
[2×10 使用说明](docs/cookie_2x10.md)。

## 资源与许可

源码包和 wheel 附带仿真配置、URDF 与必要网格，不需要其他项目目录。
数据、下载模型与输出分别放在 `datasets/`、`models/<模型名>/`、`outputs/`；
评估截图和诊断产物放在 `artifacts/`，这些产物不提交 Git。
机器人资源来源与第三方声明见 [PROVENANCE](assets/a3/PROVENANCE.md)。源码及 A3 原始资源的再分发授权尚需维护者确认；当前整理不代表已解决公开发布许可。
