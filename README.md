# A3 双臂 MuJoCo 仿真

A3 机器人双臂仿真项目，提供规则专家、键盘遥操作、可替换策略接口、LeRobot v3 数据录制，以及 SmolVLA 微调、ACT 训练和闭环评估。机器人每臂 7 个关节和一个夹爪控制量。

| 装盒任务 | 抓取计划 | 场景配置 | 采集配置 | 采集入口 |
| --- | --- | --- | --- | --- |
| 2×5，搬运 10 个 | `[5,5]` | `configs/cookie_batch.yaml` | `configs/randomization.yaml` | `examples/collect_cookie_benchmark.py` |
| 2×10，搬运 20 个 | `[n,10-n,n,10-n]` | `configs/cookie_2x10.yaml` | `configs/randomization_2x10.yaml` | `examples/collect_cookie_2x10.py` |

规则专家读取仿真位置与接触真值；SmolVLA 使用相机、关节状态和任务文本；ACT 使用相机和关节状态。仿真几何、动力学、相机及夹爪参数是功能性默认值，未完成真实硬件标定。本项目不承诺模型的任务成功率或真实机器人部署效果。

## 安装与快速运行

推荐 Linux、Python 3.12；环境要求与依赖兼容问题见 [已知限制](docs/limitations.md)。在仓库根目录执行：

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
  --task 2x5 --policy expert --profile basic --episodes 1 --seed-start 1000 --workers 1 \
  --max-steps 500 --output outputs/expert/result.json
```

训练和数据功能按需安装 `python -m pip install -e '.[train]'`。默认基座为
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)，数据为
[`Eter0109/a3-front-close-left-100`](https://huggingface.co/datasets/Eter0109/a3-front-close-left-100)。
仓库不附带数据、模型权重或训练结果；首次使用需要下载。完整命令见训练手册。

## 学生使用建议

先在独立虚拟环境安装，再按快速运行检查仿真；随后沿完整流程指南依次完成少量采集、数据审计、短训练和评测。训练依赖固定为公开版 `lerobot==0.5.1`，不要混用其他 LeRobot 开发仓库。2×5 模型用于 2×10 时属于迁移实验，不应直接作为已经解决该任务的演示模型；使用边界见 [已知限制](docs/limitations.md)。

## 文档导航

从采集到部署优先阅读 [完整流程指南](docs/workflow_guide.md)，覆盖已有策略采集、SmolVLA/ACT 训练、统一评测、六项网格测试及自定义策略。使用范围与兼容性边界见 [已知限制](docs/limitations.md)。

- [项目架构](docs/architecture.md)：模块、数据流、资源打包与策略扩展。
- [2×5 分级随机化与专家采集](docs/randomization.md)：两批各五个，配置强度、来源列和外观，采集与续采。
- [2×10 专家演示与数据采集](docs/cookie_2x10.md)：配置首夹数量，运行四次抓取、图形演示与采集。
- [动作与数据契约](docs/contracts.md)：维度、时序、统计量与成功判定。
- [SmolVLA 训练与测试全流程](docs/smolvla.md)：下载、审计、训练、部署、评估和故障排查。
- [ACT 训练与评估](docs/act.md)：安装、短训练、部署、离线误差与闭环评估。
- [入口与实验管理](docs/experiments.md)：正式入口、实验工具和结果追踪。
- [使用范围与已知限制](docs/limitations.md)：环境兼容性、任务边界及数据注意事项。

## 常用训练与测试

```bash
# 自动生成本地数据标识，默认输出 outputs/cookie_2x5_trial_smolvla
python -m a3_dual_arm_sim.cli train-smolvla --root datasets/cookie_2x5_trial

# 测试上面训练的输出；自动选择最新检查点
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy smolvla:outputs/cookie_2x5_trial_smolvla --task 2x10 --prompt-style short
```

训练默认值与自定义 JSON 配置、旧检查点补齐方法见 [简化入口说明](docs/smolvla.md#简化入口与旧检查点)。

## 统一评测与网格测试

两个任务统一使用 `examples/benchmark_cookie_batch.py`，通过 `--task 2x5|2x10` 选任务，`--policy expert|smolvla:<检查点>|act:<检查点>|module:factory` 选策略。按回合结束时的目标盒饼干数量计分，2×5 满分 10、2×10 满分 20。

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task 2x10 --policy expert --profile basic --source-column 1 \
  --episodes 1 --seed-start 13000 --max-steps 500 --workers 1

# 替换成已有完整检查点；输出目录需保持唯一。
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.evaluate_smolvla_grid_cli \
  --checkpoint outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model \
  --max-steps 500 --episodes 20 --seed-start 13000 \
  --prompt-style-2x5 training --prompt-style-2x10 short \
  --output outputs/smolvla_grid_short_500
```

网格按两个任务×三个档位执行，共 120 次，默认来源第 1 列。上例显式指定 2×5 使用训练原指令、2×10 使用 `transfer 20 cookies into target box`；短指令数量 baseline 不验证随机四次抓取计划。省略指令参数时，2×10 仍使用原生长指令。单任务入口可用 `--prompt-style short`，网格可加 `--profiles basic` 只测 basic。
结果保存到 `summary.md`、`summary.json` 和 `episodes.jsonl`；原命令加 `--resume` 可续跑。
500 步为控制预算，不保证专家或模型完成四次抓取。完整参数、可视化、录制与自定义策略示例见 [完整流程指南](docs/workflow_guide.md)。

## 资源与许可

源码包和 wheel 附带仿真配置、URDF 与必要网格，不需要其他项目目录。
数据、下载模型与输出分别放在 `datasets/`、`models/<模型名>/`、`outputs/`；
评估截图和诊断产物放在 `artifacts/`，这些产物不提交 Git。
机器人资源来源与第三方声明见 [资源来源声明](assets/a3/PROVENANCE.md)。源码与项目文档采用 [MIT 许可证](LICENSE)。维护者已确认 A3 资源的公开再分发授权；第三方资源的适用范围见资源来源声明。

## Repeatable batch skills

Run `python -m a3_batch_skills --task 2x5 --output outputs/batch_2x5` for the goal-bound contact skill. Each batch remeasures the requested five cookies, executes the contact grasp, places and verifies them, then localizes the next group without resetting physics. The module also provides paired pi0.5 queue-reset diagnostics, short RGB/left8 skill recording, LeRobot v3 export and padding audits, and an optional continuation-training wrapper. Default localization and projected goal cues explicitly use simulator truth. These geometric results are not evidence of a trained pure-vision VLA improvement. See [batch skill usage and validation](docs/batch_skills.md).
