# A3 双臂 MuJoCo 仿真
AI305-Project

## 链接发布页

SmolVLA模型数据(后续更新)：[`Eter0109/a3-cookie-multi-left8-40k-v1`](https://huggingface.co/Eter0109/a3-cookie-multi-left8-40k-v1)

SmolVLA基座：[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)

ACT模型：[`Eter0109/a3-cookie-2x5-act-20k`](https://huggingface.co/Eter0109/a3-cookie-2x5-act-20k)

π0.5双场景模型：[`yunsx114/a3-dual-arm-cookie-pi05-2x5-2x10`](https://www.modelscope.cn/models/yunsx114/a3-dual-arm-cookie-pi05-2x5-2x10)

π0.5模型：[`yunsx114/a3-dual-arm-cookie-pi05-aligned`](https://www.modelscope.cn/models/yunsx114/a3-dual-arm-cookie-pi05-aligned)

多任务示范数据集：[`Eter0109/a3-cookie-multi-tasks`](https://huggingface.co/datasets/Eter0109/a3-cookie-multi-tasks)

单任务数据集：[`Eter0109/a3-front-close-left-100`](https://huggingface.co/datasets/Eter0109/a3-front-close-left-100)

---


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

[项目文档中心](docs/README.md) 按课程评测、数据与训练、开发参考组织全部使用手册。

- [课程 Benchmark](docs/course_benchmark.md)：任务、统一指令、评分和完整评测。
- [18 个课程场景图鉴](docs/course_visual_guide/README.md)：场景截图、PDF 和逐项命令。
- [采集、训练、评测与自定义策略](docs/workflow_guide.md)：从数据准备到闭环评估。
- [开发与接口参考](docs/README.md#开发与接口)：架构、动作数据契约和入口索引。

## 常用训练与测试

```bash
# 自动生成本地数据标识，默认输出 outputs/cookie_2x5_trial_smolvla
python -m a3_dual_arm_sim.cli train-smolvla --root datasets/cookie_2x5_trial

# 用旧短提示词做 2×5 模型到 2×10 的 legacy 迁移测试；自动选择最新检查点
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy smolvla:outputs/cookie_2x5_trial_smolvla --task 2x10 --protocol legacy --prompt-style short
```

训练默认值与自定义 JSON 配置、旧检查点补齐方法见 [简化入口说明](docs/smolvla.md#简化入口与旧检查点)。

## 统一评测与网格测试

课程版包含 14 种基础任务和 4 种 Bonus 任务，统一提示词为 `Fill the target box.`。标准化评估将两个标准布局各按三个档位运行 10 次，其余 12 个基础扩展任务和 4 个 Bonus 任务各运行 5 次，共 140 回合。通过 `examples/benchmark_cookie_batch.py --task` 运行单项，或使用课程任务组入口执行全部任务；新 2×10 场景、目标盒评分、随机化和专家使用方法见 [课程 Benchmark 说明](docs/course_benchmark.md)。两个旧来源配额任务保留原有提示词和评分，见 [历史扩展场景](docs/cookie_benchmark_scenarios.md)。

直接启动完整课程任务组（默认规则专家，结果写入指定的 JSON）：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.course_benchmark_cli \
  --policy expert --seed-start 0 --output outputs/course_benchmark/expert_seed0.json
```

下面分别是 500 步单项调试和 legacy 网格示例：

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --protocol course-v1 --task 2x10 --policy expert --profile basic --source-column 1 \
  --episodes 1 --seed-start 13000 --max-steps 500 --workers 1

# 替换成已有完整检查点；输出目录需保持唯一。
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.evaluate_smolvla_grid_cli \
  --checkpoint outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model \
  --max-steps 500 --max-steps-2x10 500 --episodes 20 --seed-start 13000 \
  --protocol legacy --prompt-style-2x5 training --prompt-style-2x10 short \
  --output outputs/smolvla_grid_short_500
```

上例网格按两个任务×三个档位执行，共 120 次，默认来源第 1 列。上例显式指定 2×5 使用训练原指令、2×10 使用 `transfer 20 cookies into target box`；短指令数量 baseline 不验证随机四次抓取计划。未指定协议时，显式传入旧提示词会选择 legacy；同时省略协议和旧提示词参数时使用课程统一提示词。网格可加 `--profiles basic` 只测 basic。
结果保存到 `summary.md`、`summary.json` 和 `episodes.jsonl`；原命令加 `--resume` 可续跑。
上面两个示例均使用 500 步预算，不保证专家或模型完成四次抓取；完整课程任务组默认分别为 6000 / 12000 步。完整参数、可视化、录制与自定义策略示例见 [完整流程指南](docs/workflow_guide.md)。

## 资源与许可

源码包和 wheel 附带仿真配置、URDF 与必要网格，不需要其他项目目录。
数据、下载模型与输出分别放在 `datasets/`、`models/<模型名>/`、`outputs/`；
评估截图和诊断产物放在 `artifacts/`，这些产物不提交 Git。
机器人资源来源与第三方声明见 [资源来源声明](assets/a3/PROVENANCE.md)。源码与项目文档采用 [MIT 许可证](LICENSE)。维护者已确认 A3 资源的公开再分发授权；第三方资源的适用范围见资源来源声明。
