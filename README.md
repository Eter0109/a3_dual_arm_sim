# A3 双臂 MuJoCo 仿真

A3 机器人双臂仿真项目，提供规则专家、键盘遥操作、可替换策略接口、LeRobot v3 数据录制，以及 SmolVLA 微调、ACT 训练和闭环评估。机器人每臂 7 个关节和一个夹爪控制量；单盒装盒任务将源盒中的 10 块饼干转移至目标盒。

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
- [动作与数据契约](docs/contracts.md)：维度、时序、统计量与成功判定。
- [SmolVLA 训练与测试全流程](docs/smolvla.md)：下载、审计、训练、部署、评估和故障排查。
- [ACT 训练与评估](docs/act.md)：安装、短训练、部署、离线误差与闭环评估。
- [入口与实验管理](docs/experiments.md)：正式入口、实验工具和结果追踪。
- [发布检查与限制](docs/release_checklist.md)：验证记录与待解决的发布阻碍。

## 资源与许可

源码包和 wheel 附带仿真配置、URDF 与必要网格，不需要其他项目目录。
数据、下载模型与输出分别放在 `datasets/`、`models/smolvla_base/`、`outputs/`，这些目录中的大型产物不提交 Git。
机器人资源来源与第三方声明见 [PROVENANCE](assets/a3/PROVENANCE.md)。源码及 A3 原始资源的再分发授权尚需维护者确认；当前整理不代表已解决公开发布许可。
