# A3 Dual-Arm MuJoCo Sandbox

A3 双臂仿真、专家与遥操作、LeRobot v3 采集和 SmolVLA 训练/部署。
源码按职责分包，包路径即导入路径，不再保留顶层兼容别名。

## 快速开始

在项目根目录执行：

```bash
pip install -e '.[dev]'
a3-sim --help
env -u MUJOCO_GL python examples/run_cookie_batch.py --expert same_column --render
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py --policy same_column --episodes 1 --seed-start 1000
```

训练和数据功能按需安装 `pip install -e '.[train]'`。窗口模式取消 `MUJOCO_GL`，无头使用 `MUJOCO_GL=egl`。
专家读取仿真真值，不是视觉模型；模型缺少真实电机、摩擦、零位、夹爪及相机标定，不能视为已校准数字孪生。

## SmolVLA 全流程（采集 → 训练 → benchmark）

```bash
# 1. 采集 16 维演示（遥操作或任意策略，走同一个 recorder）
MUJOCO_GL=egl a3-sim teleop --scene cookie_transfer \
  --record datasets/<new16> --repo-id local/<new16>
MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --policy a3_dual_arm_sim.policies.base:make_hold_policy \
  --record datasets/<new16> --steps 3

# 2. 转成左臂 8 维契约（目标目录必须不存在）
python examples/prepare_left_arm_dataset.py --source datasets/<new16> --output datasets/<new8>

# 3. 训练
python -m a3_dual_arm_sim.cli train-smolvla \
  --root datasets/a3_front_close_left_100 --repo-id local/a3-front-close-left-100 \
  --output outputs/<run>/model --steps 20000 --batch-size 64 --lr 0.00005 --device cuda

# 4. benchmark（部署时必须用 --dataset-root 传入对应训练数据）
MUJOCO_GL=egl HF_HUB_OFFLINE=1 python examples/benchmark_cookie_batch.py \
  --policy smolvla:outputs/smolvla_front_close_left_20k/model/checkpoints/020000/pretrained_model \
  --dataset-root datasets/a3_front_close_left_100 \
  --episodes 20 --seed-start 1000 --n-action-steps 8 --workers 1 \
  --output outputs/smolvla_front_close_left_20k/dev20.json
```

本仓库当前只保留这一条流程所需的产物：`datasets/a3_front_close_left_100`（100 episodes，8 维左臂）
与 `outputs/smolvla_front_close_left_20k`（002000–020000 普通/EMA 权重）。8 维 checkpoint 部署时补上
reset 时固定的右臂 8 维目标，形成 16 维执行接口。SmolVLA 基座模型在兄弟仓库
`vla_ur5e_sim/assets/policy/base/pretrained_model`，不在本仓库内。

## 文档

- [架构与维护入口](docs/architecture.md)：修改某项功能应该去哪。
- [完整操作手册](docs/operations.md)：采集、训练、评估、可视化与遥操作命令。
- [数据与动作契约](docs/contracts.md)：16 维执行、8 维左臂训练、相机与录制时序。
- [入口索引与管理](docs/experiments.md)：正式入口与新增实验规范。

数据放 `datasets/`，模型、日志和报告放独立 `outputs/` 子目录；不要放入源码或提交大型产物。
