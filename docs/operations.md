# 操作手册

以下命令默认在仓库根目录和已激活的 Python 环境中执行。基础功能安装 `.[dev]`，采集与学习安装 `.[train]`。wheel 用户可在自己的工作目录运行 `a3-sim` 和 `python -m a3_dual_arm_sim.<模块>`；`examples/` 是源码仓库便利入口。

## 场景与配置

| 配置 | 用途 |
| --- | --- |
| `configs/default.yaml` | 基础 sandbox 与通用配置 |
| `configs/cookie_batch.yaml` | 单盒 5+5 批次装盒 |
| `configs/cookie_same_column.yaml` | 同列批次专家场景 |
| `configs/cookie_cooperative.yaml` | 双臂协作实验场景 |
| `configs/packed_elastic.yaml` | 接触/摩擦实验配置 |

通过 `--config` 指定 YAML；省略时入口使用其内置默认。批次场景控制 20 Hz、物理 1000 Hz，源盒 80 块、目标盒容量 10 块；具体几何和初始位置由配置定义。配置中的参数没有真实硬件标定保证。

```bash
a3-sim inspect --scene cookie_transfer
MUJOCO_GL=egl a3-sim smoke --scene cookie_transfer --steps 20
```

## 专家与窗口

```bash
# 同列：ID 0–4，再 5–9
env -u MUJOCO_GL python examples/run_cookie_batch.py --expert same_column --render
# 跨列：ID 0–4，再 20–24
env -u MUJOCO_GL python examples/run_cookie_batch.py --expert cross_column --render
# 固定布局无窗口演示；失败/超时退出码为 1
python examples/run_cookie_same_column.py --seed 0 --max-steps 6000 \
  --output outputs/expert/fixed.json
```

规则专家直接读取仿真真值。固定布局命令与随机 benchmark 不是同一协议。窗口鼠标拖动调整视角，滚轮缩放。`--snapshots` 输出阶段图片，属于诊断资料；专家演示和 benchmark 不自动录制训练数据。

## 相机与遥操作

```bash
MUJOCO_GL=egl python examples/preview_cameras.py --help
MUJOCO_GL=egl python examples/preview_cameras.py
env -u MUJOCO_GL a3-sim teleop --scene cookie_transfer --no-camera-render
```

遥操作通过独立控制面板接收键盘焦点，MuJoCo 窗口用于观察。`w/s`、`a/d`、`r/f` 控制位置，`i/k`、`j/l`、`u/o` 控制旋转；夹爪、左右臂选择、录制与停止可通过面板按钮操作。按住移动，松开停止。需要训练图像时启用相机渲染；`--no-camera-render` 不可用于录制带相机的数据。

## 采集与回放

```bash
# 人工录制到一个新目录
env -u MUJOCO_GL a3-sim teleop --scene cookie_transfer \
  --record datasets/teleop16 --repo-id local/a3-teleop16
# 规则专家采集：先查看参数，再选择规模
python examples/collect_cookie_benchmark.py --help
MUJOCO_GL=egl python examples/collect_cookie_benchmark.py \
  --root datasets/expert16 --repo-id local/a3-expert16 --episodes 10
# 如需左臂模型，转换原始 16D 数据；目标必须不存在
python examples/prepare_left_arm_dataset.py \
  --source datasets/expert16 --output datasets/expert8
MUJOCO_GL=egl a3-sim replay --root datasets/expert16 \
  --repo-id local/a3-expert16 --episode 0
```

专家采集仅保留成功 episode，并保存尝试记录和 collection summary。回放用于检查动作、录制时序和场景，不保证在不同物理配置下复现完全相同的接触。指定 Hub 数据已为左臂 8D；学习步骤见 [SmolVLA 手册](smolvla.md)。

## 策略与评估

```bash
MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --policy a3_dual_arm_sim.policies.base:make_hold_policy --steps 20
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy same_column --episodes 20 --seed-start 1000 --workers 1 \
  --max-steps 6000 --output outputs/expert/random20.json
```

使用 Python 工厂字符串加载自定义策略。benchmark 默认会扰动盒位置与饼干布局；固定布局对照使用 `--no-randomize-boxes --no-randomize-cookies`。结果包括完整成功与入盒分数；保留每 episode 结果和 seed。窗口或模型对象评估使用单 worker。

## 常见运行问题

- 无窗口：检查显示服务、驱动和桌面权限；窗口命令取消 `MUJOCO_GL=egl`。
- 离屏相机失败：检查 EGL 支持；可按机器条件使用 `MUJOCO_GL=osmesa` 并安装相应系统库。
- 录制依赖缺失：安装 `.[train]`；遥操作控制面板需要 Python Tk 支持，Ubuntu 可安装 `python3-tk`。
- 命令返回任务失败：先检查报告的失败原因、最大步数和布局；不能把启动正常标为装盒成功。
- 数据或输出目录冲突：使用新目录，保留旧产物，不覆盖已完成 run。
