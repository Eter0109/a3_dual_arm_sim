# 动作与数据契约

## 环境接口

| 字段 | 形状 / 单位 | 含义 |
| --- | --- | --- |
| `joint_position` 动作 | `(16,)` | `[L1..L7,L_gripper,R1..R7,R_gripper]`；关节绝对目标 rad，夹爪 `[0,1]`，0 关闭、1 张开 |
| `cartesian_delta` 动作 | `(14,)` | 每臂 6 个归一化位姿增量与夹爪指令，所有值位于 `[-1,1]`；经 IK 转为执行目标 |
| `observation.state` | `(16,)` | 双臂关节位置与夹爪开度 |
| `observation.velocity` | `(16,)` | 双臂状态变化率 |
| `observation.eef_pose` | `(14,)` | 每臂位置 3D 与姿态四元数 4D |
| `observation.force` | `(18,)` | 力反馈，具体排列由环境实现定义 |
| 三路图像 | `H×W×3 uint8 RGB` | `observation.images.front`、`left_wrist`、`right_wrist` |

关节顺序见 `contracts.LEFT_JOINTS/RIGHT_JOINTS`。非法维度、非有限值及越界笛卡尔指令会被拒绝。绝对关节目标不是增量、速度或力矩；不得把其他动作空间的预测直接送入本项目。

## 数据录制与左臂学习

Recorder 保存动作前观测与环境返回的 `info["applied_action"]`，并保存 task、episode 元数据和是否成功。记录的是实际应用目标，可能已经过 IK、限位和限速，不能等同未经处理的策略预测。

指定 Hub 数据为 LeRobot `v3.0`，100 episodes、42,681 帧、20 FPS；三路 256×256 RGB 以 parquet 图像字段保存。`observation.state` 和 `action` 为 8D 左臂数据，附加速度/力反馈等字段可继续保持原采集维度。具体信息以下载 revision 的 `meta/info.json` 为准。

原始双臂采集的 state/action 为 16D。左臂转换只截取前 8D 并更新统计、features 与 A3 collection summary；不要再次转换指定的 8D Hub 数据。转换目标目录必须是新目录。

部署 8D 模型时截取观测前 8D，预测左臂，再追加 reset 后首次观测中的右臂 8D 保持目标。**右臂不补零**。16D 模型保持完整双臂接口。

## 模型与时序

图像进入 LeRobot 时转换为模型需要的通道布局。状态和动作的归一化、反归一化由检查点 processor 与训练数据统计完成；工作流不实现第二套算法。测试必须指定与检查点对应的训练数据根目录。

`chunk_size` 是一次预测的动作数，`n_action_steps` 是重新推理前消费的动作数，满足 `1 <= n_action_steps <= chunk_size`。20 Hz 下执行 8 个动作对应 0.4 秒；改变控制频率会改变物理执行时长。`num_steps` 是 flow 推理迭代次数，不是训练步数。相机参数、任务文本、控制频率和动作后处理都是部署条件，改变后必须单独记录。

普通权重、训练期 EMA 权重与部署动作平滑是不同概念。公开 LeRobot 0.5.1 的本项目训练流程保存普通权重；既有 EMA 检查点可显式加载。报告必须记录最终解析到的检查点目录。

## 成功与证据

任务成功由环境的 `tasks` 判定，包括当前任务要求的计数、释放、稳定及安全条件。benchmark 的入盒数量分数不能代替完整任务成功。随机 benchmark 与固定布局专家演示的协议不同。

训练 loss、训练数据上的 teacher-forced 误差、短程部署冒烟和完整闭环成功率分别报告。仅保存了检查点或执行了若干控制步，不代表完成装盒任务。

## ACT 执行契约

ACT 与 SmolVLA 共享三相机和 8D 左臂/16D 执行接口。ACT 使用自身检查点的 MEAN_STD processor；默认 chunk_size=50、n_action_steps=8，20 Hz 下执行 0.4 秒后重预测。每个 episode reset 清空队列和时间集成，重新捕获右臂初始保持目标。时间集成只允许 n_action_steps=1。任务文本不参与 ACT 推理。

## Benchmark 成功阈值

benchmark 默认保留目标 10 块、源盒 70 块、完整槽位占用和四侧边界覆盖条件，连续保持 5 个控制步（20 Hz 下 0.25 秒仿真时间）。槽位中心容差为 x±15 mm、y±5 mm；整组边界覆盖容差为 30 mm。盒内范围、姿态和速度的计数检查照常执行。重叠容差使用一对一槽位分配，避免同一块饼干占据多个槽位或因同时匹配两个槽位而被排除。

CLI 可用 `--success-hold-steps`、`--slot-tolerance-x`、`--slot-tolerance-y`、`--wall-contact-tolerance` 调整，长度单位为米。复现旧 benchmark 协议使用 `--success-hold-steps 20 --slot-tolerance-x 0.010 --slot-tolerance-y 0.0025 --wall-contact-tolerance 0.026`。评估报告记录 `run.success_conditions`，不同阈值的成功率须注明协议。通用任务及采集的默认成功配置仍按其自身契约执行。

增加 `--diagnostic-dir outputs/<新目录>` 后，每步 `trace.jsonl` 的 `after` 包含计数、槽位占用、边界覆盖、完整排布、保持计数及 `unmet_success_conditions`；`success_diagnostics.json` 汇总首次满足计数/排布的步骤、最大保持计数和最终未满足项。步骤索引从 0 开始，诊断只观察环境，不影响判定。
