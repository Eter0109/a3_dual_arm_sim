# 2×5 和 2×10 的可组合场景

在 `configs/randomization.yaml`（2×5）或 `configs/randomization_2x10.yaml`（2×10）的副本中加入顶层 `scene`。它与 `profiles` 同级，可搭配 `basic`、`medium` 和 `advanced`。省略 `scene` 时，场景、任务指令和评分按原标准任务执行。

本分支增加的是可选的仿真场景与对应规则专家（没有新的学习模型权重）。复杂组合仍有物理失败，实际结果、版本范围与已知限制见[场景专家验证记录](scene_expert_validation.md)。

```yaml
scene:
  source_gap_groups: [2, 5]
  target_box_count: 3
  source_tilt_deg: 8
  cookie_crush_force_n: 12
```

| 参数 | 取值 | 场景行为 |
| --- | --- | --- |
| `source_gap_groups` | `0`、整数 `2–5` 或闭区间 `[2, 5]` | 从 4 列 × 20 块的源盘抽取互不重叠的五连块位置。缺失的物体退出碰撞、画面和计数。同一 seed 可重现组数与位置。 |
| `target_box_count` | `1–4` | 附加目标盒与原盒同尺寸。每个 2×5 盒装满 10 块，每个 2×10 盒装满 20 块。多盒共用本回合的平移和偏航，各盒绕自身中心转动。 |
| `source_tilt_deg` | `0–8` 度 | 任选一个完整的五连块组，同向向前或向后倾。源盘内的低矮斜垫提供物理支撑，顶部及两侧留出抓取空间。饼干保留自由体，抬离斜垫后可自由运动。 |
| `cookie_crush_force_n` | `null` 或正数，单位 N | 饼干接触使用较软的求解参数。每个物理子步按单侧左指垫和单块饼干汇总法向接触力。达到阈值后切换到不可恢复的压缩碰撞网格和外观，当前回合立即失败。 |

组合时先核对库存。四个 2×10 盒需要全部 80 块，启用缺块会报错。三个 2×10 盒需要 60 块，设置随机范围 `[2, 5]` 时实际在 `[2, 4]` 中抽样。显式指定 5 组会报错。

标准多盒布局把主盒移入左臂工作区，并在源盘侧后方布置其余目标盒。布局按较大的 2×10 盒留出间距。回归测试用高级随机化的平移和偏航边界，检查 2×10 所有原生分批方案的放置位、预放置位及两者之间的中点。

多盒回合按各盒自己的坐标系检查姿态、稳定和释放。2×5 沿用槽位及四壁覆盖判据，2×10 沿用每列十块的判据。每个盒都满足条件，且源盘余量等于初始库存减去总容量时，才可能判为成功。单盒得分以该盒容量为上限，各盒实际计数另行记录。结果记录 `scenario_metrics.target_counts`、`target_filled`、`target_score`、`crushed_cookie_ids` 和 `peak_cookie_grip_load_n`。随机化元数据记录缺口、倾斜组、实际初始倾角及各目标盒的位置。

破损是可调阈值的接触与压缩近似。触发时厚度降至原来的 50%，高度降至 80%。`12 N` 是仿真示例值，真实饼干的破损力需要另行标定。原有策略观测接口保持原样。

## 准备配置

安装及启动环境沿用[仓库快速运行说明](../README.md#安装与快速运行)。若需要录制 LeRobot 数据或运行回归测试，在仓库根目录安装相应依赖：

```sh
python -m pip install -e '.[dev,dataset]'
cp configs/randomization.yaml configs/randomization_scene_2x5.yaml
cp configs/randomization_2x10.yaml configs/randomization_scene_2x10.yaml
```

分别编辑两个配置副本：保留原有 `profiles`，将 `profile` 设为所需档位（`basic`、`medium` 或 `advanced`），在顶层加入所需 `scene` 字段。例如，仅增加两个盒子就只写 `scene: {target_box_count: 2}`；仅测试倾斜就只写 `scene: {source_tilt_deg: 8}`。全部字段可以在库存允许时组合。原配置文件中已有默认注释示例，无需修改 main 的默认场景。

## 预览

给随机化配置副本加入所需字段后，运行以下命令导出三路相机画面。预览和评测共用初始化代码。

```sh
python examples/preview_cameras.py --task 2x5 --profile advanced --seed 17 --randomization-config configs/randomization_scene_2x5.yaml --output outputs/scene_2x5
python examples/preview_cameras.py --task 2x10 --profile medium --seed 17 --randomization-config configs/randomization_scene_2x10.yaml --output outputs/scene_2x10
```

## 场景专家与数据采集

评测入口仍是 `examples/benchmark_cookie_batch.py`，使用相同的 `--task`、`--profile` 和 `--randomization-config`，并根据盒数设置 `--max-steps`。当任一 `scene` 参数启用时，`expert` 自动选择可组合场景专家；未启用时继续使用原来单盒专家。新场景按完整五连块抓取，缺失组不被选中，当前列不够时可以切换可用源列，逐盒填满全部目标。若物理空间或夹持条件确实不满足，回合会失败，不会通过物体瞬移、跳过目标或放宽判据继续采集。普通 benchmark 的提前结束协议与下文完整示范核查不同，结果不应混为同一成功指标。

新增场景的 2×10 使用五/五的分批方案，每个盒执行四批；原始无扩展场景仍支持 `first_grasp: random` 或 0–9 的可变首抓方案。场景模式记录请求的首抓模式及实际五连块计划，避免把零碎缺口两侧的饼干当成一组。

专家根据当前五块的实际倾角调整夹持方向，使用真实接触链确认抓稳、提起与释放。多盒运输在独立的运动学工作区检查连续关节路径、夹持组和指垫的离地净空，再通过真实控制步执行；放置前重新测量夹持组的相对姿态，用实际底部间隙辅助降低到盒底。易碎场景沿用物理接触破损阈值，专家在手指压力上采用保守上限。保护或拒绝失败回合不代表物理成功；仍需用完整任务回合验证。

场景采集使用原有入口。启用 `scene` 后自动采用完整示范模式及左臂 8D 状态/动作，保留三路原有 RGB、左臂速度、末端位姿和力觉相关反馈。2×5 和 2×10 分别默认每个目标盒 1600 和 3200 控制步，可通过 `--max-steps` 显式覆盖。数据只有在全部盒装满、库存正确、所有批次释放撤离并稳定结束且没有破损时才被保存。仿真真值仅供专家及审计，不作为学习策略的常规观测。

```sh
python examples/collect_cookie_benchmark.py --root datasets/scene_2x5_trial --episodes 3 --randomization-config configs/randomization_scene_2x5.yaml --max-attempts 6
python examples/collect_cookie_2x10.py --root datasets/scene_2x10_trial --episodes 3 --randomization-config configs/randomization_scene_2x10.yaml --max-attempts 6
```

## 完整物理回合核查与数据导出

`examples/audit_scene_experts.py` 使用完整示范协议，包含原场景、两种缺块分布、倾斜、易碎、2/3/4 目标盒与库存合法的组合场景，两种规格和三档随机化各一个指定种子（共 54 格）。可以先运行单例，再做完整网格。输出目录须是新目录。

```sh
# CPU 物理仿真，不创建相机渲染上下文
CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable python examples/audit_scene_experts.py \
  --output outputs/scene_expert_try --workers 1 \
  --case 2x5_basic_tilted_pick_s0 --case 2x10_basic_two_boxes_s0

CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable python examples/audit_scene_experts.py \
  --output outputs/scene_expert_grid --workers 2
```

核查记录包括 `planned_cases.json`、`results.json` 和每个 `cases/<case_id>/result.json`、`phase_trace.json`。只在完整物理回合成功时保存 `demonstration_left8.npz`；`audit_replay_only.npz` 的完整 16D 指令和反馈只供回放核查，不能作为新左8训练目标。部分失败另存 `failed_attempt_replay_only.npz`，仅供诊断。`evaluation_valid` 与 `success` 应分别统计，不能把基础设施错误或部分入盒当作完整通过。

在资源允许的仿真服务器上，使用**同一个源码版本**对已通过的轨迹进行物理回放并导出实际相机数据：

```sh
MUJOCO_GL=egl python examples/export_scene_expert_samples.py \
  --run outputs/scene_expert_try --output outputs/scene_expert_try_data --workers 1
```

输出数据按任务和场景分组，逐帧校验回放状态，记录 LeRobot v3 的三路 RGB（front、left_wrist、right_wrist）、左臂 state/action 各8维、velocity 8维、eef_pose 7维、force 9维及 20 Hz 时间戳。检查各数据集 `replay_evidence.json`、`dataset_audit.json` 和总 `exports.json`，仅纳入通过审计的示范。无渲染数值轨迹不等于已导出的图像数据，本采集入口也未默认保存三路深度图。共享服务器应按实际 CPU、账户内存与显卡分配设置并行数及 CUDA/EGL 物理设备绑定。
