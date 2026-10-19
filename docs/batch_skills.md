# 可重复的五块搬运与摆放纠偏

本模块在当前 main 场景上提供 `PICK_PLACE_FIVE` 单批接口，内部执行抓取、实测夹持姿态对准、释放、垂直撤离、连续稳定验证，以及按需触发的重新夹放。2×5 为两批，2×10 为四批。每批重新定位并清空学习策略动作队列，保留已完成物体记录，整个任务只在开始时 reset 仿真。

`--prealign` 可显式启用物理预抓取对齐，稳定后才重置并启动模型。该操作会改变整任务检查点的起始状态分布，不能假定现有权重已经适应，因此学习执行器默认保持原起点；应在单批数据适配或对照验证后启用。参考专家本身已有接近和对准阶段，不额外重复预对齐。

## 运行

在仓库根目录安装现有环境依赖，或将 `src` 加入 `PYTHONPATH`。默认执行器是当前 main 的仿真真值专家，定位也明确标记为 `simulator_truth`。

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills \
  --task 2x5 --output outputs/batch_2x5 --video

MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills \
  --task 2x10 --output outputs/batch_2x10
```

若使用已经对齐数据的三视角配置，显式传入 `--camera-reference <training_camera_profile.json>`；模块同时恢复该 profile 的腕部 target-body 相机约定。默认不改变图像抗锯齿。`--render-protocol single_sample` 是显式可选协议，用于已观察到 EGL 多重采样像素抖动的主机；比较组必须使用相同设置。

当前 30k 左臂 π0.5 检查点使用既有训练状态投影与处理器，需要其对应的 `pi05_adapter.py` 和原训练工作目录：

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills \
  --task 2x5 --executor pi05 --output outputs/batch_pi05 \
  --job /path/to/pi05_job \
  --checkpoint /path/to/030000/pretrained_model \
  --policy-module-dir /path/to/benchmark_research_v3 \
  --camera-reference /path/to/training_camera_profile.json \
  --render-protocol single_sample --policy-scope pick
```

`--policy-scope pick` 在五块抓稳并离开源盒后交给几何放置；`transfer` 要求策略先把所持批次运到目标盒附近。二者均为混合执行方式，不能标为纯 VLA。模块复用一个策略实例，在每批开始及交给几何控制器时重置队列。已有整任务模型不因更换提示词而自动变成训练好的单批策略。

来源列默认固定为 1，可跟踪该列内实际所持/已放的五块，而不错误假定每次恰好搬动预设编号。只有显式 `--source-column auto` 才允许选择其他来源列，auto 结果不能当作固定列任务成绩。规划下一批时不会把缺口两侧的零散物体拼成一个虚假的五块组。模型若自行完成放置，也会进入撤离和验证，不会无限等待持物接管。

## 摆放与纠偏

正常放置先测量物体相对工具的位姿，再对准盒内目标区域，低速进入、逐渐松手并垂直撤离。2×10 保留当前专家在相邻五块之间增加的释放间隙。

释放后检查本批、此前完成物体和下一批目标区域。当前支持对完整、轻度偏移或倾斜的一组五块进行重新夹取、提起、再次放置。偏移超限、严重倾斜、分散或接触过大时停止并记录失败；不盲推整盒、不写入物体 qpos，不声称能恢复任意坍塌状态。

```bash
# 关闭自动纠偏
python -m a3_batch_skills --task 2x5 --no-adjust --output outputs/no_adjust

# 开发检查：第一次放好后请求平移 4 mm，再实际重新夹放
python -m a3_batch_skills --task 2x5 --exercise-adjustment \
  --output outputs/regrasp_check
```

开发检查只改变所请求的目标位姿，不移动仿真物体。该标记默认关闭，结果不能混入原任务成功率。

默认每批上限 750 控制步，2×5 总上限 1500，2×10 总上限 3000；所有对准、撤离、等待和纠偏计入预算。使用 `--max-steps` 和 `--batch-steps` 显式调整。自动纠偏次数由 `--max-adjustments` 限定。

## 判据和结果

默认 `contained_unstacked`：整体水平包络在盒内、已离开夹爪、速度稳定、最低点距盒底不超过 4 mm，并连续稳定 20 步。不要求严格直立或精确占槽。4 mm 小于当前饼干的最小完整厚度，可拒绝完整第二层；这是单层高度近似，不是任意倾斜交叠检测器。

`--placement-mode strict` 还要求末次原生 benchmark 成功。原生环境的计分方法不修改，但关闭“原生成功立即结束回合”，让新流程完成实际释放和验证。安全停止始终有效。`result.json` 同时保存选定判据与原生最终分数，峰值/曾经成功另列，不替代最终结果。

- `result.json`：批次事件、目标、原生计分、选定判据、失败原因与交互成本。
- `left8_trace.npz`：每步动作前的左臂 8D 状态、实际施加的左臂 8D 目标与阶段标签；不是完整的图像模仿学习数据集。
- `rollout.mp4`：显式传入 `--video` 时录制。

学习策略只应用左臂 8D 输出。右臂由适配器保持启动时实际安全目标，不能把右臂原始关节目标全部设为零。本模块是左臂专用；原有双臂 benchmark 和右臂可选设置继续保留。

## 感知边界

`SimulatorSceneProvider` 用于明确标记的仿真参考及混合策略诊断。视觉策略自身只接收原三视角 RGB、左臂状态和指令；但几何接管、判定与整理当前使用仿真真值。因此整个新流程不能宣称为纯视觉部署。

`ExternalSceneProvider` 提供标定位姿估计接入接口；本次没有实现新的 RGB/RGB-D 检测器。不能把深度预览或 VLM 的“完成/卡住”判断当成已实现的精确位姿估计。

## 开发验证

运行 `PYTHONPATH=src python -m unittest discover -s tests/batch_skills -v`。测试包含两/四批计划、下一批阻塞、此前成果损坏、世界坐标变换、右臂保持、分散物体拒绝，以及批次衔接不重置物理场景。

2026-10-19 的开发种子 26124：参考执行器完成 2×5（556 步）、2×10（1303 步）；4 mm 显式重新夹放检查后继续完成两批（824 步）。这些是单种子的集成结果，不能推导成功率或 VLA 泛化改善。模型与定位模式须分别报告。

早期 π0.5 接入检查发现其第一批实际搬动编号 2–6，而不是预设 0–4；修正实际批次跟踪后，750 步开发检查完成了第一批，第二批尚未抓稳。该次为显式 auto 来源、尚未启用实际预抓取对齐的版本；不能标为两批成功，也不能与最终对齐版直接合并统计。新的定位和单批技能仍需要单独验证。

另一次启用物理预抓取对齐的检查，在第一批 750 步预算到达时仍未完成抓取（总计 752 步包含状态切换），因此预对齐默认关闭。不能将参考专家的成功视为旧 π0.5 权重已经适应新技能起点；应通过单批片段和批次衔接数据后训练，再做配对对照。

本模块借鉴旧 dev-cookie-manipulation 分支的技能边界和执行/验证结构，沿用当前 main 的环境、专家及动作约定；不整体回退旧相机或旧训练配置。
