# 可重复的五块搬运与摆放纠偏

本模块在当前 main 场景上提供 `PICK_PLACE_FIVE` 单批接口，内部执行抓取、实测夹持姿态对准、释放、垂直撤离、连续稳定验证，以及按需触发的重新夹放。2×5 为两批，2×10 为四批。每批重新定位并清空学习策略动作队列，保留已完成物体记录，整个任务只在开始时 reset 仿真。

`--prealign` 可显式启用物理预抓取对齐，稳定后才重置并启动模型。该操作会改变整任务检查点的起始状态分布，不能假定现有权重已经适应，因此学习执行器默认保持原起点；应在单批数据适配或对照验证后启用。参考专家本身已有接近和对准阶段，不额外重复预对齐。

## 运行

在仓库根目录安装现有环境依赖，或将 `src` 加入 `PYTHONPATH`。默认 `--executor grounded` 是目标绑定的接触抓取技能，定位明确标记为 `simulator_truth`。旧专家通过 `--executor expert` 保留作参考。

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

早期 π0.5 接入检查第一批实际搬动编号 2–6。750 步总预算中，第一批占用 322 步，第二批只有约 428 步，且早期版本仍存在跨缺口分组问题。该次显式 auto 来源检查不能作为现有模型缺乏连续抓取能力的证据。新分支已修正缺口分组，原检查记录保留。

另一次启用物理预抓取对齐的检查，在第一批 750 步预算到达时仍未完成抓取（总计 752 步包含状态切换），因此预对齐默认关闭。不能将参考专家的成功视为旧 π0.5 权重已经适应新技能起点；应通过单批片段和批次衔接数据后训练，再做配对对照。

本模块借鉴旧 dev-cookie-manipulation 分支的技能边界和执行/验证结构，沿用当前 main 的环境、专家及动作约定；不整体回退旧相机或旧训练配置。

## 目标绑定与每批重新定位

`grounded` 在任何抓取动作之前，将所选五块绑定到原生接触控制器的候选入口。控制器不再另选一组，随后根据这组物体的实际位姿计算接近、预闭合、插入、夹紧和提起动作。每批重新建立控制器局部状态，并保留已完成物体记录，仿真不 reset。接近过程中若物体发生超过 3 mm 的位移或明显转动，最多重新定位三次；开始插入后不追逐位姿噪声。

规划器优先选连续可用段的暴露端，不跨缺失编号拼组。目标绑定执行器还检查真实间隙、横向散开程度和可用夹持宽度。抓起后核对指定五块是否保持双指接触链，以及是否多带了其他物体。抓取的力、深度和滑落保护沿用原生控制器。

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills.suite \
  --output outputs/grounded_basic_dev --executors grounded \
  --tasks 2x5 2x10 --episodes 20 --seed-start 26200 --workers 4

# 改变来源列和场景扰动，不改变五块技能接口
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills.suite \
  --output outputs/grounded_medium_dev --columns 2 --profile medium \
  --episodes 4 --seed-start 26240 --workers 2
```

这是复用当前 main 接触抓取方法的几何参考，并非提出了新的抓取算法；完整工作流中的目标绑定、重新定位、放置及验证均显式记录。当前仅验证既有 2×5、2×10 场景，不声称支持任意盒型或坍塌状态。

## π0.5 的干净配对对照

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills.paired \
  --output outputs/pi05_queue_pair \
  --job /path/to/pi05_job --checkpoint /path/to/030000/pretrained_model \
  --policy-module-dir /path/to/benchmark_research_v3 \
  --camera-reference /path/to/training_camera_profile.json \
  --prompt-file /path/to/original_training_prompt.txt \
  --tasks 2x5 2x10 --seeds 26220 --video
```

该检查串行复用一份模型，比较 `continuous` 与 `queue_reset`。两组使用相同初始物理状态、RGB、原训练指令、相机、推理设置及总控制步数；逐项写入 `*_pair_check.json`。默认保留检查点自己的 `n_action_steps`，不再隐式从 16 改成 8；若显式调整 `--n-action-steps`，两组一起调整。

`queue_reset` 只在五块已松手、连续稳定并且夹爪中点撤离至少 4 cm 后清空神经动作队列。它不额外移动机械臂，不做几何放置，不改变指令，不在批次边界重设随机种子。2×5/2×10 总预算分别为 1500/3000 步。宽松判据和原生最终结果分别记录。若 2×10 继续使用写有“两批/十块”的训练指令，这一指令迁移限制会写入协议，不能忽略。

## 单批示范与明确的学习目标

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m a3_batch_skills \
  --task 2x5 --executor grounded --seed 26200 \
  --record-skills --output outputs/skill_demo \
  --camera-reference /path/to/training_camera_profile.json

# --record-skills 也可用于 suite，自动覆盖多个起点和连续批次
PYTHONPATH=src python -m a3_batch_skills.export \
  --inputs outputs/skill_demo/skill_clips --output datasets/a3_goal_batch_left8
```

每批保存为独立片段，包括实际第一批起点及上一批撤离后的起点；保留连续任务中原本的物体状态，不把每段都伪造成干净初始场景。三相机无损 RGB、动作前左臂 8D 状态、同一步实际施加的左臂 8D 关节目标、阶段、物体局部坐标及全局步号对应。失败片段保留，但默认不导出为正确动作示范。导出后每批是独立的 LeRobot v3 episode，动作窗口不会跨批次。

默认录制 `projected_batch_targets_v1`：在原三视角 RGB 上用青色框标记所选五块、绿色框标记本批目标区域，并使用固定的短技能指令。这是明确的目标输入，避免调度器知道目标而模型只收到泛化文字。状态、动作仍为 8D。当前目标框来自仿真真值投影，可能标出被遮挡物体的位置；因此这是一种有真值目标提示的协议，不能宣称为纯 RGB 感知能力。实际视觉部署仍需独立的目标定位来源。

`--unmarked-skills` 记录无标记画面，适用于独立对照，不能与有标记片段无说明混用。只把数值位姿写进旁路元信息不会自动使原 VLA 使用这些数值。

完成适配训练的检查点需同时保存 `goal_conditioning.json`，才能通过 `--goal-conditioned` 使用同一投影规则进行推理。该选项对旧 30k 权重默认关闭，不假定模型已经理解新标记。旁路中的仿真位姿和接触量不是额外的神经输入。

## 数据审计和后续适配入口

```bash
PYTHONPATH=src python -m a3_batch_skills.audit \
  --root datasets/a3_goal_batch_left8 \
  --job /path/to/pi05_job --checkpoint /path/to/030000/pretrained_model \
  --policy-module-dir /path/to/benchmark_research_v3
```

审计通过真实数据读取器检查：每批最后一帧的 50 步动作窗口中只有首行动作有效，后续均为填充；状态和动作与原始片段一致；既有 π0.5 预处理保留 `action_is_pad`。原片段目录应保持可访问，以完成精确对应审计。

`a3_batch_skills.adapt` 是现有训练启动器的包装入口。参数 `--launcher`、`--dataset`、`--output` 后用 `--` 传入原先训练参数。它校验 8D 数据，保留受训状态投影，安装同时屏蔽时间填充和 8D 以外维度的 flow loss，并保留批次末尾的真实撤离帧。学习率、优化器、解冻范围与训练预算由原配置明确给定，包装器不擅自选择。原模型及原数据不覆盖。

新训练示例的参数结构为：

```bash
PYTHONPATH=src python -m a3_batch_skills.adapt \
  --launcher /path/to/pi05_job/repo/scripts/train_pi05_launcher.py \
  --dataset datasets/a3_goal_batch_left8 --output outputs/pi05_batch_adapt \
  -- --config_path=/path/to/existing_train_config.json \
  --policy.path=/path/to/030000/pretrained_model \
  --steps=5000 --save_freq=5000 --resume=false
```

该命令仅用于后续明确配置的适配训练。本轮导出的两段样例用于验证数据通路，不能作为足够的训练量或训练效果证据；本轮没有启动新的权重训练。
