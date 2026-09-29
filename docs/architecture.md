# Architecture / 架构

## Responsibilities

`core` owns the canonical observation/action feature names and the shared
`CookieSkill`, `SkillRequest`, `SkillOutcome` definitions. It does not import
simulation, data recording or model libraries. There must not be two copies of
the skill enum: controller transitions compare enum members by identity.

`envs` owns MuJoCo geometry, contacts, joint actions, RGB and optional metric
depth. `experts` reads privileged simulator state to generate **physical** actions.
Neither module is a learned VLA. Scene parameters and expert control logic are
separate responsibilities; changing one may require revalidating the other.

`policies` owns the action-model interface. The LeRobot adapter converts three
RGB cameras, the robot state and a skill language instruction into the canonical
16-dimensional absolute joint/gripper target. It owns preprocessing and action
chunk cache clearing. A compatible A3 checkpoint and dataset metadata are required.

`agents` is split into:

| Module | Responsibility |
| --- | --- |
| `planning` | Fixed diagnostic plan or scene/language-conditioned visual plan |
| `executors` | Expert/LeRobot skill execution and safe recovery actions |
| `verification` | Image-window completion/diagnosis, plus explicit oracle baseline |
| `controller` | Skill switching, window timing, retries, timeout and safety gates |
| `backends` | Local Qwen loading or configurable HTTP model calls |
| `prompts` | Shared training/inference prompts and allowed skill grammar |

Model and expert imports are lazy. Importing a Planner/Verifier contract does not
load the recorder, MuJoCo or Transformers. The visual path is not allowed to silently
fall back to oracle truth or a fixed plan.

`data` owns both recording products, but keeps them distinct. `training` consumes
prepared datasets; `evaluation` scores independent episodes/windows. Privileged
`evaluation.skill_truth` is used only to label/audit, not as a model input.
`cli` parses arguments and calls these services; it does not own the training or
dataset implementation. `teleop` is separate from automatic execution.

## Current skill library

The supported single-box plan contains four segments:

```text
PICK_FIVE batch 0 → PLACE_FIVE batch 0 → PICK_FIVE batch 1 → PLACE_FIVE batch 1
```

A source column can be specified or sampled for expert data collection. This is
not yet a general variable-count skill library such as three cookies followed by
seven. Two-box switching is outside the supported single-box Agent skill plan.
Action-policy switching does not change the skill or observation
contract automatically.

## Storage and deployment

Source code is under `src/`, configurations under `configs/envs` and `configs/agents`,
tests grouped by functionality, and documentation under `docs/`. Data/weights/results
belong in ignored storage roots. Keep dataset metadata, checkpoint processors,
base-model references and tokenizer paths valid when transferring artifacts to
another machine. The public entry point is `a3-sim` (or
`python -m a3_dual_arm_sim`); old entry scripts are retired.

This organization follows [LeRobot's functional packages](https://github.com/huggingface/lerobot/tree/main/src/lerobot)
and [Agentic Robot's responsibility split](https://github.com/Agentic-Robot/agentic-robot).
It adapts those ideas to A3 rather than replacing this project with LIBERO/OpenVLA.

## 中文速读

共享定义放 `core`，仿真放 `envs`，规则专家放 `experts`，学习动作模型放 `policies`。
Agent 只做规划—执行—验证的调度，数据采集与训练另放功能包，入口只解析参数。
同一个技能定义贯穿所有模块，不再让 Agent 为了读技能类型而加载整套数据/专家代码。
真值负责标注和评估，不作为视觉模型输入。场景、专家和学习策略分别配置，修改后应
检查数据采集、训练与推理的接口是否一致。部署到另一台机器时同时迁移权重所需的
预处理、数据元信息、骨干和 tokenizer，检查保存路径，不能只复制一个权重文件。
