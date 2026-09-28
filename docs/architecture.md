# 架构与维护入口

| 包 | 维护内容 | 不应包含 |
| --- | --- | --- |
| `config/contracts/paths` | 配置、协议、项目资源路径 | 训练或工作流 |
| `sim` | MuJoCo 环境、动作 IK、URDF、机器人及物体构建 | benchmark 与训练 |
| `tasks` | 抓取、槽位、计数、贴壁、成功稳定保持 | 模型推理 |
| `controllers` | 专家、扶盒、遥操作和接管 | 训练优化循环 |
| `policies` | 插件加载、SmolVLA 推理、动作后处理 | 任务成功判定 |
| `data` | 录制、采集、审核、左臂转换 | 模型优化循环 |
| `learning` | 训练启动、配置和检查点准备 | 物理任务逻辑 |
| `workflows` | runner、benchmark、诊断、实验编排 | 重复物理/归一化/判定 |

基础配置/协议 → 仿真 → 任务 → 控制器 → 工作流。策略实现基于协议，训练依赖数据和策略。
数据采集编排允许使用控制器；底层 recorder 不依赖训练。Torch/LeRobot 在实际使用时加载。

## 大文件拆分

- `sim/model.py` 总装；`urdf.py` 读取；`model_types.py` 类型；`robot_builder.py` 机器人/夹爪；`object_builder.py` 桌面物体。
- `controllers/phases.py` 阶段；`grasp_expert.py` 抓取；`cookie_expert.py` 单块；`batch_expert.py` 和 `same_column_batch_expert.py` 保留原继承关系。
- `workflows/policy_adapters.py` 策略适配；`episode_execution.py` 单次运行；`benchmark_results.py` 汇总类型；`benchmark.py` 批量执行。
- `cli.py` 只保留参数与分发，具体处理在 `workflows/cli_handlers.py`。

## 新增与维护规则

包路径即导入路径，不再保留顶层兼容别名：`a3_dual_arm_sim.env`、`a3_dual_arm_sim.policy`
等旧模块已删除，请使用 `sim.env`、`policies.base` 等正式路径；policy 工厂字符串同样使用
`a3_dual_arm_sim.policies.base:make_hold_policy` 与 `a3_dual_arm_sim.policies.smolvla:make_policy`。

examples 只转发正式实现，不在入口脚本里写业务。新增功能先确定归属；成功判定只改 tasks；
相机/物理改 sim 与 YAML；训练和统计改 learning/data；部署改 policies。
每次只改一个子系统并运行对应 tests 与全量回归。
资源路径通过 `paths.project_root()` 等工具定位；不从移动后的文件层级推算项目根目录。
