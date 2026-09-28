# 数据与动作契约

本次整理不改变接口、模型几何、奖励、成功标准或归一化算法。

- 环境执行 `joint_position` 16 维：`[L1..L7,L_gripper,R1..R7,R_gripper]`。
- 关节为绝对目标 rad，夹爪为 [0,1] 开度，不是增量或力矩。`cartesian_delta` 为 14 维，通过 IK 转换后执行。
- recorder 保存动作前观测和环境 `info["applied_action"]`，不是未经限位/限速的模型输出。
- 原始采集 `observation.state/action` 为16维；三路256×256 RGB：front、left_wrist、right_wrist；另存速度、末端位姿、力反馈、task和episode元数据。
- 左臂训练转换为前8维状态/动作，模型输出8维；部署补上 reset 时固定的右臂8维目标，形成16维。**不是补零**，零关节目标会移动右臂。
- 双臂16维模型沿用16维，不追加填充值。维度由检查点契约识别。
- 归一化/反归一化使用训练对应的 LeRobot processor 和统计；不在工作流重复实现。
- 图像字段、通道与像素约定、控制频率以数据 metadata 和当前配置为准；相机变更应显式记录配置，不能混同旧数据。

任务成功由 tasks 中环境定义。benchmark 的方块分数不是完整成功：需同时满足原任务的源/目标计数、稳定保持及安全条件。
policy 不拥有成功判定，诊断中的真值不得作为纯视觉 policy 输入。
