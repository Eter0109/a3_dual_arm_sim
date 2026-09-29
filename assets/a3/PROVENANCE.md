# A3 仿真资源来源

## A3 机器人

`source/A3.urdf`、`source/A3_mujoco.urdf` 与 A3 网格来自提供的 A3 外观 URDF 包（包名标识 `lanxin-urdf-20260204`）。原包包含每臂七个旋转关节。

`L_LAST_S.STL` 和 `R_LAST_S.STL` 为无有效几何的 header-only 文件，因此未收录；运行时以参数化法兰、相机、力反馈和夹爪元素替代空链接。

原始包未提供电机传动、摩擦辨识、控制器参数、零位标定、相机标定或生产夹爪模型。本项目相应参数为仿真默认值。当前未找到 A3 原始 URDF/网格明确的再分发许可，公开发布前需维护者确认授权。

## Robotiq 2F-85 外观

五个 `robotiq_arg2f_85_*` STL 来自 robosuite 1.4.0 的 Robotiq 85 gripper mesh 资源，保留 MIT 声明于 [ROBOSUITE_LICENSE.txt](ROBOSUITE_LICENSE.txt)。这些网格用于外观；碰撞与驱动采用简化平行夹爪模型。

第三方声明只覆盖对应资源，不能据此推定 A3 原始资产或项目源码的许可证。
