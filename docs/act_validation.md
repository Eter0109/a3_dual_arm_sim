# ACT 接入验证记录

验证日期：2026-09-29。产物保存于 `outputs/act_integration_20260929/`，不纳入 Git 或发行包。使用独立验证虚拟环境中的公开 PyPI LeRobot 0.5.1，未使用相邻 LeRobot 开发仓库。

## 环境与输入

- Python 3.13.12，Torch 2.10.0，torchvision 0.25.0，MuJoCo 3.6.0；NVIDIA RTX 4070 Ti SUPER，CUDA。
- 数据：`Eter0109/a3-front-close-left-100`，revision `8fdd8a7caf8275eba8be8a83b53a99176a084062`；完整数据路径 `/tmp/a3-release-audit/hub-data`。临时目录不是发行依赖，新用户按 [ACT 手册](act.md) 下载。
- 骨干：`ResNet18_Weights.IMAGENET1K_V1`，SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- ACT 参数：chunk_size=50，n_action_steps=8，batch_size=1，lr=1e-5，seed=1000，10 步优化，使用三相机和 8D 左臂状态动作。

## 实际结果

| 检查 | 结果与证据 |
| --- | --- |
| 配置生成与公开训练器 | 10 步训练退出 0；`smoke_launch.json`、`train.log` |
| loss 日志 | step 10 日志 loss=55.320，仅用于确认优化链路，不表示收敛 |
| 保存与严格重载 | `smoke/checkpoints/000010/pretrained_model` 含完整权重和 processor；`strict=True` 重载成功 |
| 仿真部署 | seed=1000、20 控制步完成；`deploy.json`、`deploy_trace/seed_1000/` 三相机视频及逐步诊断 |
| 动作契约 | 20×16 输出均有限；右臂目标保持初始值，raw 与 processed 一致；`validation.json` |
| 离线误差入口 | 训练集抽样 10 帧，左臂关节 MAE=0.152545 rad，夹爪绝对误差=0.204122；`offline.json` |
| wheel 安装 | wheel 构建、独立验证环境安装和 `pip check` 通过；`build.log`、`wheel_install.log` |
| 仓库外运行 | 在 `/tmp` 清除 PYTHONPATH 后从已安装 wheel 加载资源和 ACT；n_action_steps=1、时间集成系数 0.01、seed=1001 完成 20 步；`wheel_ensemble.json` |
| CLI 与静态检查 | 训练/benchmark 帮助通过；Ruff check/format 通过；`train_help.txt`、`benchmark_help.txt`、`lint.log`、`format.log` |

检查点 SHA256：`398e842003295a92a792f080d3415f6e4829b4d69fca0f9232adbc38488ba141`。`validation.json` 记录依赖版本、实际模块路径、数据 revision、权重哈希和动作检查结果；部署报告另外保留有效配置、Git revision 与 dirty diff。

## 验证边界

两次短部署均因 20 步上限终止，装盒数 0/10，未完成任务。此处只验收训练、重载和执行链路；离线误差来自训练集抽样，不能解读为泛化或闭环成功率。没有启动 20,000 步完整训练，没有运行完整时长的 20 episode 成功率评估，没有验证真实机器人。CPU 参数可配置，本次真实训练与部署使用 CUDA。

自动测试覆盖 ACT 左臂切片、右臂保持、非法输出、horizon 约束、真实 LeRobot 队列/时间集成 reset、配置生成和 benchmark 路由；保留 SmolVLA 原有接口。最终完整测试为 **150 passed、2 xfailed**（既有密集布局专家预期失败），耗时 88.16 秒；见产物 `tests_final.log`。
