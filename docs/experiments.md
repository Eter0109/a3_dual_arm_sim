# 入口索引与实验管理

正式实现位于 `src/a3_dual_arm_sim/`，`examples/` 仅转发对应模块。源码用户运行脚本，wheel 用户使用 `python -m a3_dual_arm_sim.<实现模块>`。所有入口支持 `--help`。

## 主流程

| 入口 | 实现模块（包名之后） | 用途 |
| --- | --- | --- |
| `a3-sim inspect/smoke/run/teleop/replay` | `cli` | 通用仿真与策略、遥操作及回放 |
| `a3-sim download-smolvla/train-smolvla` | `cli` | Hub 下载及公开 LeRobot 训练 |
| `a3-sim train-act` | `learning.act_training` | ACT 配置适配及公开 LeRobot 训练 |
| `run_cookie_batch.py` | `workflows.run_cookie_batch_cli` | 同列/跨列专家 |
| `run_cookie_same_column.py` | `workflows.run_cookie_same_column_cli` | 同列专家便利入口 |
| `run_cookie_transfer.py` | `workflows.run_cookie_transfer_cli` | 单块/协作搬运专家 |
| `evaluate_cookie_transfer.py` | `workflows.evaluate_cookie_transfer_cli` | 单块任务评估 |
| `benchmark_cookie_batch.py` | `workflows.benchmark_cookie_batch_cli` | 批次随机闭环评估 |
| `preview_cameras.py` | `workflows.preview_cameras_cli` | 三相机预览 |
| `collect_cookie_benchmark.py` | `data.collect_cookie_benchmark_cli` | 成功专家演示采集 |
| `prepare_left_arm_dataset.py` | `data.prepare_left_arm_dataset_cli` | 16D→8D 数据转换 |
| `audit_smolvla_data.py` | `data.audit_smolvla_data_cli` | 数据及检查点统计审计 |
| `smolvla_offline_error.py` | `workflows.smolvla_offline_error_cli` | 训练数据首动作误差，`--policy-type act` 支持 ACT |
| `summarize_smolvla_trace.py` | `workflows.summarize_smolvla_trace_cli` | 汇总诊断动作轨迹 |

主流程指维护入口，不代表所有任务已获得成功率或完整训练验收。单块顺序专家在密集 80 块布局上的两个测试为既有预期失败；批次任务优先使用同列/跨列专家，不能据此宣称单块专家已通过该布局验收。

## 高级实验入口

| 脚本 | 实现模块（包名之后） | 限制 |
| --- | --- | --- |
| `diagnose_smolvla.py` | `workflows.diagnose_smolvla_cli` | 多阶段诊断；部分阶段假设普通/EMA 两类已有检查点 |
| `smolvla_deployment_sweep.py` | `workflows.smolvla_deployment_sweep_cli` | 部署配置对照 |
| `collect_smolvla_recovery.py` | `data.collect_smolvla_recovery_cli` | 恢复示范采集 |
| `collect_smolvla_dagger.py` | `data.collect_smolvla_dagger_cli` | 专家接管与 DAgger 实验 |
| `prepare_smolvla_retraining.py` | `learning.prepare_smolvla_retraining_cli` | 再训练数据与命令准备 |
| `probe_smolvla_retrain.py` | `learning.probe_smolvla_retrain_cli` | 再训练候选短验证 |
| `train_after_collection.py` | `learning.train_after_collection_cli` | Linux PID/锁编排，要求特定 100 episode 采集契约 |
| `run_smolvla_improvement.py` | `workflows.run_smolvla_improvement_cli` | 带门控的实验流水线 |
| `improve_smolvla_50.py` | `workflows.improve_smolvla_50_cli` | 扩展实验编排 |

这些入口保留已有研究功能，其整条流水线不属于本次发布冒烟验收。使用前检查 `--help`、数据与实际检查点；不存在的 EMA 权重不得视为已产生，单阶段准备成功也不等于整条流水线成功。部分编排默认依赖仓库内 `examples/`，适用于源码安装；wheel 主流程使用模块入口。

## 结果追踪

每个实验使用独立 `outputs/<run>`，保留命令、配置、环境版本、Git revision/dirty diff、数据 revision、实际检查点及 SHA256、场景与推理 seed。普通/EMA 权重和部署后处理分开标记。开发 seed 与最终验收 seed 隔离。

比较时保持场景、随机化、控制频率、最大步数、数据统计及 seed 一致。模型评估显式传 `--dataset-root`；默认目录仅是便利约定。原始数据、权重、缓存及日志不纳入 Git。新增功能按架构职责归属正式包，不在示例转发脚本中写业务。
