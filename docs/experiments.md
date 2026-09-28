# 入口索引与管理

正式实现都在 `src/a3_dual_arm_sim/`，`examples/` 只做转发。历史实验脚本（`experiments/legacy`）
与旧模块兼容层已在清理中删除，下表入口全部指向维护中的实现。

| 入口 | 实现 | 状态 |
| --- | --- | --- |
| `examples/audit_smolvla_data.py` | [src/a3_dual_arm_sim/data/audit_smolvla_data_cli.py](../src/a3_dual_arm_sim/data/audit_smolvla_data_cli.py) | maintained entrypoint |
| `examples/benchmark_cookie_batch.py` | [src/a3_dual_arm_sim/workflows/benchmark_cookie_batch_cli.py](../src/a3_dual_arm_sim/workflows/benchmark_cookie_batch_cli.py) | maintained entrypoint |
| `examples/collect_cookie_benchmark.py` | [src/a3_dual_arm_sim/data/collect_cookie_benchmark_cli.py](../src/a3_dual_arm_sim/data/collect_cookie_benchmark_cli.py) | maintained entrypoint |
| `examples/collect_smolvla_dagger.py` | [src/a3_dual_arm_sim/data/collect_smolvla_dagger_cli.py](../src/a3_dual_arm_sim/data/collect_smolvla_dagger_cli.py) | maintained entrypoint |
| `examples/collect_smolvla_recovery.py` | [src/a3_dual_arm_sim/data/collect_smolvla_recovery_cli.py](../src/a3_dual_arm_sim/data/collect_smolvla_recovery_cli.py) | maintained entrypoint |
| `examples/diagnose_smolvla.py` | [src/a3_dual_arm_sim/workflows/diagnose_smolvla_cli.py](../src/a3_dual_arm_sim/workflows/diagnose_smolvla_cli.py) | maintained entrypoint |
| `examples/evaluate_cookie_transfer.py` | [src/a3_dual_arm_sim/workflows/evaluate_cookie_transfer_cli.py](../src/a3_dual_arm_sim/workflows/evaluate_cookie_transfer_cli.py) | maintained entrypoint |
| `examples/improve_smolvla_50.py` | [src/a3_dual_arm_sim/workflows/improve_smolvla_50_cli.py](../src/a3_dual_arm_sim/workflows/improve_smolvla_50_cli.py) | maintained entrypoint |
| `examples/prepare_left_arm_dataset.py` | [src/a3_dual_arm_sim/data/prepare_left_arm_dataset_cli.py](../src/a3_dual_arm_sim/data/prepare_left_arm_dataset_cli.py) | maintained entrypoint |
| `examples/prepare_smolvla_retraining.py` | [src/a3_dual_arm_sim/learning/prepare_smolvla_retraining_cli.py](../src/a3_dual_arm_sim/learning/prepare_smolvla_retraining_cli.py) | maintained entrypoint |
| `examples/preview_cameras.py` | [src/a3_dual_arm_sim/workflows/preview_cameras_cli.py](../src/a3_dual_arm_sim/workflows/preview_cameras_cli.py) | maintained entrypoint |
| `examples/probe_smolvla_retrain.py` | [src/a3_dual_arm_sim/learning/probe_smolvla_retrain_cli.py](../src/a3_dual_arm_sim/learning/probe_smolvla_retrain_cli.py) | maintained entrypoint |
| `examples/run_cookie_batch.py` | [src/a3_dual_arm_sim/workflows/run_cookie_batch_cli.py](../src/a3_dual_arm_sim/workflows/run_cookie_batch_cli.py) | maintained entrypoint |
| `examples/run_cookie_same_column.py` | [src/a3_dual_arm_sim/workflows/run_cookie_same_column_cli.py](../src/a3_dual_arm_sim/workflows/run_cookie_same_column_cli.py) | maintained entrypoint |
| `examples/run_cookie_transfer.py` | [src/a3_dual_arm_sim/workflows/run_cookie_transfer_cli.py](../src/a3_dual_arm_sim/workflows/run_cookie_transfer_cli.py) | maintained entrypoint |
| `examples/run_smolvla_improvement.py` | [src/a3_dual_arm_sim/workflows/run_smolvla_improvement_cli.py](../src/a3_dual_arm_sim/workflows/run_smolvla_improvement_cli.py) | maintained entrypoint |
| `examples/smolvla_deployment_sweep.py` | [src/a3_dual_arm_sim/workflows/smolvla_deployment_sweep_cli.py](../src/a3_dual_arm_sim/workflows/smolvla_deployment_sweep_cli.py) | maintained entrypoint |
| `examples/smolvla_offline_error.py` | [src/a3_dual_arm_sim/workflows/smolvla_offline_error_cli.py](../src/a3_dual_arm_sim/workflows/smolvla_offline_error_cli.py) | maintained entrypoint |
| `examples/summarize_smolvla_trace.py` | [src/a3_dual_arm_sim/workflows/summarize_smolvla_trace_cli.py](../src/a3_dual_arm_sim/workflows/summarize_smolvla_trace_cli.py) | maintained entrypoint |
| `examples/train_after_collection.py` | [src/a3_dual_arm_sim/learning/train_after_collection_cli.py](../src/a3_dual_arm_sim/learning/train_after_collection_cli.py) | maintained entrypoint |


## 新实验规范

1. 使用独立 `outputs/<experiment-id>`，禁止复用旧报告冒充新结果。
2. 保存完整命令、配置快照、Git revision 与 dirty diff、检查点实际路径与 SHA256、数据根路径/版本、场景和推理 seed。
3. 普通/EMA 权重分开标记，按完整 episode 恢复；原始数据/模型不纳入 Git。
4. 通用采集、训练、诊断调用正式包；新实验专有实现放 `experiments/`，不继续扩大 `examples` 或顶层业务模块。
5. 部署评测必须传 `--dataset-root`，报告 checkpoint 路径与 seed；短程启动测试不等于任务成功率。

当前保留的产物只有 `datasets/a3_front_close_left_100` 与 `outputs/smolvla_front_close_left_20k`；
其它入口在需要时应把 `--root`/`--dataset-root` 指向新采集的数据与新 run。
