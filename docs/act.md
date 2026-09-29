# ACT 训练、部署与评估

ACT 与 SmolVLA 共用 A3 数据和仿真接口，推理实现彼此独立。ACT 使用三路图像和关节状态，不使用任务文本。本项目采用公开 LeRobot 0.5.1 的 ACT；新建 ACT 策略，视觉骨干默认下载 torchvision 的 ImageNet ResNet18 权重。它不加载 `lerobot/smolvla_base`。

## 1. 安装和数据

在仓库根目录的独立 Python 3.12+ 环境执行：

```bash
python -m pip install -e '.[act,dev]'
python -m pip check
a3-sim train-act --help
hf download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 \
  --local-dir datasets/a3_front_close_left_100
python -m a3_dual_arm_sim.data.audit_smolvla_data_cli \
  --root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 --output outputs/act_data_audit.json
```

数据为 100 episodes、42,681 帧、20 FPS；`observation.state` 和 `action` 都是 8D 左臂，三相机键为 `observation.images.front/left_wrist/right_wrist`，每路 256×256 RGB。默认无需采集或转换。已按 SmolVLA 手册下载相同 revision 的数据可直接复用。审计工具名称保留兼容性，其数据检查也适用于 ACT。手动下载需保留上述 revision 和下载日志；项目下载器生成的 `.a3_download_revision.json` 会自动写入训练启动记录。

审计失败时检查下载完整性和数据维度，不跳过校验。数据内容审计与训练启动时的轻量元数据审计是两步独立检查。

## 2. 短训练和完整训练

```bash
mkdir -p outputs/act_smoke
set -o pipefail
a3-sim train-act --root datasets/a3_front_close_left_100 \
  --output outputs/act_smoke/model --steps 10 --batch-size 1 \
  --num-workers 0 --save-freq 10 --device cuda \
  2>&1 | tee outputs/act_smoke/train.log
```

可先加 `--dry-run` 检查配置；它写入输出同级的 `.model_act_config/train_config.json` 和 `model_launch.json`，不优化或下载骨干。真实训练在线下载约 45 MB 的 ResNet18 权重到 `$TORCH_HOME/hub/checkpoints`（默认 `~/.cache/torch/hub/checkpoints`）。离线训练需先缓存该权重；`--no-pretrained-backbone` 明确改为随机初始化，会改变实验条件。部署使用完整训练检查点，不再次下载骨干。

CUDA 不可用时改 `--device cpu`；内存不足先降 batch size。目录必须不存在，避免覆盖用户模型。期望进程退出码 0，并生成 `model/checkpoints/000010/pretrained_model/`，其中含 `config.json`、`model.safetensors`、输入/输出 processor 及统计量；训练状态位于同一步的 `training_state/`。

完整训练起点：

```bash
a3-sim train-act --root datasets/a3_front_close_left_100 \
  --output outputs/act_20k --steps 20000 --batch-size 8 \
  --chunk-size 50 --n-action-steps 8 --lr 1e-5 \
  --num-workers 2 --save-freq 2000 --seed 1000 --device cuda
```

批大小按实测显存调整。默认 `n_obs_steps=1`，每次预测 50 步、执行 8 步后重新观察；在 20 Hz 下分别覆盖 2.5 秒和 0.4 秒。ACT 自带 VAE/KL 损失及 MEAN_STD 归一化，不能套用 SmolVLA 的 processor。可用 `--temporal-ensemble-coeff 0.01 --n-action-steps 1` 启用每个控制步重预测与时间集成；该模式必须执行步长为 1。

## 3. 检查点重载与 20 步仿真冒烟

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli \
  --policy act:outputs/act_smoke/model \
  --dataset-root datasets/a3_front_close_left_100 --device cuda \
  --episodes 1 --seed-start 1000 --max-steps 20 --workers 1 \
  --n-action-steps 8 --inference-seed 123 \
  --diagnostic-dir outputs/act_smoke/deploy_trace \
  --output outputs/act_smoke/deploy.json
```

`act:` 接受训练 run 目录或明确的 `pretrained_model` 目录；run 目录选择最新有效数字步骤。为兼容已有产物，该步骤同时有 EMA 时优先 EMA；本项目公开 ACT 训练器只生成普通权重，精确复现请传明确目录。报告记录实际检查点、SHA256、保存配置、有效推理配置、数据目录和已知 revision。部署时也可传 `--temporal-ensemble-coeff 0.01 --n-action-steps 1` 启用时间集成。诊断目录必须是新目录。

8D 模型只读取左臂 state。每次 episode reset 清空 ACT 动作队列/时间集成历史，并在第一次观测中保存右臂 8D 初始目标；输出补齐成 16D，环境继续执行限位与限速。`raw_action` 为反归一化后含右臂保持的输出；ACT 无额外几何修正或 EMA 动作平滑，环境约束后的 `applied_action` 可能不同。

通用插件入口也可使用：

```bash
A3_ACT_CHECKPOINT=outputs/act_smoke/model \
A3_ACT_DATASET_ROOT=datasets/a3_front_close_left_100 \
A3_ACT_DEVICE=cuda MUJOCO_GL=egl a3-sim run --scene cookie_transfer \
  --policy a3_dual_arm_sim.policies.act:make_policy --steps 20
```

## 4. 离线误差与多 episode 评估

```bash
python -m a3_dual_arm_sim.workflows.smolvla_offline_error_cli \
  --policy-type act --root datasets/a3_front_close_left_100 \
  --checkpoint outputs/act_smoke/model --device cuda --max-samples 50 \
  --output outputs/act_smoke/offline.json
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli \
  --policy act:outputs/act_20k --dataset-root datasets/a3_front_close_left_100 \
  --device cuda --episodes 20 --seed-start 1000 --max-steps 6000 \
  --workers 1 --output outputs/act_20k_eval.json
```

离线工具复用既有模块，`--policy-type act` 选择 ACT，默认仍为 SmolVLA；每个抽样观测前 reset，报告训练集首动作误差，不能当作独立测试集表现。训练 loss、离线关节/夹爪误差和闭环装盒成功率分别度量不同结果。10 步训练及 20 步部署仅验证执行链路，不能证明任务已学会。完整评估比较应固定场景、seed、最大步数、控制频率及执行 horizon，并将开发 seed 与最终验收 seed 分开。

故障定位：类型不匹配检查 `config.json.type`；维度错误检查数据与权重是否同为 8D；归一化异常检查 processor 是否与权重配套；加载缺键需恢复完整检查点；EGL 失败检查驱动和渲染后端；空抓/闭环失败先比较观测、原始动作与实际动作，再决定是否补数据或训练。ACT 的短训练实测记录见 [接入验证](act_validation.md)。
