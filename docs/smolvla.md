# SmolVLA 训练与测试全流程

本流程在 A3 MuJoCo 仿真中微调和评估左臂装盒策略。数据固定为
[`Eter0109/a3-front-close-left-100`](https://huggingface.co/datasets/Eter0109/a3-front-close-left-100)，
基座固定为 [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)。
默认直接使用 Hub 的 8D 数据，无需采集或转换。

## 1. 环境准备

在仓库根目录使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
python -c "import torch, lerobot; print(lerobot.__file__); print(torch.__version__, torch.cuda.is_available())"
python -m pip check
a3-sim train-smolvla --help
```

项目要求 Python 3.12 或更新版本，与公开 LeRobot 0.5.1 的最低要求一致。本项目锁定公开 LeRobot 0.5.1，避免依赖本机开发分支。基础仿真无需 Torch；训练需要 Torch、SmolVLA 和数据依赖。CUDA 命令要求支持当前 Torch 的驱动；无 CUDA 可用 `--device cpu` 做少量冒烟，但完整训练会很慢。批大小取决于显存，先从 1 开始，再根据实测增加。

## 2. 下载与版本记录

发布验证使用的 Hub revision：

| 资源 | revision |
| --- | --- |
| 数据 | `8fdd8a7caf8275eba8be8a83b53a99176a084062` |
| SmolVLA 基座 | `d9f33c94a60fb382c90dea2164c96845bd955e28` |
| VLM 配置/tokenizer | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

```bash
a3-sim download-smolvla \
  --dataset-root datasets/a3_front_close_left_100 \
  --model-root models/smolvla_base \
  --dataset-revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 \
  --model-revision d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --vlm-revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --manifest outputs/download_manifest.json
```

下载器保存数据、完整基座及其 VLM 配置/tokenizer 依赖，记录真实 commit 与本地路径。完整 SmolVLA 基座已含 VLM 权重；辅助快照只下载配置和 tokenizer，不重复下载 VLM 权重或示例视频。VLM 放 Hugging Face 缓存，模型目录的 `a3_dependencies.json` 指向对应快照。可用 `HF_HOME`、`HF_HUB_CACHE` 指定缓存位置；移动缓存后需重新下载/更新依赖记录。下载目录必须为空，或由本下载器标记为同一 repo/revision；相同 revision 的中断下载可重试，不混入旧数据。

也可以使用标准 Hugging Face CLI 手动下载：

```bash
hf download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 \
  --local-dir datasets/a3_front_close_left_100
hf download lerobot/smolvla_base \
  --revision d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --local-dir models/smolvla_base
```

手动下载还需按基座 `config.json` 的 `vlm_model_name` 下载 VLM 配置和 tokenizer；在线训练时允许自动下载，离线前必须将依赖完整缓存。数据/模型下载错误不是训练结果；检查网络、磁盘与 Hub 权限；Xet 传输持续停滞时可在重试命令前设置 `HF_HUB_DISABLE_XET=1` 使用标准 HTTP，公开资源通常不需登录。不要在命令或日志中写入访问令牌。

## 3. 数据审计

```bash
python examples/audit_smolvla_data.py \
  --root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 \
  --output outputs/data_audit.json
```

预期数据契约：v3.0、100 episodes、42,681 帧、20 FPS，8D state/action，三路 256×256 RGB。审计遍历 parquet 检查维度、有限值、episode/frame 数、连续索引、时间戳及统计量；每 episode 抽检一帧三路图像。`collection_summary.json` 和 `a3_episode_metadata.jsonl` 必须与数据一致且满足成功演示契约。图像抽样审计不代表全部图像已解码验证，动作对齐还需回放检查。

失败时先检查是否下载完整、选错 revision 或把 16D 数据误当作 8D。不要通过跳过校验强行训练。

## 4. 配置适配与训练冒烟

```bash
a3-sim train-smolvla \
  --root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 \
  --model models/smolvla_base --output outputs/smolvla_smoke/model \
  --steps 10 --batch-size 1 --num-workers 0 --save-freq 10 \
  --seed 1000 --device cuda --dry-run
```

`--dry-run` 审计数据并生成适配配置及权重链接，输出训练命令，不运行优化。它会写入适配目录，不是完全无副作用的只读操作。输出模型目录必须不存在；适配目录为输出同级的独立隐藏目录。

适配将相机键替换为 A3 三相机，将状态/动作改为数据的 8D 或 16D，沿用基座架构和容量，从完整基座恢复 VLM 与 action expert 权重；`load_vlm_weights=false` 仅避免初始化时重复读取辅助 VLM 权重，不表示从零训练。训练使用学习率 `5e-5`、推理迭代 25；短训练将 warmup 限制在总步数以内。基座 config 和权重不被修改。

去掉 `--dry-run` 运行相同命令，并记录日志：

```bash
set -o pipefail
a3-sim train-smolvla \
  --root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 \
  --model models/smolvla_base --output outputs/smolvla_smoke/model \
  --steps 10 --batch-size 1 --num-workers 0 --save-freq 10 \
  --seed 1000 --device cuda 2>&1 | tee outputs/smolvla_smoke/train.log
```

运行前确保 `outputs/smolvla_smoke` 已由 dry-run 创建，或手动 `mkdir -p outputs/smolvla_smoke`。预期输出：`model_launch.json`、训练日志、检查点中的配置、processor、权重及训练状态。末次训练退出必须为 0，并保存有效检查点。公开训练入口禁用环境评估（`eval_freq=0`）；本项目用自己的 MuJoCo benchmark 测试。

## 5. 完整训练与恢复

确认冒烟后在新目录启动 20,000 步训练：

```bash
mkdir -p outputs/smolvla_20k
set -o pipefail
a3-sim train-smolvla \
  --root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 \
  --model models/smolvla_base --output outputs/smolvla_20k/model \
  --steps 20000 --batch-size 8 --num-workers 2 --save-freq 2000 \
  --lr 0.00005 --seed 1000 --device cuda \
  2>&1 | tee outputs/smolvla_20k/train.log
```

批大小 8 是教程起点，不是经过完整训练验证的最优值；显存不足降低为 4/2/1。`--steps` 使用所锁定公开训练器的更新循环定义，未配置梯度累积。每 2,000 步保存，末次也保存；检查 `train_config.json` 与日志确认实际设置。

启动器拒绝覆盖已存在的输出目录。中断后通过公开训练器读取保存的训练配置恢复，路径以实际检查点为准：

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path=outputs/smolvla_20k/model/checkpoints/002000/pretrained_model/train_config.json \
  --resume=true
```

恢复依赖同一环境、数据、源模型和 optimizer 状态；不要修改训练配置后冒充同一次 run。保留 launch、下载 manifest 和各阶段日志。

## 6. 检查点与离线误差

通过启动器末尾 JSON 的 `checkpoint` 字段获取检查点，不硬编码步数目录的位数。以下例子先在 shell 设置实际路径：

```bash
export CHECKPOINT="outputs/smolvla_20k/model/checkpoints/020000/pretrained_model"
python examples/audit_smolvla_data.py \
  --root datasets/a3_front_close_left_100 --checkpoint "$CHECKPOINT" \
  --output outputs/smolvla_20k/checkpoint_data_audit.json
python examples/smolvla_offline_error.py \
  --root datasets/a3_front_close_left_100 --checkpoint "$CHECKPOINT" \
  --repo-id Eter0109/a3-front-close-left-100 --device cuda --max-samples 500 \
  --output outputs/smolvla_20k/offline_error.json
```

检查点需包含 `config.json`、`model.safetensors`、processor 配置与状态。训练公开版只保存普通权重，不传入本地分支专用 EMA 参数；可显式部署已有 `pretrained_model_ema`。传 run 根目录时解析器选择最新数字步数目录，并优先同一步 EMA；可重复比较必须传完整目录，不能依赖自动选择。

离线脚本在训练数据上抽取每 episode 至多五帧，报告 teacher-forced 首动作关节 MAE 与夹爪误差，不是 held-out 泛化指标，也不是装盒成功率。

## 7. 20 控制步部署冒烟

将 `CHECKPOINT` 设置为刚保存的冒烟检查点，再运行：

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy "smolvla:$CHECKPOINT" \
  --dataset-root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 --device cuda \
  --episodes 1 --seed-start 9000 --max-steps 20 --workers 1 \
  --n-action-steps 8 --inference-seed 1000 \
  --output outputs/smolvla_smoke/deploy20.json
```

预期模型重新加载、三相机渲染、动作反归一化和 8D→16D 执行链路正常，并写报告。20 步通常无法完成任务，`success=false` 不自动表示加载故障；检查实际执行步数和失败原因。该结果只用于接口冒烟。

## 8. 多 episode 闭环评估与诊断

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy "smolvla:$CHECKPOINT" \
  --dataset-root datasets/a3_front_close_left_100 \
  --repo-id Eter0109/a3-front-close-left-100 --device cuda \
  --episodes 20 --seed-start 1000 --max-steps 6000 --workers 1 \
  --n-action-steps 8 --inference-seed 1000 \
  --output outputs/smolvla_20k/dev20.json
# 单 episode 动作与三路视频追踪；diagnostic-dir 必须为新目录
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --policy "smolvla:$CHECKPOINT" --dataset-root datasets/a3_front_close_left_100 \
  --episodes 1 --seed-start 1000 --max-steps 6000 --workers 1 \
  --n-action-steps 8 --inference-seed 1000 \
  --diagnostic-dir outputs/smolvla_20k/trace_seed1000 \
  --output outputs/smolvla_20k/trace_result.json
```

正式验收用单独 seed 区间（例如 3000–3019），开发调参不要提前使用。对照专家时保持 seed、随机化、场景和最大步数一致。报告完整成功数/总数、逐 episode 入盒数、失败原因、控制/物理频率、预测与执行 horizon、检查点类型、权重 SHA256、数据 revision 与推理 seed。默认扰动盒和饼干；固定布局需要两个 `--no-randomize-*` 开关。

诊断区分 `raw_action`、`processed_action`、`applied_action`：分别是反归一化预测、策略后处理结果与环境应用动作。部署默认不启用夹爪锐化、右臂规则锚定、动作平滑或几何修正；左臂模型的右臂保持仍是必要接口行为。高级对照工具见 [入口索引](experiments.md)。

## 9. 故障排查

| 症状 | 检查与处理 |
| --- | --- |
| Hub SSL/超时 | 重试相同 revision，检查网络与代理；不关闭 TLS 校验 |
| 离线找不到 tokenizer/VLM | 先完整下载依赖；确认 `HF_HOME` 与下载时一致，在线模式默认允许获取缺失文件 |
| `unrecognized arguments` | 检查是否使用锁定公开 LeRobot 0.5.1，避免加载本机分支 |
| CUDA 不可用 / OOM | 核对 Torch 与驱动，降低批大小；CPU 只建议少量验证 |
| 8D/16D 或相机键不匹配 | 重跑审计，确认使用适配后的检查点与对应训练数据 |
| processor stats 不匹配 | 不要混用其他数据统计或旧检查点，核对保存的 processor |
| EGL 相机加载失败 | 检查驱动和渲染后端，先跑相机预览 |
| loss 很低但空抓/失败 | 检查动作轨迹、执行 horizon、首抓与恢复；不要用训练 loss 代替闭环对照 |
| 输出目录已存在 | 新训练用新目录；恢复用训练器 resume，不覆盖旧结果 |

本次实际完成的下载、短训练和部署验证，以及未完成事项，统一见 [发布检查](release_checklist.md)。
