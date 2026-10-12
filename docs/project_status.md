# 项目完整度检查（2026-10-12）

检查对象是当前 `codex/randomization-profiles` 工作区，包含尚未提交的统一 benchmark 和网格脚本。检查依据为源码、CLI、回归测试及本机已有评测产物；不把旧发布记录当成本次验证。

## 已具备的流程

| 流程 | 当前能力 | 边界 |
| --- | --- | --- |
| 仿真任务 | 独立 2×5 与 2×10 环境、专家、三档随机化 | 参数未完成实机标定 |
| 专家采集 | 两种任务均可成功筛选、LeRobot v3 录制、续采、尝试日志与审计 | 采集 CLI 的 policy 固定为对应专家 |
| 模型训练 | SmolVLA/ACT 入口、8D/16D 适配、保存 processor、短训练和恢复说明 | 当前本机 LeRobot 版本偏离声明依赖，见下文 |
| 仿真评测 | 公共 `--task`、`--policy`、可视化、诊断、多 episode 数量统计 | 成绩依赖指令、seed、步数与来源列 |
| 网格评测 | 两任务×三档、500 步、逐次落盘、哈希校验、续跑 | 当前为单 SmolVLA 检查点、来源第 1 列 |
| 策略扩展 | `module:factory`、reset/act/close 与动作契约 | 自定义模型负责自己的配置及 8D→16D 适配 |

## 优先改进项

### P1：恢复可重复的训练依赖环境

`pyproject.toml` 声明 `lerobot==0.5.1`，但本机当前实际导入 `/home/eter/桌面/workspace/lerobot/src/lerobot`，元数据版本为 **0.5.2**。

本次 ACT dry-run 测试报 `TrainPipelineConfig` 不接受 `eval_freq`，当前开发版使用 `env_eval_freq`。SmolVLA 命令生成器同样仍输出 `--eval_freq=0`，因此也需要在正式训练前核对当前训练器参数；本次没有把它当作已通过完整训练验证。

建议在独立虚拟环境安装项目声明的依赖并跑训练 dry-run/短训练；若决定升级到开发版，应同步两种训练入口、依赖约束及测试。此次未修改全局环境或本地 LeRobot checkout。版本排查命令：

```bash
python -c "import importlib.metadata as m, lerobot; print(m.version('lerobot')); print(lerobot.__file__)"
python -m pip check
```

### P1：统一“任意 policy 采集”的训练数据契约

通用 `a3-sim run --record` 能保存策略轨迹，但默认包含失败回合，且不创建标准采集摘要。2×10 审计还要求四次抓取的 lifted/released/cookie ID 等专家记录。已有神经策略能跑 benchmark，不代表它能直接产出当前审计认可的专家数据。

建议新增与 benchmark 共用任务选择的采集入口，明确成功筛选、失败轨迹用途、续采协议和策略无关的任务元数据。不要给失败轨迹补一个虚假的成功摘要。

### P1：在评测前校验任务文本与检查点兼容性

2×5 旧模型只见过短任务指令，新的按列指令已经在同 seed 诊断中改变搬运行为。2×10 使用更长的数量计划指令；旧检查点可能截断文本，新 2×10 训练配置会设置 128 tokens。

建议启动时输出实际 prompt、token 数和截断情况，记录训练任务/来源列覆盖。2×5→2×10 迁移成绩与专门训练的 2×10 模型分开报告，不应因模型“能加载”就视为任务兼容。

### P2：增强网格实验配置和进程保护

当前网格支持显式 checkpoint/data/seed/steps 和严格续跑，但任务、档位及来源第 1 列固定。它没有独占输出目录的进程锁，强制断电也可能留下不完整 JSONL 尾行。

建议支持任务到检查点的映射、来源列和指令模式参数；增加输出目录锁、可审计的尾行恢复，以及逐项完成通知。恢复时仍应保持协议校验，不能默默混入不同代码/模型结果。

### P2：补充独立验证集和持续集成

当前示例训练不自动保证 held-out 验证划分。建议固化 train/dev/final seed 清单和数据划分，并在固定依赖环境建立 CI（当前仓库未发现 `.github/workflows`），至少覆盖两种任务的入口、插件、录制往返和训练 dry-run。CPU/无头测试与真实 GPU 冒烟分别记录。

## 本次验证

- 统一 benchmark 的 2×10 专家、并行任务参数传递、20 块分母和诊断视频已有专门回归；SmolVLA 已执行真实 GPU 单步冒烟。
- 全量回归首次为 237 passed、4 failed、2 xfailed。两项失败是统一入口后测试仍 mock 旧构造器，已更新；一项是只读 Hugging Face 缓存，重测改用临时缓存。
- 使用可写临时缓存复测：**240 passed、1 failed、2 xfailed**；唯一失败为 `test_act_dry_run_public_config`，原因是上述 LeRobot 版本/API 不一致，没有跳过或修改断言掩盖它。
- 7 个文档使用的 CLI 帮助入口已检查；自定义 policy 文档模板在 2×10 完成单步调用；文档相对链接及 `git diff --check` 通过。修正一处既有测试 import 排序后，Ruff 检查通过。
- 本次未执行新的完整训练、权重发布、原生 Windows 验证或实机验证。既有两项预期失败不表示任务整体已通过验收。

## 已完成的数量评测快照

本机已有 40k 普通 SmolVLA，500 控制步、来源第 1 列，每项 20 个相同场景 seed（13000–13019），2×5 为训练短指令、2×10 为原生指令/随机首夹计划。主指标是结束时目标盒计数：

| 任务 | basic | medium | advanced |
| --- | ---: | ---: | ---: |
| 2×5，平均块数 / 10 | 3.90 | 3.75 | 2.45 |
| 2×10，平均块数 / 20 | 0.00 | 0.00 | 0.00 |

此处是**本机实验快照**，不代表所有 SmolVLA 模型；2×10 是旧十块模型的迁移测试。原始记录位于本机 `smolvla_grid_unified_500_20261012` 输出目录，包含 `protocol.json`、`episodes.jsonl`、`summary.json`；模型与完整运行产物未随 Git 发布。文档中的训练命令可运行性与上述模型效果是两类证据。

该快照权重 SHA256：`fbe59160f6d7cef205dc9ecc36f791a56d6344b276e1e9738473f9aac42d5347`。代码 Git 基线：`5c2ca502876a8d79831de1301459ba8c71b74551`；运行包含当时的未提交共享 benchmark 修改，精确共享代码哈希见原始 `protocol.json.shared_source_sha256`。
