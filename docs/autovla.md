# AutoVLA：Agent 驱动的 SmolVLA 实验循环

当前分支已有训练器和 MuJoCo benchmark。本工具将其接成：

```text
固定基座与数据 → 基线训练与开发评测
                       ↓
失败报告 + 最佳配置 + 最近结果 → API Agent 提出一个假设和参数改动
                       ↓
配置校验 → 训练或复用检查点 → 固定开发评测 → KEEP / DISCARD
                       ↑                              ↓
                       └──────── 简短状态更新 ─────────┘
```

v1 只研究现有 SmolVLA 配置，不新增 π0.5、LoRA、loss、架构、数据采集或采样策略。
Agent 只返回 JSON，程序负责应用白名单改动、训练与评分。新训练从同一基座开始。
丢弃候选后，下轮从当前最佳配置继续；所有历史产物保留，无需回退 Git 工作目录。

## Linux GPU 服务器准备

在仓库根目录安装依赖，按 [SmolVLA 手册](smolvla.md) 准备固定 revision 的数据、
基座及 VLM 配置/tokenizer。默认使用 `datasets/a3_front_close_left_100` 和
`models/smolvla_base`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
python -m autovla --help
```

`autovla` 是仓库内工具，请从本仓库根目录运行。编排只用标准库；初始化需要
PyYAML，真实训练/评测需要项目的完整训练依赖和 CUDA。

## 接入自己的 API

默认 `--agent api` 支持兼容 OpenAI Chat Completions 的服务，接受
`POST <base_url>/chat/completions`，从 `choices[0].message.content` 读取 JSON：

```bash
export AUTOVLA_API_BASE_URL='https://your-provider.example/v1'
export AUTOVLA_API_MODEL='your-model-name'
read -rsp 'API Key: ' AUTOVLA_API_KEY
export AUTOVLA_API_KEY
```

Key 只从环境读取，不写进配置、实验日志或 Git。远程端点要求 HTTPS，本地可用
`http://127.0.0.1:<port>/v1`；本地服务不验证 Key 时可设置非空占位值。
HTTP 重定向被拒绝，避免转发凭据。

默认发送 `response_format={"type":"json_object"}`。服务不接受该选项时设置：

```bash
export AUTOVLA_API_JSON_MODE=none
```

两种模式都在本地严格校验提案。默认 `max_tokens=1600`，每轮一次请求，发送研究
规则、最佳配置、最近结果、失败统计和已试配置；不发送完整聊天、训练日志、视频、
数据文件或本地目录。训练/评测期间不调用 LLM。私有消息协议、其他鉴权形式或不支持
`max_tokens` 的服务需要在 `autovla/agent.py` 额外适配；这不是任意 API 的通用适配器。

可选 `--agent codex` 使用本机 Codex CLI 的临时会话、只读沙箱和 JSON Schema，
沿用 CLI 认证，见 [官方说明](https://learn.chatgpt.com/docs/non-interactive-mode)。
API 后端无需安装 Codex CLI。

## 配置与预算

初始化前编辑 `autovla/protocol.json` 和 `autovla/experiment.json`，也可复制后使用
`init --protocol <file> --config <file>`。相对路径均以仓库根目录解析。

| 固定项 | 默认值 |
|---|---|
| 新训练预算 | 2,000 updates，全训练数据 |
| 总实验数 | 11，含基线与失败/无效提案 |
| 累计活动时间 | 48 小时，含提案、训练、开发评测 |
| 单轮 / 单次 Agent 超时 | 4 小时 / 180 秒 |
| 开发集 | basic/medium/advanced 各 8 次，seed 10000–10007 |
| 最终验收 | 三档各 20 次，seed 30000–30019 |
| 源列 | 第一列，与默认左臂数据匹配 |
| 单 episode 上限 | 6,000 控制步 |
| 视频追踪 | 每档前 1 个 episode，三路相机 |

三档共用 seed，分档报告后按所有 episode 等权汇总。要研究其他列，初始化前改
`cases` 并核对数据覆盖。默认是同预算的低成本筛选，尚未实现多级预算调度。
完整 20,000 步训练应新建 study，并按该预算重新跑基线及候选，不能跨预算混比。
评测可能比训练耗时更长，应按实际 GPU 和执行块长度设置单轮超时。

| 可改参数 | 范围 | 作用 |
|---|---|---|
| `lr` | 1e-6–1e-3 | 学习率 |
| `decay_lr` | 1e-7–1e-4，且不大于 lr | 衰减末值 |
| `warmup_steps` | 0–2000，且小于训练步数 | 预热 |
| `batch_size` | 1–32 | 训练批大小 |
| `n_action_steps` | 1–50 | 推理执行块长度 |

只改 `n_action_steps` 时复用当前最佳检查点直接评测；它不是训练预测 horizon。
其他参数变化时重新调用原训练器。三路相机、网络、优化器类型和动作契约保持固定。

## 启动

```bash
python -m autovla init --study outputs/autovla_smolvla
python -m autovla doctor --study outputs/autovla_smolvla
MUJOCO_GL=egl python -m autovla loop --study outputs/autovla_smolvla --rounds 0
MUJOCO_GL=egl python -m autovla loop --study outputs/autovla_smolvla --agent api --rounds 3
python -m autovla status --study outputs/autovla_smolvla
```

`--rounds` 是候选次数；首次还会自动跑基线。先单独跑基线便于确认训练/评测链路。
长任务可在 `tmux` 中启动；无需保持聊天窗口打开。程序每 30 秒提示日志位置。
非法提案、API 错误、OOM 或缺失评测立即停止，保留上一最佳，不在故障状态继续消耗 GPU。

初始化快照场景、随机化配置、研究规则和基线，并计算源码、物理资源、完整本地数据
与基座文件的 SHA256，下载 `.cache` 除外。每轮前后核验内容，变化即拒绝继续，须新建
study。大数据集核验需要磁盘读取时间。doctor 首次固定环境版本，后续要求一致。
外部 VLM/tokenizer 缓存仍由原下载 manifest 管理，不属于此工具的内容校验范围。
真实训练仍执行原数据审计；初始化另检查训练侧 seed 与开发/验收区间互不重叠。
不要修改 `seal.json` 绕过校验。修改模板不影响已经初始化的 study。

## 产物与评分

```text
outputs/autovla_smolvla/
  contract.json / seal.json / environment.json   固定输入与环境
  research_state.json                           可重建的简短状态
  results.csv                                   各轮状态、指标与耗时
  latest_failure_report.json                     最近开发失败统计
  best_config.json / best.json                   最佳配置、检查点与指标
  experiments/exp_0001/
    result.json / config.json                    权威日志及本轮配置
    proposal.json / proposal.usage.json           假设、改动和 API 用量
    train.log / train.json                       训练结果或检查点复用说明
    evaluate.log / evaluate.json                 全量逐 episode 开发结果
    evaluate.partial.json                        中途结果，仅供诊断
    failure_report.json / traces/                失败统计、动作和相机视频
```

KEEP 先比较完整整任务成功数，同分比较平均入盒数，完全相同则 DISCARD。
耗时/动作步数只报告。成功仍由现有仿真判定：目标 10 块、源盒 70 块、满足稳定条件
且无安全终止。缺失、重复或不符合 seed/case 的报告无效。

失败统计包括分档/列结果、超时和零/部分搬运。VLA 没有可靠专家 phase 标签，不声称
得到 grasp/place 成功率、毫米级误差或滑落率。预先指定的视频未必是失败 episode；
当前 API Agent 只读统计，自动视频诊断未实现。小开发集增益不代表统计显著提升。

## 恢复与最终验收

Ctrl+C 停止子进程并保存中断状态。重启 loop 从权威日志重建状态并从最佳继续，
不重跑已完成配置；失败/中断配置也视为已试。失败基线需排查后新建 study。
`kill -9` 或宿主崩溃无法保证清理子进程，重启前确认旧训练已结束。恢复未完成记录
时，启动至恢复的时间保守计入预算。单个 study 只能有一个控制器。

可手动提交同样的提案：

```json
{
  "hypothesis": "缩短执行块可能减少接触前累计偏差",
  "change": {"parameter": "n_action_steps", "value": 4},
  "expected_effect": "提高固定开发集整任务成功数",
  "finding": "主要观察到超时，需结合视频确认原因",
  "stop": false
}
```

```bash
python -m autovla submit --study outputs/autovla_smolvla --proposal my_proposal.json
MUJOCO_GL=egl python -m autovla accept --study outputs/autovla_smolvla
```

验收冻结最佳，核验检查点所有文件后运行隔离 seed。启动后禁止继续调参和重复验收；
失败也不自动重用 holdout。报告在 `acceptance.json`。查看结果后再研究须新建 study
并选择新验收 seed。API Agent 不会读取最终验收结果。

## 无权重演示

```bash
python -m autovla init --study outputs/autovla_demo --demo
python -m autovla loop --study outputs/autovla_demo --agent mock --rounds 3
python -m autovla status --study outputs/autovla_demo
python -m pytest -q tests/test_autovla.py
```

演示全是确定性合成指标，标记 `synthetic=true`，不训练、不调用 API，不是性能证据。
测试用模拟 HTTP 返回验证 API 提案到判定的链路；真实服务和 GPU 训练需在服务器验证。
