# RECAP-on-A3：完整实现计划

参照 `RLinf/RLinf` 的四阶段离线 RECAP 管线（`docs/source-zh/.../recap.rst`），适配本项目的
π0.5 + history 架构。本文档只描述**要做什么、为什么这样适配、怎么验证**，不重复 RLinf 的
算法细节。

状态：**阶段 0 采集器已交付并验证**（见 §11 的落地状态）。目标策略：
`a3_clean_v9_30000`（2x10 基线 13.18，严格成功 6.2 %）。

---

## 0. 判据先行

开工前先定死"成功"是什么，否则后面的每一步都会被一个好看的数字骗。

| 指标 | 基线（v9，n=80） | 目标 | 判定方式 |
|---|---|---|---|
| **严格成功率**（主指标） | 6.2 % (5/80) | ≥ 15 % | 种子 300–399，n=56 起 |
| 平均入盒数 | 13.18 | ≥ 15.0 | 同上 |
| 严格成功率的增益**不能**来自「更慢」 | — | 平均步数不增 | 记录 steps |

主指标取严格成功率而不是分数，理由见 §9.75：分数到 20 的回合里有 34 % 根本没过
`hold_steps`，76 % 的 2x5 满盒是槽位没占满。分数会骗人，`Status:` 不会。

**测试集隔离**：种子 0–111 已用于多轮消融对比，全部污染。新评测一律用 **300–399**。

---

## 1. 与 RLinf 的对照

| 阶段 | RLinf | 本项目 | 差异理由 |
|---|---|---|---|
| 1. Compute Returns | 每步 −1，成功末端 0，失败末端 `failure_reward=-300` | 同，但 `failure_reward = -2000` | 见 §2.2：−300 在 1500 步预算下会让早崩的失败轨迹回报高于慢成功轨迹 |
| 2. Value Model SFT | SigLIP2-400M + Gemma3-270M + critic expert(1M)，201 bin C51 | **复用 v9 的 PaliGemma（冻结）+ 新 value head** | 本地无 SigLIP2/Gemma3 权重且 `HF_HUB_OFFLINE=1`；v9 已有 `history` latent（R² 0.853），价值头能直接吃到 |
| 3. Compute Advantages | $A_t=\mathrm{norm}(\sum_{k<N}\gamma^kr)+\gamma^NV_{t+N}-V_t$，top 30 %，N=10 | 同，N=10 | 我们的 `execute=16`，N=10 ≈ 半个 chunk，与 RLinf 同量级 |
| 4. 条件化训练 | CFG：正样本用 positive prompt，负样本无条件，`unconditional_prob=0.1`，推理双前向外推 | **阶段 4A 文本后缀（零改动）→ 4B 加 dropout + CFG 外推** | 见 §5：文本后缀可由 `task_index` 实现，连 lerobot 都不用改 |
| 分布式 | Ray + FSDP 多卡 | 单卡 | 32 GB 单卡，`train_expert_only` 已经是当前唯一可行的 pi0.5 配置 |

**RLinf 的两处设计直接采用**：sidecar parquet（不改原数据集）、`tag` 版本化（每个产物带
参数指纹，避免"这个 advantage 是哪次算的"）。

---

## 2. 阶段 0：数据

### 2.1 现有数据不够，必须采 rollout

`a3_clean_v9h` 的 400 episodes **全部是成功轨迹**（accepted set）。纯成功数据上做 RECAP 等于
再做一遍 BC——价值函数学不到"什么状态会失败"，而这正是唯一的增量。所以第一阶段必须是采集。

| 项 | 值 |
|---|---|
| 采集对象 | v9 在 2x10 上的自主 rollout |
| 规模 | **300 episodes**（约 15 % 严格成功、45 % 分数 ≥ 15） |
| 并行 | **10 路**，共享同一个 bridge（`threading` + `lock`，已验证支持多客户端） |
| 单集预算 | 1500 步 |
| 预估耗时 | 300 / 10 × 700 s ≈ **6 h** |
| 必须开 | `--diagnostic-dir`（现在只有 38 个 `trace.jsonl`，几乎全是 v6，等于没有） |
| 字段 | `is_success`、`cookies_in_target`、`cookies_in_source`、逐帧 `observation.*` |

**关键改造**：`scripts/bench_pi05_on_main.py` 目前只写 `result.json`，不写 LeRobot 数据集。
需要一个新入口把 rollout 写成 LeRobot v3（复用 `src/a3_dual_arm_sim/collection.py` 的录制器），
**并且在同一循环里写 `trace.jsonl`**，保证一次 `env.step` → 一帧数据集 + 一条 trace，帧数严格
对齐（§2.4 的对齐要求）。

### 2.2 奖励：为什么是 −2000 而不是 −300

RLinf 用 −300 是因为 LIBERO episode 只有几百步。我们的 2x10 上限 **1500 步**，直接照抄会出
现：失败在 t=1 就崩 → $G_0\approx-301$；成功走满 1400 步 → $G_0=-1400$。**失败轨迹的回报
高于成功轨迹**，价值函数学出来是反的。

约束：对任意 $T_{\text{fail}}\ge1$ 和 $T_{\text{succ}}\le1500$，
$$L - T_{\text{fail}} < -T_{\text{succ}} \quad\Longrightarrow\quad L < T_{\text{fail}}-T_{\text{succ}} = 1-1500 = -1499$$
取 **`failure_reward = -2000`**，留 500 步余量。

**必须加断言**（阶段 1 的验收项）：
```
max(return over failed episodes) < min(return over successful episodes)
```
这一条不成立，后面全部作废。

### 2.3 但这个 return 不能直接当主价值目标（归一化失衡）

$L=-2000$ 满足了"不交叠"，却不满足"分辨率"。假设实测分布是：失败 $G_0\in[-3500,-2000]$、
成功 $G_0\in[-1500,-1200]$，按 RLinf 的 `(x-ret_min)/(ret_max-ret_min)-1` 映射到 $[-1,0]$ 后：

| | 归一化后的区间 | 占 201 个 bin |
|---|---|---|
| 失败 | $[-1,\ -0.35]$ | **约 131 个** |
| 成功 | $[-0.13,\ 0]$ | **约 26 个** |

成功区间只有 26 个 bin，分辨率约 0.038（相当于 1/26 个"成功度"），而这两个区间的边界正是
我们最需要分清的。C51 在失败区浪费 65 % 的容量去区分"输得多惨"，那对我们毫无价值。

**主方案改为：价值直接预测最终入盒数（0–20），不用 return。**

$$
V(o_t)=\mathbb{E}[\,\text{cookies\_in\_target at } T \mid o_t\,]
$$

| | return 方案（RLinf） | **最终计数方案（主方案）** |
|---|---|---|
| 值域 | 依赖 episode 长度，实测 $[-3500,-1200]$ | 固定 $[0,20]$ |
| 归一化 | 分段失衡（上表） | 均匀，201 bin 覆盖 0–20 |
| 失败轨迹标签 | 由 `failure_reward` 这个魔法数决定 | **直接就是最终计数**，不需要魔法数 |
| 与 judge 对齐 | 间接 | 直接（`cookies_in_target` 就是它） |
| 时间惩罚 | 有 | **无**（我们不追求吞吐，1500 步固定预算） |
| 优势含义 | "这一步之后还剩多少亏损" | **"这一步之后还能多放几块"** ← 正是放置阶段需要的 |

优势公式不变：$A_t = r_t + \gamma V(o_{t+1}) - V(o_t)$，其中 $r_t$ 改为**该步实际新增的饼干数**
（0 或 1，或释放瞬间的批量增加）。这样 $A_t>0$ 精确表示"这一步让最终计数变多了"。

**这是本计划相对 RLinf 的最重要偏离**，理由是可测量的：把 201 个 bin 的 65 % 花在"输得多惨"
上，而我们只需要知道"还能不能多放"。return 方案仍保留为 §6 的 A/B 对照（`--value-target return`），
但主表跑计数方案。

### 2.4 计数方案的标签已经现成（`trace.jsonl`）

`--diagnostic-dir` 写出的 `trace.jsonl` 每一行都有 `after` 字段，实测包含：

```
cookies_in_target, cookies_in_source, success, success_hold_count,
target_slot_occupancy, target_touches_all_walls, unmet_success_conditions, ...
```

所以计数方案**不需要 `compute_returns`、不需要 `failure_reward`、不需要归一化统计量**：

| 量 | 从哪来 |
|---|---|
| $r_t$ | `after.cookies_in_target[t+1] - after.cookies_in_target[t]`（整数 0/1/2/…） |
| $V$ 的标签 | 该 episode **最后一帧**的 `after.cookies_in_target` |
| `is_success` | 最后一帧的 `after.success` |
| 槽位占用（可选辅助头） | `after.target_slot_occupancy` |

**唯一的坑**：`trace.jsonl` 的 `observation_before` 只有 `eef_pose`/`force`/`state`/`velocity`，
**没有图像**。图像在三路 mp4 里。所以训练时必须把 trace（标签）与 LeRobot 数据集（图像）按
`(episode_index, frame_index)` 对齐，且两边帧数必须相等——采集器要保证一次 `env.step` 同时写
一条 trace 和一帧数据集。

**验收**：随机抽 5 个 episode，检查 `trace` 行数 == 该 episode 的 `length`，且
`after.cookies_in_target` 单调不减。

### 2.5 `is_success` 现在不存在，必须先补

实测 `a3_clean_v9h` 的 15 个 feature 里**没有 `is_success`，`meta/episodes/*.parquet` 里也没有**
（只有 `episode_index`/`tasks`/`length`/`stats/*`）。`compute_returns` 和新数据集的过滤都依赖它。

| 数据 | 怎么补 |
|---|---|
| 现有 400 条专家集 | 全部来自 accepted set → 置 `True`。**但要先跑一遍合约复核**：§9.75 记录了 11/10 的超额投递，说明来源判据可能与当前合约不一致 |
| 新 rollouts | 采集时写入，直接取 `EpisodeScore.success`（`workflows/episode_execution.py:322` 已有） |
| 新专家演示 | 同上，从 `result.json` 的 `success` 字段读 |

补列落在 `data/*.parquet` 和 `meta/info.json`（后者要加 feature 声明，否则 lerobot 加载会报
未知列）。这一步是阶段 0 的硬前置。

### 2.6 阶段 0 验收

- [ ] 300 条 rollout + 100 条 expert 写入 LeRobot v3，`is_success` 列存在
- [ ] 成功/失败回报区间不重叠（§2.2 的断言）
- [ ] 训练集 / 测试集种子不重叠（rollout 用 0–111 之外的，评测用 300–399）
- [ ] 数据能喂进现有 `train_pi05_a3.sh`（回归：跑 100 步不报错）

---

## 3. 阶段 1：compute_returns

**若走计数方案（主方案），本阶段可跳过**——标签在 `trace.jsonl` 里已经现成（§2.4）。
本节只在跑 return 对照臂时需要，或作为完整实现的参考。

新文件 `scripts/recap/compute_returns.py`，输出 sidecar：`meta/returns_{tag}.parquet`
（列：`episode_index`、`frame_index`、`return`、`reward`、`is_success`、`prompt`）。

```
r_t = -1            for t < T-1
r_{T-1} = 0         if is_success
r_{T-1} = -2000     otherwise
G_t = r_t + G_{t+1}                 (gamma = 1.0，与 RLinf 一致)
```

**不修改原 parquet**（RLinf 的 sidecar 设计，理由是可回溯、可重算）。

`tag` 命名带参数指纹：`fail2000_g1.0`。

### 验收
- [ ] `G_0` 的分布按 `is_success` 分成两簇，无交叠
- [ ] 成功轨迹的 $G_t$ 单调递增（每步 −1）；
- [ ] 数值检查：$G_0 = -(T-1)$ 对成功轨迹成立（容许末步 0 的一步误差）

---

## 4. 阶段 2：价值模型

### 4.1 架构选择：复用 v9，不新建

```
输入 ─┬─ 3 相机 ──┐
      ├─ 语言 ────┼─→ v9 的 PaliGemma（冻结）+ history encoder（冻结）
      └─ history ─┘         ↓
                     prefix pooled (2048)
                            ↓
                    LayerNorm → MLP(2048→512→201)
                            ↓
                      C51 分布 → 期望值
```

只训 value head（约 **1.05 M** 参数）。C51 常量照 RLinf：`NUM_BINS=201`、`V_MIN=-1`、
`V_MAX=0`、two-hot 交叉熵。

**为什么冻结 VLM**：现有训练脚本已经是 `--policy.train_expert_only=true`（PaliGemma 全冻结），
解冻会撞 32 GB 显存墙（全量微调约 28 GB，加价值头就爆）。而且冻结能让价值模型读到的表示与
策略一致——我们要估的是"这个策略会怎样"，不是"一个独立模型怎么看"。

**为什么加 history**：`history` latent 对 progress 的 R² 是 0.853、corr 0.927（v9，§9.45）。
RLinf 的价值模型只有图像+语言，它会缺"已经做了几批"这个信息，而我们的任务恰恰是四批次的。
这是本项目相对 RLinf 的**唯一结构性优势**，必须用上。

**禁止使用 `observation.progress`。** 数据集里有这一列（2 维），但它是 v5 遗留的**特权通道**——
直接从仿真读饼干计数（`bench_pi05_on_main.py:389` 的 `_state_with_progress` 就是把它当特权量
重算出来喂给 v5 的）。价值模型一旦用它，训练时 Spearman 会很漂亮，部署时那个通道是零，
价值判断整体失效，而 advantage 错标是**静默的**——没有任何地方会报错。

价值模型的输入清单，逐项都必须在 bridge 推理时可得：

| 输入 | 推理可得 | 备注 |
|---|---|---|
| 3 相机 | ✅ | 与策略同源 |
| 语言 | ✅ | 带优势后缀 |
| `observation.history`（480 维） | ✅ | 由本机轨迹维护，不含特权量 |
| `observation.state`（16） | ✅ | 机器人自身关节 |
| `observation.progress`（2） | ❌ **特权** | 禁用 |
| `cookies_in_target` 等 | ❌ **特权** | 禁用 |

验收时加一条**泄漏测试**：把 `progress` 列整列置零，Spearman 变化应 < 0.02；若明显下降，
说明某处仍在读它。

归一化：计数方案直接把 $[0,20]$ 线性映射到 $[-1,0]$（`V = x/10 - 1`），无需数据集统计量。
若走 return 对照方案，则用数据集 min/max（RLinf 的 `normalize_to_minus_one_zero`）。

### 4.2 训练

| 项 | 值 |
|---|---|
| 数据 | 400 专家 + 400 rollout = 约 **60 万帧** |
| batch | 32 |
| epoch | 3–5（约 55 K–95 K 步） |
| lr | 1e-4（只训 head，可以激进） |
| 显存 | 约 12 GB（VLM 冻结 + 无梯度） |
| 预估耗时 | **1.5–3 h** |
| 验证 | 留出 20 % episodes（**按 episode 切，不按帧切**），报 Spearman |

### 4.3 验收（这一步是整条管线的命门）

- [ ] **`eval/spearman` ≥ 0.6**。低于 0.5 说明价值函数排不出好坏，后面全部无意义，**必须停下**。
- [ ] 价值在成功轨迹上单调上升（看几张 episode 的曲线，与 π 博客的示意图对照）
- [ ] $V(o_0)$ 在成功/失败回合上的均值差 ≥ 0.2
- [ ] 无崩溃：$V$ 的输出 std 在 0.05–0.4（塌缩到常数或发散都算失败）

---

## 5. 阶段 3：compute_advantages

```
A_t = normalize(Σ_{k=0}^{9} r_{t+k}) + V(o_{t+10}) - V(o_t)
threshold = percentile(A, 70)          →  top 30 %
```

原子（atom）索引换算：`advantage_{tag}` 里存 `(episode_index, frame_index) → bool`，作为
sidecar `meta/advantages_{...}.parquet`，**不改原数据集**。

`tag` 编码全部参数：`{returns_tag}_N10_q30_ckpt{step}`。

**人工接管帧强制 positive**（民间版 `label_advantage_from_vlm.py` 的做法）。我们现在还没有
接管数据，但 §7 的 DAgger 修复落地后就适用——届时 `intervention=1` 的帧直接标正，不看 $A_t$。

### 验收
- [ ] 正样本率 = 30 % ± 1 %（阈值定义保证）
- [ ] 正样本在**成功轨迹**里的比例 > 在失败轨迹里的比例
- [ ] 正样本在时间上不均匀（应当集中在每批次放置的后半段），若完全均匀说明 $V$ 没学到东西

---

## 6. 阶段 4：优势条件化训练

### 6.1 关键设计：用 `task_index` 实现条件，不改 lerobot

我们的数据集已经有 11 个 `task_index`（0–9 是 2x10 的 11 个 plan 变体，10 是 2x5）。
文本后缀方案可以完全落在现有机制里：

| 变体 | `tasks.parquet` 新增行 | 用途 |
|---|---|---|
| 无条件 | 原句（0–9） | CFG 的 uncond 分支 / 阶段 4A 的负样本 |
| positive | 原句 + `. Advantage: positive`（20–29） | 正样本 |
| negative | 原句 + `. Advantage: negative`（30–39） | 仅阶段 4A 用 |

训练时只需把 data parquet 的 `task_index` 按 advantage 重映射。**零代码改动**。

### 6.2 两个子阶段

**4A：文本后缀（先跑通，1 天）**

| 项 | 值 |
|---|---|
| 正样本 prompt | `+ ". Advantage: positive"` |
| 负样本 prompt | `+ ". Advantage: negative"` |
| 初始化 | v9（不是 pi05_base） |
| 步数 | 10 K |
| lr | 1e-5（比 v9 的 2.5e-5 低，避免覆盖已有能力） |
| 预估 | **3–4 h** |

推理：只用 positive 后缀，单前向，成本与现在相同。

**4B：CFG（完整版，再加 1–2 天）**

| 项 | 值 |
|---|---|
| 训练路由 | 正样本 → positive prompt；负样本 → 原始 prompt；正样本再以 `unconditional_prob=0.1` 降级为原始 |
| 推理 | 双前向 + 外推 $v=(1-w)v_{\text{uncond}}+w\,v_{\text{cond}}$，$w$ 扫 1.0 / 1.5 / 2.0 |
| 成本 | 推理 **2×**（两份 KV cache）；显存 +约 1 GB |

4B 的架构改动只在**推理侧**和数据集构造，模型本身不动——因为我们用的是"两条不同的
prompt"，不是新增条件通道。

### 验收
- [ ] 4A 在测试集 300–399 上严格成功率 > 基线（配对，n≥40）
- [ ] 正样本率 30 % 的训练没有让**负样本行为变差**（检查 ≤5 分的比例不上升）
- [ ] 4B 的 $w$ 扫描给出单调趋势；若 $w>1$ 反而变差，记录并放弃 CFG（保留 4A）

---

## 7. 与 DAgger 的协同（可选，但收益最大）

§9.65 的算术说明整链条 DAgger 的天花板是 15–24 %，但**单批次干预**（`adopt`，已验证 4/4）
可以到 40 %+。两者结合的正确顺序：

1. 先修 `collect_graft.py` 的触发（`stall` → "已抓起一批即接管"）
2. 采 100 条接管轨迹，`intervention=1`
3. 这些帧**强制 positive**（不看 $A_t$），且天然落在策略真正会失败的状态上
4. 混进 4A 的训练集

这一步比调 advantage 阈值更有价值，因为它同时解决了"数据分布"和"标签质量"两件事。

---

## 8. 预算总表

| 阶段 | 耗时 | 显存 | 阻塞项 |
|---|---|---|---|
| 0. 采集 | 6 h（rollout）+ 4 h（expert） | 10 GB/bridge | 需写 LeRobot rollout 录制器；**需补 `is_success` 列**（现数据集没有） |
| 1. returns | 10 min | — | 无 |
| 2. value SFT | 1.5–3 h | ≈12 GB | 需 value head 实现 |
| 3. advantages | 0.5–1 h（60 万帧推理） | ≈12 GB | 无 |
| 4A. 后缀训练 | 3–4 h | ≈26 GB | 无 |
| 4B. CFG | 3–4 h + 推理 2× | ≈27 GB | 无 |
| 评测（每臂 56 种子） | 11 h | 10 GB | 无 |
| **合计（不含评测）** | **约 2 天** | | |

**关键：显存必须串行。** π0.5 训练要 26 GB，价值模型要 12 GB，bridge 要 10 GB。当前两条
benchmark 臂跑完（约 21:00）后才能开始采集，且训练期间不能同时开 bridge。

---

## 9. 风险与回退

| 风险 | 概率 | 症状 | 回退 |
|---|---|---|---|
| 价值函数 Spearman < 0.5 | **中** | 阶段 2 验收失败 | ①核对是否误用 `progress` 特权列（先跑泄漏测试）②加大 value head 容量 ③换独立小模型（需先解决权重下载）④退回 return 目标做 A/B |
| 价值模型读了特权通道 | 中 | Spearman 虚高、部署后 advantage 全错，**且无报错** | 强制泄漏测试（§4.1）与输入白名单，`progress` 不得出现在 input_features |
| 正样本只有 30 %，且集中在少数简单回合 | 中 | 正/负样本的 `cookies_in_target` 分布几乎不重叠 | 提高阈值到 50 %，让正负更接近；或按 episode 分层采样 |
| 10 K 步微调覆盖了 v9 的能力 | 中 | 分数上升但严格成功率下降 | lr 降到 5e-6；加 KL 锚定到 v9（我们的 `A3_HISTORY_KL_WEIGHT` 可以复用，锚的是 latent 不是策略，需另做） |
| CFG 外推发散 | 低 | $w>1$ 时动作幅度异常 | 只用 4A |
| 采集的 rollout 质量太差（85 % 是低分） | 中 | 负样本全是"完全没动静" | 分层采集：先用 `stall` 触发保证每集至少动起来 |

**最可能失败的地方是阶段 2**。如果 Spearman 上不去，说明"用 v9 的冻结表示 + 小 head"不足
以判断好坏，那时才需要 RLinf 那套独立价值 VLM。计划里把这道门设在价值模型之后，就是为了
在花掉 8 小时训练之前先知道答案。

---

## 10. 明确不做的事

| 不做 | 原因 |
|---|---|
| 在线 RL / PPO | 1200 episodes/天，horizon 94（按 16 步块），样本效率不够 |
| 从 `pi05_base` 重训 | 10 h/30 K 步，且 4A 的增益不需要重训；续训 v9 才有对照 |
| 训独立价值 VLM（SigLIP2+Gemma3） | 本地无权重、`HF_HUB_OFFLINE=1`、网络拉不动；留作阶段 2 失败后的备选 |
| 用 `-300` 作为失败奖励 | §2.2 的算术：会让失败轨迹回报高于慢成功轨迹 |
| 用归一化 return 当**主**价值目标 | §2.3 的实测：成功区间在 201 bin 里只剩 26 个，65 % 容量浪费在"输得多惨"上 |
| 让价值模型读 `observation.progress` | 特权通道，部署时为零，advantage 会静默全错（§4.1 泄漏测试） |
| 用分数当主指标 | §9.75：34 % 的满盒没过 hold，76 % 的 2x5 满盒槽位没占满 |
| 在种子 0–111 上评测 | 已被多轮消融使用，污染 |

---

## 11. 代码落地清单

状态图例：✅ 已交付并验证，⬜ 未开工。

```
scripts/recap/
  collect_rollouts.py       ✅ 阶段 0 采集器（LeRobot v3 + trace，失败保留）
  run_collect.sh            ✅ 多进程并行编排（共享一个 bridge，种子轮转分配）
  build_value_labels.py     ✅ 逐帧 reward + 价值目标（计数方案）+ 不变量检查
  merge_parts.py            ⬜ 合并 part_*（LeRobot create 拒绝多写者）
  compute_returns.py        ⬜ 仅 return 对照臂需要
  train_value.py            ⬜ 阶段 2
  compute_advantages.py     ⬜ 阶段 3
  build_task_variants.py    ⬜ 阶段 4：tasks.parquet 变体 + task_index 重映射
scripts/recap/tests/
  test_collect.py           ✅ 15 项（含 a_previous 双向钉死、progress 防漂移）
```

### 已交付的三个文件，各自的非显然点

**`collect_rollouts.py`** — 自己写循环而不是调用 `bench.run_episode`，原因有三，前两条是 bug：

1. `episode_execution.run_episode` 从不调用 `policy.observe_step`，于是 history 的
   `a_previous` 通道整集全零，而训练数据里是真实的上一步动作（§9.76）。
2. 它在失败时调 `recorder.discard_episode()`，而 RECAP 要的正是失败。
3. trace 必须与数据集帧在同一循环迭代内写出，否则两者会漂移（§2.4）。

`RecapRecorder` 不是 `LeRobotV3Recorder` 的子类：后者的 `features` 硬编码了
`lerobot_features()`，没有 `observation.history`/`.progress`，产出的数据集无法与
`a3_clean_v9h` 混训。冒烟验证：12 个数值列 + 3 个视频 = 15 features，与训练集逐名一致。

`_ProgressTracker` 是 policy 内那条规则的**副本**，因为原函数非幂等（会推进投递计数）
且只在每 16 步的 chunk 里跑一次，而数据集要每帧一个值。副本必然漂移，所以
`test_progress_tracker_matches_policy` 用一条 400 帧的合成轨迹把两者的输出钉死。

**`run_collect.sh`** — 每个 worker 一个独立 `part_XX` 目录（`LeRobotDataset.create` 独占
root），种子按 `i % workers` **轮转**分配而不是连续分块：2x10 的指令变体由 seed 决定
（`Cookie2x10Plan.from_seed`），连续分块会把某个 plan 变体整个塞进一个 part，中途收工时
各 part 的指令分布就不平衡了。

**`build_value_labels.py`** — 输出 `value_target = cookies_in_target at T`，`rewards[t]`
是逐帧计数差。`--check` 强制四个不变量：trace 帧数 == label 的 steps、末帧计数 == judge 的
计数、success 一致、计数单调不减。最后一条不是形式主义：计数下降意味着有饼干离开盒子，
那么 `A_t` 会把「离开」算成负奖励，语义就变了。

每个 `*.py` 只做一件事，纯逻辑进 `learning/recap/`，I/O 留在 `scripts/recap/`。这样
`test_returns.py` 里的反例（−300 必须让断言失败）能在毫秒内跑完，不必等采集。
