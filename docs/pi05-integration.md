# π0.5 接入与微调

**日期**：2026-09-29
**零样本清单**：`configs/policies/pi05_base.yaml`
**微调清单**：`configs/policies/pi05_a3_finetuned.yaml`
**适配器**：`/root/autodl-tmp/a3-collect/adapters/pi05_a3.py`
**训练脚本**：`scripts/train_pi05_a3.sh` + `scripts/train_pi05_launcher.py`

---

## 一句话结论

π0.5 已完整接入、训练完成（20,000 步，loss 0.008）、已评测。过程中发现并修复了 **两个**缺陷，两者都属于"看起来像模型不行，其实是接线错了"：

1. **数据集统计量缺陷**：`action` 第 8–14 维的 `q01 == q99`（真实 span 0.078 rad）。π0.5 用 QUANTILES 归一化 → 除以 eps → **训练 loss 1.2e12**。
   **只有分位数塌陷、`min`/`max` 正常**，所以 GR00T（用 MIN_MAX）完全不受影响——这个缺陷在本项目里只有接 π0 系列才可能暴露。
2. **适配器缺陷**：`dataset_stats=` 被静默忽略 + state 被填充到 32 维。零样本数字因此**错了 12 倍**（`corrected` 0.345 实测应为 0.028）。

修完之后，π0.5 base 是全部四个模型里行为最规矩的之一，微调后是**唯一移动了饼干的**。

但**任务仍未完成**：`cookies_in_target` 全为 0。诊断显示微调检查点的单次预测**在自我抵消**（净位移只有总行程的 4%，示范中位数是 25%）——与 GR00T 是同一类失败，更严重。

---

## 1. 选型：为什么是 `pi05_base` 而不是 `pi05_libero_base`

| | 相机 | state | action | 适配 A3 |
|---|---|---|---|---|
| `lerobot/pi05_base` | 3：`base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` | 32（填充） | 32（填充） | ✅ 相机数一致 |
| `lerobot/pi05_libero_base` | 2 | 8 | 7 | ❌ 单臂，不匹配 |

`pi05_base` 是 openpi 的通用具身配置，三个相机正好对应 A3 的「第三人称 + 双腕」。检查点里 `pi05_robocasa` 的 state/action 是 **16/12（未填充）**，说明 lerobot 的 `pad_vector` 是**尾部填充**——所以 A3 的 16 维放在索引 0–15。

**必须说清的限制**：`pi05_base` 是**微调底座**，不是可部署策略。它的模型卡写明 "Intended use: Base model to fine tune on your specific use case"，动作空间 32 维且 `action_names: null`——没有任何已发布的 π0.5 权重训练过双臂 16 维关节布局。所以零样本数字是**接线测试**，不是能力测试。

---

## 2. 三个集成陷阱（报错都不指向真因）

### 2.1 tokenizer 是 gated 仓库

检查点的 `policy_preprocessor.json` 指定 `google/paligemma-3b-pt-224`，**该仓库需要授权**。报错是 `OSError: You are trying to access a gated repo`，出现在构建处理器管线时，离"tokenizer"很远。

两个场景要走不同的解法，这是关键：

- **推理**：适配器直接传本地目录（`leo009/paligemma-3b-pt-224` 的非 gated 镜像），绕开 gated 名字。
- **训练**：`lerobot_train.py` **自己**构建管线，tokenizer 名来自检查点 JSON，**没有 CLI 钩子**。所以必须让**名字本身**能解析——`scripts/setup_paligemma_tokenizer_cache.py` 在 HF 缓存里为 gated 名字建一个条目，指向镜像的同一批文件。

另外一个坑：即使缓存条目建好，**联网时**它仍会去要 `chat_template.jinja`（镜像没有这个文件）→ 403。所以训练脚本设 `HF_HUB_OFFLINE=1`，离线时该请求不会发出。

> π0.5 原样使用 PaliGemma 的 tokenizer，所以同一批文件的副本是**等价**而非近似。

### 2.2 检查点的 `device_processor.device` 是 `cpu`

不覆盖的话，语言 token 留在 host 而权重在 GPU，报的是 `F.embedding` 里的 device mismatch——**看起来像模型 bug，实际是 JSON 里的一个字段**。

### 2.3 处理器步骤被改名

检查点存的是 `relative_actions_processor`，本版 lerobot 注册名不同。报 `ImportError: Failed to load processor step from registry`——**看起来像缺依赖**。

解法是 `scripts/train_pi05_launcher.py` 在转交前注册别名（`setdefault`，幂等）。安全因为该步在检查点里 `enabled: false`：类会被实例化但什么都不做，所以**只有名字需要匹配**。

---

## 3. 数据集统计量缺陷（本次最重要的发现）

### 3.1 症状

第一次训练 **第 1 步 loss = 1.2e12**，梯度范数 3.5e7。这不是发散——还没有东西可以发散。

### 3.2 根因

`meta/stats.json` 里 `action` 第 8–14 维（右臂 7 关节）存的是：

```
q01 == q99 == mean        span 恰好 0
```

而真实 span 是 **0.078 rad**。π0.5 用 QUANTILES 归一化：

```
x_norm = 2 * (x - q01) / (q99 - q01) - 1
```

分母被 `eps = 1e-8` 兜住 → 0.02 rad 的偏差被放大成 ~4e6 → loss 1e12。

### 3.3 为什么一直没被发现

**只有分位数塌陷，`min`/`max` 是正常的**：

| 维度 | max − min | q99 − q01 |
|---|---|---|
| 0–7（左臂） | 0.87 – 1.65 | 0.52 – 0.76 |
| **8–14（右臂）** | **0.0796（正常）** | **0.00000（塌陷）** |
| 15（右夹爪） | 0（真恒定） | 0（真恒定） |

- **GR00T 微调用 MIN_MAX** → 除以 `max − min` → **完全不受影响**，所以它的 loss 能正常降到 0.0123
- **π0.5 用 QUANTILES** → 除以 `q99 − q01` → 爆炸

所以这个缺陷在 π0.5 之前**不可能**被这个项目发现。

### 3.4 成因

右臂在**回合内恒定**、**跨回合**才变（`arm_home_rad` ±0.04 rad 抖动）：

```
action dim 8 的唯一值数量 = 200        (= 回合数)
action dim 8 的 std       = 0.0226    (跨回合的真实变化)
```

只有 200 个不同的值分散在 110,485 帧里，外层聚合代码把分位数算成了均值。

### 3.5 识别信号（值得记住）

> **存储的 `q01` 和 `q99` 严格相等，不是"分布很窄"，而是"统计量缺失"。**

真实数据几乎不可能给出严格相等的分位数对。

### 3.6 修复

新增 `src/a3_dual_arm_sim/statistics.py:recompute_quantiles()`：

- 只读 `action` / `observation.state` 两列 parquet，几秒完成（lerobot 自带的 `augment_dataset_quantile_stats` 会遍历视频并在最后 push 到 Hub，两者都不适用）
- 首次写入前备份 `meta/stats.json.bak`
- 报告区分**"缺失"**（存储 span ≈ 0 但实际有变化 → 修复）和**"真恒定"**（唯一值 = 1 → 保持不动）

两个实现细节：
- **判定用阈值不用 `<= 0`**：`action` 存的是精确 `0.0`，而 `observation.state` 存的是 `1.5e-9`（两条聚合路径不同），但两者是**同一个缺陷**。`DEGENERATE_SPAN = 1e-3`——本项目最小的真实跨度是 0.075 rad，是阈值的 75 倍。
- **`np.percentile` 必须显式 `axis=0`**：默认 `axis=None` 会**先展平**，逐维分位数静默变成"所有维度混在一起的一个标量"。

已接入 `train_pi05_a3.sh` 做预检，幂等。

### 3.7 修复效果

| | loss | 梯度范数 |
|---|---|---|
| 修复前 | 1.2e12 | 3.5e7 |
| 修复后 | 0.05 – 0.38 | 1 – 3 |

---

## 4. 训练配置

### 4.1 `train_expert_only` 不是优化，是前提

π0.5 = PaliGemma-2B（~2B）+ 动作专家（~300M）。**全量微调放不下**：

| 项 | 大小 |
|---|---|
| bf16 权重 | 4.6 GB |
| 梯度 | 4.6 GB |
| AdamW fp32 动量 + 主权重 | ~18 GB |
| **小计（激活之前）** | **~28 GB** |
| 可用 | 32 GB |

`--policy.train_expert_only=true` 冻结整个 PaliGemma（`self.paligemma.eval()` + 所有参数 `requires_grad = False`），只训动作专家和投影层。这也是 openpi 作者推荐的 π0.5 适配方式。

**实测**：

| | 值 |
|---|---|
| 可训参数 | **693M** |
| 总参数 | 4.14B |
| 峰值显存 | **15.7 GB** |
| 速度（batch 8） | **1.03 step/s** |
| 20,000 步耗时 | **~5.3 小时** |

### 4.2 另外两个必须加的参数

- **`--policy.push_to_hub=false`**：默认 `True`。失败不在开头而在**训练成功结束之后**——`OfflineModeIsEnabled` 让进程退出码为 1，看起来像训练失败。
- **`--dataset.video_backend=pyav`**：`get_safe_default_codec()` 只检查 `importlib.util.find_spec("torchcodec")`，而这里 torchcodec 装了但**动态库加载失败**（`libavutil.so.58`，容器的 FFmpeg soname 不同）。症状是 DataLoader worker 崩溃 + 一屏 FFmpeg 版本噪音。pyav 也是唯一能读本数据集 AV1 编码的后端。

### 4.3 `rename_map`：数据集键 → 策略键

```
observation.images.front         → observation.images.base_0_rgb
observation.images.left_wrist    → observation.images.left_wrist_0_rgb
observation.images.right_wrist   → observation.images.right_wrist_0_rgb
```

`lerobot_train.py` 同时用它改策略特征键和批次键，所以是 flag 而不是改数据集——重写 26 GB 视频只为改三个字符串，还会毁掉 GR00T 仍能读这个数据集的价值。

### 4.4 检查点与磁盘

- 每个检查点 **~11 GB**，lerobot **没有** `save_total_limit` → 10 个就 110 GB，而可用 92 GB
- `save_freq=2000` 保留中间检查点（长微调的判断依据是评估中间结果，不是 loss 曲线）
- `scripts/prune_checkpoints.sh` 后台保留最新 3 个，带 **120 秒 guard** 防止删掉正在写的

**踩到的坑**：检查点目录名是**裸的 `002000`**，不是 `checkpoint-002000`。写 `-name 'checkpoint-*'` 的 glob 永远匹配不到 → 清理器每轮都"成功"却什么都没删，磁盘会在某个小时静默填满。另外**改后台守护的脚本必须重启守护**（bash 启动时就解析了脚本内容）。

---

## 5. 适配器缺陷二：统计量被静默忽略（零样本数字因此错了 12 倍）

第一次零样本评测跑出 `corrected 0.345`、`saturated 0.089`、`motion 0.0362`（示范速率的 11 倍）。**这三个数字测的是适配器 bug，不是模型。**

### 5.1 根因一：`dataset_stats=` 被忽略了

`make_pre_post_processors(cfg, repo, dataset_stats=...)` 看起来是官方接口，**但在管道配置里已有 `normalizer_processor` 步骤时它是无效的**——工厂会从检查点的 JSON 构建该步骤，参数永远到不了 `_tensor_stats`。

症状不是报错而是**缺失**：归一化根本没发生。于是
- 未归一化的弧度（A3 关节到 ±1.6）被离散成 256 bin 写进语言提示 → bin 索引越界 → 提示不是模型训练过的那个
- 模型输出被当作物理关节值直接使用，**没有反归一化**

两者都表现为"模型动作怪"，而不是"适配器坏了"。

正确做法是注入到**步骤自己的 override**（`lerobot_train.py` 就是这么做的，微调检查点也因此拿到了统计量）：

```python
preprocessor_overrides["normalizer_processor"] = {
    "stats": dataset_stats,
    "features": {**policy.config.input_features, **policy.config.output_features},
    "norm_map": policy.config.normalization_mapping,
}
```

### 5.2 根因二：state 被填充到了 32 维

检查点声明 `observation.state` 是 32 维，适配器（和文档）因此把 A3 的 16 维**尾部填充**到 32。这错两次：

- **宽度必须匹配统计量**，因为归一化器要和张量广播。检查点自带的 state 统计量是 **16** 维，所以 32 维张量直接崩：`The size of tensor a (32) must match the size of tensor b (16)`。声明的宽度是**容量**，不是形状。
- **π0.5 没有 state projector**。state 只以空格分隔的整数出现在语言提示里，填充等于往句子里塞了 16 个虚构的关节值。

### 5.3 修正后的零样本结果（这才是模型）

| 指标 | 修复前（bug 读数） | **修复后** | 变化 |
|---|---|---|---|
| `corrected` | 0.345 | **0.028** | **↓ 12×** |
| `rate limited` | 0.320 | **0.028** | ↓ 11× |
| `saturated` | 0.089 | **0.000** | ↓ 至零 |
| `correction max` (rad) | 1.070 | **0.085** | ↓ 13× |
| `motion mean` (rad) | 0.0362 | **0.0091** | ↓ 4× |

**教训**：一个"模型表现不好"的结论，如果它是在一个从未被独立验证过的管线上取得的，那它可能就是管线自己的读数。这次差点把适配器的 bug 写成 π0.5 的能力缺陷。

---

## 6. 全模型对比

`a3_cookie_same_column`，seeds 0/1/2，400 步，`--visual-style plain`，`execute: 16`。

| 指标 | π0.5 base | π0.5 微调 | GR00T 零样本 | GR00T 微调 |
|---|---|---|---|---|
| `corrected` | 0.028 | **0.007** | 0.396 | 0.0225 |
| `rate limited` | 0.028 | **0.004** | 0.408 | 0.0209 |
| `saturated` | 0.000 | **0.000** | 0.524 | 0.000 |
| `correction max` (rad) | 0.086 | **0.055** | 0.227 | 0.061 |
| `motion mean` (rad) | 0.0091 | 0.0067 | 0.0399 | 0.0081 |
| **源盒变化** | 80 → 80 | **80 → 78** | 80 → 80 | 80 → 80 |
| `cookies_in_target` | 0 | 0 | 0 | 0 |
| 推理 (ms) | 277 | 236 | 175 | 171 |
| 启动 (s) | 117 | 110 | 18 | 16 |

**怎么读**：

- **示范的自然速率是 0.0032 rad/步**。所有四个模型都高于它（0.007–0.040），但 π0.5 两个都只高 2–3 倍，GR00T 零样本高 12 倍
- **只有 π0.5 微调移动了饼干**（seed 2 源盒 80 → 74，即 6 个离开了原位）。GR00T 微调三个种子全是 80——一个都没碰
- π0.5 微调是**唯一 `corrected` 低于 0.01** 的
- **π0.5 base 修正后与 GR00T 微调相当**（0.028 vs 0.0225）——一个微调底座在没有归一化问题的前提下，行为规矩程度已经接近另一个项目专门后训练过的权重
- **仍不是"π0.5 vs GR00T"的对比**：GR00T 零样本是可部署权重，π0.5 是微调底座，可比之处只有"接线是否正常"

---

## 7. 微调检查点的诊断

### 7.1 开环精度（训练集上）

`scripts/eval_pi05_openloop.py`，192 个 16 步 chunk，基线在**不参与评分的** 160–199 号回合上拟合：

| 预测器 | MSE | 相对 `mean` |
|---|---|---|
| **模型** | **0.00167** | **0.06×** |
| `hold` | 0.00347 | 0.12× |
| `mean` | 0.02822 | 1.00× |

模型比 `mean` 好 **16.9×**、比 `hold` 好 **2.08×**。分块看右臂 MSE 0.00002、右夹爪 0.00000——与 GR00T 完全相同的模式，原因也相同：右臂跨回合只有 200 个值、回合内恒定，预测它是平凡的。

**最难的帧上 `model mse ≈ hold mse`**（0.0779 vs 0.0767），说明难度全在左臂。

### 7.2 闭环：同一类失败，更严重的震荡

用 `A3_PI05_TRACE` 拿到 25 次请求的全部输入与输出，重建实际执行轨迹（**注意：trace 记的是完整 50 步 chunk，而 `execute: 16`——必须只取前 16 步**，我第一次用 50 步拼路径得到了错误结论）。

与 8 个示范回合对比：

| | 示范 | 模型 | 比 |
|---|---|---|---|
| 活动范围合计 | 5.282 | 2.559 | **0.48×** |
| 总行程合计 | 13.3 | 15.3 | **1.15×** |

**行程够、范围只有一半** → 在小包络里来回摆动。与 GR00T 同类（0.85× / 0.63×）但更严重。

### 7.3 根因：模型自己的单次预测在自我抵消

关键指标是 **chunk 内部的 净位移 / 总行程**：

| | 净/行 |
|---|---|
| 示范的 50 步滑动窗口（174 个） | 中位数 **0.254**，p10 0.168 |
| **模型** | **0.040** |

模型落在示范分布的 **2.3% 分位**。也就是说：**模型一次前向预测出的 50 步轨迹，走了一圈又回到原地**（净位移只有总行程的 4%）。

所以**不是**重规划冲突（chunk 之间的跳跃只有 0.01–0.06，很小），**是模型自己的单次预测就是自相消的**。

一个附带观察：`call 0` 的净/行是 0.195（和示范相当），之后骤降到 0.01–0.07。**第一次预测在真移动，之后就只在原地抖。**

### 7.4 夹爪

| | 低点 | 高点 | 行程 |
|---|---|---|---|
| 示范 | **0.282** | 1.000 | 2.0 |
| 模型 | 0.309 | 1.020（仅起始） | 2.6 |

模型的开合**范围**和示范相当，但第 16 步闭到 0.31 后，**再没回到 1.0**，只在 0.31–0.53 之间震荡。示范每个批次走一次 `1.0 → 0.28 → 1.0` 的完整循环。

**与 GR00T 一模一样的失败模式**：闭合之后不再张开，一个完整的抓取-释放周期都走不完。

### 7.4 完整 chunk：`execute: 50` 的实测（结果与预测相反）

既然模型预测 50 步而只执行 16 步，自然该问完整 chunk 是否更好。**答案是：任务结果完全相同，但机制变了。**

| | `execute: 16` | `execute: 50` |
|---|---|---|
| 请求数 | 25 | 8 |
| `corrected` | 0.007 | 0.008 |
| 源盒 / 入盒 | **78 / 0** | **78 / 0** |
| 左臂 范围比 | 0.46 | 0.57 |
| **左臂 行程比** | 1.11 | **2.66** |
| chunk 间跳跃（均值） | 0.030 | **0.058** |
| 相邻 chunk 方向翻转 | 19/24 | **22/23** |

**行程比从 1.11 涨到 2.66，而范围只从 0.46 涨到 0.57**——只多覆盖 24% 的范围，却多走了 2.4 倍的行程。**整体效率反而更差。**

而且这暴露了真正的机制：

- `execute=16` 的 chunk 内 net/travel 是 **0.040**（自相消），但 chunk 之间跳跃小（模型每步小幅修正）
- `execute=50` 的 chunk 内 net/travel 是 **0.179**（更连贯），但 chunk 之间跳跃翻倍

也就是说：**震荡发生在 chunk 之间，不在 chunk 之内。模型每次重新观测就掉头**——所以震荡周期绑定的是**重规划间隔**，不是 chunk 长度。延长 commit 时间只是把掉头间隔从 16 步拉到 50 步，来回的幅度也跟着变大。

> `execute: 16` 那次的 net/travel 曲线有个可疑形态：call 0 = 0.195（正常），之后骤降到 0.014–0.071 再没恢复。`execute=50` 那次全程 0.09–0.26，没有这个骤降。**同一次运行内从连贯变不连贯**，这不是随机噪声能解释的。

### 7.5 决定性实验：采样器不是原因

既然方向在 chunk 之间翻转，最直接的问题是：**固定一个观测多次采样，方向是否一致？**（`scripts/probe_pi05_direction.py`，每个观测采 12 次）

| 观测（取自示范） | 方向 | net/travel |
|---|---|---|
| step 20 | **12/12 前进** | 0.829 |
| step 200 | **12/12 后退** | 0.798 |
| step 340 | **12/12 后退** | 0.971 |

**均值 0.866 —— 连贯、单一意图的移动。**

（不同观测方向不同是**正常的**，不构成发现：靠近饼干和远离是相反的计划。要看的是**同一个观测**是否只有一个答案。）

**所以采样器没问题。** 同一个模型、同一个采样器、同样的 10 步去噪：

| 输入 | 计划的 net/travel |
|---|---|
| 示范观测 | **0.87** |
| 闭环观测 | **0.04–0.18** |

**差别只在输入分布。** 模型的动作头工作正常，是它看到的场景让它产生了自相消的计划。

这直接否定了一个原本很诱人的候选（`num_inference_steps`：更充分的去噪会让轨迹更一致）——**去噪已经很充分了，在示范观测上它给出 0.87。**

### 7.6 θ 一个待验证的机制

数据支持的解释是**双稳极限环**：机器人漂到一个位置，模型说"退回去"；它退回去，模型又说"过去"。两边都是模型给出的、各自连贯的计划，但任何一侧都不能结束这个环。

要验证它，需要**闭环观测**——即 trace 里的那些 `state` 配上当时的图像。当前 trace 只存了状态和动作，没存图像。加图像会让 trace 变成几十 MB，但这是唯一能直接测量"闭环输入的模型响应"的办法。

---


## 8. 适配器的两种模式

`pi05_a3.py` 会读检查点的 `policy_preprocessor.json` 判断统计量从哪来：

| 检查点 | `features` | 统计量来源 |
|---|---|---|
| `lerobot/pi05_base` | `{}` | A3 数据集的（`--config stats=...`） |
| 微调检查点 | 非空 | 检查点自带的（训练时存的） |

这个区分是必要的：微调检查点自带描述**它自己动作范围**的统计量，用数据集的反而是错的（尤其数据集此后被修复过）。读字段而非取 flag，调用方就不可能弄反。

---

## 9. 复现命令

```bash
# 1. 一次性：让 gated tokenizer 名字能离线解析（训练需要，推理不需要）
HF_HOME=/root/autodl-tmp/hf /root/autodl-tmp/a3-collect/.venv/bin/python \
    scripts/setup_paligemma_tokenizer_cache.py

# 2. 零样本评测（3 回合约 6 分钟，其中模型加载 116s）
cd /root/a3_dual_arm_sim
MUJOCO_GL=egl a3-sim compare --scene a3_cookie_same_column \
    --manifest configs/policies/pi05_base.yaml \
    --seeds 0,1,2 --steps 400 --visual-style plain \
    --output outputs/reports/compare_pi05_base.json

# 3. 微调（约 5.3 小时）
bash scripts/train_pi05_a3.sh \
    /root/autodl-tmp/gr00t-runs/b1/merged \
    /root/autodl-tmp/pi05-runs/a3_b1_20k 20000 8 a3_pi05_b1

# 4. 后台守护：只保留最新 3 个检查点
nohup bash scripts/prune_checkpoints.sh \
    /root/autodl-tmp/pi05-runs/a3_b1_20k 3 300 --daemon \
    > /root/autodl-tmp/logs/prune_checkpoints.log 2>&1 &

# 5. 微调评测（训练完成后）
MUJOCO_GL=egl a3-sim compare --scene a3_cookie_same_column \
    --manifest configs/policies/pi05_a3_finetuned.yaml \
    --seeds 0,1,2 --steps 400 --visual-style plain \
    --output outputs/reports/compare_pi05_finetuned.json
```

---

## 10. 产出物

| 路径 | 内容 |
|---|---|
| `configs/policies/pi05_base.yaml` | 零样本清单 |
| `configs/policies/pi05_a3_finetuned.yaml` | 微调清单 |
| `/root/autodl-tmp/a3-collect/adapters/pi05_a3.py` | 适配器（自动识别统计量来源） |
| `/root/autodl-tmp/a3-collect/smoke_pi05.py` | 冒烟测试：加载 + 单批推理 |
| `scripts/train_pi05_a3.sh` | 训练脚本（含统计量预检） |
| `scripts/train_pi05_launcher.py` | 训练启动器（处理器步骤别名） |
| `scripts/prune_checkpoints.sh` | 检查点清理守护 |
| `scripts/setup_paligemma_tokenizer_cache.py` | gated tokenizer 缓存条目 |
| `scripts/eval_pi05_openloop.py` | 开环精度评测（带 `hold`/`mean` 基线） |
| `scripts/diagnose_pi05_loss.py` | 诊断脚本（定位 1e12 的过程） |
| `src/a3_dual_arm_sim/statistics.py` | `recompute_quantiles()` 新增 |
| `outputs/reports/compare_pi05_base.json` | 零样本 — **修复前的错误读数**，保留作对照 |
| `outputs/reports/compare_pi05_base_normalized.json` | 零样本 — 修复后，**这才是模型** |
| `outputs/reports/compare_pi05_finetuned.json` | 微调闭环结果 |
| `outputs/videos/pi05_finetuned_seed2.mp4` | 唯一移动了饼干的回合 |

---

## 11. 待办

前两项已经做完，结论写在 §7.4 和 §7.5，列在这里是为了说明下一步**不该**做什么。

- [x] ~~`execute: 50`~~ — 已测，**任务结果完全相同（78/0），整体效率反而更差**（行程比 1.11 → 2.66）。
      失败在 chunk 之间，不在 chunk 之内。
- [x] ~~`num_inference_steps`~~ — 已排除，**不该试**。固定观测下 12/12 采样方向一致、net/travel 0.87；
      去噪已经足够，问题在输入分布。
- [ ] **给 trace 加图像**（`A3_PI05_TRACE` 只存了 state 和 action）。这是唯一能直接测量
      "闭环输入 → 模型响应"的办法，用来验证 §7.6 的双稳极限环假设。代价是 trace 从 390 KB 涨到几十 MB，
      值得。
- [ ] **中间检查点扫描**：016000 / 018000 / 020000 都在。若 16k 的开环指标和 20k 一样，说明 20k 步不是瓶颈，
      失败在别处（这与 §7.5 的"输入分布"结论一致）。
- [ ] **对照实验定位输入分布的差异**：用示范观测但**故意喂错位的图像**（比如配上一个时刻的帧），
      看 net/travel 是否从 0.87 掉到 0.1 量级。如果能复现，就说明模型对图文一致性极度敏感，
      而这正是闭环里会持续发生的事（它的状态偏离示范后，画面与状态不再匹配）。
