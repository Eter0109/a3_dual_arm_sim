# GR00T-N1 微调评测报告

**日期**：2026-09-29
**模型**：`gr00t-n1-a3-finetuned`（`/root/autodl-tmp/gr00t-runs/b1/checkpoints/checkpoint-14000`）
**场景**：`a3_cookie_same_column`，400 步/回合
**清单**：`configs/policies/gr00t_a3_finetuned.yaml`
**适配器**：`/root/autodl-tmp/ACG/adapters/gr00t_a3_finetuned.py`

---

## 一句话结论

**接线全部正确，模型也确实在预测——但闭环里它把夹爪闭上后不再张开，手臂在比示范小 37% 的范围内摆动，一块饼干都没搬。**

任务成功率 0/10，**但这不是又一次"零样本失败"**：环境对指令的纠正率从零样本的 39.6% 降到 2.25%（17.6×），饱和率降到 0%，说明模型已经完全在自己的训练分布里输出关节角。失败发生在更下游——它学会了"像专家那样移动"，但没学会"完成一个抓取周期"。

---

## 1. 与零样本基线的对比

同一场景、同一提示词、同一步数、同一执行节奏（`execute: 16`），唯一变量是权重。

| 指标 | 零样本 `gr00t_bimanual` | 微调 `gr00t_a3_finetuned` | 变化 |
|---|---|---|---|
| 成功回合 | 0 / 3 | 0 / 3 | — |
| `cookies_in_target` | 0 | 0 | — |
| **corrected** | 0.396 | **0.0225** | **↓ 17.6×** |
| **rate limited** | 0.408 | **0.0209** | **↓ 19.5×** |
| **saturated** | 0.524 | **0.000** | **↓ 至零** |
| correction max (rad) | 0.227 | 0.061 | ↓ 3.7× |
| motion mean (rad) | 0.0399 | 0.0081 | ↓ 4.9× |
| 启动 (s) | 18.2 | 16.4 | — |
| 单次推理 (ms) | 175 | 171 | — |

**怎么读这张表**

`corrected` 是环境必须改写指令的步数占比——"模型没有在开这台机器人"的最直接信号。零样本是 39.6%，微调是 2.25%，落到和 A3 自己的专家策略一个量级。零样本的 0.0399 rad/步运动速率是示范的 **12 倍**，41% 的步在撞限速器；微调的 0.0081 落在示范分布的 p90 上。**速率是对的。**

所以要小心一个误读：`motion mean` 下降 4.9× 看着像"模型更保守了"，其实是从病态回到正常。

---

## 2. 运动速率基准（解读一切运动指标的前提）

从 200 回合示范里实测：

| | 值 |
|---|---|
| 示范每控制步位移（左臂 7 关节，\|q(t+1) − q(t)\|） | **mean 0.0032**，p90 0.0082，max 0.05 |
| 零样本模型 | 0.0399（**12× 示范速率**，撞限速） |
| 微调模型 | 0.0081（**2.5× 示范均值，≈ 示范 p90**） |

`motion_max_rad = 0.05` 在两边都出现，那是环境的每步关节上限，不是行为特征。

---

## 3. 逐层排除

每一条都是实测，不是推断。

### 3.1 不是接线问题——开环预测精度

`adapters/eval_openloop.py`：用训练集观测喂模型，与专家的 16 步动作块对比，192 个样本。

| 预测器 | MSE | 相对 `mean` |
|---|---|---|
| **模型** | **0.00058** | **0.02×** |
| `hold`（保持当前位姿） | 0.00214 | 0.08× |
| `mean`（数据集均值） | 0.02767 | 1.00× |

模型比 `mean` 好 **48×**，比 `hold` 好 **3.7×**。分块看，右臂 MSE 0.00000（MAE 0.0013 rad），左臂 MAE 0.0209 rad ≈ 1.2°。

> `mean` 基线在**不参与评分的 160–199 号回合**上拟合，所以对比不是循环论证。
>
> 但要注意这些观测**就是训练数据**。这个分数是必要条件而非充分条件：它排除了适配器接错和训练没起来，但它对"能否自己走 400 步"什么也没说。

**副产品发现**：数据里**右臂起始位姿几乎固定**（跨回合 std 仅 0.025，而左臂是 0.16），chunk 内却有固定轨迹——右臂承担的是脚本化的固定角色，左臂才随布局变化。所以任务的难度集中在左臂。

### 3.2 不是提示词问题——三组对照

`adapters/probe_prompt.py`，同一观测只改一个变量：

```
噪声底噪 0.0419 | 视觉响应 0.5727 | 语言响应 0.0965
  摄像头: 在用 (13.7x 噪声底噪)
  语言  : 在用 ( 2.3x 噪声底噪)
```

- **噪声底噪**：动作头是扩散模型，同样输入两次调用就不完全相同（max\|Δ\| 0.0419）。任何跨提示词的差异必须先和这个比。
- **视觉**：同一提示词换不同帧（step 20 / 200 / 360），输出差 0.5727 = **13.7× 底噪**。模型**强烈**依赖相机。
- **语言**：换成 `"do nothing"` 只差 0.0965 = **2.3× 底噪**。语言**确实在用，但是个小旋钮**。

结论：改写提示词对结果的影响上限大约是视觉的 1/6。值得试，但别指望它救场。

### 3.3 不是渲染风格问题

发现训练数据用的是 `plain`（`collection.py` 默认值，`collect_cookie.sh` 没有覆盖），而首次评测用的是 `detailed`——**模型从未见过槽位网格和双色饼干**。这是真实的分布错配，值得记下。

但换成 `plain` 重跑：`corrected 0.025`、`cookies_in_target 0`，与 `detailed` 无差别。**风格不是原因。**

### 3.4 不是布局随机化问题

`--randomize`（匹配训练时的随机布局）：`corrected 0.024`、`cookies_in_target 0`。无差别。

### 3.5 不是重规划频率问题

| `execute` | corrected | motion mean | cookies_in_target |
|---|---|---|---|
| 4 | 0.061 | 0.015 | 0 |
| 8 | 0.035 | 0.011 | 0 |
| 16 | 0.028 | 0.009 | 0 |

**重规划越勤反而越差**——每次都引入新的采样噪声。这不是误差累积型失败。

---

## 4. 根因：逐请求轨迹

`A3_GR00T_TRACE` 转储了 25 次请求的全部输入状态与输出动作块。

### 4.1 夹爪：闭上之后不再张开

| | 示范 | 模型（闭环） |
|---|---|---|
| 右夹爪 | **恒为 1.000**（右臂本来就不抓取，这是任务设计） | 恒为 1.000 ✓ |
| 左夹爪 | `1.00 → 0.28 → 1.00` 每个批次循环一次 | `1.00 → 0.45 → 0.31`，**之后在 0.31–0.44 之间震荡 380 步，再没回到 1.0** |

任务要求反复开合：抓住五块 → 插进槽位 → 松开 → 回去再抓五块。**模型在第 16 步闭了一次夹爪就再没松开**，一个完整的抓取-释放周期都没走完。

### 4.2 手臂：行程够，范围不够

左臂 7 关节，模型轨迹 vs 三个示范回合平均：

| | 示范 | 模型 | 比值 |
|---|---|---|---|
| 各关节运动范围合计 (max−min) | 4.194 | 2.645 | **0.63×** |
| 总行程合计 Σ\|Δq\| | 9.594 | 8.117 | **0.85×** |

**总行程接近（85%），但活动范围只有 63%。** 它在比示范更小的包络里来回摆动——不是"停住了"，是**震荡**。配合 `|a[0] − state| ≈ 0.04`（即模型每一步都在说"往这边挪一点"，而示范每步只挪 0.0032），图像是一致的：策略锁定在一个悬停姿态附近抖动，迟迟不进入下一阶段。

---

## 5. 训练侧的解释

1 个 epoch / 14,000 步 / 200 回合，`train_loss` 收在 **0.0123**。这个数**低到可疑**——对一个扩散动作头来说，它意味着模型已经能很好地复现动作块的**条件均值**，也就是"看起来像专家轨迹的那条平均轨迹"。开环 48× 优于 `mean` 证实了它学到了比均值多得多的东西，但闭环里它退化成了一个**悬停吸引子**。

最可能的原因不是模型容量，而是**数据**：200 个回合里每个回合只包含约 2 个抓取批次（5 块 ×2），也就是说夹爪只开合 2 次。**模型见过的"抓取-释放"完整周期总共约 400 次**，而它见过 110,169 帧的"在去抓的路上"。动作分布严重偏向过渡态，抓取瞬态被淹没。这解释了为什么闭合动作只学了一次就锁死。

---

## 6. 下一步（按性价比排序）

1. **不用重训就能确认的诊断**：用 `A3_GR00T_TRACE` 把模型轨迹与**同布局**的示范叠加画图，看它是在跟踪一条更慢的轨迹，还是已经脱轨。前者说明是时间尺度问题（重采样 `dt` 或增加 `execute` 之外的时序对齐），后者说明是分布问题。

2. **加大数据量到 448 回合**（磁盘上已有，`watch_and_train.py --episodes 448`）。若"抓取瞬态被淹没"是对的，则收益应该**非线性**——每回合的抓取次数不变，但布局多样性翻倍。

3. **改变数据构成而非只是数量**：采集时让每个回合包含更多抓取批次（更长的回合成更少的块/更多目标位），直接提高抓取瞬态在动作分布里的占比。这是针对根因的改动。

4. **暂时不做**：改提示词、换相机、调 `execute`。第 3 节已经把这三条排除掉了。

---

## 7. 复现命令

```bash
# 开环精度（几分钟，无模拟器）
MUJOCO_GL=egl /root/autodl-tmp/ACG/.venv/bin/python \
    /root/autodl-tmp/ACG/adapters/eval_openloop.py

# 提示词/视觉敏感性对照
MUJOCO_GL=egl /root/autodl-tmp/ACG/.venv/bin/python \
    /root/autodl-tmp/ACG/adapters/probe_prompt.py

# 闭环评测（3 回合，约 2 分 40 秒）
MUJOCO_GL=egl a3-sim compare --scene a3_cookie_same_column \
    --manifest configs/policies/gr00t_a3_finetuned.yaml \
    --seeds 0,1,2 --steps 400 --visual-style plain \
    --output outputs/reports/compare_gr00t_finetuned_plain.json

# 逐请求轨迹（在清单 env 块里取消注释 A3_GR00T_TRACE 后）
MUJOCO_GL=egl a3-sim compare --scene a3_cookie_same_column \
    --manifest configs/policies/gr00t_a3_finetuned.yaml \
    --seeds 0 --steps 400 --visual-style plain --output /tmp/traced.json

# 回合视频
MUJOCO_GL=egl python scripts/render_bridge_policy_video.py \
    --manifest configs/policies/gr00t_a3_finetuned.yaml \
    --scene a3_cookie_same_column --seed 0 --steps 400 \
    --visual-style detailed \
    --output outputs/videos/gr00t_finetuned_cookie_seed0.mp4
```

---

## 8. 产出物清单

| 路径 | 内容 |
|---|---|
| `configs/policies/gr00t_a3_finetuned.yaml` | 清单（含本次结论的摘要注释） |
| `/root/autodl-tmp/ACG/adapters/gr00t_a3_finetuned.py` | 适配器（含 `A3_GR00T_TRACE` 轨迹转储） |
| `/root/autodl-tmp/ACG/adapters/smoke_finetuned.py` | 冒烟测试：加载 + 合成观测推理 |
| `/root/autodl-tmp/ACG/adapters/probe_prompt.py` | 三组对照：噪声底噪 / 视觉 / 语言 |
| `/root/autodl-tmp/ACG/adapters/eval_openloop.py` | 开环精度评测（带 `hold`/`mean` 基线） |
| `outputs/reports/compare_gr00t_finetuned.json` | 闭环评测（`detailed`，`execute 16`） |
| `outputs/reports/compare_gr00t_finetuned_plain.json` | 闭环评测（`plain`，训练契约） |
| `outputs/reports/compare_gr00t_finetuned_random.json` | 闭环评测（`--randomize`） |
| `outputs/videos/gr00t_finetuned_cookie_seed0.mp4` | 400 步回合视频（896×640 @ 20fps） |

---

## 附：这次踩到的两个契约坑（都值得记住）

1. **`MAX_NUM_EMBODIMENTS`**：检查点的动作头有 35 个 embodiment 槽位，vendored GR00T 默认 32。不设的话不报"配置错了"，而是 `from_pretrained` 报 14 处形状不匹配——**读起来像检查点损坏**。适配器在 import 时设置，清单 `env` 里也写了一份。

2. **`units` 不是单位标签**：它声明的是"模型输出处在哪个归一化刻度"，`"raw"` 才表示"已经是物理值、什么都不用做"。写 `"radians"` 等于在说"这些数在一个叫 radians 的归一化刻度上，请清单去还原它们"。桥接层会拒绝 `units != "raw"` 且 `normalize.mode: none` 的组合——**这是对的**。单位事实应该写进 `info`，那是给人看的。
