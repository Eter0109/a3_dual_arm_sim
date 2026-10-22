# 复现 π0.5 在 main 的 benchmark 上的分数：逐步手册

这一版替换上一份同名文档。上一版**指令缺了一个开关**、**带了一个错文件**，对方报的两个现象
（"pi0.5 根本不动"、"视角偏远"）都能解释，而且都不需要改 main。

先看第 0 节的两张表，再按第 1→7 节走。每一步都有**可粘贴的命令**和**必须看到的输出**；看不到就不要往下走。

---

## 0. 上一版到底错在哪

| 现象（对方报的） | 真实原因 | 这一版怎么改的 |
|---|---|---|
| `--workers 2` 起不来：`ModuleNotFoundError: No module named 'bench_main_20k'` | 驱动里把**自己文件名**写死成投递前的 `bench_main_20k`，别人拿到的是 `bench_pi05_on_main.py` | 改成从文件名推导（`Path(__file__).stem`） |
| **视角偏远**：main 的视角近，我们给的视角远 | 交付的 `scene_fingerprint.json` 是从**未对齐**的场景（`cookie_batch_second_batch.yaml`）导出的：相机 `[0.7076, -0.4232, 1.5748]`、fovy 54°。而我们**训练 / 录视频**用的是 main 的相机 `[0.5, -0.18, 1.25]`、fovy 48°（距离差 0.35 m、视场差 6°） | 指纹按 `cookie_batch_main_aligned.yaml` **重新导出**。现在 main 里装上它之后，三路相机与我们自己的渲染**逐像素一致（MAE = 0.00 / 255）** |
| **pi0.5 根本不动**，20 集全是 `in_target 0 / in_source 80` | 指令不是训练那一句。main 的 harness 默认发 `"transfer 10 cookies into target box"`，这个 checkpoint 没见过，**一个 Cookie 都不碰** | 驱动内置训练指令（不再依赖本机数据集路径）；指令与训练不一致时**打印 WARNING** |
| 要跑完 20 集才知道成不成，中途看不出来 | 没有"到底动没动"的单步观测 | 新增 `--diagnose N`：一分钟内打印每步的 `\|cmd\|`（指令幅度）与 `\|dq\|`（机械臂实际位移），并导出三路相机首帧 |

判据只有一句：**`--diagnose` 里 `|dq|` 必须爬到 0.3 rad 以上**。它是 0，后面所有数字都没有意义。

---

## 1. 前置：四样东西

| # | 东西 | 检查命令 | 通过的样子 |
|---|---|---|---|
| 1 | 本仓库，分支 `vla-policy-interface-v2` | `git -C <本仓库> log --oneline -1` | 最新一条是本文这次提交 |
| 2 | main 的代码树（**不要改它**） | `ls <MAIN>/src/a3_dual_arm_sim/workflows/benchmark.py` | 文件存在 |
| 3 | 权重：一个 checkpoint 目录，内含 `pretrained_model/` | `ls <CKPT>/model.safetensors <CKPT>/train_config.json` | 两个都在 |
| 4 | 能加载 LeRobot π0.5 的 Python 环境 | `<PY> -c "import lerobot, torch; print(lerobot.__file__, torch.cuda.is_available())"` | 打印路径 + `True` |

本机（我们这边）的取值，换成你自己的即可：

```
<本仓库>  = /root/a3_dual_arm_sim
<MAIN>    = /tmp/a3-main
<CKPT>    = /root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
<PY>      = /root/autodl-tmp/a3-collect/.venv/bin/python
<ADAPTER> = /root/autodl-tmp/a3-collect/adapters/pi05_a3.py
```

> `<CKPT>` 必须是 **4.05 那个 fine-tune**（`a3_b2_aligned_v3_4000/004000`），用它才复现 4.05 / 5.00。
> 换成 base `a3_b2_unfreeze_20k/020000` 是另一条曲线（A 行 0.00）。

---

## 2. 环境变量（桥接终端和评测终端都要设）

```bash
export MUJOCO_GL=egl
export HF_HOME=<你的权重缓存>          # 本机 /root/autodl-tmp/hf
export HF_HUB_OFFLINE=1                # 不联网也能起
export A3_MAIN_ROOT=<MAIN>             # 例 /tmp/a3-main
```

`A3_MAIN_ROOT` 设错的表现是 `benchmark.py` 找不到、或 `a3_dual_arm_sim` 解析到别的树，**不是**模型不动。

---

## 3. 步骤 1：不加载模型，先验 main 的 benchmark 能跑（30 秒）

```bash
cd <本仓库>
$PY scripts/bench_pi05_on_main.py --selftest --selftest-steps 20
```

必须看到这两行（第二行会被截断，前缀对上即可）：

```
[bench] instruction: 'grasp five yellow blocks together from the source bin, insert them upright ...'
[selftest] BenchmarkResult(policy_name='hold', total_episodes=2, total_score=0, ...)
```

这一步不需要 GPU、不需要桥接、不需要权重。**它失败就别往下走**：99% 是 `A3_MAIN_ROOT` 或 main 的依赖。

---

## 4. 步骤 2：起模型服务（约 4 分钟，显存约 20 GB）

```bash
cd <本仓库>
$PY scripts/a3_bridge_server.py \
    --adapter <ADAPTER> \
    --listen unix:///tmp/pi05.sock \
    --config ckpt=<CKPT>
```

必须等到这一行，再去开第二个终端：

```
listening on unix:///tmp/pi05.sock (model loaded once)
```

模型不在评测进程里：benchmark 每 16 步通过这个 socket 问一次；服务端一次加载，可被多个客户端共用。

---

## 5. 步骤 3：一分钟判定"动不动 + 视角对不对"（**最重要的一步**）

```bash
cd <本仓库>
$PY scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock \
    --diagnose 200 \
    --training-scene scripts/scene_fingerprint.json --scene-mode full \
    --diagnose-shots /tmp/bench_diag
```

我们这边的实测输出（4.05，200 步，同一套代码、同一台机器）：

```
[bench] training scene: scripts/scene_fingerprint.json mode=full (18 joints, 80 cookies) | cameras from /tmp/cookie_batch_training_scene.yaml
[bench] instruction: 'grasp five yellow blocks together from the source bin, insert them upright ...'
[diag] step    1  |cmd|=0.0965 rad  |dq|=0.0418 rad  in_target=0 in_source=80
[diag] step   40  |cmd|=0.0103 rad  |dq|=0.5444 rad  in_target=0 in_source=80
[diag] step  200  |cmd|=0.0502 rad  |dq|=0.6926 rad  in_target=5 in_source=75
[diag] worst |cmd| = 0.0965 rad over 200 steps; worst |dq| = 0.6933 rad
[diag] VERDICT: the arm moves (0.693 rad).
```

三个读法：

| 你看到的 | 含义 | 怎么办 |
|---|---|---|
| `\|dq\|` 一路 0.00x | 机械臂没动 | 看 `\|cmd\|`：它也是 0 → 桥接回的是"保持指令"，**指令不对**；`\|cmd\|` 明显但 `\|dq\|` 是 0 → 动作空间不对（关节目标 vs 力矩） |
| `\|cmd\|` 每次都 < 0.02 | 模型没按这句话动作 | 确认打印的 instruction 与训练集 `meta/tasks.parquet` 里那句**逐字符相同**；不一致就传 `--task-prompt "..."`
| 200 步后 `in_target` 仍为 0 | 动了但没往目标去 | 看视角（下一段） |

### 视角对不对：拿图比，不要靠描述

`--diagnose` 会把三路相机首帧写到 `/tmp/bench_diag/diag_{front,left_wrist,right_wrist}.png`（256×256）。
本仓库里有参考图：

```
docs/repro/expected_view_front.png
docs/repro/expected_view_left_wrist.png
docs/repro/expected_view_right_wrist.png
```

一行命令给数字：

```bash
cd <本仓库>
$PY - <<'PY'
import cv2, numpy as np
for n in ("front", "left_wrist", "right_wrist"):
    a = cv2.imread(f"/tmp/bench_diag/diag_{n}.png").astype(float)
    b = cv2.imread(f"docs/repro/expected_view_{n}.png").astype(float)
    print(f"{n:11s} MAE = {np.abs(a - b).mean():.2f} / 255")
PY
```

**期望：三路都 ≤ 3 / 255**（我们这边 main 装指纹后是 0.00，因为两边相机参数完全相同）。
两张图都是同一套渲染出来的 PNG，没有视频压缩，所以对得上就是 0.00；大于 3 就说明相机块没装上。

> 如果你看到的是"箱子更小、桌面更多"，那就是上一版指纹的远相机（`[0.7076,-0.4232,1.5748]`, fovy 54°）。
> 打开这一版的 `scripts/scene_fingerprint.json` 看一眼：`front_position_m` 必须是 `[0.5, -0.18, 1.25]`，
> `front_fovy_deg` 必须是 `48.0`。不是就说明拿的还是旧文件。

---

## 6. 步骤 4：2 集短测（约 4 分钟），确认真跑也能动

```bash
cd <本仓库>
$PY scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 2 --max-steps 300 \
    --training-scene scripts/scene_fingerprint.json --scene-mode full
```

期望：`max_score` 不为 0（300 步通常到 5）；行尾 `bridge forwards` 是几十次而不是 0。
`total_score=0` 但 `forwards>0` = 模型在动、只是没放进槽，这时才回步骤 3 查。

---

## 7. 步骤 5：正式三口径

同一个 checkpoint，三种口径给出三个完全不同的分数。**先把口径对上，再对数字。**
"加不加 `--training-scene`"是最容易搞混的一维。

### A) main 的默认 benchmark（随机化，论文里看到的那个）

```bash
$PY scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 \
    --output /tmp/bench_A.json
```

**不要**加 `--training-scene`。预期：`mean ≈ 4.05`，分布 `{0:5, 1:1, 4:1, 5:10, 6:1, 10:2}`，严格成功 2/20。
同一命令换成 base `020000` 权重是 `0.00`（`{0:20}`）。

### B) main 的冻结布局

```bash
$PY scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 \
    --no-randomize --output /tmp/bench_B.json
```

预期：`mean = 5.00`，分布 `{5:20}`，**20/20 全部停在 5**。这是 main 判据下的天花板，不是跑错了。

### C) 冻结布局 + 把训练场景装进 main

```bash
$PY scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 \
    --no-randomize --training-scene scripts/scene_fingerprint.json --scene-mode full \
    --output /tmp/bench_C.json
```

> **这一行的数字这一版需要重测。** 上一版记录的 `7.75 {5:9, 10:11}` 是在**错误指纹**（远相机、
> 布局差 23 mm）下测出来的，不是训练分布。修好指纹后先用步骤 3 确认 `|dq|≈0.69 rad`、三路相机
> MAE≈0，再跑这 20 集，把分布贴回来我们更新这张表。

`--scene-mode full` 每次 reset 之后还原相机 + 机械臂起始位姿 + 两个箱子 + 80 个 Cookie；
`--scene-mode cameras` 只换相机块、保留 main 自己的随机化。

### 关于 `--workers`

**第一遍一律用默认（串行）。** 并行路径依赖 worker 进程按模块名重新 import 这个驱动文件，
这一版修好了模块名（上一版就是这里炸的），但**没有在别人的机器上验证过**；
串行 20 集约 45 分钟，先拿到一组可信数字更重要。跑通之后再加 `--workers 4` 试。

---

## 8. 症状对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: bench_main_20k` | 上一版把文件名写死 | 用这一版仓库的驱动 |
| `ModuleNotFoundError: scene_compat` | 上一版包没带 | 用这一版仓库的驱动（它会把自身目录加入 `sys.path`） |
| **pi0.5 完全不动**，`in_target=0, in_source=80` | 指令不对（最常见），或拿到的是 base 权重 | 看 `[bench] instruction:` 那行；不对就 `--task-prompt "<训练集里那句>"` |
| **视角偏远 / 箱子偏小** | 用了上一版的指纹 | 用这一版 `scripts/scene_fingerprint.json`（`front_position_m` 应为 `[0.5,-0.18,1.25]`） |
| 全都停在 5 | B 行的**预期结果** | 想看到 10 就换 C 行口径 |
| `A3_MAIN_ROOT` 指错 | `a3_dual_arm_sim` 解析到别处 | `$PY -c "import a3_dual_arm_sim; print(a3_dual_arm_sim.__file__)"` 应在 main 树下 |
| 显卡 OOM | 权重 20 GB + 每个 worker 一个 MuJoCo env | 降 `--workers`；env 占 90% 时间，串行不慢太多 |
| `bridge replied 'ready' with keys []` | 客户端没先读握手帧 | 驱动已处理；自己写客户端时第一帧必须先 `read_frame` |

---

## 9. 文件清单

```
scripts/bench_pi05_on_main.py    驱动：把外部策略喂给 main 的 CookieBatchBenchmark；含 --diagnose
scripts/scene_compat.py          把训练场景装进 main，并提供 --verify-scene 的像素校验
scripts/scene_fingerprint.json   训练场景指纹（**这一版按对齐场景重导**：相机 = main 的，布局 = main 的）
scripts/a3_bridge_server.py      模型服务端：一次加载，按请求回动作块
docs/repro/expected_view_*.png   期望的三路相机首帧（256×256），用来判断"视角对不对"
docs/reproduce-pi05-on-main-benchmark.md   本文
```
