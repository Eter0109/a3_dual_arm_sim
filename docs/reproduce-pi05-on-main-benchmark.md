# 复现 π0.5 在 main 的 benchmark 上的分数

这份文档替换 `handover-pi05-4.05/CHANGES_vs_your_benchmark.md`。上一版交付物在别人机器上跑不起来，
原因有三个，都是**缺文件**而不是缺步骤：

1. 驱动 `bench_main_20k.py` 里把 main 的路径写死成 `/tmp/a3-main/src`；
2. 它依赖的 `scene_compat.py` **没有打进包**（`ImportError: scene_compat`）；
3. `--training-scene` 需要的场景指纹 `ours_scene.json` **也没有打进包**（缺它就没有 7.75 那一行）。

现在这三样都在本仓库里，路径全部改成可配置参数。

---

## 0. 你需要什么

| 项 | 说明 |
|---|---|
| 本仓库 | 分支 `vla-policy-interface-v2`，提供 `scripts/bench_pi05_on_main.py`、`scripts/scene_compat.py`、`scripts/scene_fingerprint.json` |
| main 的代码树 | 任意路径即可，用 `A3_MAIN_ROOT` 指过去（默认 `/tmp/a3-main`）。**不需要修改 main** |
| 权重 | 一个完整 checkpoint 目录（含 `pretrained_model/`），本机是 `/root/autodl-tmp/pi05-runs/a3_b2_unfreeze_20k/checkpoints/020000/pretrained_model` |
| Python 环境 | 能加载 LeRobot π0.5 的那个 venv（本机 `/root/autodl-tmp/a3-collect/.venv`） |
| 指令文本 | 训练用的那一句；驱动默认从训练集 `meta/tasks.parquet` 读，拿不到时用 `--task-prompt` 显式给 |

环境变量（每台机器都要设，且**桥接与评测都要设**）：

```bash
export MUJOCO_GL=egl
export HF_HOME=/root/autodl-tmp/hf          # 权重缓存位置，按自己的改
export HF_HUB_OFFLINE=1                      # 不联网也能起
export A3_MAIN_ROOT=/path/to/your/main       # main 的代码树
```

---

## 1. 先做一次不带模型的连通性检查（30 秒）

这一步不加载权重、不需要 GPU、不需要桥接，只验证 main 的 benchmark 能被驱动起来：

```bash
cd <本仓库>
MUJOCO_GL=egl python scripts/bench_pi05_on_main.py --selftest --selftest-steps 20
```

打印出 `BenchmarkResult(policy_name='hold', total_episodes=2, total_score=0, ...)` 就说明
main 的 benchmark、场景、评分口径都跑通了。**如果这一步就失败，不要继续往下看**，
99% 是 `A3_MAIN_ROOT` 指错，或者 main 的依赖没装。

---

## 2. 起模型服务（约 4 分钟，显存约 20 GB）

模型不在评测进程里，而是通过一个 unix socket 提供服务；benchmark 每 16 步来问一次。

```bash
cd <本仓库>
MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 \
python scripts/a3_bridge_server.py \
    --adapter /path/to/adapters/pi05_a3.py \
    --listen unix:///tmp/pi05.sock \
    --config ckpt=/path/to/checkpoints/020000/pretrained_model
```

看到 `listening on unix:///tmp/pi05.sock (model loaded once)` 之后，另开一个终端跑评测。

> `--adapter` 指向把 A3 观测翻译成 π0.5 输入的那个适配器（本仓库交付物里的 `pi05_a3.py`）。
> 它做三件事：三路相机缩放到 224×224（`cv2.INTER_AREA`）、`observation.state` 取前 16 维、
> 把 50 步的动作块按 `chunk` 交给评测端。

---

## 3. 三种口径，三行数字

同一个 checkpoint，三种口径给出三个完全不同的分数。**先把口径对上，再对数字。**
下面每一行都标明它用不用训练场景（`--training-scene`），因为这是最容易搞混的一维。

### A) main 的**默认** benchmark（随机化，读者在论文里看到的那个）

```bash
MUJOCO_GL=egl python scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 --workers 20 \
    --output /tmp/bench_A.json
```

预期：`mean ≈ 4.05`，分布 `{0:5, 1:1, 4:1, 5:10, 6:1, 10:2}`，严格成功 **2/20**。
`20k` 那一代权重在这里是 **0.00**（`{0:20}`）。
**不要**加 `--training-scene`：加了就是另一个口径。

### B) main 的**冻结布局**（`--no-randomize`）

```bash
MUJOCO_GL=egl python scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 --workers 20 \
    --no-randomize --output /tmp/bench_B.json
```

预期：`mean = 5.00`，分布 `{5: 20}`，**20/20 全部停在 5**。
这是 main 判据下的天花板，不是跑错了。同样**不加** `--training-scene`。

### C) main 的布局 + **把训练场景装进去**（7.75 那一行）

```bash
MUJOCO_GL=egl python scripts/bench_pi05_on_main.py \
    --socket /tmp/pi05.sock --episodes 20 --max-steps 900 --workers 20 \
    --no-randomize --training-scene scripts/scene_fingerprint.json --scene-mode full \
    --output /tmp/bench_C.json
```

预期：`mean ≈ 7.75`，分布 `{5: 9, 10: 11}`。

`--scene-mode full` 会在**每次 reset 之后**把相机、机械臂起始位姿、两个箱子和 80 个 Cookie
一起还原成训练时的布局（`scripts/scene_fingerprint.json` 是这个布局的指纹）；
`--scene-mode cameras` 只换相机、保留 main 自己的随机化。

装之前先花一分钟确认观测真的对上了：

```bash
MUJOCO_GL=egl python scripts/bench_pi05_on_main.py \
    --training-scene scripts/scene_fingerprint.json --scene-mode full --verify-scene
```

## 4. 三个分数为什么不一样

| 口径 | 分数 | 原因 |
|---|---|---|
| A 随机化 | 4.05 | 箱子 ±1 cm、Cookie 0.3 mm/0.9°，训练集是单一固定布局 → 一半的集数连第一批都放不进槽 |
| B 冻结 + main 判据 | 5.00 | 判据严格（15° 直立、精确入槽、贴壁、hold 5 步），模型停在"5 进盒、不进槽"的平台 |
| C 冻结 + 我们的场景 | 7.75 | 观测分布对上了，第二批的插入余量也变宽；仍然 9/20 卡在 5 |

一句话：**A 是任务难度，B 是判据严格度，C 是观测分布**。三者不要混着比。

---

## 5. 常见失败与对策

| 现象 | 原因 | 方法 |
|---|---|---|
| `ModuleNotFoundError: scene_compat` | 上一版包没带这个文件 | 用本仓库的 `scripts/bench_pi05_on_main.py`（它会把自身目录加入 `sys.path`） |
| 20 集全是 `cookies_in_target: 0`、`cookies_in_source: 80` | 指令不对，或没装训练场景 | 传 `--task-prompt`；确认给了 `--training-scene` |
| 全都停在 5 | 这是 B 行的**预期结果** | 想看到 10 就换 C 行口径 |
| 评测卡住不动 | `--workers > 1` 但共用了一个客户端 | 必须让每个 worker 自建连接（驱动已用 `worker_policy_spec` 做这件事；不要手工传共享 client） |
| `A3_MAIN_ROOT` 指错 | `benchmark` 里 `a3_dual_arm_sim` 解析到别处 | `python -c "import a3_dual_arm_sim, sys; print(a3_dual_arm_sim.__file__)"` 应在你的 main 树下 |
| 显卡 OOM | 权重 20 GB + 每 worker 一个 MuJoCo env | 降 `--workers`；env 侧约 90% 的时间，串行也不会慢太多 |

---

## 6. 文件清单

```
scripts/bench_pi05_on_main.py    驱动：把外部策略喂给 main 的 CookieBatchBenchmark
scripts/scene_compat.py          把训练场景装进 main，并提供 --verify-scene 的像素级校验
scripts/scene_fingerprint.json   训练场景指纹（相机 + 位姿 + 箱子 + 80 个 Cookie 的布局）
scripts/a3_bridge_server.py      模型服务端：一次加载，按请求回动作块
docs/reproduce-pi05-on-main-benchmark.md   本文
```
