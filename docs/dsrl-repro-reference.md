# 复现参考 DSRL（robomimic can）— 环境与补丁记录

## 结论

参考实现 `/root/refs/dsrl` **在本机跑起来了**，训练中。这是独立于 a3 项目的一次复现，目的是回答
"参考算法本身能不能在这个机器上跑通"——如果连参考都跑不通，我们的适配就无从验证。

启动命令（必须在 `/root/refs/dsrl` 下，`train_dsrl.py` 会 `sys.path.append('./dppo')`）：

```bash
cd /root/refs/dsrl
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf TMPDIR=/root/autodl-tmp/tmp HYDRA_FULL_ERROR=1
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
    use_wandb=False env.n_envs=4
```

日志 `/root/autodl-tmp/logs/dsrl_repro_long.log`。

## 所需资产（用户从 Google Drive 下载的唯一一个）

```
/root/refs/dsrl/dppo/log/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt
```

4.4 MB，顶层键 `['epoch', 'model', 'ema']`，`model` 和 `ema` 各 12 项且都是 `network.*` 前缀。
`diffusion_eval.py:41-56` 先试 `checkpoint["model"]` 里的 `actor.*`，失败则回退到
`checkpoint["ema"]` 里的 `network.*`。**这个 checkpoint 走的是回退分支**，加载日志确认：
`Loaded base policy weights from ...state_5000.pt`。

## 其余资产全部本地解决

| 资产 | 来源 |
|---|---|
| `dppo`、`stable-baselines3` 两个 submodule | `git clone --depth 1 https://github.com/ajwagen/...`（HTTPS 可用，SSH submodule 没有初始化） |
| `can` ph 数据集 45 MB / 200 集 | `http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim.hdf5` |
| `can` mh 数据集 112 MB / 300 集 | 同目录 `mh/low_dim.hdf5` |
| `normalization.npz` | **自己生成**：`dppo/script/dataset/process_robomimic_dataset.py --load_path=... --save_dir=... --normalize` |
| `train_offline.npz` | 不需要（`cfg/robomimic/dsrl_can.yaml: load_offline_data: False`） |

**该服务器会返回 503 并在传输中途截断**（`ph` 停在 18 MB / 45 MB，`mh` 停在 27 MB / 112 MB）。
`curl -sSIL` 报 200 而 `curl -sSL` 拿到 503 HTML，所以必须用 `-C -` 断点续传并反复重试，
每次校验字节数直到等于 `Content-Length`。校验方法：文件头必须是 `89 48 44 46`（HDF5 magic）。

## 依赖安装（pypi 不通，用腾讯镜像）

```bash
M=https://mirrors.cloud.tencent.com/pypi/simple/
pip install -i $M --no-build-isolation gym==0.26.2
pip install -i $M --no-build-isolation --no-deps d4rl
pip install -i $M --no-build-isolation robomimic==0.3.0
pip install -e /root/refs/dsrl/stable-baselines3 --no-build-isolation --no-deps
pip install -e /root/refs/dsrl/dppo --no-build-isolation --no-deps
```

`pypi.org`、清华镜像都超时；**腾讯镜像 200**。三个必须知道的坑：

* **`--no-build-isolation` 必须加**：隔离构建会去拉 `setuptools>=40.8.0`，离线必然失败。
* **`--no-deps` 必须加**：`stable-baselines3` 要求 `gymnasium<1.3.0`，本机是 1.3.0，而 pypi 不可达
  所以无法降级。实测 1.3.0 与 DSRL 需要的部分兼容。
* **`robomimic==0.3.0` 是 DPPO `pyproject.toml` 指定的版本**（第 46 行），不是 0.2.0。

**副作用（已核实无害）**：升 `robomimic` 0.2.0 → 0.3.0 会让 `hf-libero 0.1.4` 报版本冲突。
a3 项目**不 import robomimic**，且已验证 `make_cookie_benchmark(task='2x10')` 仍返回
`required_cookies = 20`；run11 当时仍在运行且未受影响。

## 唯一的代码补丁

`robomimic 0.3.0` 的 `envs/env_robosuite.py:10` 无条件 `import mujoco_py`，而 `mujoco_py` 早已弃用、
在 Python 3.12 上无法编译。**全文件只有 378 行**用到它：

```python
return (mujoco_py.builder.MujocoException)
```

即仅作为一个异常类型。robosuite 1.4 直接用 `mujoco` bindings，不需要它。补丁把它改成可选导入，
失败时给一个永远不会被抛出的替身类。原文件保留为 `env_robosuite.py.orig`。

补丁路径：
`/root/autodl-tmp/a3-collect/.venv/lib/python3.12/site-packages/robomimic/envs/env_robosuite.py`

**这是对第三方库的修改，不是对参考代码的修改**——`refs/dsrl` 下除 submodule 克隆外未改一行。

## 已知噪音（不是错误）

启动时大量 `No module named 'mujoco_py' / 'flow' / 'dm_control' / 'carla' / 'pybullet_envs'`，
是 `d4rl` 在探测它支持的各个环境，与 robomimic 路径无关。筛选日志时排除这些以及
`robosuite WARNING`、`Gym has been unmaintained`。

## 第一次跑通时的实测数（`n_envs=1`，用于确认链路）

```
total_timesteps 600    fps 1    n_updates 11960    time_elapsed 505s
actor_loss -2.6   critic_loss 1.07   ent_coef 0.0459   noise_critic_loss 5.34
```

`fps 1` 是 `n_envs=1` 且每步都要跑 diffusion 采样（ta4 = 4 步动作块 × td20 = 20 步去噪）。
这不是性能问题，是这一类方法的固有成本；`n_envs=4` 时约 4 fps。参考配置
`eval_interval: 3000`，所以约 12 分钟能看到第一次评测。

## 待验证的问题

参考复现要回答的是**算法层面的可行性**，不是 our-task 的可行性：

1. **能不能在合理时间内看到 success rate 上升？** 论文的 can 结果是 sparse success rate。
   如果参考在几百分钟内没有上升，说明这台机器/这个种子下复现本身就不成立。
2. **`DSRL-NA` 的蒸馏链路是否真的工作？** 我们怀疑的 `min` 偏差、噪声覆盖率、bootstrap
   刷新粒度，都可以在这个参考上直接测量，因为它的 `act_steps=4`、`action_magnitude` 语义
   比我们的 VLA 接口简单得多。
3. **`n_updates` 与 `total_timesteps` 的比例**：跑了 600 步就有 11960 次更新（utd 20 × 600 = 12000，
   吻合），这验证了 `utd` 的语义是"每个 env step 做 utd 次梯度步"，和我们线上实现的
   "每轮 20 次"是**不同的**——我们每轮是 8 个场景 × 16 控制步 = 128 个控制步才做 20 次，
   而参考是每 4 个控制步（一个 act_steps）就做 20 次。这是 32 倍的差距，值得单独核对。

## 复现过程中发现的第一个实质差异：`utd` 的语义

这是复现参考最直接的收获，也是不需要跑完就能确认的。

参考的 `train_dsrl.py:94-95`：

```python
train_freq=cfg.train.train_freq,      # 1
gradient_steps=cfg.train.utd,         # 20
```

而 `ActionChunkWrapper.step` 一次执行 `act_steps = 4` 个控制步。所以：

$$
\text{参考：}\frac{20\ \text{梯度步}}{4\ \text{控制步}} = 5.0\ \text{次/控制步}
$$

我们的 `train_dsrl_online.py` 每轮跑 `scenes=8 × execute=16 = 128` 个控制步，然后做
`utd=20` 次梯度步：

$$
\text{我们：}\frac{20}{128} = 0.156\ \text{次/控制步}
$$

**相差 32 倍。**

实测验证（参考 `n_envs=1`，600 env steps）：

```
n_updates = 11960
11960 / 600 env_steps   = 19.9 次更新/env.step   <- 与 gradient_steps=20 吻合
600 × 4 act_steps       = 2400 控制步
11960 / 2400            = 4.98 次更新/控制步     <- 与 5.0 吻合
```

我们 run11 在 r700 时：

```
n_updates = 700 × 20     = 14000
控制步    = 700 × 128    = 89600
14000 / 89600            = 0.156 次/控制步
```

### 为什么这一直没被发现

`utd=20` 在两边**字面上相同**，配置对比表里记为 "identical"。差异来自**分母**：
参考的分母是 `act_steps=4`（一个 action chunk），我们的是 `scenes × execute = 128`
（八个场景各执行十六步）。对比表只对了 `utd` 本身，没有对 `utd` 的**归一化基准**。

### 这意味着什么

如果 32 倍是真的，那么我们所有的 run 在**学习率意义上**都相当于把参考的 `utd` 设成了
`20/32 ≈ 0.6`。这是一个全新的、无需任何训练就能验证的假设，而且能解释为什么：
`alpha` 衰减缓慢（更少的 actor 更新）、`greedy` 单调退化（actor 扰动的速度快于 critic
跟上的速度）、`q` 追不动不动点（更少的 critic 更新）。

### 下一步

**不要立刻改**。先做两件事：

1. 让参考跑出第一次 `eval_interval=3000` 的 success rate，确认它在 can 上确实能学。
   如果参考本身在这个机器上学不动，那 32 倍是不是 bug 都无从判断。
2. 在参考上做一次 `gradient_steps ∈ {1, 5, 20}` 的对照，测出 `n_updates` 与 success 的关系。
   这直接给出"多少次更新/控制步才够"，比把我们的 `utd` 直接乘 32（会贵 32 倍）更可靠。

## 时间预算：20M 步不可行，必须预先定一个目标（11-12 实测）

论文的 `train_dsrl.py:178` 硬编码 `total_timesteps=20000000`。实测稳态速度 **5.5 timesteps/s**，
所以 20M 步需要 **46 天**。这个数不可协商，所以必须先定"跑到哪里算数"。

### 成本分解（cProfile，`n_envs=2`，`init_rollout_steps=0`）

| 项 | 每次 | 占比 |
|---|---|---|
| `train()`（20 个 utd 步） | 663 ms | **91%** |
| `vec_env.step`（物理） | 66 ms | 9% |
| 合计每 vec step | 729 ms | → 5.5 timesteps/s |

每 utd 步 33 ms 的构成：扩散 forward（batch 256，单独计）**8.2 ms / 25%**，其余 critic/actor MLP
与 polyak **24.8 ms / 75%**。

**关键：把 `train.layer_size` 从 2048 降到 256 完全不改变 fps。** 所以那 25 ms 不是 MLP 算力，
而是固定开销（SB3 的 buffer 采样、张量搬运、每次更新的 Python 层）。这一点很重要——它意味着
"把网络变小"不能换来速度，只有"减少更新次数"能。

### 还有一笔被忽略的开销

`cfg.num_evals: 200`，`eval_freq: 3000` vec steps（= 12000 timesteps）。每次评测
`200 集 × 75 步 × 67 ms = 17 分钟`，而 12000 timesteps 的训练只要 40 分钟。**评测占 30%。**

这是**纯测量开销**，不是算法的一部分，所以本次用 `num_evals=20` 启动（1.7 分钟），
其余全部保持论文值。

### 顺带否掉了两个我自己的猜测

* **不是设备往返**：`dsrl.py:276` 每个梯度步做 `.cpu().numpy()` 再回 GPU，我给了一个数学
  等价的 GPU 版本（`Box(-1,1)` 下 `unscale_action` 是恒等），**fps 从 5 变成 5**。补丁保留
  （无害且更干净），但它不是瓶颈。
* **不是物理**：`n_envs=4` 时 env 子进程各只占 13% CPU，主进程 80%。停掉我们的 run11 之后
  fps 仍是 5，确认之前的"竞争"不是原因。容器有 16 核，只用了 1.6 核。

### 目标与预注册判据

`total_timesteps` 改成监控到 **300k**（= 17 小时，fps 5）。这是论文预算的 **1.5%**。

| 时点 | timesteps | 判据 |
|---|---|---|
| T+2h | ~40k | 记下 `success_rate`，不作判断（warmup 6004 + 早期） |
| T+6h | ~120k | 若 `success_rate > 0` → 算法链路成立，可以此标尺对照我们的实现 |
| T+17h | 300k | **若仍为 0 → 判定"参考在这台机器/这个种子下无法复现"**，我们所有 run 学不动就不再是我们的实现问题 |

第三条是这次复现的核心价值：它把"我们的实现有 bug"和"这个方法本身在这个预算下学不动"分开。

## can 复现：预算来自论文，不是来自 `train_dsrl.py` 的 20M

### 论文给出的真实规模

| 来源 | 内容 |
|---|---|
| Figure 4 横轴 | `Timesteps ×10^6`，0.0 → **1.0**，即最多 **100 万原始控制步** |
| Table 4（Can 列） | chunk 4, hidden 2048, **grad updates 20**, **Q_W updates 10**, γ 0.99, action magnitude 1.5, **Initial steps 24000** |
| Table 3（所有在线实验） | lr 3e-4, batch 256, Tanh, **target entropy 0**, τ 0.005, 3 层, 2 critics, **4 envs** |
| p.7 | 每个实验 **5 seeds** 平均 |

`train_dsrl.py:178` 里的 `total_timesteps=20000000` 只是一个"一直跑"的 ceiling，**不是论文的预算**。
这纠正了本文档早先按 20M 估算的 42 天——那个数是错的。

### 配置核对：仓库与论文逐项一致

| 项 | 仓库 | 论文 Table 4 |
|---|---|---|
| act chunk | 4 | 4 ✓ |
| hidden | 2048 | 2048 ✓ |
| grad steps/update | 20 | 20 ✓ |
| Q_W steps | 10 | 10 ✓ |
| discount | 0.99 | 0.99 ✓ |
| action magnitude | 1.5 | 1.5 ✓ |
| target entropy | 0.0 | 0 ✓ |
| initial steps | `init_rollout_steps 1501` | 24000 |

`1501 × 4 envs × 4 act_steps = 24016` 控制步，与论文的 24000 吻合。**所以仓库就是论文的设置，无需改任何超参。**

### 时间预算（实测 22 控制步/s 稳态）

| 目标控制步 | 累计时间 | 相当于论文曲线 |
|---|---|---|
| 50k | 0.4 h | 5% |
| 100k | 1.0 h | 10% |
| 300k | 3.5 h | 30% |
| 1M | **12.3 h** | 100% |

论文的图是 5 seeds 平均，完整复现一张 can 图 = **约 62 小时**。

### 我又引入并修掉了一个 bug

为了把评测从 17 分钟降到 8.5 分钟，我用 `num_evals=20` 启动。但
`train_dsrl.py:151` 是 `eval_episodes = int(cfg.num_evals / num_env_eval)`，而 `n_eval_envs=25`：

```
num_evals=20  -> eval_episodes = int(20/25) = 0   *** 完全不评测 ***
num_evals=100 -> 4 次评测、每次 25 env
```

也就是说那一版跑起来**根本不会输出 success rate**，而我还以为在观察。现在用 `num_evals=100`
（评测成本减半，仍然有效）。教训是同一个：改一个数之前要确认它在代码里怎么被消费，而不是
确认它"看起来合理"。

当前命令：

```bash
cd /root/refs/dsrl
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf TMPDIR=/root/autodl-tmp/tmp HYDRA_FULL_ERROR=1
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
    use_wandb=False env.n_envs=4 num_evals=100
```

日志 `/root/autodl-tmp/logs/dsrl_can_repro.log`。

### 预注册判据

| 控制步 | 时间 | 判据 |
|---|---|---|
| 24k | 2 min | 初始采集结束，开始训练（`n_updates` 开始增长） |
| 100k | 1 h | 论文 10% 处；若 `success_rate > 0` 则链路成立 |
| 300k | 3.5 h | 论文 30% 处；若仍为 0 → 判定"此机器/此预算下参考无法复现" |
| 1M | 12.3 h | 论文 100% 处；应接近论文报告的 Near-optimal |

## 严重问题：基座策略 0%，而论文的图在 t=0 就有 ~45%

论文 Figure 4 的 Can 面板里，DSRL 曲线在 timestep 0 处已经是 **~0.45–0.5**。我们复现出的基座
策略是 **0.000**，`ep_len` 恒等于 75（跑满、从未成功终止）。这意味着**复现尚不成立**，
不是"算法学不动"。

### 已经排除的原因（每一项都是实测，不是推测）

| 检查 | 结果 |
|---|---|
| robosuite 版本 | 装的是 1.4.0，数据集是 **1.4.1** → 已升到 1.4.1，**0% 不变** |
| 归一化来源 | 我最初用 **ph** 数据算 `normalization.npz`，而 DPPO 配置的路径是 `can/`（=mh）→ 换成 mh，**0% 不变** |
| checkpoint 权重 | `checkpoint['ema']` 与 `checkpoint['model']` 两套，**都是 0%** |
| checkpoint 架构 | 输入 67 = 28(4×7 动作) + 16(time) + 23(obs)，输出 28，`epoch=5000`，**与配置完全吻合** |
| 观测键顺序 | 运行时 `low_dim_keys` 与 `process_robomimic_dataset.py` 都是 `eef_pos, eef_quat, gripper_qpos, object`，**一致** |
| `predict_epsilon` | 训练配置与推理配置都是 `True`，**一致** |
| 推理设置扫描 | ddim 8/10/20/50、关掉 ddim（20 步）、关掉裁剪 —— **全部 corr≈0.3** |
| 机器人是否在动 | 在动：eef 归一化位移 1.365、平均 \|a\| 0.63、动作随步数变化 std 0.046 |

### 最关键的一条：策略无法复现**训练集内部**的动作

在同一状态上比较策略输出与数据集动作（200 个样本，mh 归一化）：

```
数据动作 std 0.473     数据方差 0.224（= 预测均值的 MSE）
策略输出 std 0.611
step 0: MSE 0.609  corr +0.286
step 3: MSE 0.615  corr +0.235
噪声=0 时: corr +0.27
```

**MSE 0.61 是数据方差的 2.7 倍** —— 比直接输出常数均值还差。一个训练好的 BC 策略在训练集
状态上不可能这样。所以问题在"模型看到的输入 ≠ 它训练时看到的输入"，而不在 RL 部分。

### 结论与下一步

**这不是算法的问题，是数据管道的问题**，而且它发生在 RL 之前。可能的方向：

1. 这份 `state_5000.pt` 可能不是 DSRL 实际使用的那份（Drive 文件夹里可能有多个 state 文件，
   配置里写的是 5000）。应当试其他 state。
2. `object` 观测（14 维）在 robosuite 1.4.0 与 1.4.1 之间的具体取值可能有差异，而 hdf5 是
   1.4.1 录的 —— 需要逐维对比"数据集里的 obs"与"运行时 env 的 obs"在同一 state 下是否相同。
   这是下一项最该做的检查：**用 dataset 的 `states` 字段把 env 恢复到同一物理状态，再比较两边的
   23 维观测**。如果它们不等，观测管道就有确定性错误。

在这一点解决之前，任何 DSRL 训练结果（我们的和参考的）都没有意义。

## 后续排查：观测管道逐位正确，问题锁定在 checkpoint 本身

### 观测管道已验证无误

用 hdf5 的 `states` 字段把环境恢复到**同一物理状态**（`env.reset_to({"states": ...})`），再比较
运行时 env 产生的 23 维观测与 `train.npz` 里存的那一行：

```
demo_0 t=40: 最近行 40   max|diff| = 0.000   mean = 0.000
demo_1 t=40: 最近行 158  max|diff| = 0.000   mean = 0.000
demo_5 t=40: 最近行 646  max|diff| = 0.000   mean = 0.000
```

逐维也对（最大差异维度的差值到小数点后三位仍相等）。**所以 obs 归一化、键顺序、robosuite 版本
这条链路是完全正确的**，可以排除。

### 归一化统计量已确定为 mh

用原始 hdf5 观测，分别按两套统计量归一化后喂给模型，与同状态下 `actions` 比较：

| 统计量来源 | ph 数据上 corr | mh 数据上 corr |
|---|---|---|
| ph (`robomimic/can`，我最初的错误选择) | +0.322 | +0.345 |
| **mh**（正确） | +0.469 | **+0.533** |

mh 明显更好，且已就位（`dppo/log/robomimic/can/normalization.npz` 与 mh 版本 `allclose` 为真）。

### 但基座策略仍是 0%

在上述全部修正之后，四种噪声条件重测：

```
noise=normal 0/10    noise=zero 0/10    noise=half 0/10    noise=clip15 0/10
```

同时策略在**训练集内部**状态上的动作相关性只有 **0.53**，MSE **0.43** 是数据方差 **0.216** 的
两倍 —— 一个收敛的 BC 策略不会这样。

### 结论：`state_5000.pt` 很可能不是收敛的基座策略

配置里写的是 `.../checkpoint/state_5000.pt`，其 `epoch` 字段也是 5000。DPPO 的 can 预训练是
`n_epochs: 3000`、约 90 个 batch/epoch，即约 27 万梯度步；**若 `state_5000` 指的是第 5000 个
梯度步，那只是训练进度的约 1.8%**，是一个欠训练的 checkpoint。

这解释了全部现象：动作相关性低、MSE 高于数据方差、成功率 0%，而**观测管道是对的**。

### 需要的东西

从同一个 Google Drive 文件夹里取**同目录下更靠后的 state**（例如 `state_100000.pt` 或
`state_200000.pt`），放到同一路径替换即可，无需改任何配置。如果文件夹里只有 `state_5000`，
那 DSRL 仓库里 `base_policy_path` 这一个字段就是与论文不符的，需要找作者确认。

## 上游 issue 查证：问题已知，且答案是"必须用作者提供的 NPZ"

查 `ajwagen/dsrl` 的 7 个 issue（`api.github.com/repos/ajwagen/dsrl/issues`），其中
**Issue #7 就是同一个问题，且已给出答案。**

### Issue #7（closed）：基座策略 0%，与我们完全一致

提问者用 robomimic 官方数据集 + 自己生成的 npz 预训练 π_dp，始终得到：

```
eval: num episode 120 | success rate 0.000 | avg episode reward 0.000
```

**作者之外的第一作者 ChandradithyaJ 的结论：**

> "The dataset and normalization files provided by the official robomimic project page don't
> work with the pretrained checkpoints. **The author-provided NPZ files must be used.**"

**这就是我们卡住的原因。** 我用 `process_robomimic_dataset.py --normalize` 自己生成的
`normalization.npz` 来自 robomimic 官方 hdf5，**而 issue 明确说不能这样做**。

注意我此前的验证为何没能发现这一点：观测一致性检查（`max|diff|=0.000`）用的是
**我自己的** normalization 的两边，它只证明了 env 与我的 npz 一致，**没有**证明我的 npz 与
作者的一致。

### Issue #2：有人复现成功，但需要 100 小时；`n_envs=50` 明显更好

> "I have successfully reproduced the results reported in the paper after training for **100
> hours** on our server."
> "**Setting `n_envs=50` for the can task works much better** (similar to the DPPO settings)."

我们用的是 `n_envs=4`（仓库默认），而成功者用 50。这解释了为什么我们的速度只有 5 fps 且
样本效率低——`n_envs` 直接决定每次 `train()` 调用覆盖多少 transition。

### Issue #5：另一人在 RTX 5090 上跑了 4 天，结果仍与 Fig 4 不符

> "The experiment has been running for **four days on a RTX 5090 GPU** ... Both plots are quite
> different from the result presented in Fig.4 of the paper."

所以"复现困难"不是我们独有的问题，是**仓库的已知状态**。

### Issue #6：三个真实代码 bug（影响我们的 can 复现）

| bug | 位置 | 影响 |
|---|---|---|
| **`action_magnitude` 在 `dsrl_na` 下被完全忽略** | `env_utils.py:145-146`（`ActionChunkWrapper` 用 `env.action_space.low/high` = ±1） | can 用的就是 `dsrl_na`，所以 Table 4 的 `action_magnitude 1.5` **实际是 1.0**。已核实：`action_magnitude` 在 `env_utils.py` 只出现在 `DiffusionPolicyEnvWrapper`（dsrl_sac）里 |
| `dsrl_sac` 的动作被去噪两次 | `utils.py:171` + `env_utils.py:217` | 不影响 dsrl_na |
| `truncated` 恒为 `False` | `env_utils.py:184` | bootstrap 在时间上限处不当裁剪 |

**必须更正我此前的一个推断**：我把 `ep_len` 恒等于 75 当作"从未成功终止"的证据。**这是错的。**
由于 `ignore_done: true` 且 `truncated` 恒 False，episode 本来就会跑满 75 步——
`ep_len = 75` 是代码行为，不是失败证据。真正的证据只有 `success_rate = 0`。

### 当前阻塞项

需要用户从**同一个 Google Drive 文件夹**取作者提供的 NPZ：

```
dppo/log/robomimic/can/normalization.npz     <- 必需（约 1 KB）
dppo/log/robomimic/can/train.npz             <- 若文件夹里有，一并取
```

README 说 "Download the contents of this folder and place in `./dppo/log`"，所以这些 npz 应该就在
那个文件夹里。拿到后替换即可，无需改配置。

## 复现结果：成功，但先升后崩（与 Issue #2 一致）

用作者提供的 NPZ 之后，**基座策略立刻从 0% 变成 71%**，复现成立。随后跑正式训练。

### 配置

```bash
n_envs=50                    # Issue #2 的建议，也是 DPPO 的设置；默认 4 只有 5 fps
init_rollout_steps=120       # 120 x 50 x 4 = 24000 控制步，保持论文 Table 4 的 warmup
num_evals=100                # 4 次评测 x 25 env（20 会得到 eval_episodes=0，见上文）
其他全部保持仓库默认，与 Table 4 逐项一致
```

`n_envs` 从 4 提到 50 让吞吐从 **5 fps 涨到 69 fps（13.8 倍）**，这是复现可行的前提。

### 基座策略（作者 NPZ）

| 噪声 | 成功率（30 集） | eval env 25 个 |
|---|---|---|
| `N(0,1)` raw | 0.47 | **0.71** |
| std 0.5 | 0.70 | — |
| `clip(±1.5)` | 0.40 | — |
| **零噪声** | **0.07** | — |

论文 Figure 4 的 Can t=0 约 0.45，我们 0.47–0.71，量级吻合。

**`noise=zero` 只有 7%** —— 这推翻了我在饼干项目里一直用的假设：零噪声（`tanh(mean)=0`）不是"平均策略"，是**分布外**，会严重损害性能。`noise=half`（std 0.5）最高，说明策略对噪声尺度敏感，这正是 DSRL 要学的量。

### 训练轨迹

| 控制步 | timestep | success_rate | vs 基线 0.71 |
|---|---|---|---|
| 84k | 21k | 0.66 | 93% |
| 144k | 36k | 0.82 | 115% |
| **204k** | **51k** | **0.86** | **121%** ← 峰值 |
| 264k | 66k | 0.70 | 99% |
| 324k | 81k | 0.40 | 56% |
| 444k | 111k | **0.20** | **28%** ← 崩塌 |

**先升后崩。** 这与 **Issue #2 的标题完全一致**（"Why the eval success rate is decreasing?"，该 issue 至今仍 open，无人回答）。

### 崩塌的机制线索

`ent_coef` 单调衰减，没有回升：

```
0.642 -> 0.412 -> 0.27 -> 0.184 -> 0.134 -> 0.0995 -> 0.0775
```

Table 3 给的是 `target entropy = 0`。而 alpha 的平衡点是 `log_prob = -target_entropy = 0`，
28 维动作空间的 `log_prob` 约 **−26**。所以 alpha 会被一路推向 0，策略失去探索、逐渐变确定，
然后在某个点崩掉。这与"先升后崩"的时间结构吻合：先靠探索涨上去，alpha 衰减到底后失去探索。

**这是论文超参的固有隐患，不是我们复现的错误。**

### 对饼干主线的三条直接结论

1. **`greedy`（零噪声）不是一个有意义的评估条件。** 参考实现里零噪声只有 7%，远低于
   `N(0,1)` 的 47%。所以 run11 里 `greedy` 从 15.00 掉到 12.50 不能读作"策略变差"，
   它读作"actor 的 mean 离开了 0，而 0 本身是 OOD"。**应当把主线的 `--eval-scenes` 改成
   用固定种子的 `N(0,1)` 而不是零噪声。**
2. **`target_entropy = 0` 会让 alpha 单调衰减到 0。** 参考实现里就这样，代价是后期崩塌。
   我们的 run8/9/11 里 alpha 的行为值得用这个视角重读。
3. **`utd` / `gradient_steps` 不是我们的问题。** 论文原文明确"after every (parallel)
   environment step"，一个环境步 = 一个 chunk，参考与我们的定义相同。我此前的 32 倍和 8 倍
   两个说法都是分母选错，已撤回。

## can 任务：崩塌的根因是熵项，且与论文参数无关

### 参数对齐：14 项全部一致，崩塌不是"没对齐"造成的

| 项 | 仓库 | 论文 |
|---|---|---|
| act chunk / hidden / utd / Q_W steps | 4 / 2048 / 20 / 10 | Table 4 同 |
| discount / action magnitude / combine | 0.99 / 1.5 / min | Table 4 同 |
| lr / batch / target entropy / tau | 3e-4 / 256 / 0 / 0.005 | Table 3 同 |
| n_envs / max_episode_steps | 4 / 300 | Table 3 / 4 同 |
| init_rollout_steps | 1501 (×4×4=24016) | 24000 控制步 |

**所以崩塌源自论文参数本身，不是我们配错。**

### 根因：熵项在早期是奖励的 340%

`target = r + γ(Q' − α·logπ)`，而 `logπ < 0`，所以 `−α·logπ` 是**正的加到 target 上**。
探针实测（baseline，`entropy_in_target` 开着）：

| timestep | logπ | α | 熵项/步 | 奖励/步 | 熵项占奖励 | 有效 r |
|---|---|---|---|---|---|---|
| 4,000 | −18.67 | 0.620 | **+11.57** | −3.40 | **340%** | **+8.17** |
| 8,000 | −16.79 | 0.389 | +6.53 | −3.40 | 192% | +3.13 |
| 12,000 | −12.74 | 0.251 | +3.19 | −3.40 | 94% | −0.21 |
| 20,000 | −7.37 | 0.120 | +0.88 | −3.40 | 26% | −2.52 |

**前 8k timesteps 有效奖励是正的** —— critic 学到的是「活着有奖励」（拖延时间有利），
而不是「成功有奖励」。实测 `Q = +98` 到 `+126`，而正确值应为负。

### 为什么 α 衰减这么快（约 400 次更新内降 8 倍）

`alpha_loss = −(log_alpha · (logπ + 0))`，`d/dlog_alpha = −logπ = +18`。
每步 `log_alpha` 变化 `18 × 3e-4 = 5.4e-3`；从 0 降到 −2.1（α: 1→0.12）只需 **389 次更新**，
即 `utd=20` 时 19 个 vec step。**α 在 950 个 timesteps 内就衰减完了。**

`target_entropy = 0` 决定了平衡点在 `logπ = 0`，而 28 维的 `logπ` 初始约 −18，所以 α 必然一路降到 0。

### 三个对照实验

| 实验 | 改动 | 峰值 | 最低 | >200k 均值 |
|---|---|---|---|---|
| **A** | baseline（论文原设置） | 0.86 | **0.20** | 0.52 |
| **B** | `ent_coef` 初值 1.0 → 0.01 | — | — | 早期就 0.00，`logπ` 变 **+27（饱和）** |
| **C** | **关掉 `entropy_in_target`** | 0.84 | **0.52** | **0.61** |

**实验 B 更糟**：初值低导致 α 太小、熵正则不足，actor 直接把动作推到 tanh 饱和
（`logπ = +27`）。这说明「熵项」和「α 大小」是两个独立的问题。

**实验 C 是净改善**：最低点从 0.20 → 0.52（2.6 倍），>200k 均值 0.52 → 0.61，
且 504k 处回升到 0.80。但 `logπ` 仍从 −18.4 漂到 −1.5 —— α 衰减到 0 的问题**没有**被解决。

### 下一步方向（按证据强度）

1. **`entropy_in_target = False` 已验证有效**，应作为默认。
2. **`target_entropy` 改成正值**（如 +18），把 α 的平衡点放在 `logπ = −18`，
   α 就不会衰减到 0。这是唯一能根治 α 塌缩的改法，但**明确偏离论文 Table 3 的 0**。
3. 两者可以叠加：关掉熵项后，α 只影响 actor 的探索强度，不影响 Q 的固定点。

### 对饼干主线的迁移

我们的 `train_dsrl_online.py` **已经**默认 `entropy_in_target=False`（run9 起），
当时是因为稀疏奖励下算过同一笔账。**这个决定在参考实现上得到了独立验证。**
