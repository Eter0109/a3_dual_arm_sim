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
