# 外部 VLA / WAM 模型接入方案

状态：设计提案，2026-09-27（VLA workstream）
前置文档：`docs/policy-interface.md`（LeRobot 系适配器，已实现）

## 实施状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **Phase 0** | 协议 + manifest + gateway + 四类 transport + `policy check` | **已完成** |
| **Phase 1** | 打通第一个真实外部模型（Cosmos Policy ALOHA 2B） | **已完成** |
| Phase 2 | Subprocess 隔离的实机验证 | 已完成（subprocess 已在真模型上跑通） |
| **Phase 3** | 按清单逐个加 manifest | **已完成两个**：Cosmos Policy、GR00T-N1-2B（ACG） |
| **Phase 4** | `a3-sim compare` 横向评测 | **已完成**，见 4.2 的实测表（sandbox 与饼干场景） |

### Phase 1 结果：Cosmos Policy 已经能驱动 A3

```bash
a3-sim run --scene sandbox --policy-manifest configs/policies/cosmos_policy_aloha.yaml --steps 32
```

实测（RTX 5090 / 32GB / sm_120）：

| 指标 | 值 |
| --- | --- |
| 模型 | NVIDIA Cosmos Policy, ALOHA 2B（`nvidia/Cosmos-Policy-ALOHA-Predict2-2B`） |
| 传输 | `subprocess`（独立 Python 3.10 + torch 2.7.0+cu128 环境） |
| 启动时间 | **21.7 s**（权重已在缓存） |
| 推理延迟 | 均值 **1321 ms** / 峰值 2091 ms（10 步去噪，chunk 50） |
| 显存 | **4.87 GB**（低于官方估计的 6.0 GB） |
| 相机 | 3 路全用（`dropped_cameras: []`） |
| 动作 | ALOHA 14 维 → A3 16 维，映射正确 |
| 模型自评价值 | 0.863（`cosmos_value`，写入 episode info） |

这一条命令验证了整条设计：观测投影、跨进程协议、维度映射、chunk 节拍、
动作可行性、episode 落盘。

### Phase 1 遇到的五个真实问题（全部已修）

按发现顺序，每一个都是"静默错误"类：

1. **`torch` 的 cu126/cu128 陷阱**
   清华镜像的 `torch==2.7.0` 是 **cu126** 标签，arch list 到 `sm_90` 为止，
   在 5090 上直接 `no kernel image is available`。必须显式
   `torch==2.7.0+cu128` 且只从 `download.pytorch.org/whl/cu128` 取。
   指南警告的"CUDA wheel 硬绑定"，方向是反的：新卡不能用旧 wheel。

2. **平台检测靠 `sys.argv` 猜，默认 LIBERO**
   `detect_robot_platform()` 搜 argv 里的 `libero`/`robocasa`/`aloha`，
   **都不是则默认 LIBERO（ACTION_DIM=7）**。
   处理：manifest 显式传 `platform=aloha`，适配器 import 后**复查**，
   把静默错配变成启动失败。

3. **重依赖其实有预编译 wheel（我一开始找错了地方）**
   `flash-attn`/`natten`/`transformer-engine` 在 GitHub Releases，
   直连不通、NVIDIA 官方索引只有源码 tarball。但 **Cosmos Policy 自己的依赖索引**
   托管了全部三个的预编译 wheel（torch 2.7/cu128/cp310 正好匹配）：
   `https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu128_torch27/simple/`
   实测三个融合内核在 sm_120 上都能执行。**教训：先查项目自己的索引。**

4. **协议不能与模型共用 stdout（设计缺陷）**
   Cosmos Policy 在加载前就往 stdout 打 `"Using ALOHA constants: ..."`。
   协议写在同一个管道后，A3 侧把日志的四个字符当成了帧长度（`"Usin"`
   ≈ 14 亿字节），双方一起卡在 `pipe_read` —— **死锁，无报错**。
   修复：协议改用专用文件描述符（`A3_BRIDGE_REQ_FD`/`RESPONSE_FD`），
   子进程的 stdout/stderr 全部让给模型日志。
   回归测试：`test_a_chatty_model_cannot_corrupt_the_protocol`。

5. **`hold` 逐步重读等于跟随漂移**
   模型不能驱动的关节（ALOHA 没有的第七腕关节）原本每步读"当前位置"
   并命令它，等于边漂边跟。实测 32 步内漂到 0.09–0.10 rad。
   修复：回合开始时**锁定**（gateway 的 `reset` 调用 `latch_hold_values`），
   漂移降到 0.0067 rad。

另外修掉的两个自己埋的 bug：

- 适配器默认的 `dataset_stats` 用 `f"{ckpt}/..."` 拼接，当 `ckpt` 是**文件**
  路径时拼成 `...pt/aloha_dataset_statistics.json`。改为从文件取父目录。
- 适配器兜底的 T5 标签写的是 `"fold the shirt"`，而缓存里的真实键是
  `"fold shirt"`。不匹配会触发下载 **45 GB 的 T5-11B** 来现场编码。
  改为**从缓存读取真实键名**来匹配，并把替换事实写进 episode info。

### 环境事实（本机）

| 项 | 事实 |
| --- | --- |
| GPU | RTX 5090, sm_120, 32GB, 单卡 |
| 可用 torch | **必须 cu128+**（cu121/cu124/cu126 都无 sm_120 kernel） |
| 代理 | `gh-proxy.com` **6–11 MB/s**；`ghfast.top`/`ghproxy.net` 仅 25–30 KB/s |
| HF | 直连不通；`HF_ENDPOINT=https://hf-mirror.com` 可用 |
| gated 模型 | `nvidia/Cosmos-Predict2-2B-Video2World` 需 HF token（基座 DiT + VAE） |
| GL 后端 | `MUJOCO_GL=egl`（需 `apt install libegl1`）；glfw 无 X display 不可用 |

### Phase 1 进展：Cosmos Policy

选它的理由（均经核实）：

| 事实 | 来源 | 意义 |
| --- | --- | --- |
| ALOHA 任务仅需 **6.0 GB 显存** | 官方 README 的 System Requirements | 本机 32GB 绰绰有余 |
| 锁定 **torch 2.7.0+cu128** | `pyproject.toml` 的 `cu128` extra | cu128 是首个含 sm_120 kernel 的 wheel，RTX 5090 能跑 |
| **双臂**（ALOHA 2）、**3 路相机** | `ALOHA_CONSTANTS`、`deploy.py` | 与 A3 的 3 相机 + 双臂一一对应，零丢弃 |
| 自带 server-client | `deploy.py` / `run_aloha_eval.py` | 正好对应方案的 `subprocess`/`socket` transport |
| 权重 2B | `nvidia/Cosmos-Policy-ALOHA-Predict2-2B` | 下载量可控 |

集成中发现的三个**真实陷阱**（都已处理，都是"静默错误"类）：

1. **`torch` 的 cu126/cu128 陷阱（最严重）**
   同时配置 PyPI 镜像与 pytorch 索引时，uv 会从镜像拿 `torch==2.7.0`（即
   **cu126** 标签）——它的 arch list 是 `sm_50..sm_90`，**没有 sm_120**，在 5090 上直接
   `CUDA error: no kernel image is available for execution on the device`。
   必须显式 `torch==2.7.0+cu128` 且只从 `download.pytorch.org/whl/cu128` 取。
   这正是指南第五节警告的"CUDA wheel 版本硬绑定"，只不过方向相反：
   新卡不能用旧 wheel。

2. **Cosmos Policy 的平台检测靠 `sys.argv` 关键字**
   `detect_robot_platform()` 搜索 argv 里有没有 `libero`/`robocasa`/`aloha`，
   **都不是则默认 LIBERO（ACTION_DIM=7）**。若启动命令里没出现 "aloha"，
   就会拿着 ALOHA 权重跑 LIBERO 常量，得到**形状不对的动作而不是报错**。
   处理：manifest 显式传 `platform=aloha`，而且适配器在 import 后**复查**
   `ROBOT_PLATFORM`，把静默错配变成启动失败。

3. **`flash-attn` / `natten` / `transformer-engine` 在 GitHub Releases 上，本地不可达**
   逐一核查后确认可跳过：`get_backend_list(arch_tag>=80)` 返回
   `["flash2", "cudnn", "natten"]`，且 flash2/natten 的 check 在包缺失时**返回 False**，
   attention 会回退到 **torch 内置的 cudnn**。ALOHA 推理路径用的是
   `CosmosPolicyVideo2WorldModel`，不引用 multiview camera 模块（那是
   `transformer_engine` 的硬 import 点）。因此手工建环境、只装 torch + 主依赖即可。

4. **关节数不同是真实的语义不匹配，不是配置问题**
   A3 每臂 7 关节 + 夹爪（16 维），ALOHA 每臂 6 关节 + 夹爪（14 维）。A3 布局是
   `[L0..L6, L_grip, R0..R6, R_grip]`，所以 ALOHA 无对应的是索引 6 与 14
   （`L_WRIST_Y` / `R_WRIST_Y`）。观测方向需要**非连续**选择——为此给 manifest
   加了 `observation.vectors.*.indices`；动作方向已有的 `joint_map` 恰好够用。
   两者使用同一张映射表 `[0,1,2,3,4,5,7,8,9,10,11,12,13,15]`。

配套的 manifest 能力扩展（均由真实模型倒逼出来）：

| 新增 | 解决什么 |
| --- | --- |
| `observation.vectors.*.indices` | 非连续选取（关节数不同的模型），与 `slice` 互斥 |
| `action.default: hold` | 模型不控制的关节**保持实测值**，而非写常量 0 |
| `action.dt` | 记录模型训练频率（ALOHA 50 Hz）与 A3 控制率（20 Hz）的差异 |

### Phase 1 结论（历史记录：阻塞阶段）

以下是我最初误判为硬阻塞、后来实测证明可解的过程，保留下来是因为
"先查项目自带索引"这个教训具有普遍性。

当时的判断与纠正：

| 当时的判断 | 实际情况 |
| --- | --- |
| "TE 是硬依赖，只能源码编译" | 项目自带索引有预编译 wheel，sm_120 可直接用 |
| "NVIDIA 索引无 wheel" | 那个索引确实只有 tarball；wheel 在 **cosmos-dependencies** 索引 |
| 结论：需装 nvcc 编译 1–1.5h | 实际：三个 wheel 共 1.16 GB，1 分钟下完 |

真正的阻塞（随后已解决）是 **gated 基座模型**：
`nvidia/Cosmos-Predict2-2B-Video2World` 需要 HF 账号接受 NVIDIA 许可。


已完成的验证：

- **cu128 torch 在 sm_120 上跑通**：arch list 含 `sm_120`/`compute_120`，
  bf16 matmul 与 SDPA 均正常（前一次装的 cu126 直接 `no kernel image`）；
- **适配器加载链走通到模型构造前一步**：平台检测→ALOHA 常量 14/14/50 ✓、
  `PolicyServer` 实例化、megatron/cosmos_policy 导入 ✓；
- **错误上报跨进程完整**：失败时服务端把**完整 traceback** 回传，
  A3 侧得到的是精确的 `fused_adam.py:17: import transformer_engine as te`，
  而不是挂起或泛化错误 —— 这正是 Phase 0 设计里 `error` 帧存在的意义；
- **14→16 维映射已被端到端验证**（`tests/test_bridge_cosmos_mapping.py`）：
  ALOHA 的 14 个值精确落到它们命名的 14 个 A3 关节，
  两个 `*_WRIST_Y` 保持实测值；16 维输入被**拒绝而非截断**。

阻塞根因：

| 环节 | 事实 |
| --- | --- |
| 谁需要 TE | `predict2` 的 DiT 骨架 `minimal_v4_dit.py`：**模块级** `import transformer_engine`、
从 TE 取 `apply_rotary_pos_emb`、且 `transformer_engine` 是 attention 的**默认后端** |
| 怎么被引入的 | `config.py → config_v2 → ema → text2world_model → optim_instantiate
→ fused_adam.py`，即**仅配置导入链就会碰到**，与是否训练无关 |
| 有 wheel 吗 | NVIDIA 索引（`pypi.nvidia.com` → 重定向到 `pypi.nvidia.cn`）
上 `transformer_engine_torch` 共 **28 个包，全部是 .tar.gz，零 wheel** |
| 能源码编译吗 | 本容器**无 nvcc、无 docker**。`conda install -c nvidia cuda-nvcc=12.8.93`
可得，即理论可行，但需自建 CUDA 12.8 工具链 + 长时间编译 TE，且上游自己的
安装文档就是让用户用 Docker（`docker build -t cosmos-policy docker`） |

值得注意：`megatron.core` 在缺 TE 时会明确提示
*"Transformer Engine and Apex are not installed. **Falling back to local
implementations**"* —— 所以 megatron 那一层确实可跳过，
但 Cosmos Policy 自己的 DiT 没有这个回退。

后续选项（按成本排序）：

1. **装 nvcc 并编译 TE**（约 1–1.5 小时，且 CUDA 12.8 工具链需自建）→
   然后 Cosmos Policy 应能跑；
2. **改用官方 Docker 镜像**（`nvcr.io/nvidia/pytorch:25.10-py3` 对应 cu130 extra），
   但本容器无 docker；
3. **换一个不依赖 CUDA 扩展编译的模型**。按指南的依赖清单，
   ViFailback（8B，20–24 GB，指南称"依赖最干净"）是下一个候选。

Phase 0 交付的模块：

| 文件 | 职责 |
| --- | --- |
| `bridge/protocol.py` | `a3-bridge/1` 帧格式、`Request`/`ActionChunk`、流式读写 |
| `bridge/manifest.py` | YAML 解析 + `validate_manifest` 审计 |
| `bridge/observation.py` | 观测投影（取键、裁剪、resize、类型） |
| `bridge/action_bridge.py` | 反归一化 → 维度重排 → 空间转换 |
| `bridge/normalize.py` | `mean_std` / `min_max` / `q01_q99` 的逆运算 |
| `bridge/chunk.py` | chunk → 逐帧节拍 |
| `bridge/transport.py` | `in_process` / `subprocess` / `socket` / `replay` |
| `bridge/gateway.py` | 组装成 `Policy`，暴露 `metrics` / `success_override` |
| `scripts/a3_bridge_server.py` | 外部环境侧的自包含服务端（仅 numpy + 标准库） |
| `scripts/make_replay_recording.py` | 生成 replay 录制（hold 姿态或数据集 episode） |
| `configs/policies/*.yaml` | 示例与模板 |

### 用法

```bash
# 不加载任何模型，只审计接线是否自洽
a3-sim policy check configs/policies/subprocess_template.yaml

# 用录制数据跑通全链路（无需模型）
python scripts/make_replay_recording.py --source hold --steps 200 \
    --output configs/policies/replay_hold.json
a3-sim run --scene sandbox --policy-manifest configs/policies/replay_hold.yaml --steps 100

# 跑真实外部模型
a3-sim run --scene cookie_transfer --policy-manifest configs/policies/cosmos_policy.yaml \
    --steps 2000 --record /data/a3-cosmos
```

`policy check` 能在**不加载模型**的前提下回答：必填项是否齐全、`cwd`/`env` 是否存在、
观测键是否是 A3 能提供的、动作维度与空间是否自洽、归一化统计量文件是否存在。
这一条命令把"跑起来才发现"的成本降到"配置时发现"。

## 目标

让清单里 **16 个外部仓库**（DreamZero、LingBot-VA、Fast-WAM、ACG、Cosmos Policy、
AtomicVLA、OptimusVLA、ViFailback、Action-Sketcher、VITRA、WoG、InSpire、MM-ACT、
LLaVA-VLA、Dexora、TrajBooster）能在**不改 A3 代码、不改外部项目源码**的前提下，
被 `a3-sim` 用同一条命令驱动，并把 rollout 落进同一个 LeRobot v3 数据集。

判定标准：

1. 新增一个模型 = 新增一个 **manifest 文件**，不新增 Python 模块；
2. 外部模型跑在**它自己的 conda 环境 / 进程 / 机器**里，与 A3 的 torch 版本互不干扰；
3. 落盘数据永远是 A3 原生契约（`observation.state` 16 维、`action` 16 维、3 路相机），
   因此不同模型的数据集可以**直接横向对比**。

---

## 一、现状诊断

### 1.1 已有接缝是对的，但太窄

`EpisodeRunner` 是唯一的 rollout 循环，它只依赖 `policy.Policy` 协议
（`action_mode` / `reset` / `act` / `close`）：

```
a3-sim CLI / evaluation / collection
              │
              ▼
       EpisodeRunner                     ← 不变，是正确接缝
              │  Policy: act(obs, task) -> ndarray
              ▼
   ┌──────────────────────────┐
   │ LeRobotPolicyAdapter     │  ← 唯一的学习策略适配器
   │  · PreTrainedConfig      │     ← 这四样都是 lerobot 专有
   │  · LeRobotDatasetMetadata│
   │  · make_policy           │
   │  · prepare_obs_for_infer │
   └──────────────────────────┘
              │
              ▼
      lerobot 的 16 种 policy
```

协议本身够用，**问题在于它以下只有一条实现路径，而那条路径是 lerobot 专用的**。
外部模型不提供 `PreTrainedConfig`、不提供 `ds_meta`、不提供 `make_policy`，
所以 `LeRobotPolicyAdapter` 覆盖不到清单里的任何一个仓库。

### 1.2 真正的约束是版本不可共存，不是接口形态

`租GPU环境选型指南.md` 的版本矩阵给出硬边界：

| | 范围 | 后果 |
| --- | --- | --- |
| PyTorch | 2.1.0（Dexora）→ 2.9.1（DreamZero） | 无法在同一进程 import |
| Python | 3.10（多数）→ 3.13（MM-ACT） | 无法共用一个解释器 |
| CUDA wheel | cu121 / cu124 / cu126 / cu128 / cu129 / cu130 | 硬绑定，不可混装 |
| flash-attn | 2.5.5 / 2.7.1.post4 / 2.7.3 / 2.8.3，且与 torch 配套编译 | 每簇一个 |

**结论：外部模型不可能与 A3 MuJoCo 仿真同进程加载。**
统一接口的主战场因此不是"再写一个 in-process 适配器"，而是**跨进程边界**——
边界之上是 A3 契约，边界之下是各模型自己的世界。

### 1.3 现成资产（不要重造）

| 资产 | 位置 | 在新方案中的角色 |
| --- | --- | --- |
| `EpisodeRunner` 只记录 `info["applied_action"]` | `runner.py:83` | **免费的一致性保证**：无论外部模型输出什么空间，落盘永远是 A3 原生 16 维 joint。数据集天然可比。 |
| `CartesianDeltaAdapter` | `actions.py` | 14 维双臂 EEF delta → 16 维 joint（含 IK）。**openpi 系（AtomicVLA / OptimusVLA）直接复用。** |
| `validate_action` / `validate_observation` | `contracts.py` | 边界校验，跨进程数据进来先过这一关。 |
| `load_policy(spec, env)` + `module:factory` | `policy.py:47` | 新 gateway 就用这个挂进去，CLI 不用改。 |
| `jsonable()` | `contracts.py` | 序列化辅助。 |
| `checkpoint_embeds_vlm` | `checkpoint.py` | 判断 checkpoint 是否自带 VLM，避免重复下载基座。 |

---

## 二、方案总体结构

三层 + 一张注册表。每层职责单一，外部模型只需触碰最底层的 protocol。

```
        a3-sim CLI / evaluation / collection          (不变)
                          │
                          ▼
                 EpisodeRunner                        (不变)
                          │  Policy: act(obs, task) -> ndarray[16]
                          ▼
    ┌───────────────────────────────────────────────────────┐
    │              ExternalPolicyGateway   (新)             │
    │  由 manifest 驱动；不含任何模型专有代码                  │
    │                                                       │
    │   ┌────────────────┐  ┌────────────────┐              │
    │   │ObservationView │  │  ActionBridge  │              │
    │   │ 取键/裁剪/缩放  │  │ 空间/维度/归一化│              │
    │   └────────────────┘  └────────────────┘              │
    │   ┌────────────────┐  ┌────────────────┐              │
    │   │  ChunkBuffer   │  │  SuccessProbe  │              │
    │   │ chunk→20Hz节拍 │  │  任务级成功判定 │              │
    │   └────────────────┘  └────────────────┘              │
    └───────────────────────────────────────────────────────┘
                          │  Transport（可替换）
        ┌─────────────────┼──────────────────┬─────────────┐
        ▼                 ▼                  ▼             ▼
   InProcess         Subprocess          Socket         Replay
   同环境直调        另一 conda env      远程/独立服务   离线回放
   (lerobot 系)      (DreamZero 等)      (openpi server) (回归测试)
```

### 2.1 为什么必须分 Transport

| Transport | 用于 | 为什么不能合并 |
| --- | --- | --- |
| `InProcess` | lerobot 系（SmolVLA/ACT/pi0），torch 与 A3 兼容 | 零拷贝、无序列化开销，调试最方便 |
| `Subprocess` | DreamZero、LingBot-VA、Fast-WAM…（版本冲突） | 必须换解释器/conda env；但不需要开端口 |
| `Socket` | openpi 系（AtomicVLA/OptimusVLA 自带 websocket policy server）、真机 | 模型已在别处跑着；或跨机器 |
| `Replay` | 测试 | 无模型也能验证 gateway/数据集链路，CI 用 |

`Replay` 不是玩具：手册第五条"避坑清单"里一半问题（归一化统计量、动作尺度）
只能靠固定输入→固定输出的回归测试发现。

### 2.2 Gateway 的职责边界（只做机械的四件事）

1. **投影观测** —— 按 manifest 取键、裁剪、缩放、类型转换；缺键时**报错而不是静默降级**
   （`docs/policy-interface.md` 的 finding 8：lerobot 会静默接受缺失相机）。
2. **适配动作** —— 空间转换（`cartesian_delta`→`joint_position` 等）、维度重排、
   反归一化、chunk 展开。
3. **管理时序** —— 把模型的 chunk 拆成 `control_hz` 的逐帧动作；re-plan 时机；
   可选 temporal ensemble。
4. **持有生命周期** —— `reset()` 清 chunk 与外部侧状态；`close()` 回收进程/连接。

**不做的事**：不解析 checkpoint、不构造模型、不 import torch。
一旦 Gateway 里出现模型名，方案就退化成"每个模型一个插件文件"，即当前状态。

---

## 三、Wire Protocol（跨进程契约）

跨进程边界上只传 numpy 与标量，格式与任何框架无关。

### 3.1 请求（A3 → 模型）

```jsonc
{
  "protocol": "a3-bridge/1",
  "episode_id": "0f3a…",          // 一个 episode 内恒定，用于外部侧重置
  "step": 137,
  "task": "cookie batch",          // 语言指令，直接转发
  "observation": {
    "images.front":      { "dtype": "uint8", "shape": [224, 224, 3], "data": "<bin>" },
    "images.left_wrist": { "…": "…" },
    "images.right_wrist":{ "…": "…" },
    "state":             { "dtype": "float32", "shape": [16], "data": "<bin>" },
    "velocity":          { "dtype": "float32", "shape": [16], "data": "<bin>" }
  },
  "timestamp": 6.85
}
```

### 3.2 响应（模型 → A3）

```jsonc
{
  "protocol": "a3-bridge/1",
  "actions": { "dtype": "float32", "shape": [50, 14], "data": "<bin>" },
  "action_space": "cartesian_delta",   // joint_position | cartesian_delta | eef_pose | …
  "reference_frame": "current_eef",    // 仅 delta 空间需要
  "units": { "translation": "normalized", "rotation": "normalized", "gripper": "[-1,1]" },
  "horizon": 50,                        // 执行多少帧
  "replan_after": 50,                   // 下一次请求前执行几步
  "latency_budget_ms": 200,             // 外部侧自报，A3 用它决定是否超时
  "extra": { }                          // 原样并入 EpisodeResult.final_info
}
```

设计要点：

- **`units` 必须显式声明**。手册避坑清单第 4 条（VITRA 归一化统计量、Fast-WAM 的
  `*_dataset_stats.json`）说明动作尺度是最常见的静默错误来源。声明了才能校验。
- **`action_space` 允许非 A3 空间**。转换在 Gateway 做，A3 侧永远只见 16 维 joint。
- **一次返回 chunk**。手册 finding 12 已证明开环 chunk 优于逐步重规划，
  且 VLA 推理慢（3–10 Hz）而 A3 控制 20 Hz，必须支持 chunk。
- **`reset` 需要确认帧**。只写不读会在管道里留下一个未读回复，下一次请求会读到它。
- 传输编码：自定义**长度前缀帧**（magic + JSON header + 裸二进制），
  不依赖 msgpack：比 JSON+base64 小、比 pickle 安全、且外部环境只需 numpy。

---

## 四、Manifest（每模型一个文件）

放 `configs/policies/<name>.yaml`。**新增模型 = 新增此文件。**

```yaml
# configs/policies/example.yaml
name: example
protocol: a3-bridge/1

transport:
  kind: subprocess              # in_process | subprocess | socket | replay
  runtime:
    kind: conda                 # conda | docker | local | ssh
    env: vla27                  # 见选型指南 3.2：python3.10 + torch2.7.1 + cu128
    cwd: ~/models/Example
    cmd: [python, -m, a3_bridge.server, --ckpt checkpoints/example]
    startup_timeout_s: 300      # 大模型加载慢，超时要给足
  # socket 专有：
  # url: ws://127.0.0.1:8000

observation:
  images:
    front:       { source: "observation.images.front",       resize: [224, 224] }
    left_wrist:  { source: "observation.images.left_wrist",  resize: [224, 224] }
    # 模型只要 2 路时显式丢弃第三路（见 policy-interface.md 开放问题 4）
  state:
    source: "observation.state"
    slice: [0, 16]              # 取哪一段；外部只要关节角时给 [0, 14]
    dtype: float32
  rename: {}                    # A3 键 → 模型期望键，最终映射

action:
  source_space: cartesian_delta # 模型原生输出空间
  target_space: joint_position  # A3 契约，一般不改
  dim: 14                       # 模型原生维度
  chunk: 50
  execute: 50                   # receding horizon
  normalize:
    stats: ~/models/Example/norm_stats.json
    mode: mean_std

success:
  probe: env                    # env | policy | none

env:
  # conda run 包装；Gateway 负责生成实际命令
  PYTHONPATH: .
```

### 4.1 Manifest 校验

`a3-sim policy check <manifest>` 应能在**不加载模型**的情况下回答：

- 文件是否合法、必填项是否齐全；
- `cwd` / `env` / `cmd` 是否可达；
- `observation.*.source` 是否是 A3 能提供的键（对照 `ENVIRONMENT_KEYS`）；
- `action.dim` 与 `source_space` 是否自洽（如 `joint_position` 必须是 16 或 14）；
- `normalize.stats` 文件是否存在且维度匹配。

这一条命令能把"跑起来才发现"的成本降到"配置时发现"。

---

## 五、外部项目侧的桥（Bridge）

外部项目**不改源码**，只加一个薄客户端。两种模式：

### 模式 A：命令行服务（推荐，隔离最彻底）

外部环境里安装一个约 200 行的 `a3_bridge` 包（我方维护，随 A3 仓库分发）：

```
外部 env:  python -m a3_bridge.server --adapter adapters/example.py --ckpt ...
                │  stdin/stdout 或 unix socket
A3 env:    ExternalPolicyGateway(transport=Subprocess)
```

`adapters/example.py` 是**用户唯一要写的东西**，只有两个函数：

```python
def load(cfg):                       # 在外部 env 里 import torch / 构造模型
    from example import ExampleModel
    return ExampleModel.load(cfg.ckpt)

def infer(model, obs, task, state):  # obs 是 numpy dict，无框架类型
    return model.predict(obs["images.front"], task, return_chunk=True)
```

### 模式 B：直接挂已有 server

openpi 系（AtomicVLA / OptimusVLA）自带 websocket policy server，
`transport.kind: socket` 直接连，**零外部改动**：

```yaml
transport:
  kind: socket
  url: ws://127.0.0.1:8000
```

### 5.1 为什么是"外部侧薄、A3 侧厚"

归一化统计量、动作空间转换、chunk 展开全部留在 A3 侧，因为：

- 这些是**评测口径**，必须所有模型用同一套实现，否则横向对比无意义；
- 外部侧代码跑在隔离 env 里，出错只能靠日志，越薄越好排查。

---

## 六、动作空间适配矩阵（针对清单）

Gateway 需要覆盖的转换，按难度排序：

| 转换 | 实现位置 | 状态 |
| --- | --- | --- |
| 16 维 joint 直通 | 无 | 已有 |
| 14 维双臂 EEF delta → 16 维 joint | `CartesianDeltaAdapter` | **已有，直接复用** |
| chunk `[H,D]` → 逐帧 `[D]` | `ChunkBuffer` | 待实现（约 80 行） |
| 反归一化（mean/std、min/max、q01/q99） | `Normalizer` | 待实现（约 60 行） |
| 7 维单臂 → 14/16 维（另一臂保持 home） | `SingleArmBridge` | 待实现 |
| `eef_pose` 绝对位姿 → joint | IK，`CartesianDeltaAdapter` 的变体 | 待实现 |
| 二值夹爪 → `[0,1]` 连续 | `GripperMap` | 待实现 |
| 14 维 ALOHA joint → 16 维（夹爪映射） | `GripperMap` 变体 | 待实现（openpi 系需要） |

**优先做 `ChunkBuffer` + `Normalizer`**：手册第五条列出的一半坑（归一化统计量、
动作尺度、`n_action_steps` 调参）都落在它们身上，而且与具体模型无关。

### 6.1 待核实项（不猜测，用探针脚本量）

Phase 1 已完成 Cosmos Policy；其余待核实的项：

| 仓库 | 需确认 | 状态 |
| --- | --- | --- |
| Cosmos Policy | 动作维度、显存、加载路径 | **已确认**：14 维、6.0GB（官方）、torch 2.7.0+cu128 |
| LingBot-VA | `attn_mode` 取值；diffusers pipeline 输入键名 | 仓库已定位：`Robbyant/lingbot-va` |
| Fast-WAM | `*_dataset_stats.json` 的统计量口径 | 权重已定位：`ellenzhou/fastwam` |
| ACG | GR00T-N1 的 embodiment tag 与动作维度 | 待查 |
| AtomicVLA / OptimusVLA | ALOHA 14 维是 joint 还是 EEF；server 端点 | 待查 |
| ViFailback | 8B 模型 + transformers 4.57 的输入键名 | 待查 |
| DreamZero | 推理入口；是否强制 2 卡 | **本机不可行**（单卡 32GB vs 14B/2 卡） |

### 6.2 本机硬件筛出的不可行项（重要）

RTX 5090 是 **sm_120**，而 CUDA 12.8 是首个支持 sm_120 的 toolkit。这把清单**反向**筛了
一遍——被淘汰的不是新框架而是**旧框架**：

| 仓库 | 锁定版本 | 在 5090 上 |
| --- | --- | --- |
| Dexora | torch 2.1.0 + cu121 | ✗ 无 sm_120 kernel |
| Action-Sketcher | torch 2.5.1 + cu124 | ✗ 无 sm_120 kernel |
| LLaVA-VLA | 未 pin，但 flash-attn 2.7.1.post4 | 需重编 flash-attn |
| Cosmos Policy | torch 2.7.0 + cu128 | ✓ **可行** |
| Fast-WAM / ACG / openpi 系 | torch 2.7.1 + cu128 | ✓ 理论可行 |
| LingBot-VA | torch 2.9.0 + cu126 | ✗ cu126 无 sm_120，需改 cu128 |
| DreamZero | torch 2.9.1 + cu129 | 版本可以，但 **14B 要 2 卡**，本机单卡 |

---

## 七、实施路线

### Phase 0 —— 接口底座（不依赖任何外部模型）—— 已完成

1. `docs/external-policy-bridge.md`（本文）
2. `src/a3_dual_arm_sim/bridge/protocol.py` — schema 定义 + 帧编解码
3. `src/a3_dual_arm_sim/bridge/manifest.py` — YAML 解析 + 校验（`policy check`）
4. `src/a3_dual_arm_sim/bridge/observation.py` / `action_bridge.py` / `chunk.py` / `normalize.py`
5. `src/a3_dual_arm_sim/bridge/gateway.py` — 组装成 `Policy`
6. `src/a3_dual_arm_sim/bridge/transport.py` — 四类 transport
7. `scripts/a3_bridge_server.py` — 外部环境侧服务端（自包含）

交付判据（已验证）：

- `Replay` transport 下 `a3-sim run` 跑通 episode，动作与输入逐帧对应；
- 跨真实进程边界（`subprocess` + 官方服务端）跑通，`startup_s`/`request_mean_ms`
  出现在 `EpisodeResult.final_info` 中；
- 声明宽度不符、录制耗尽、归一化缺失、依赖缺失等情形均**报错而非静默降级**。

测试期间发现并修正的真实缺陷（值得记录，因为都属于"静默错误"类）：

- `mean_std` 反归一化把 `std - mean` 当作缩放（应为 `std`）—— 由
  `test_mean_std_inverse_restores_physical_units` 抓到；
- `reset` 帧只写不读，导致后续请求读到上一个回复而失步 —— 表现为
  "模型返回了错误的帧类型"；
- `_FramedTransport.infer` 漏传 `chunk`，服务端回落到每步 1 个动作，
  一次 chunked rollout 静默退化为 50 倍推理。

### Phase 1 —— 打通第一个真实外部模型

选 **Cosmos Policy**（手册推荐第 1 个：权重 2B、6.8GB 显存、uv 一键环境）。
目标是暴露 Phase 0 设计里所有想当然的地方。

交付判据：一条命令跑出 episode，`applied_action` 在 A3 契约内，数据集可读。

### Phase 2 —— Subprocess transport + conda 隔离

验证 `conda run -n <env>` 包装、启动超时、崩溃恢复、stdin/stdout 粘包处理。

### Phase 3 —— 按清单顺序逐个加 manifest

顺序照抄手册第四节（Cosmos Policy → ViFailback → ACG → Fast-WAM → VITRA →
LingBot-VA → DreamZero），因为那是按"能跑起来"的成本排的。
每个模型只允许新增 `configs/policies/<name>.yaml` + `adapters/<name>.py`。

#### 3.1 候选筛选的实测记录（不要凭 README 判断）

逐个核查依赖后，手册里的候选在本机（sm_120 / 32GB / 单卡）的实际情况：

| 候选 | 实测结论 |
| --- | --- |
| **ViFailback** | **不是动作模型**——Qwen3-VL-8B 微调的失败诊断 VLM，输出文字与视觉标记，桥接无从驱动。**排除** |
| **Fast-WAM** | 需要完整 Wan2.2 栈：`Wan2.2-TI2V-5B` 34 GB（含 11.36 GB 文本编码器）+ 策略 12 GB ≈ **35 GB 下载**，显存约 26 GB。本机数据盘与 32GB 显存都吃紧。**暂缓** |
| **Action-Sketcher** | torch 2.5.1 + cu124，**无 sm_120 内核**。**不可行** |
| **Dexora** | torch 2.1.0 + cu121，同上。**不可行** |
| **DreamZero** | 14B，需 2 卡。**不可行** |
| **LingBot-VA** | pin cu126，**无 sm_120 内核**；需换 cu128 且未验证 |
| **VITRA** | 需 PyTorch3D 编译，风险最高 |
| **ACG** | torch 2.7.1+cu128 ✓、2B、基于 GR00T-N1，**双臂 DexMimicGen 变体与 A3 同构**。**已选** |

关键教训：手册按 H100/1TB 规划，而本机是 RTX 5090/32GB/**sm_120**。
sm_120 需要 CUDA ≥12.8 的 wheel，这把筛选**反向**做了一遍——淘汰的是**旧**框架
（cu121/cu124/cu126），不是新框架。

#### 3.2 ACG / GR00T-N1 集成（已完成）

```bash
a3-sim run --scene sandbox --policy-manifest configs/policies/gr00t_bimanual.yaml --steps 32
```

实测：

| 指标 | 值 |
| --- | --- |
| 模型 | `DAVIAN-Robotics/GR00T-N1-2B-tuned-DexMG-MG100-CrossEmbodiments`（2B） |
| 具身标签 | `bimanual_panda_gripper` |
| 启动 | **17.5 s** |
| 推理 | **415 ms** 均值 / 686 ms 峰值（chunk 16） |
| 传输 | `subprocess`，专用 fd |
| 相机 | 3 路（front + 双腕），无丢弃 |
| 动作 | GR00T 14 维 EEF delta → A3 `cartesian_delta` → A3 自带 IK |

这一条比 Cosmos 更有价值：GR00T 的 `bimanual_panda_gripper` 是**双臂 + 3 相机**，
且 A3 的 `observation.eef_pose`（14 维 = 2 臂 × (pos3 + quat4)）**恰好**就是
GR00T 需要的 `eef_pos`/`eef_quat` 分解，`observation.state` 的夹爪位置也直接可用。
零相机丢弃、零维度捏造。

##### 集成中发现的四个问题（前三个是我的错，第三个最隐蔽）

1. **夹爪观测映射方向错误**
   两个手指是镜像的（finger 0 = [0.0009, 0.0406]，finger 1 = [-0.0407, 0.0003]）。
   统一的 `low + t*(high-low)` 会让张开时 finger 1 停在 ≈0，模型看到的夹爪**永远不是全开**。
   修正判据：**每个手指的张开端 = 离零更远的那一端**（这是平行夹爪的物理事实，
   不是该数据集的特性）。

2. **夹爪动作的仿射算错**
   我写了 `scale=-2, offset=+1`，它只在两端正确（靠 clip 兜住），中点是错的
   （GR00T 0 会变成全开而不是半开）。正确关系是 `A3 = -GR00T`，即
   `scale=-1, offset=0`。教训：**从约定推导，不要用端点拟合**。

3. **左右臂的命名约定相反（最隐蔽）**
   DexMimicGen 按**观察者视角**命名，A3 按**解剖学**命名：

   | | +y 侧 | −y 侧 |
   | --- | --- | --- |
   | A3 | 左臂 | 右臂 |
   | GR00T | `right_arm`（统计量 y 均值 **+0.23**） | `left_arm`（y 均值 **−0.235**） |

   两者都朝 +x（robosuite 的 `base_xpos_offset["table"] = (-0.16 - len/2, 0, 0)`，
   机器人位于 −x 朝桌面），所以**这是命名差异而非坐标系差异**，不需要镜像或旋转。
   但**按名配对**会把每只臂的命令发给另一只，而沿途每个数字看起来都合理。
   按**物理侧**配对后，GR00T 的右臂输入落在自身统计量 +0.23 附近（A3 左臂实测 +0.25），
   而不是 −0.1（分布外）。
   测试 `test_the_shipped_gr00t_manifest_pairs_arms_by_side` 断言的是结构性不变量——
   "观测为 right 的臂必须是动作驱动的同一只臂"，而非固定映射表。

4. **需要 flash-attn（我判断错了）**
   Eagle2 视觉主干的 config 里写死了 `attn_implementation="flash_attention_2"`，
   所以 transformers 强制要求该包。我先前看到 `_supports_flash_attn_2 = True` 就
   判断"只是能力声明"，错了。用 cosmos-dependencies 索引的同款 wheel 装上即可。

##### 结果解读：链路正确，但迁移行为退化

| 现象 | 证据 |
| --- | --- |
| 夹爪**双臂全程饱和**在 +1（闭合） | `gr00t_spans` 里 `*_gripper_close = [1.0, 1.0]` |
| 手臂被推到**关节限位** | `L_ELBOW_P = -1.919` 恰等于下限，`R_WRIST_P = 0.698` 恰等于上限 |

**对照实验**（重要）：把 `execute` 从 16 改为 1（每步重规划），关节值几乎相同
（L3: −1.788 vs −1.919）。所以**这不是开环漂移，而是策略本身的意图**。

预期之中的原因，如实记录：

- GR00T 在 DexMimicGen 的桌面双臂任务上训练，A3 是不同机器人、不同场景；
- A3 的前置相机在 `(1.3, −0.8, 1.55)`——**偏轴**（y = −0.8），而 robosuite 的
  `agentview` 通常在 y ≈ 0，主视觉输入视角不同；
- A3 的作业高度 z ∈ [0.83, 0.92]，位于 DexMG 训练范围 [0.81, 1.29]（均值 1.0）
  **偏低端**；
- **不存在任何 A3 演示数据**——这是零样本跨具身迁移。

因此这次验证的结论是：**桥接链路正确**（观测投影、协议、维度映射、IK、限位、
episode 落盘全部工作），而不是任务成功。这与 Cosmos Policy 的结论一致，
也正是 Phase 4 要显式声明口径的原因（见第七之二节）。

### Phase 4 —— 对比评测

`a3-sim compare` 已实现。它把多个 manifest 放在**同一场景、同一组 seed、同一任务字符串**
下运行，并按 regime 分组报告。

```bash
a3-sim compare --scene sandbox \
    --manifest configs/policies/replay_hold.yaml \
    --manifest configs/policies/cosmos_policy_aloha.yaml \
    --manifest configs/policies/gr00t_bimanual.yaml \
    --seeds 0 --steps 32 --output report.json
```

#### 4.1 为什么不能只报成功率

零样本 regime 下每个模型的成功率都预期接近 0——**16 个模型没有一个为 A3 训练过**。
一张全 0 的表无法区分"没教过"和"学不会"。所以每一步还额外评分：

| 指标 | 含义 |
| --- | --- |
| `corrected_fraction` | 环境在应用前**不得不修改命令**的步数占比（限位裁剪或速率限制）。**最清晰的"模型没在驱动这台机器人"的信号** |
| `rate_limited_fraction` | 上述修正中由**每步关节上限**（而非关节限位）造成的部分。高值意味着模型要求的每步行程超过机器人允许——是**尺度不匹配**而非决策错误 |
| `saturated_fraction` | 有臂关节**顶在限位**的步数占比 |
| `correction_max_rad` | 单步最大修正量（弧度） |
| `motion_mean_rad` | 平均每步关节行程。高行程 + 高修正 = 被命令猛动，而非真的动了 |

`replay_hold` 作为**对照组**参与：它是一段录制，不是模型，所以它的行为指标应当全为 0。
这验证了测量本身不引入噪声。

#### 4.2 实测结果（2026-09-28，RTX 5090，sandbox，**3 seeds × 400 步**）

| model | prompt | 丢弃相机 | success | corrected | saturated | correction max | motion mean | infer ms | startup s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay-hold（对照） | none | 2 | 0 | **0.000** | 0.000 | 0.000 | 0.000 | 0 | 0.0 |
| cosmos-policy-aloha | fixed_label | 0 | 0 | **0.098** | 0.000 | 0.611 | 0.028 | 886 | 21.0 |
| gr00t-n1-bimanual | free_text | 0 | 0 | 0.458 | **0.760** | 0.417 | 0.040 | 174 | 17.5 |

逐 seed 值（方差见 4.2.6）：

| model | corrected | saturated |
| --- | --- | --- |
| replay-hold | 0 / 0 / 0 | 0 / 0 / 0 |
| cosmos-policy-aloha | 0.090 / 0.095 / 0.108 | 0 / 0 / 0 |
| gr00t-n1-bimanual | 0.490 / 0.428 / 0.458 | 0.620 / 0.923 / 0.738 |

读法：

- **对照组精确为 0**，说明测量链路本身不产生噪声——这是这张表可信的前提。
- **两个模型都被速率限制主导**，即它们要求的每步行程都超过 A3 的
  `max_joint_step_rad = 0.05`。这是**尺度不匹配**：两个模型的动作尺度都为各自机器人的
  控制频率标定，而 A3 的每步上限更保守。
- **Cosmos 在 sandbox 上明显更"听话"**（corrected 0.098 vs 0.458），而 GR00T 有
  76% 的步数顶在关节限位上。
- **推理成本差 5 倍**（174 ms vs 886 ms），启动差 3.5 s。
- **相机丢弃数不同**（2 vs 0）——接线差异，必须与能力差异分开读。

##### 4.2.1 视频印证（`scripts/render_bridge_policy_video.py`）

四段各 400 帧 / 20 秒的视频（`outputs/videos/`）目视确认了数字，而且显示出
**两种截然不同的失效模式**：

| | GR00T | Cosmos |
| --- | --- | --- |
| sandbox 画面 | 双臂**向内塌陷、交叉贴在桌面**上 | 双臂**向左侧伸出工作区、几乎水平** |
| sandbox 腕部相机 | 紧贴桌面 | **空场景**（手臂已移出视野）|
| sandbox 数字 | `saturated 0.76` | `saturated 0.00` |
| 解释 | 把手臂推向行程末端，被限位挡住 | 移向不受约束的空域，故不被限位阻挡 |

两者都不是"接近完成任务"，但**失败的形状完全不同**——单看成功率的 0 完全看不出来。
这正是行为指标存在的理由。

**视频是一次 episode，不是表格的一行。** 由于两个模型都随机（见 4.2.6），
视频里的数字与该 seed 在表格里的值不会逐位相同：
GR00T 的 sandbox 视频给出 `corrected 0.388`，而它在 3-seed 表里 seed 0 是 `0.490`。
这不是矛盾，是方差。

##### 4.2.2 一个方法论问题：episode 长度会改变结论

同一组模型在 32 步与 200 步下的数字**不同**，且变化方向不一致：

| 指标 | Cosmos 32 步 | Cosmos 200 步 | GR00T 32 步 | GR00T 200 步 |
| --- | --- | --- | --- | --- |
| corrected | 0.656 | **0.165** | 0.906 | **0.665** |
| inference ms | 1369 | **915** | 422 | **196** |
| saturated | 0.000 | 0.000 | 0.500 | **0.575** |

原因：短 episode 被**初始瞬态**主导——机器人从归位姿态出发，模型第一步的输出必然
离目标很远；同时首次推理包含内核选择与预热，未摊薄。

**结论：对比必须声明 episode 长度，且不应使用过短的 episode。** 32 步的数字不是错的，
但它回答的是"模型在刚接手时的表现"，而不是"模型稳定下来后能否驱动这台机器人"。
本节的表格因此统一使用 200 步，并与视频一致。

延伸实测（400 步，Cosmos 在 sandbox 场景）进一步降到 `corrected 0.090`，
说明该指标在 200 步后仍在缓慢下降但已接近稳定。

##### 4.2.3 饼干场景（单源盒 → 单目标盒）

`a3_cookie_same_column` 是"一个源盒搬到一个目标盒"的场景：源盒 4×20 = 80 块，
目标盒 2×5 = 10 块，专家分两批各 5 块装入。

渲染脚本现在接受全部注册场景（`--scene`），并通过**场景注册表**构建环境，
而不是在脚本里重复任务配置——那份配置决定什么算成功
（`require_exact_slots` / `require_released` / `terminate_on_success`），
复制一份到脚本里迟早会与数据集采集时用的那份不一致。

实测（**3 seeds × 400 步**，seed 0，确定性布局）：

| model | 收到的话语 | in target | corrected | 逐 seed | saturated | correction max |
| --- | --- | --- | --- | --- | --- | --- |
| gr00t-n1-bimanual | `transfer exactly ten upright square cookie blocks...` | **0/10** | **0.541** | 0.590 / 0.525 / 0.508 | 0.526 | 0.291 |
| cosmos-policy-aloha | `pick up the cube and place it in the bowl` | **0/10** | 0.838 | 0.845 / 0.833 / 0.838 | 0.296 | **1.838** |

**两个模型都没有搬动任何一块饼干。**

Cosmos 在饼干场景上与 sandbox 相比**显著变差**（corrected 0.098 → 0.838，8.5 倍），
而它的提示词全程没有变——所以这个退化**不是提示词造成的，是场景本身更难**
（饼干场景有 80 块密集摆放的物件，而 sandbox 只有机器人）。

##### 4.2.3.1 换相机之后，同样的两个模型

上表是**旧相机**（俯角 21.9°、距离 1.289 m）下的数字。
相机与外观改动（见 `CameraConfig` 的 docstring）之后重测，同样 3 seeds × 400 步：

| model | 相机/外观 | corrected | 逐 seed | 变化 |
| --- | --- | --- | --- | --- |
| gr00t-n1-bimanual | 旧 + 旧提示词 | 0.541 | 0.590 / 0.525 / 0.508 | — |
| gr00t-n1-bimanual | 新 + 新提示词 | **0.396** | 0.432 / 0.432 / 0.323 | **−27%** |
| cosmos-policy-aloha | 旧（`pick up the cube...`） | 0.838 | 0.845 / 0.833 / 0.838 | — |
| cosmos-policy-aloha | 新（同一个标签） | **0.413** | 0.400 / 0.417 / 0.422 | **−51%** |

两件事需要分开说：

1. **相机 + 外观的改动对两个模型都有帮助，对 Cosmos 帮助更大**（−51% 对 −27%）。
   这与“视角决定了模型能看到什么”一致：旧视角下目标箱的内腔几乎全被
   它自己的侧壁挡住（见 `CameraConfig` 里那 339 像素的测量），
   Cosmos 要预测 ALOHA 动作却看不到放置区域。
2. **外观也一起变了**（`plain` → `detailed`），所以严格说这是
   “相机 + 外观”的合并效果，不是相机单独的效果。对 Cosmos 而言两者都改动了它的输入
   （饼干由琥珀色变黄色、箱内多了槽位标记），所以这里不做单因子归因。

对 GR00T 只单独测了**提示词**的增量：新相机下旧提示词 0.487、
新提示词 0.396，且两组区间不重叠（`[0.508, 0.590]` 对 `[0.323, 0.432]`）。
相机单独的增量区间重叠，n=3 分辨不了。

##### 4.2.4 一个必须先说清的混淆：两者被问的不是同一件事

上表第二列是本节的要点。**Cosmos Policy 在饼干场景上仍然使用它自己的固定标签**
（`prompt_kind: fixed_label`），而 GR00T 收到的是真正的饼干指令（`free_text`）。
所以这张表比较的是"被正确告知任务的模型"与"被告知另一件事的模型"，
**不是**两个模型解同一道题的能力。

这不是需要修掉的缺陷，而是这个 checkpoint 的固有属性：Cosmos Policy 的语言条件是
**预计算的 T5 embedding 查表**，发布的 ALOHA checkpoint 只带 6 个标签，
没有一个与饼干相关。要让它收到饼干指令，只能微调它（并重新预计算 embedding）。

视频画面上会显示每个模型**实际收到的话语**（取自模型的 `*_task` 自评），
所以这一点在视频里也是可见的，而不是只写在文档里。

##### 4.2.4.1 场景的提示词从哪里来（完整链路）

四种饼干场景**共用同一条提示词**，定义在 `src/a3_dual_arm_sim/tasks.py`：

```python
COOKIE_PROMPT = "transfer exactly ten upright square cookie blocks into the 2x5 box"
```

链路是：

| 环节 | 值 |
| --- | --- |
| `tasks.py` 的 `COOKIE_PROMPT` | `transfer exactly ten upright square cookie blocks into the 2x5 box` |
| `TaskSpec.prompt` | 四个饼干场景全部指向这一条（`a3_cookie_transfer` / `_batch` / `_same_column` / `_two_box`） |
| `a3-sim compare --scene <饼干场景>` | 取 `_default_task(scene)` → `scene_by_name(scene).task.prompt` |
| 协议帧的 `task` 字段 | 每帧都带 |
| 模型侧 | 见下 |

由 `tasks.py` 自己的注释可知这是有意为之：*"a task name identifies a data contract,
not a goal. Every cookie variant shares one prompt -- from a policy's point of view
they are the same task"*。

**但"发给模型"的字符串取决于 `prompt_kind`：**

| 模型 | `prompt_kind` | 实际用于 conditioning 的字符串 |
| --- | --- | --- |
| gr00t-n1-bimanual | `free_text` | 就是上面那条饼干指令 |
| cosmos-policy-aloha | `fixed_label` | **`pick up the cube and place it in the bowl`**（它自己的标签表里的一个） |

所以要改"模型被要求做什么"，对 GR00T 改 `COOKIE_PROMPT` 或 `--task` 即可；
对 Cosmos Policy **两个都不起作用**——它的标签表里没有饼干相关的条目，
换成别的标签只会让它被要求做另一件事，而不会让它理解饼干。

##### 4.2.5 排名随场景翻转（且翻转量超出噪声）

| 场景 | Cosmos `corrected` | GR00T `corrected` | 差距 |
| --- | --- | --- | --- |
| sandbox | **0.098**（spread 0.018） | 0.458（spread 0.062） | 0.36 → Cosmos 胜 |
| a3_cookie_same_column | 0.838（spread 0.012） | **0.541**（spread 0.083） | 0.30 → GR00T 胜 |

**两次差距都远大于各自 spread**（3 seeds，400 步），所以这不是噪声，是真实的翻转。

原因可以部分归因，但要诚实标出边界：

- **Cosmos 的提示词在两个场景上是同一个**（`pick up the cube and place it in the bowl`，
  固定标签），而它的 `corrected` 从 0.098 变成 0.838。**场景变化单独就足以造成这个退化。**
- **GR00T 在两个场景上收到不同的提示词**（sandbox 是 `move the A3 dual-arm robot`，
  饼干场景是真正的饼干指令），所以它的变化混合了场景与提示词两个因素，无法分离。

**单场景对比不足以给模型排名。** 这也说明 `compare` 的价值在于同一组 manifest 可以跑多个
场景，而不是一次跑出一个"总分"。

##### 4.2.6 单 episode 不足以给模型排名：两个模型都有运行间方差

同一配置**重复运行**给出的数字不同：

| model | sandbox `corrected` 两次运行 |
| --- | --- |
| cosmos-policy-aloha | 0.107 / 0.090 |
| gr00t-n1-bimanual | 0.512 / 0.390 |

原因不同：

- **GR00T 是随机采样**。`flow_matching_action_head.py` 用未播种的
  `torch.randn(actions.shape, ...)` 生成初始噪声，每次推理都重新抽。这与
  `docs/policy-interface.md` 对 LeRobot 记录的 finding 11 是同一类问题，
  方差大到足以翻转一个未经多 seed 比较的排名。
- **Cosmos Policy 有小的 GPU 非确定性**。它声明了 `deterministic=True` 与固定
  `seed`，但卷积/归约类算子的原子加与 cudnn 选择仍会引入差异，实测 spread 0.017。

3 个 seed 的结果（sandbox，400 步）：

| model | corrected | 逐 seed | saturated | 逐 seed |
| --- | --- | --- | --- | --- |
| replay-hold | **0.000** | 0 / 0 / 0 | 0.000 | 0 / 0 / 0 |
| cosmos-policy-aloha | 0.098 | 0.090 / 0.095 / 0.108 | 0.000 | 0 / 0 / 0 |
| gr00t-n1-bimanual | 0.458 | 0.490 / 0.428 / 0.458 | 0.760 | 0.620 / **0.923** / 0.738 |

读法：

- **对照组在三个 seed 上精确为 0**，再一次说明测量链路本身不引入噪声；
- Cosmos 的 spread 是 **0.018**，GR00T 是 **0.062**；
- Cosmos 与 GR00T 在 sandbox 上的差距（0.098 vs 0.458）**远大于噪声**，结论稳健；
- 但 **GR00T 的 `saturated` spread 达 0.303**（0.620 / 0.923 / 0.738），
  单 episode 的这个数字不可用。

**结论：`compare` 应至少用 3 个 seed，并报告逐 seed 值。** `--seeds 0-2` 就是为此存在的，
JSON 报告里保留了逐 episode 的明细。

##### 4.2.7 三个被修掉的不一致

这一轮暴露出三处"表格与视频说的不是同一件事"，都已修正：

1. **`compare` 与渲染脚本的随机化默认值不同**。前者默认 `randomize_objects=True`，
   后者 `False`，于是同一 seed 下同一模型给出不同数字（GR00T 0.512 vs 0.393）。
   统一为**确定性布局**，并提供 `--randomize` 显式开启。这也让数字可复现。
2. **`compare` 在注册场景上用错了任务字符串**。它的默认值是硬编码的
   `"move the A3 dual-arm robot"`，所以跑饼干场景时**模型会被问错误的任务**，
   而画面上却是饼干。改为取场景自己的 prompt。
3. **对照组录制太短**。400 步的对比需要 400 行动作，而录制只有 200 帧，
   于是 `replay-hold` 报 `replay recording ran out after 200 frames`。
   失败是**明确报错**而非静默产出全 0（`episodes: 0` 加上 `failure` 字段），
   这正是设计意图；录制已重新生成为 600 帧。


#### 4.3 实现中修掉的三个 bug（值得记录）

**其一：`PolicyTrace.reset()` 丢掉绑定。**

它原本用 `BehaviorTrace()` 替换整个 trace，**丢掉了已绑定的关节限位与速率上限**。
后果：从第二个 episode 起，`saturated_fraction` 与 `rate_limited_fraction` 恒为 0——
**看起来像"模型从不碰限位"，而不是像测量坏了**。

这个 bug 之所以危险，是因为它产出的数字**完全合理**。修正后 GR00T 的
`saturated_fraction` 从 0.000 变成 0.500，Cosmos 的 `rate_limited_fraction`
从 0.000 变成 0.645——两个模型的结论都变了。

回归测试 `test_resetting_keeps_the_limits_and_the_rate_cap` 断言的是
"机器人的常量跨 episode 存活"，并已用**变异测试**验证：把 bug 放回去，测试即失败。

**其二：视频读错了自评的来源。**

渲染脚本自己驱动 rollout（不走 `EpisodeRunner`），所以 `env.step` 返回的 `info` 是
**原始环境信息**，不含策略的 `metrics`——把 `metrics` 合并进 `info` 是 `EpisodeRunner`
做的事。脚本却从 `info` 里找自评，于是 Cosmos Policy 的 `cosmos_value` 和它**实际使用的
提示词**一直没有显示，画面上只有一句 `no model self-report`。

后果不是数字错，而是**该显示的信息没显示**——而 4.2.4 那条混淆正是靠这个字段才能被看到。
修正为读 `gateway.metrics`。

**其三：视频与表格对"自评"的定义不同。**

修好来源之后仍不一致：`compare._self_report` 采集 `_value` / `_calls` / `_task` 三类，
而视频的 `_model_line` 只认 `_value` / `_task`。于是 GR00T 在表格里有 `gr00t_calls`，
在视频里却仍显示 `no model self-report`。两处的后缀已统一。

#### 4.4 饼干的外观：`--visual-style`（纯视觉，不动物理）

用户反馈"饼干建模比较粗糙"。实地核查后的结论是**分层**的：

| 层 | 实际情况 |
| --- | --- |
| 碰撞几何 | **不粗糙**。饼干是**八边形截面挤出**（`_add_beveled_cookie_mesh`），上下缘各有真实倒角，碰撞模式可用 `mesh` 或 `compound`（后者把平面部分退化为 box-box，斜面保留 mesh） |
| 视觉几何 | **与碰撞几何同尺寸**（注释说明了原因：更小的视觉网格会让正确接触的饼干看起来悬空） |
| 视觉呈现 | **这一层确实粗糙**：饼干只有内联 `rgba` 棋盘色，用的是 MuJoCo 默认材质；而默认 **headlight 的 diffuse 是 0.4，比场景自己的补光（0.45）还亮**，且它跟随相机、不投阴影、对所见面均匀着色 → 两盏真实方向光只剩一片平光，80 片薄饼干糊成一整块 |

所以优化做在**材质与光照**上，新增 `visual_style`（默认 `plain`）：

```bash
python scripts/render_bridge_policy_video.py \
    --manifest configs/policies/gr00t_bimanual.yaml \
    --scene a3_cookie_same_column --visual-style detailed \
    --output outputs/videos/gr00t_cookie_detailed.mp4 --steps 200
```

`detailed` 做三件事：压低 headlight（0.4 → 0.16）恢复方向性明暗；给饼干、盒子各自
专用材质（带 specular/shininess，而不是默认的统一高光）；把棋盘对比调柔和
（`0.90 0.58 0.12` vs `1.0 0.79 0.24`，原来的对比读起来像编织纹而非独立饼干）。

**默认之所以是 `plain`：渲染出的像素是数据集契约。** 所有已采集的图像与已跑过的对比
都是在 `plain` 下产生的，静默换样式会让既有数字不可复现。要换就显式声明，
这样对比报告也能记录自己用的是哪种。

物理不变性已逐项验证（`plain` vs `detailed`）：

| 检查项 | 结果 |
| --- | --- |
| `nbody` / `ngeom` / `njnt` / `nq` | 相同 |
| `body_mass` / `geom_size` / `geom_pos` | 逐元素相同 |
| `geom_contype` / `dof_damping` | 逐元素相同 |
| 视觉几何的碰撞位 | 本来就是 `contype="0" conaffinity="0"` |

**实测有效的开关（用像素差量化，避免留下装饰性配置）：**

| 改动 | 平均像素差 | 结论 |
| --- | --- | --- |
| headlight `0.4` → `0.16` | **6.80** | 这就是关键 |
| `quality/shadowsize` `4096` → `8192` | 0.03 | 无用，默认已足够细，**已移除** |
| `map/shadowclip` `1.0` → `3.0` | 0.00 | 完全无影响，**已移除** |

**一个诚实的限制：阴影边缘的锯齿无法用配置修掉。** MuJoCo 3.3.7 的 `mjtRndFlag`
里没有软阴影标志（只有 `mjRND_SHADOW` 等 11 个），所以桌面上那条硬阴影边界会一直存在。
它在视频里可见，代码注释里也写明了。

#### 4.4 报告刻意不做的事

- **不把不同 regime 平均在一起**。微调模型赢说明它被教过，零样本模型输说明它没被教过，
  两者都不是关于架构的陈述。
- **不归一化掉差异**。相机数量、prompt 类型、实际使用的标签都进入报告，
  这样读者能区分接线差异与能力差异。
- **不把缺失值渲染成 0**。没报告成本的模型显示 `—`，而不是看起来像"成本为 0"。


`a3-sim compare` 读多个 manifest、同 seed 跑同场景、输出成功率/步数/指垫力对比表。
这是本方案的最终目的：**让清单里的模型在同一把尺子下比较**。

但"同一把尺子"要先定义。Phase 1 跑通后暴露出一个必须先回答的问题，见下。

---

## 七之二、评测口径：零样本还是微调（必须显式声明）

> 状态：**设计**，尚未实现。下面的 manifest 字段属于 Phase 4 的待做项；
> 本章先把它写清楚，因为它的结论会影响 `compare` 怎么定义\"成功\"，
> 而那个定义一旦做错，后面所有对比都要重做。

### 7.1 问题的来源

Phase 1 跑通 Cosmos Policy 后，它的 episode 里出现了一个不能说小的事实：

```json
"cosmos_task": "pick up the cube and place it in the bowl"
"cosmos_value": 0.7868
```

而 A3 的任务是"把饼干块夹进盒子"。这个 checkpoint 的训练数据是折衣服、
装袋、摆盘——**没有一个与饼干相关**。它产生的动作是"合理但不对题"的。

这不是 Cosmos Policy 的问题：**清单里的 16 个模型，没有一个是为 A3 训练的**。
所以"成功率"这个指标在当前状态下几乎必然接近 0，而接近 0 的成功率
**不能区分"模型不行"和"模型没被教过这件事"**。

### 7.2 因此 `compare` 必须同时报告两组指标

| 指标类 | 含义 | 回答什么问题 |
| --- | --- | --- |
| **行为可用性** | 路面下的：动作是否有限、是否在关节限内、是否触发安全停止、指垫力是否超限、是否有抖动/震荡 | "这个模型的输出能不能安全地驱动机器人" |
| **任务进度** | 目标盒内的饼干数、源盒剩余数、抬升次数、`success` | "它在多大程度上完成了任务" |
| **模型自陈** | `cosmos_value` 这类模型自带的自评量 | "模型自己觉得做得怎么样"（用于发现口径分歧） |
| **成本** | 启动时间、推理延迟、显存峰值、chunk/execute | "在什么预算下能跑" |

**行为可用性是零样本阶段的主要指标**。它既不需要任务成功，也不是无意义的
——一个输出 NaN、超关节限或乱抖的模型，即使"没被教过"也应被判为不合格。

### 7.3 三段式口径，manifest 必须声明

```yaml
evaluation:
  regime: zero_shot          # zero_shot | finetuned
  # 零样本阶段：环境判定不可能通过，用它只会得到全 0 的表
  primary_metric: behavior   # behavior | task | both
  # 本次配置喂给模型的是「查表得到的固定标签」还是「自由文本」。
  # 注意：这是**本次选择**的描述，不是模型能力上限——例如 Cosmos Policy
  # 的标签来自一个可扩充的缓存，改标签或重烤即可换话语（见 7.4）。
  prompt_kind: fixed_label   # free_text | fixed_label | none
  prompt_labels: ["pick up the cube and place it in the bowl"]
```

| regime | 何时用 | 能回答 |
| --- | --- | --- |
| `zero_shot` | 直接拿发布权重跑 | 行为的可用性、先验的接近程度、成本 |
| `finetuned` | 在 A3 数据上微调后 | 真正的任务能力（这才是通常意义上的"谁更强"） |

`compare` 默认 **不混合** 两种 regime 的结果。把零样本和微调的模型放在同一张
成功率表里，是把"没教过"和"学不会"混为一谈。

### 7.4 为什么语言输入要单独记录

早期实测（那时只跑了两条标签）：

| 标签 | `cosmos_value` |
| --- | --- |
| `'put purple eggplant on plate'`（兑底，任意选中） | 0.863 |
| `'pick up the cube and place it in the bowl'`（显式配置） | 0.787 |

同一个观测、同一个权重，**仅换标签，自评价值就变了 0.076**。这说明两件事：

1. 语言输入**真的影响输出**，不是装饰；
2. 所以"用了哪个提示词"必须进 episode 元数据，否则两次 rollout 的差异无法归因。

但也要看清限制——**这一节的初版把限制说错了，特此更正**。

初版写的是"Cosmos Policy 的条件是预先算好的 T5 embedding 查表，只带 6 个标签，
传表外字符串会现场下载 45 GB"。前半句是**机制误读**：那个 pkl 是
`get_t5_embedding_from_cache()` 维护的**缓存**（`cosmos_utils.py` L387），
未命中会调用 `get_text_embedding()` **现场编码并写回**，而不是拒绝。
实测确认（固定种子，同标签重复逐位相同）：

| 检验 | 结果 |
| --- | --- |
| 传一个缓存里没有的字符串 | 日志打印 `Computing T5 embedding for new instruction: '...'`，随即进入编码器 |
| 在 pkl 副本上写回 | 创建 `.backup`、写入 7 条、重载确认 |
| 直接传 `(1,512,1024)` ndarray | 被接受，且与传字符串路径**逐位相同**（`max|diff| = 0`，两条标签各测一次） |
| 换标签是否改变动作 | 是，且很强：`fold shirt` vs `put purple eggplant on plate` = **0.0927** |

“逐位相同”这一点值得强调：它把路线 C 从“能跑通”升级为**等价计算**。
如果两条路径只是“都返回了 50×14”，那 ndarray 就是第二条未经校验的代码路径；
逐位相同说明它是真替代，将来换掉 T5-11B（换中文文本塔、换 Qwen/CLIP）
不会引入与官方实现不一致的第二套行为。

注意一个偏差：平均两个缓存的 embedding 得到的向量
距两个父向量**不等距**（距 `candies` 0.0142、距 `fold shirt` 0.0411）。
平均是在向量空间而非方向空间做的，所以“取中点”不等于“语义中间”。

所以**接口并不限制指令**，`prompt_kind: fixed_label` 描述的是
**我们当前怎么用它**，不是它的能力上限。真正的限制是成本：
烤一条自己的 embedding 需要 T5-11B，**45.23 GB**（已用 HEAD 请求核实），
而本机可用 8.4 G（数据盘）+ 28 G（系统盘）= 36.4 G，装不下。

至于"对'饼干'这个词没有理解能力"——**这句话是错的**。
只要把它送进 T5-11B 编码一次，它就有了那个词的条件向量；卡住的是磁盘不是架构。
而且这台机器上根本没有走通这条路。

**结论：`prompt_kind` 仍应在 manifest 里声明，且 `compare` 报告里要并列显示**，
因为"这次实验实际喂了哪句话"确实是两个模型不可比的一个维度。
但声明的是**本次配置的选择**，不是模型的天花板——把两者混为一谈，
就会像本节初版那样，从"我们没喂"误推出"它不能吃"。

#### 7.4.1 六条标签的实测排序（并更正一个自以为是的猜测）

既然标签确实影响行为，那就不该“随便选一个”，而该测。
**6 条标签 × 3 seeds × 400 步**，同一场景、当前相机：

| 标签 | 均值 corrected | 逐 seed 区间 | 区间宽 |
| --- | --- | --- | --- |
| `put brown chicken wing on plate` | 0.388（名义最优） | 0.325 .. 0.507 | **0.182** |
| `put candy in bag` | 0.404 | 0.398 .. 0.417 | 0.020 |
| `put purple eggplant on plate` | 0.412 | 0.403 .. 0.422 | 0.020 |
| **`pick up the cube and place it in the bowl`（manifest 当前值）** | 0.413 | 0.400 .. 0.422 | 0.022 |
| `fold shirt` | 0.438 | 0.432 .. 0.445 | 0.013 |
| `put candies in bowl` | 0.611 | 0.605 .. 0.615 | 0.010 |

**要看区间，不要看均值。** 这是本节最重要的方法教训：
`brown chicken wing` 均值最低，但它自己的区间宽 **0.182**，
是其他标签（0.010–0.022）的 **9–18 倍**——它的“胜出”来自 seed 0 的 0.325，
而 seed 1 给出 0.507，比四条标签的均值都差。**它是最不稳定的标签，不是最好的。**
只看均值会把它选成“最优”。

按区间重叠只能说三件事：

1. `put candies in bowl` **可靠地最差**（它的区间不与任何一个重叠）；
2. `fold shirt` 可靠地差于前三名（区间 0.432–0.445，与 0.398–0.422 不交）；
3. `candy in bag` / `eggplant` / `pick up the cube` **三者不可分**。

**所以 manifest 的当前取值保留，但理由变了**：它原本是语义猜测
（“pick-up-and-place 在类型上最接近饼干任务”），现在是实测结果。

而那个语义猜测**被测伪了**：我最看好的 `put candies in bowl`
（“把多个小物件放进容器”，读起来就是饼干任务）是六条里**最差的**，
0.611 对 0.413。语言条件的影响方向不能靠读句子猜。

#### 7.4.2 `cosmos_value` 不能当质量信号读

上表还有一个副产品：把模型自评与实测行为并排，两者**排序不一致**。

| 标签 | 自评 `value` | 实测 corrected |
| --- | --- | --- |
| `put purple eggplant on plate` | **0.861（自评最高）** | 0.412（行为第 3） |
| `put brown chicken wing on plate` | 0.847 | 0.388（行为名义最优） |
| `pick up the cube...` | 0.231 | 0.413 |
| `put candy in bag` | 0.230 | 0.404 |
| `put candies in bowl` | 0.229 | **0.611（行为最差）** |
| `fold shirt` | **0.189（自评最低）** | 0.438（行为不是最差） |

两点值得记下：

- **自评主要由标签决定，而不是由局面决定。** 这些调用用的是**同一个观测**，
  场景状态一模一样，自评却横跨 0.189–0.861（4.5 倍）。
- **自评接近不代表行为接近。** 三条自评几乎相同的标签
  （0.229 / 0.230 / 0.231，相差 <0.002）行为上相差 0.611 vs 0.413。

所以适配器里那句“它比一个光秃秃的 success 标志更有信息量”是**过度声称**，
已在适配器里更正。`cosmos_value` 仍然记录（免费，且在固定采样种子下完全可复现），
但 manifest 的 `success.probe: env` 仍然是判定依据。

#### 7.4.3 想换提示词，编码器是**没有选择**的

上面说“接口接受任意字符串”，容易被读成“随便怎么编码都行”。**不是。**
论文 §3 写明了基座的条件方式：

> 𝐜 represents the textual description encoded as **T5-XXL** embeddings

T5-XXL 就是 `google-t5/t5-11b`（模型卡与 `get_t5_emb.py` 用的同一个），
所以论文与代码一致。但这句话的含义是**硬约束**：

- **用 T5-XXL 编码 → 天然对齐**。因为基座就是拿这种向量预训练的，
  微调也只见过这种几何。这不是“效果更好”，是“同一个东西”。
- **用任何别的文本塔编码 → 必然 OOD**。把 Qwen / CLIP / 中文 T5 的输出
  塞进 `crossattn_emb`，cross-attention 会照常做点积，但那个几何它从没见过，
  输出是静默的垃圾——**而且不会报错**，因为形状对得上。

这条把“路 C”的边界划清楚了：**直接传 ndarray 是 T5-XXL embedding 的
*传输*机制（实测与传字符串逐位等价），不是*替代编码器*。**
两者差一个字，后果是从“等价”变成“无声乱码”。

#### 7.4.4 一个意外发现：我们钉的标签不是训练标签

`aloha_dataset.py` 里藏着 ALOHA 的指令白名单（`raw_file_string` 那一串分支）：

```
fold_shirt / put_candies_in_bowl / put_candy_in_bag /
flatten_shirt / put_brown_chicken_wing_on_plate / put_purple_eggplant_on_plate
```

对照发布的 pkl 与官方训练数据（`configs/default.yaml` 同名的 4 任务混合）：

| pkl 里的标签 | 是训练标签吗 |
| --- | --- |
| `put brown chicken wing on plate` | ✓ task 1「put X on plate」 |
| `put purple eggplant on plate` | ✓ task 1 |
| `fold shirt` | ✓ task 2 |
| `put candies in bowl` | ✓ task 3 |
| `put candy in bag` | ✓ task 4 |
| **`pick up the cube and place it in the bowl`** | **✗ 仓库、模型卡、论文里都没有** |

而 `flatten shirt` 在白名单里、却不在 pkl 里——这条是合理的，因为它不在
上面那个 4 任务混合中（15+45+45+80 = 185，正好等于公布的总演示数）。

所以第 6 条是个来历不明的条目，**而它正是 manifest 当前钉住的那条**。
它的行为并不差（0.413，并列第一梯队），value 也与 `put candy in bag` 同簇
（0.231 vs 0.230），说明它不是坏输入。但它**不是这个 checkpoint 学过的任务**，
这一点应当记下来，而不是当成“挑了最合适的一个”。

#### 7.4.5 为什么“对齐”这件事在本机测不了

论文的实验设置里有一条关键信息（§5.1）：

> “put X on plate” (80 demos): place objects on a plate **based on language
> instructions, testing language following**

也就是说 **ALOHA 的任务 1 本身就是语言跟随测试**：桌上同放茄子和鸡翅，
模型必须按指令选对哪一个。Cosmos Policy 在这项拿了 **100.0**
（in-distribution 与 OOD 都是）。**这个 checkpoint 的名词级语言接地是作者验证过的。**

但要注意论文的 OOD 是**空间 OOD**（“the objects are not constrained to be in the
same row and the positions are more varied”）——**不是新名词 OOD**。
换一个训练中从未出现的物体名词，论文没有测。

于是本机为什么测不了语义对齐，就有两条独立的理由：

1. **场景里没有任何一个标签提到的指称物。** 没有茄子、鸡翅、衬衫、糖果、袋子、
   盘子。在这种场景里测“指令 A 与指令 B 的动作差异”，测的是
   “模型对着不存在的物体说话会怎样”，而不是“它理解得对不对”。
   对齐实验**必须在有指称物的场景里做**。
2. **探针太少。** 6 个标签给出 15 个两两距离，但它们不独立（只有 6 个自由度）。
   实测 Spearman ρ = **+0.286**（动作距离 vs embedding 距离），
   在 n=6 上毫无显著性。这个数字既不能支持“模型按几何读 embedding”，
   也不能否定它。**唯一值得记下的具体观察**是：embedding 最接近的一对
   （`put candies in bowl` ↔ `put purple eggplant on plate`，余弦距离 0.5564，
   全场最小）动作距离反而**高于平均**（0.0528 > 0.0457），
   且两者的 value 差得最远（0.229 vs 0.861）——
   至少说明 **`cosmos_value` 不是 embedding 几何的函数**。

**所以"自定义提示词"目前的能力边界是：**

- **能选** —— 六条实测排序过，选了第一梯队，这是可信的。
- **能自定义字符串** —— 机制上通，但需要 T5-XXL 编码一次。
  **磁盘 45.23 GB 是唯一瓶颈，显存不是**：
  `T5EncoderModel` 只需编码器那一半，3.35 B 参数、fp32 约 13.4 GB，32 GB 装得下；
  但下载的是一个含 decoder 的 45.23 GB 单文件（可用 36.4 GB）。
- **能自定义 embedding** —— 只有当你自己就有一份 T5-XXL 的输出时成立。
  换编码器不行（见 7.4.3）。

**实际可行的一条路**：在别的机器上烤一次
`generate_t5_embeddings([我们的提示词])`，拿回约 2 MB 的 pkl，
再用 `--config t5_embeddings=<path>` 指过去。
一次性 45 GB 路费换永久的 2 MB 缓存——而且 `save_t5_text_embeddings_cache()`
本来就会把新编码写回，所以烤一次就够。

#### 7.4.6 磁盘到账后真的做了：结果与**一个必须防的指标陷阱**

数据盘扩到 250 GB 后，走完了整条路（`integrations/cosmos_policy/README.md`
有完整步骤），**端到端成功**：

```
first inference took 87.7 s        ← 含 T5-XXL 加载 + 一次编码
cosmos_task = '<我们的饼干提示词>'   ← 原样送达，未被替换
VERDICT: accepted -- conditioned on our own prompt
```
第二次运行 24.1 s 且不再出现 `Computing T5 embedding` → **缓存命中**；
pkl 从 6 条增到 7 条（7.3 MB → 9.4 MB）。

**顺带解开了那条来历不明标签的来源。** pkl 里的 dtype 分两组：官方脚本
（`generate_t5_embeddings`）显式 `.to(bfloat16)`；而
`get_t5_embedding_from_cache` 自动写回时**不做 dtype 转换**。所以：

| 标签 | dtype | 进来的方式 |
| --- | --- | --- |
| 5 条官方标签 | `bfloat16` | 离线烤好 |
| `pick up the cube and place it in the bowl` | **`float32`** | **自动写回** |
| 我们的饼干提示词 | **`float32`** | **自动写回** |

dtype 就是"这条是运行时补进来的"的指纹。功能上没问题（`crossattn_proj` 是
Linear 层，bf16 autocast 会把 fp32 cast 下去），但那条标签确实不是训练标签。

**然后是最重要的部分：6.3 倍改善是假的。**

用我们自己的提示词跑 3 seeds × 400 步，对比清单钉住的标签：

| model | corrected | rate limited | saturated | motion mean | in target |
| --- | --- | --- | --- | --- | --- |
| `pick up the cube...`（钉住） | 0.425 | 0.154 | 0.292 | 0.034 | 0/10 |
| **我们的饼干提示词** | **0.068** | 0.065 | **0.000** | 0.018 | 0/10 |

`corrected` 低了 **6.3 倍**，看起来像大胜。但有三个信号要求先怀疑：

1. **三个 seed 的结果几乎逐位相同**（0.068 / 0.068 / 0.068），
   比模型已知的 GPU 非确定性还要稳定——而场景我验证过是逐位相同的
   （`randomize` 关闭，585 个 qpos 全等），所以差异只能来自模型；
2. `saturated` **恰好 0.000**；
3. 运动量只有一半。

于是记录了**每步运动包络**（而不是只看整段聚合）：

| | 钉住标签 | 我们的提示词 |
| --- | --- | --- |
| 每步请求变化（均值） | 0.042 | 0.026 |
| 8 段窗口的运动趋势 | 0.079 → … → **0.053（上升）** | 0.088 → … → **0.011（衰减）** |
| 命令与前一步完全相同的步数 | 0 / 399 | 0 / 399 |

**两者都没冻结**，但形状相反：钉住标签**持续甚至加速**运动（所以 29% 的步
有轴顶着限位），我们的提示词**逐步衰减到近乎静止**（所以 `saturated` 为 0）。

所以 `corrected 0.068` 测的不是"模型干得更好"，而是"**模型收工了**"。
两者都是 0/10，都没搬动任何饼干。

**这次实验真正的结论**是两句话，分开说：

- **自定义提示词确实被读取，而且改变行为。** 它选出了一个**可复现的不同行为模式**
  （衰减式、零饱和），不是噪声，也不是被忽略。这是对 7.4 节机制的端到端确认。
- **但它没有让任务变好。** "衰减到静止"和"一直顶着限位推"在任务完成度上
  没有区别——都是 0/10。把 `corrected` 的下降读成改进，
  等于把"放弃"读成"温柔"。

**`corrected_fraction` 是本次会话里第三次误导我们的单一标量。**
它是整段的分数，所以"猛动 50 步、然后静止 350 步"会比"持续运动 400 步"
得分好得多，而前者并不更有用。**引用它时必须同时给出运动包络**，
否则无法区分"学会了轻拿轻放"和"停下来了"。


### 7.5 三条可选的路线

| 路线 | 做法 | 代价 |
| --- | --- | --- |
| **A. 零样本行为对比** | 直接跑发布权重，只比行为可用性 + 成本 | 低（现在就能做） |
| **B. 全部微调后对比** | 先用 `a3-sim collect` 采 A3 数据集，各模型各微调一次 | 高（每模型一次训练，本机单卡偏紧） |
| **C. 混合：零样本筛，微调定** | 先用 A 筛掉不可用的，再用 B 出结论 | 中（推荐） |

本机是单卡 32GB，路线 B 对多数 2B+ 模型不现实（手册建议 8×80GB）。
**建议先做 A**，并把结论作为"要不要花这个钱"的依据。

---

## 八、风险与开放问题

1. **硬件与手册不符（最高优先级）**
   当前机器是 **RTX 5090 / sm_120 / 32GB / 单卡**，手册建议 H100 80GB（DreamZero 要 2 张）。
   后果：DreamZero（14B，2 卡起）**当前跑不了**；Fast-WAM（推理 >24GB）可能 OOM；
   ViFailback（20–24GB）处于边界。方案本身不受影响，但 Phase 3 的**验收顺序要重排**。
   另：磁盘仅 29GB + 50GB，而手册要求 1TB —— 一次只装一个模型，用完即删。

2. **flash-attn 与 sm_120（已解决，保留作为方法记录）**
   手册的 flash-attn 版本（2.5.5 / 2.7.1.post4 / 2.8.3）都是为 sm80/sm90 编译的，
   而 sm_120 需要较新的构建。**但不必自己编译**：Phase 1 证明
   `flash_attn-2.7.3+cu128.torch27-cp310` 有预编译 wheel，实测内核可在 sm_120 执行。
   它不在 PyPI、也不在 NVIDIA 官方索引（那里只有源码 tarball），而在
   **项目自己的依赖索引**里。推而广之：接一个模型前，先读它的 `pyproject.toml`
   看有没有声明私有 index——这一步能省掉一次半小时的编译。

3. **成功判定权属**
   `EpisodeRunner` 已支持 `policy.success_override`。外部模型若自带成功判定
   （WoG / InSpire 有明确任务定义），应走这个口子，还是统一用 A3 环境判定？
   **建议：默认用环境判定，manifest 可显式开启 `success.probe: policy`**，
   并在 `final_info` 里同时记录两者，便于发现口径分歧。

4. **相机数量与视角差异**
   A3 是 3 路（front + 双臂 wrist）。单相机模型（多数）只能看到 front，
   这会让对比不公平。**建议 manifest 必须显式声明丢弃哪几路**，
   且报告中标注"输入视角数"，避免把视角差异误读成模型能力差异。

5. **语言指令格式**
   `task` 字符串直接转发，但各模型的 prompt 模板不同（有的要 `"pick up the block"`，
   有的要模板包裹）。**Phase 1 的初版结论在这里也错了，已在 7.4 更正**：
   Cosmos Policy 不是"只能查表"，那个 pkl 是缓存，未命中会现场编码。
   已加 `prompt_kind` 声明（见 7.3），且适配器仍从缓存**读取真实键名**而不是猜——
   猜错的表现是下载而不是报错，这个防御是对的，但要和"模型会拒绝"区分开。

   顺带记下真正的教训：适配器原本在**两条路径**上都强制替换调用方的字符串
   （`configured` 优先、以及未匹配时兜底），于是"模型不接受我们的提示词"
   这个现象其实是**适配器不接受**。一个为了防止 45 GB 下载而加的保险，
   被我自己读成了模型的能力边界。**防御性代码要写清楚它在防什么，
   否则下一个人会把它当成规格。**

6. **时间语义**
   模型假设的 `dt` 各不相同（50 Hz 数据训练的模型在 20 Hz 下行为不同）。
   manifest 需要 `action.dt` 与重采样策略，否则动作会被静默地当作正确频率执行。

7. **`n_action_steps` / `execute` 如何调**
   手册 finding 12 说明 chunk 越长越省算力但越不反应。这不是猜出来的参数，
   应该在 Phase 4 做 sweep，并记录在 manifest 里作为**该模型在该任务上的实测最优**。
