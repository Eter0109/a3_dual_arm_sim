# 饼干扩展场景 benchmark

> 课程版 `course-v1` 的当前任务、统一提示词、目标盒评分和专家入口以 [课程 Benchmark 说明](course_benchmark.md) 为准。下文保留历史 legacy 协议说明和来源配额任务；如需按下文的旧提示词和成功条件运行单项，请显式使用 `--protocol legacy`。

以下六种初始场景使用既有 `benchmark_cookie_batch` 入口、相机、动作接口和 JSON 报告。
它们是独立的 2×5 扩展任务，不改变 `2x5` 或 `2x10` 的评分协议。

| `--task` | 初始化 | 目标 | 满分 |
| --- | --- | --- | ---: |
| `missing-columns` | 随机移除 1 或 2 整列 | 从剩余列搬 10 块 | 10 |
| `partial-gaps` | 每列移除前端 3 块和内部连续 3 块 | 搬 10 块 | 10 |
| `cross-column` | 每列仅保留连续 8 块 | 跨列搬 10 块 | 10 |
| `tilted` | 两列各选一块施加倾斜并物理沉降 | 搬 10 块 | 10 |
| `prefilled` | 目标盒随机一列预放 5 块 | 保留预放饼干，再补 5 块 | 5 |
| `two-boxes` | 一个源盒、两个空目标盒 | 默认两盒各 5 块；可选两盒各 10 块 | 10 或 20 |

`tilted` 默认 `--tilt-level mild`，在沉降后至少有一块选中的饼干倾斜 5° 或以上；
`--tilt-level fallen` 至少有一块倾斜 60° 或以上。平躺模式在每块目标饼干两侧
各移除一块，为平躺腾出空间，共移除 4 块、源盒初始剩 76 块。报告记录沉降后的
实际角度和被移除的 ID。倾斜构造可能引起相邻饼干的真实接触移动。场景生成失败时抛出错误，
不会将无效初态当作评测失败。缺失饼干不可见、无碰撞，并且不参与库存统计。
平躺模式不要求必须抓取平躺的饼干，因此主要衡量存在倒伏物体时的场景泛化，
不能单凭成功率宣称策略具备扶正能力。

在仓库根目录运行（为结果选择新的输出路径）：

```bash
MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --protocol legacy --task partial-gaps --policy expert --episodes 20 --seed-start 13000 \
  --profile basic --max-steps 1000 --workers 4 \
  --output /tmp/cookie_partial_gaps.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --protocol legacy --task two-boxes --policy smolvla:/absolute/path/to/checkpoint \
  --episodes 20 --seed-start 13000 --profile basic --max-steps 1000 \
  --output /tmp/cookie_two_boxes.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --protocol legacy --task tilted --tilt-level fallen --policy expert --episodes 20 \
  --seed-start 13000 --profile basic --max-steps 1000 \
  --output /tmp/cookie_fallen.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --protocol legacy --task two-boxes --box-quota 10 --policy expert --episodes 20 \
  --seed-start 13000 --profile basic --max-steps 3200 \
  --output /tmp/cookie_two_full_boxes.json
```

六种任务使用各自的 `task` 文本，记录在每回合 `randomization.task_instruction`。
若显式传入 `--prompt-style`，应使用 `task`。现有 ACT 接口不消费任务文本；
它与文本条件策略的对比应注明输入条件。所有扩展场景仅支持 `--profile basic`；
显式指定 medium 或 advanced 会在启动前报错。`--no-randomize-boxes`、
`--no-randomize-cookies` 可关闭相应基础扰动，但不会关闭场景本身的缺失、
倾斜、预填充、目标盒大角度旋转或来源配额。自定义配置中的 basic 值仍会被使用并记录。`--source-column` 控制旧专家的
选列，不改变缺失场景的随机生成；推荐不传，让策略自行决定取料位置。

既有 `expert` 是原任务规则专家，仍可作为基线，但没有为缺料重规划、半满盒或
双盒切换新增专家控制器，不能预期它能完成这些任务。模型权重、推理参数、步数、
随机化档位和指令应在比较时对齐。`--workers > 1` 支持多进程；加载策略实例、
录制诊断或交互渲染时沿用现有单进程限制。

结果 `summary` 和 `episodes` 保持原有格式。每回合额外提供 `scenario_metrics`：

- `target_counts` 是每个盒内满足直立、稳定、释放条件的数量；`new_target_counts`
  只统计初始源盒里的饼干。`prefilled` 的初始 5 块不计分。
- `two-boxes` 的数量得分按每盒各自的配额封顶。默认每盒 5 块时，`[10, 0]`
  为 5/10，`[5, 5]` 为 10/10；`--box-quota 10` 时各盒放满 10 块，满分 20。
- `cookies_outside_containers` 是回合结束时位于源盒和目标盒范围之外的活动饼干数；
  它包含被夹在空中等未放置状态，并不等同于确认掉到桌面的数量。
- `disturbed_source_cookie_count` 统计仍在源盒内、相对初态位移超过 10 mm 或倾角
  变化超过 15° 的饼干；这是观察指标，不是成功判定。
- `randomization.scenario` 记录初始库存、移除/预填充的物体 ID、沉降后倾角等；
  `run.scenario_protocol.source_sha256` 记录场景和评测代码的哈希。

完整成功要求目标数量满足各盒配额、源盒保留 `初始源盒数量 − 新放数量`、没有
活动饼干落在盒外，并连续满足配置的保持步数。单盒任务还要求完整槽位匹配和
边界覆盖；两盒各 10 块时每盒均需满足完整槽位匹配和边界覆盖。`prefilled` 要求
原有 5 块仍在目标盒中。数量得分与完整成功率分别报告。
通过测试或运行 1 步冒烟不代表模型完成任务。

## 新增任务

以下六个扩展任务也仅支持 basic。使用统一评测入口传入任务名，默认使用任务指令。

| 任务名 | 数量评分 | 满分 |
| --- | --- | ---: |
| variable-prefill | 随机预填 1–9 块，保留原件并补齐目标盒，只计新增数量 | 10 减预填数 |
| sparse-exact-ten | 从四个源列取完剩余的 1、2、3、4 块 | 10 |
| whole-row-lean | 从整排同向轻倾的 20 块中取 10 块，只计指定排 | 10 |
| rotated-target | 目标盒水平旋转 50°–85°，沿盒内槽位填满 | 10 |
| 2x10-four-source | 四个源列各贡献 5 块，两个目标列各放 10 块 | 20 |
| 2x10-paired-columns | 按当回合指令，将两个指定源列分别放入对应目标列，各 10 块 | 20 |

两种 2×10 扩展均记录四行两列的来源去向矩阵。自定义策略默认收到 RGB，纯状态策略可声明 requires_cameras=False。

## 扩展场景布局版本 2

默认 2×5 扩展场景使用 54.4 mm 源列间距，源盒中心为世界坐标 `(0.155, 0.315)` 米；
同时平移全部源饼干。`rotated-target` 的目标盒中心为 `(0.005, 0.100)` 米；
`two-boxes` 的主盒和第二盒中心分别为 `(0.095, 0.140)`、`(0.070, 0.045)` 米。
`sparse-exact-ten` 的目标盒 Y 半尺寸由 35 mm 改为 40 mm，目标槽纵向间距由
约 6.33 mm 改为 12 mm，为分列的十块饼干逐块放置和夹爪退出留出空间。
二维坐标不包含每回合 basic 盒位扰动，第二目标盒沿用固定位置。2×10 扩展场景
沿用既有 2×10 源盒布局。原始 `2x5` 和 `2x10` 的配置及评分协议未因该调整改变。

采样期间检查同列不重叠、相邻源列 X 投影间隙至少 3 mm；场景设置和物理沉降后，
再次检查仍在源盒中的活动饼干，并检查不同盒子之间的碰撞穿透。无效初态会报错，
不会静默当作策略失败。自定义场景的源盒和单目标位置会保留，双目标盒仍按主盒位置
计算相对偏移；所有自定义配置也必须满足上述几何检查。
`randomization.source_column_clearances_m` 和 `source_column_clearance_pairs` 记录实测
间距及列号；`randomization.scenario.version` 和 `run.scenario_protocol.version` 为 2，
新旧布局的评测结果应分别记录。

扩展任务的绝对位姿 IK 在首次求解残差偏大时会使用固定的其他初值重试；这些
初值不消耗场景随机数。它不检查碰撞或路径，也不会替换笛卡尔动作适配器。

布局和端点 IK 检查不能证明携物路径可行或任务能完整完成。原 `expert` 仍是基线：
它不会根据稀疏库存、旋转目标、双目标盒或 2×10 来源配额自动规划完整任务。
