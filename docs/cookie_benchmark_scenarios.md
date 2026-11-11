# 六种饼干场景 benchmark

六种任务使用既有 `benchmark_cookie_batch` 入口、相机、动作接口和 JSON 报告。
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
  --task partial-gaps --policy expert --episodes 20 --seed-start 13000 \
  --profile basic --max-steps 1000 --workers 4 \
  --output /tmp/cookie_partial_gaps.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task two-boxes --policy smolvla:/absolute/path/to/checkpoint \
  --episodes 20 --seed-start 13000 --profile basic --max-steps 1000 \
  --output /tmp/cookie_two_boxes.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task tilted --tilt-level fallen --policy expert --episodes 20 \
  --seed-start 13000 --profile basic --max-steps 1000 \
  --output /tmp/cookie_fallen.json

MUJOCO_GL=egl python examples/benchmark_cookie_batch.py \
  --task two-boxes --box-quota 10 --policy expert --episodes 20 \
  --seed-start 13000 --profile basic --max-steps 3200 \
  --output /tmp/cookie_two_full_boxes.json
```

六种任务使用各自的 `task` 文本，记录在每回合 `randomization.task_instruction`。
若显式传入 `--prompt-style`，应使用 `task`。现有 ACT 接口不消费任务文本；
它与文本条件策略的对比应注明输入条件。`--profile basic|medium|advanced` 和
`--no-randomize-boxes`、`--no-randomize-cookies` 继续控制原有环境扰动；
这些选项不会关闭场景本身的缺失、倾斜或预填充。`--source-column` 控制旧专家的
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
