# 分级随机化与整任务采集 / Randomization profiles

随机化强度和抓取列是两个独立选项。保持单盒、同列两次各五块、共十块的
整任务 LeRobot v3 数据格式；不使用 Planner/Verifier，不切技能片段。

| 参数 / Parameter | basic | medium | advanced |
|---|---:|---:|---:|
| 源盒、目标盒 XY 各轴 / box XY | ±10 mm | ±15 mm | ±20 mm |
| 目标盒 yaw | ±0.030 rad | ±0.050 rad | ±0.080 rad |
| 饼干 XY、yaw | ±0.3 mm、±0.015 rad | 相同 | 相同 |
| 固定相机位置各轴 / fixed camera | 0 | ±4 mm | ±9 mm |
| 腕相机安装位置各轴 / wrist camera | 0 | ±2 mm | ±3 mm |
| 固定／腕相机局部旋转各轴 | 0 | 0 | ±2° / ±1° |
| FOV | 0 | ±1° | ±2° |
| 灯光强度 / light intensity | 原值 | ±10% | ±20% |
| 饼干颜色 / cookie palette | 原值 | 原值 | 金黄、可可棕、抹茶绿、莓果粉四选一 |

每轮统一一种饼干颜色，整个 episode 保持不变。颜色不代表任务类别。相机扰动
以各自安装坐标系为准。重置先恢复标称参数，不会累积漂移。布局、外观、选列
使用独立随机流。没有增加初始倾斜、间距、尺寸、摩擦、质量或延迟随机化；
第一批搬走后自然发生的倾斜仍由同列专家处理。

## 采集 / Collection

在仓库根目录执行；每组使用独立 root 和 repo-id：

```bash
MUJOCO_GL=egl python examples/collect_cookie_benchmark.py \
  --profile advanced --source-column 1 --episodes 100 --max-attempts 200 \
  --root datasets/cookie_advanced_col1 --repo-id local/cookie-advanced-col1
```

- `--profile basic|medium|advanced`：控制强度。
- `--source-column 1|2|3|4|random`：按源布局 X 从小到大编号，不按相机画面左右。
- `random`：每个对齐的连续四 seed 组随机排列四列，保证尝试数均衡，不保证
  成功保存数均衡。任务文本包含具体列号，固定列和随机列使用相同的指令模板。
- `--resume`：仅允许配置、强度、列模式、seed 起点等一致时续采；旧版没有
  随机化元数据的数据集请不要原地续采，使用新目录。
- `--max-attempts`：限制总尝试次数。连续 25 次失败时停止供检查。

三档共用同列专家主体；多列使用列相关倾角、速度和放置补偿。没有为每档复制
独立专家。第四列仍是困难场景，先小规模评估，再决定采集量。

## Provenance and evaluation

`collection_summary.json` records the profile, numeric ranges, source-column
selection, config and seed origin. `attempts.jsonl` includes every completed or
failed attempt and its sampled scene parameters. Saved episode sidecars contain
the same privileged provenance; it is **not** added to policy observation features.
Images, state and absolute applied joint actions retain their original schema.

Measure success by profile and selected column, not only the total. Failed
attempts must remain in the denominator. Color balance among saved episodes is
not guaranteed; inspect the saved sidecars before training. Small smoke matrices
validate connectivity, not an 80–90% success claim. Train/test seeds must be disjoint.

Programmatic evaluation uses the same implementation:

```python
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark
b = CookieBatchBenchmark(profile="advanced", source_column="random", max_steps=1600)
result = b.evaluate("same_column", num_episodes=20, seed_start=10000, workers=2)
```

Omitting `profile` from the existing Python API preserves its explicit numeric
arguments; specifying a profile selects that profile's box ranges. Physics and
geometry are otherwise unchanged. Use `source_column=None` only for the legacy
automatic-column instruction behavior.

## 验证范围 / Validation scope

相关回归测试共 39 项通过，可在安装开发依赖后运行：

```bash
python -m pytest -q tests/test_benchmark.py tests/test_cookie_same_column.py \
  tests/test_runner_and_recording.py tests/test_randomization_profiles.py
```

初步专家测试覆盖三档强度 × 四列，每组合一次，共 12/12 成功；高级档四列
分别使用 seed 9001–9004，均搬入十块。另以高级第一列 seed 9001 验证了一条
实际 LeRobot v3 视频与动作采集，因此该采集不是额外独立测试样本。
之后修复了正视目标跟踪相机覆盖角度扰动的问题，并通过回归测试及实际渲染检查；
上述 12 次物理搬运试跑在这一相机修复之前完成。规则专家不读取相机图像。

这只是功能连通性验证，不代表达到 80%–90% 的采集成功率，也不是 VLA 成绩。
正式采集前建议用独立 seed 每列至少尝试 25 次，保留失败记录，并报告各列
成功次数/总尝试次数。不要仅以保存下来的成功 episode 计算成功率。
