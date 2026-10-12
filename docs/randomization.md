# 2×5 分级随机化与专家采集

2×5 任务从选定来源列分两批各搬五个，共搬十个，使用整任务 LeRobot v3 数据格式。
随机化强度和来源列是两个独立选项；`source_column: random` 只随机选列，
每批数量固定为五个。2×10 用法见 [2×10 专家演示与数据采集](cookie_2x10.md)。

| 参数 | basic | medium | advanced |
|---|---:|---:|---:|
| 源盒、目标盒 XY 各轴 | ±10 mm | ±15 mm | ±20 mm |
| 目标盒偏航角 | ±0.030 rad | ±0.050 rad | ±0.080 rad |
| 饼干 XY、偏航角 | ±0.3 mm、±0.015 rad | 相同 | 相同 |
| 固定相机位置各轴 | 0 | ±4 mm | ±9 mm |
| 腕相机安装位置各轴 | 0 | ±2 mm | ±3 mm |
| 固定／腕相机局部旋转各轴 | 0 | 0 | ±2° / ±1° |
| 相机视场角 | 0 | ±1° | ±2° |
| 灯光强度 | 原值 | ±10% | ±20% |
| 饼干颜色 | 原值 | 原值 | 金黄、可可棕、抹茶绿、莓果粉四选一 |

每轮统一一种饼干颜色，整个回合保持不变。颜色不代表任务类别。相机扰动
以各自安装坐标系为准。重置先恢复标称参数，不会累积漂移。布局、外观、选列
使用独立随机流。没有增加初始倾斜、间距、尺寸、摩擦、质量或延迟随机化；
第一批搬走后自然发生的倾斜仍由同列专家处理。

## 采集

配置统一放在 `configs/randomization.yaml`，默认选择 basic、第一列。
修改顶部 `profile: advanced`、`source_column: 1` 后，在仓库根目录执行；
每组使用独立 root 和 repo-id：

```bash
MUJOCO_GL=egl python examples/collect_cookie_benchmark.py \
  --randomization-config configs/randomization.yaml --episodes 100 --max-attempts 200 \
  --root datasets/cookie_advanced_col1 --repo-id local/cookie-advanced-col1
```

- YAML `profile: basic|medium|advanced`：控制强度，可在 profiles 中新增命名档位。
- YAML `source_column: 1|2|3|4|random`：按源布局 X 从小到大编号，不按相机画面左右。
- `profiles` 中设置各项幅度，`colors` 中设置 RGBA 颜色；数值设为 0 可关闭该项，
  `palette: false` 关闭颜色变化。位置单位米、盒子角度弧度、相机角度度数。
- 2×5 与 2×10 的装盒采集命令不接受 `--profile`、`--source-column`，只使用 YAML；统一评测支持这两个覆盖项。
- 可复制此 YAML，再通过 `--randomization-config` 指定实验配置。启动时读取一次；
  运行中修改文件不影响当前任务，重启后生效。完整解析配置会随采集记录保存，
  续采时配置不一致会拒绝，避免不同实验混入同一数据集。
- `random`：每个对齐的连续四随机种子组随机排列四列，保证尝试数均衡，不保证
  成功保存数均衡。任务文本包含具体列号，固定列和随机列使用相同的指令模板。
- `--resume`：仅允许配置、强度、列模式、随机种子起点等一致时续采；旧版没有
  随机化元数据的数据集请不要原地续采，使用新目录。
- `--max-attempts`：限制总尝试次数。连续 25 次失败时停止供检查。

三档共用同列专家主体；多列使用列相关倾角、速度和放置补偿。没有为每档复制
独立专家。第四列仍是困难场景，先小规模评估，再决定采集量。

## 采集溯源与评测

`collection_summary.json` 保存档位、数值范围、来源列选择、完整配置和随机种子起点。
`attempts.jsonl` 保存包括失败在内的尝试记录及实际采样的场景参数。
成功回合的附加元数据也保留这些仿真真值，但不会加入模型的观测特征；
图像、状态和实际应用的绝对关节动作沿用原有数据格式。

采集质量应按档位和来源列分别检查。计算采集成功比例时，失败尝试必须保留在分母中；
该比例只用于采集质量诊断，策略评测仍以最终放入数量计分。
成功数据中的颜色比例不保证均衡，训练前应检查回合元数据。
训练与测试使用互不重叠的随机种子，小规模流程测试不能证明稳定的任务表现。

Python 调用与命令行共用评测实现：

```python
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

benchmark = make_cookie_benchmark(
    task="2x5",
    randomization_config="configs/randomization.yaml",
    profile="basic",
    source_column=1,
    max_steps=500,
)
result = benchmark.evaluate("same_column", num_episodes=20, seed_start=10000, workers=2)
```

旧 Python 接口未指定 `profile` 时保留显式数值参数；指定档位后使用该档位的盒位置范围。
`source_column=None` 仅用于兼容旧的自动选列指令行为。物理模型与几何不会因档位切换而改变。

## 统一评测

两任务的模型/专家评测、500 步数量统计及六项网格使用统一入口，命令见 [完整流程指南](workflow_guide.md#4-统一仿真评测)。以最终放入数量为主指标；采集仍按完整成功条件保存演示。
