# 课程饼干 Benchmark

`course-v1` 包含 14 个基础任务和 4 个 Bonus 任务。全部任务向语言策略传入同一句
`Fill the target box.`；ACT 不使用文本。每回合按目标盒内直立、释放且稳定的物体判定：
2×5 目标盒两列各 5 块，2×10 两列各 10 块，双盒任务每盒 5 块。释放检查覆盖左右夹爪。
源盒库存、取料列、
抓取次数、原预填物身份和盒外物体仅记录，不作为成功条件；物理安全终止仍会导致失败。
默认判定要求倾斜角不超过 15°，相对目标盒的线速度不超过 0.09 m/s、角速度不超过
0.30 rad/s，并连续满足 5 个控制步；这些数值是实现中的判定阈值，不是 PDF 指定的分数。

课程任务组：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.course_benchmark_cli \
  --policy expert --episodes 5 --seed-start 13000 --output course_results.json
```

默认标准 2×5 和 2×10 使用 `advanced`；通过 `--standard-profile basic|medium|advanced`
可修改两个标准任务的档位。所有扩展和 Bonus 任务只支持 `basic`。每个任务默认 5 回合，
基础任务共 70 次，Bonus 共 20 次；报告中的 Bonus 每个完整成功记 1 分。未完成或抛错的
任务标记为 incomplete，不计作已完成回合。`--episodes` 更改样本量时不计算课程 Bonus 分。

单项任务继续通过现有入口运行：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli \
  --task 2x10-sparse-exact-twenty --policy expert --profile basic \
  --episodes 5 --seed-start 13000 --workers 1 --output task_results.json
```

新增的 2×10 任务名为 `2x10-missing-columns`、`2x10-partial-gaps`、
`2x10-cross-column`、`2x10-sparse-exact-twenty`、`2x10-tilted`、
`2x10-whole-row-lean`。2×10 稀疏源列的块数为 2、4、6、8；目标盒更宽，
使逐块释放有空间。默认回合上限为 2×5 的 6000 步、2×10 的 12000 步；
`--max-steps` 可覆盖，实际回合长度与环境 horizon 均记录在报告中。

两个历史来源配额任务 `2x10-four-source` 和 `2x10-paired-columns` 不在课程任务组，
继续使用各自的原提示词、评分和 basic 档位。历史单任务提示词可显式使用
`--protocol legacy --prompt-style ...`；旧数据采集器保持原有协议。课程协议与 PDF 中
“所有任务 advanced”“从倾斜列取料”和“保留预填物”的描述不同，原因是课程协议遵循
后续确认的 basic 限制和目标盒最终状态判定。不要混合新旧协议的评测结果。

`expert` 是读取模拟器特权状态、通过真实接触搬运的规则控制器，不是训练出的模型。
稀疏 2×10 专家按源列现有块数分成 8+2 和 6+4 四批搬运，填满目标盒的两列；
稀疏 2×5 专家逐块搬运，并按现场状态选择抓取姿态和必要的关节路径。
一条专家成功回合只证明该设置下存在一次成功执行，不代表对所有 seed 可达或模型成功率。
