# 课程饼干 Benchmark

本文描述当前实现的 `course-v1` 课程评测：**14 种基础任务、4 种 Bonus 任务**。
全部 18 种任务向策略接口传入同一条任务指令：

```text
Fill the target box.
```

对于实现 `reset(context)` 和 `act(observation, task)` 的策略，指令分别传入每回合的
`context.task` 和每步的 `task`，并记录在回合的 `randomization.task_instruction` 中。
单参数策略可以忽略任务文本，ACT 不使用任务文本；规则专家读取仿真器特权状态，并非训练出
的模型。是否能加载某个模型，不代表它已经适配这条指令或能够完成任务。

## 任务清单

18 种任务的提示词均为上面的 `Fill the target box.`。每个场景的初态截图、说明和单项
运行命令见 [18 个场景图鉴](course_visual_guide/README.md)。

| 基础场景 | 2×5 任务 ID | 2×10 任务 ID | 初始场景 |
| --- | --- | --- | --- |
| 标准搬运 | `2x5` | `2x10` | 完整源盒、空目标盒 |
| 缺失整列 | `missing-columns` | `2x10-missing-columns` | 随机移除 1 或 2 个源列 |
| 局部缺口 | `partial-gaps` | `2x10-partial-gaps` | 各源列的前端和内部出现缺口 |
| 跨列取料 | `cross-column` | `2x10-cross-column` | 四列各保留连续 8 块 |
| 稀疏库存 | `sparse-exact-ten` | `2x10-sparse-exact-twenty` | 四列分别保留 1/2/3/4 块或 2/4/6/8 块，列顺序随机 |
| 局部倾斜 | `tilted` | `2x10-tilted` | 两个源列各有一块倾斜物体 |
| 整列轻倾 | `whole-row-lean` | `2x10-whole-row-lean` | 一个源列的物体同向轻倾 |

| Bonus 场景 | 任务 ID | 初始场景和目标 |
| --- | --- | --- |
| 固定预填 | `prefilled` | 目标盒预放 5 块，最终填满 2×5 盒 |
| 双目标盒 | `two-boxes` | 两个目标盒最终各有 5 块 |
| 旋转目标盒 | `rotated-target` | 目标盒水平旋转，最终填满 2×5 盒 |
| 可变预填 | `variable-prefill` | 目标盒随机预放 1–9 块，最终填满 2×5 盒 |

## 完整评测计划

默认评测计划为 `course-140-v1`，共 **22 个任务与档位组合、140 回合**：

| 任务组 | 随机化档位 | 回合数 |
| --- | --- | ---: |
| 标准 `2x5`、`2x10` | 每个任务分别使用 `basic`、`medium`、`advanced` | 2 × 3 × 10 = 60 |
| 其余 12 种基础扩展任务 | `basic` | 12 × 5 = 60 |
| 4 种 Bonus 任务 | `basic` | 4 × 5 = 20 |
| 合计 | | 140 |

默认每回合上限：2×5 及其扩展任务为 **6000 控制步**，2×10 及其扩展任务为
**12000 控制步**。这是评测计划而非已经取得的成绩；实际耗时取决于策略、渲染和硬件。

## 一条命令启动完整评测

在仓库根目录、已安装本项目依赖的环境中运行：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.examples.course_benchmark
```

示例脚本默认运行 `expert` 和上述完整的 140 回合，种子从 0 开始，使用 1 个 worker。
它在 `outputs/course_benchmark/` 下自动生成报告文件名，并在终端显示路径；每个任务与
档位组合结束后更新 JSON 报告。

先查看完整计划、不运行仿真：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.examples.course_benchmark --dry-run
```

`--dry-run` 在终端输出计划，也将计划 JSON 保存到报告路径。可以指定策略、种子和新
报告路径；参数转交课程评测入口：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.examples.course_benchmark \
  --policy smolvla:/绝对路径/pretrained_model --seed-start 13000 --workers 1 \
  --output outputs/course_benchmark/smolvla_13000.json
```

策略还可使用 `act:/绝对路径/pretrained_model`、`pi05:/绝对路径/pretrained_model` 或
`module:factory`。路径和模型依赖需与实际检查点一致；示例脚本拒绝使用已存在的输出路径，
请为每次运行选择新路径。模型适配器使用其默认推理配置，需自定义时可提供策略工厂。

`--episodes N` 覆盖所有组合的回合数，`--standard-episodes`、`--expanded-episodes` 和
`--bonus-episodes` 分别覆盖各组；`--standard-profile basic|medium|advanced` 只运行该
标准档位，`--max-steps` 覆盖每项的控制步上限。这些选项可用于自定义测试；
报告中的 `is_standard_assessment` 标明其计划是否仍为完整的默认方案。可用 `--help`
查看全部参数。

也可直接调用底层入口；底层入口会覆盖指定的已有输出文件：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.course_benchmark_cli \
  --policy expert --seed-start 13000 --output outputs/course_benchmark/course_13000.json
```

单个课程任务可通过统一 benchmark 入口运行，例如：

```bash
MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli \
  --protocol course-v1 --task 2x10-sparse-exact-twenty --policy expert --profile basic \
  --episodes 5 --seed-start 13000 --workers 1 --output outputs/course_benchmark/sparse20.json
```

## 成功判定与结果

`course-v1` 以目标盒最终状态和完整成功 `success` 为主要判定：2×5 目标盒两列各 5 块，
2×10 两列各 10 块，双盒任务每盒 5 块。计入的物体须直立、释放且稳定；释放检查覆盖左右
夹爪。默认倾斜角不超过 15°，相对目标盒的线速度不超过 0.09 m/s、角速度不超过
0.30 rad/s，并连续满足 5 个控制步。这些是实现阈值，不是 PDF 中指定的分数。

源盒库存、取料列、抓取次数、原预填物身份和盒外物体仅记录，不作为课程成功条件；
物理安全终止仍会导致失败。物体入盒数量可用于诊断，不能代替完整成功数。

报告分别列出标准、扩展、基础、Bonus 和总计的 `completed`、`expected`、`successes`；
`tasks` 保留每个任务与档位的状态及逐回合数据。只有 4 个 Bonus 任务各 5 回合均完整执行
时才报告 Bonus `points`，每个完整成功记 1 分，不推算课程总分。失败和未完成要分别看：
任务报错时报告会标记 `error`；源码未变化时保留已完成回合，源码发生变化时会将当前任务的
回合标记为无效。未运行的回合不当作失败。

报告在每个任务与档位组合结束时更新。若运行被 Ctrl+C 或进程终止中断，当前尚未写入的
组合可能丢失，已经写入的组合仍在报告中；当前入口不支持自动续跑。完整评测应检查报告的
`status`、各组已完成数和逐回合记录，不应将单步冒烟或部分回合结果当作完整成绩。

## 历史协议与课程 PDF

两个历史来源配额任务 `2x10-four-source` 和 `2x10-paired-columns` 不在 18 个课程任务中，
继续使用各自的 `legacy` 提示词和评分；旧数据采集器也保持原有协议。单任务旧协议可显式
使用 `--protocol legacy --prompt-style ...`。课程协议与 PDF 中“所有任务 advanced”、
“从倾斜列取料”和“保留预填物”的描述不同，项目当前遵循后续确认的档位安排和目标盒最终
状态判定。对照课程要求时应保留这些差异，不能混合新旧协议的评测结果。
