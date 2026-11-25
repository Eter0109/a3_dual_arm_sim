# A3 项目文档中心

本目录集中存放项目使用文档。安装和最短运行示例见[项目首页](../README.md)；
命令默认在仓库根目录、已激活的 Python 环境中执行。

## 课程 Benchmark

| 我想做什么 | 文档 |
| --- | --- |
| 了解 18 个课程任务、统一提示词、评分及 140 回合评测计划 | [课程 Benchmark 说明](course_benchmark.md) |
| 看各场景图片、任务说明和单项命令 | [18 个场景图鉴](course_visual_guide/README.md)（[PDF](course_visual_guide/course_benchmark_18_scenes.pdf) / [HTML](course_visual_guide/report.html)） |
| 查看旧扩展场景、原提示词及来源配额规则 | [历史扩展场景与 legacy 协议](cookie_benchmark_scenarios.md) |

`course-v1` 的 18 个任务统一使用 `Fill the target box.`，以完整成功为主要指标。
旧来源配额任务和采集指令有各自的协议；与课程 PDF 的差异见课程说明。

## 数据采集、训练与评测

| 我想做什么 | 文档 |
| --- | --- |
| 按采集、训练、评测的顺序操作，或接入自己的策略 | [完整流程指南](workflow_guide.md) |
| 配置 2×5 随机化、来源列和专家采集 | [2×5 随机化与采集](randomization.md) |
| 运行 2×10 专家、设置抓取计划并采集数据 | [2×10 场景与采集](cookie_2x10.md) |
| 下载数据、训练和评估 SmolVLA | [SmolVLA 手册](smolvla.md) |
| 训练、部署和评估 ACT | [ACT 手册](act.md) |

## 开发与接口

| 我想查什么 | 文档 |
| --- | --- |
| 仿真、策略、数据与训练模块的职责 | [项目架构](architecture.md) |
| 动作和观测维度、时序、数据与成功判定 | [动作与数据契约](contracts.md) |
| CLI 入口、实验工具与结果追踪 | [入口索引与实验管理](experiments.md) |
| 依赖、渲染、模型和实机适用范围 | [使用范围与已知限制](limitations.md) |
| 仿真资源及第三方许可 | [资源来源声明](../assets/a3/PROVENANCE.md) |

课程规则和命令以课程 Benchmark 说明为准；历史协议按其文档区分。`outputs/` 等运行
目录保存本地实验数据，单步冒烟、开发结果和完整闭环成绩应分别记录。
