# 2×10 场景与可控抓取数量

2×10 装盒任务从源盒的 80 个饼干中搬运 20 个，目标两列各装十个。
场景配置为 `configs/cookie_2x10.yaml`，模型文件为 `models/a3_cookie_transfer_2x10.xml`，
规则专家按四次计划抓取。

## 配置

演示与采集默认读取 `configs/randomization_2x10.yaml`。三个选项并列，彼此独立：

```yaml
profile: basic         # basic / medium / advanced，控制环境随机化强度
source_column: 1       # 1 / 2 / 3 / 4 / random，选择源盒列
first_grasp: random    # 0–9 / random，选择 2×10 任务首夹数量
```

- `profile` 的数值幅度在 `profiles` 下配置，颜色在 `colors` 下配置。
- `source_column` 按源布局 X 从小到大编号；随机选列沿用独立的均衡抽样规则。
- `first_grasp` 不属于任何强度档位。切换档位或来源列不会改变它。
- `first_grasp: 3` 的四次计划为 `[3,7,3,7]`。一般形式为 `[n,10-n,n,10-n]`：
  前两次填满目标第 1 列，后两次填满目标第 2 列，每列十块。
- `first_grasp: 0` 跳过第 1、3 次抓取，第 2、4 次各搬十块。
- `first_grasp: random` 每个 episode 均匀抽取一次 0–9，首夹和第三夹共用该数量。
  抽样由 episode seed 决定，与环境扰动及选列使用独立随机流。

源盒初始化会重抽相交的饼干摆放，位置和角度仍在配置的扰动幅度内。

可复制 YAML，通过 `--randomization-config path/to/config.yaml` 指定另一份配置。
文件启动时读取一次，修改后需要重新启动。配置必须包含 `first_grasp`。
`--first-grasp` 参数可临时覆盖 YAML，正常使用只需编辑配置。

## 检查与专家演示

按 [README](../README.md) 安装依赖，在仓库根目录、已激活的 Python 环境中执行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 显示配置、数量计划与实际指令，不运行仿真、不写数据。
python examples/collect_cookie_2x10.py --dry-run
python examples/run_cookie_2x10.py --dry-run --seed 0

# 运行一次专家，输出最终数量和成功判定。
python examples/run_cookie_2x10.py --seed 0 --output artifacts/cookie_2x10/result.json

# 桌面窗口演示。先固定布局，去掉 --no-randomization 可查看配置中的扰动。
env -u MUJOCO_GL python examples/run_cookie_2x10.py --render --no-randomization --seed 0
```

`--render` 打开 MuJoCo 交互窗口，可旋转、平移和缩放视角，观察完整抓取过程。
该入口运行规则专家，无需下载神经策略权重。窗口在回合结束时关闭，结果保留在终端，
可用 `--output` 保存。`--seed` 可复现数量、选列和环境扰动。

Windows PowerShell 在仓库根目录执行：

```powershell
$env:PYTHONPATH = "$PWD/src"
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python examples/run_cookie_2x10.py --render --no-randomization --seed 0
```

图形窗口需要可用的桌面显示服务。远程 Linux 先连接已配置的远程桌面，
再在该桌面的终端执行演示命令。普通 SSH 会话需要连接相应的显示服务；仅设置
`DISPLAY` 不会创建桌面。相机离屏渲染使用 EGL 时，需有支持该后端的驱动和库。
交互窗口使用桌面 OpenGL，应清除此前设置的 `MUJOCO_GL=egl`。

## 数据采集与续采

安装数据依赖 `python -m pip install -e '.[dataset]'`，根据运行环境选择相机渲染后端。
以下为 Linux EGL 示例，数量和来源列均从 YAML 读取：

```bash
MUJOCO_GL=egl python examples/collect_cookie_2x10.py \
  --randomization-config configs/randomization_2x10.yaml \
  --episodes 5 --max-attempts 15 \
  --root datasets/cookie_2x10_trial --repo-id local/cookie-2x10-trial

# 相同配置续采。
MUJOCO_GL=egl python examples/collect_cookie_2x10.py \
  --randomization-config configs/randomization_2x10.yaml \
  --episodes 5 --max-attempts 15 \
  --root datasets/cookie_2x10_trial --repo-id local/cookie-2x10-trial --resume
```

默认启用视频，每次尝试最多 3200 控制步，只保存成功 episode。
`--episodes` 是成功保存条数，`--max-attempts` 包括失败尝试；连续 25 次失败时停止。
修改数量模式、随机化设置或 seed 起点后应使用新的数据目录，续采会检查这些设置。
Ctrl+C 会在当前 episode 完成后关闭记录器。

专家根据实际数量调整夹爪开度、抓力和放置位置，并检查接触链、抬升、释放及最终数量。
成功要求目标两列各十块、共二十块饼干直立、稳定且已释放，源盒保留六十块。
物体通过真实接触搬运，不绑定或瞬移；目标判定允许列内堆放偏离标称槽中心。

## 任务指令与训练兼容

首夹 3、源第 1 列时，实际写入 LeRobot 每帧的指令为：

> Transfer 20 cookies from source column 1 into the 2x10 target box. Grasp 3 cookies first, then 7 to fill target column 1 with 10. Repeat: grasp 3, then 7 to fill target column 2 with 10. Release all 20 cookies upright in the target box and leave 60 in the source box.

随机模式写入当前 episode 的实际数量及补数，指令中的数量不会写成 `random`。
数量为 0 时，指令明确跳过第 1、3 次抓取。

数据采用 LeRobot v3 格式，训练入口见 [SmolVLA 使用说明](smolvla.md)。
训练入口识别 2×10 数据后，将文本长度设为 128 tokens，保留完整四次抓取指令。
`collection_summary.json` 记录 2×10 布局及数量模式；episode 元数据记录实际数量、
四次计划、饼干 ID、抬升与释放检查和目标各列数量。审计核对指令、seed 和计划。
原十块任务的 benchmark 与二十块任务分别评估，不共用成功分数。

例如，采集目录为 `datasets/cookie_2x10_trial`、repo-id 为 `local/cookie-2x10-trial` 时，
先验证训练读取、视频解码和 GPU 优化：

```bash
python -m a3_dual_arm_sim.cli train-smolvla \
  --root datasets/cookie_2x10_trial --repo-id local/cookie-2x10-trial \
  --output outputs/cookie_2x10_smoke/model \
  --steps 20 --batch-size 8 --save-freq 20 --device cuda
```

短训练正常退出并保存检查点后，再对正式采集的数据运行完整训练。
批大小按显存实测调整，训练和测试使用对应的 2×10 场景及任务指令。

## 采集前试跑

先按需要设置 YAML 的强度、来源列和首夹数量，再运行几个不同 seed：

```bash
python examples/run_cookie_2x10.py --seed 100 --output artifacts/cookie_2x10/seed100.json
python examples/run_cookie_2x10.py --seed 101 --output artifacts/cookie_2x10/seed101.json
```

结果中的 `success` 应为 `true`，`target_column_counts` 应为 `[10,10]`，
`cookies_in_source` 应为 `60`。`grasp_reports` 记录每夹的计划数量、实际抬升数量和
释放结果；数量为零的夹次会标记 `skipped`。

采集结束后，查看 `collection_summary.json` 中的成功数和总尝试数，并抽查回放。
用新的 seed 评估 SmolVLA，避免使用训练数据中的布局作为最终测试集。
