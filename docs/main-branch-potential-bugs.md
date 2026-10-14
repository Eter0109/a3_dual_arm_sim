# main 分支潜在 bug 清单（2026-10-14 调查）

对照分支 `vla-policy-interface-v2` 已修复的问题，`ssh-origin/main`（29fe90e）上仍可能触发的项。

## 高危（随机化/发散时静默出错，无报错）

### 1. reset 后臂位姿不重申
- **位置**：`sim/env.py:165-206`（基类 reset）、`tasks/cookie_transfer.py`（子类 reset）
- **问题**：settle 步数内若仿真发散，MuJoCo 内部 `mj_checkPos/mj_checkVel` 会把臂 qpos 静默打回初始值，reset 后无人重写回 home，策略看到的 eef 状态与真实不符（本地实测曾偏 841 mm / 200°）。
- **触发条件**：settle 期间任何发散（重排布、大随机化、新场景）。
- **修复参考**：`cookie_transfer.py _assert_arm_at_home`（写回 qpos + 1e-6 rad 校验）。

### 2. 盒位姿被 settle 擦除
- **位置**：`tasks/cookie_transfer.py` reset——target 盒走 `target_bin_free` 自由关节 qpos（会被 MuJoCo 复位）；source 盒走 `model.body_pos`（免疫）。
- **问题**：box 随机化写入 free joint qpos → 放饼干 → 50 步 settle 中若发散，MuJoCo 把盒位姿静默打回 XML 名义值；已放入的饼干按记录偏移放置 → 与盒壁穿插被弹飞（本地实测 3/10 种子、最多 20/80 块出盒）。无任何报错。
- **触发条件**：`randomize_boxes=True`（现在默认 False，所以暂未爆发）+ settle 发散。
- **修复参考**：`_assert_box_poses`（重放抽取的盒位姿）。

## 中危（一旦配置变化即触发）

### 3. 批专家 transit 位姿是世界坐标常量
- **位置**：`controllers/cookie_expert.py:196`——`q_transit = self._solve_l(np.array([0.10,  0.30, 0.89]), ...)`，无布局偏移补偿。
- **问题**：场景布局平移后，调优 transit 点落到盒/恐推范围外 → 专家回退到"盒中心"位姿 → IK 差 4.33 mm 超出容差必抛错。
- **触发条件**：`cookie_source_positions_m` / `source_bin_center_m` 等大幅偏移（当前 noise 仅 ±1-2 mm 暂不触发；若做过布局实验或换放位就触发）。
- **修复参考**：`expert.py reset` 里 `layout_offset = ...`；`tuned_transit_xy = fixed_transit_xy + layout_offset`。

### 4. settle 前后顺序风险（50 步无保护窗口）
- **位置**：`tasks/cookie_transfer.py` reset——settle(100) → 放 80 块 → settle(50)。
- **问题**：第二个 settle 无保护，即 bug 2 的触发窗口。**settle(100) 与放置的先后顺序本身正确**。
- **触发条件**：同 bug 2。

## 低危（已知未修，两分支同病）

### 5. `actions.py` IK 失败无校验
- **位置**：`sim/actions.py` 的 `solve`/`convert`——`res = least_squares(...)` 后 **直接返回 `res.x`**，不检查 `res.success`、不检查残差、不回退 `initial_q`。
- **问题**：IK 解算失败时按发散解下发关节指令（抖动或突然甩臂）；仅在专家层有校验，走 adapter 直接路径无保护。
- **触发条件**：目标位姿越界或奇异点附近，命令来回跳。本地 `actions.py:130` 同样未修。

### 6. Panda/libero 接入时需随行的修复（若合并 π0.5 栈到 main）
main 目前**没有** LIBERO/pi0.5 相关代码，以下坑本地已踩过、main 接入时会直接踩：

- **动作空间混用**：π0.5 LIBERO 输出是 7 维 eef 增量（[-1,1]），当关节角读**数值差 ~7 倍**→ 步步撞速率上限，trace 里 `rate_limited≈0.9975`，无报错。需 `action.space: cartesian_delta` + `dim: 14`（双臂 padding 约定：零 delta = 保持）。
- **旋转增量坐标系**：OSC 与 A3 IK 对同一 delta 旋转部分解释不同（world 帧的工具局部帧，实际差 120°）。修复在 `actions.py`（`R.T @ delta` 转换），必须随行。
- **扭矩臂 vs 位置伺服混淆**：`cartesian` 旗标不跟随 `panda` 时给 OSC 扭矩臂建关节命令策略，**零报错只是臂够不着**。main 的旗标配对逻辑需对照检查。
- **π0.5 launcher 解冻默认值陷阱**：`_unfreeze_request()` 默认 18 层+vision tower 解冻，与脚本注释声明的 expert-only 矛盾，不设 env 时实际训 3.09B 而是 693M，会 OOM。合并时需把默认改 frozen。

## 排除项（查过、main 无此问题）

- 批专家 CLOSE 阶段指垫接触丢失——LIBERO Panda 三-mesh 引入的坑，main 无 Panda。
- `left_finger_pad_half_thickness_m: 0.0005` 三个 cookie YAML 全带。
- 乱抖根因（执行器 kv 阻尼 b08296c）main 已修（sim/model.py:278 同数值）。
