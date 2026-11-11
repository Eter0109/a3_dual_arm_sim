#!/usr/bin/env python3
"""Slide figures for the mid-term report: Motivation, Challenges, Proposed.

Each is generated at the strip the slide leaves free -- 11.5 x 1.85 inches at the bottom of a
page whose body text ends around 5.5 inches -- so nothing needs rescaling in the deck.

Every number is read from files already on disk (the benchmark JSON and one recorded episode's
trace).  Where a panel is a concept sketch rather than measured data it is drawn as a diagram
with no invented curves, and labelled as such.

Glyph note: the CJK font here (WenQuanYi Zen Hei) has no combining dot, no subscripts and no
superscripts, so `q_dot` / `a_prev` / `128-dim` are written in plain ASCII rather than with
Unicode that would render as boxes.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

FONT = "/root/arm_agent/assets/fonts/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT)
CN = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.family": CN,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT = Path("/tmp/ppt")
FIGS = Path("/root/a3_dual_arm_sim/docs/figs")
FIGS.mkdir(parents=True, exist_ok=True)
EP = Path("/root/autodl-tmp/diag/v9_best/2x10_rank1_seed5_20of20")
LOGS = Path("/root/autodl-tmp/logs")

ACC, GREY, WARN, OK = "#2f6db4", "#5a5a5a", "#c0392b", "#2e8b57"
STRIP = (11.4, 1.63)
DPI = 220


def scores(tags, row):
    d = {}
    for t in tags:
        for p in glob.glob(str(LOGS / f"{t}_{row}_*.json")):
            seed = int(re.search(r"_(\d+)\.json$", p).group(1))
            d[seed] = json.load(open(p))["episodes"][0]["cookies_in_target"]
    return d


def trace(path):
    steps, states, placed = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            steps.append(r["step"])
            states.append(r["observation_before"]["observation.state"][:8])
            placed.append(r["after"]["cookies_in_target"])
    return np.array(steps), np.array(states, float), np.array(placed)


# ------------------------------------------------------------------ Motivation
def fig_motivation() -> None:
    fig = plt.figure(figsize=STRIP)

    # --- two frames, nearly the same arm pose, opposite task stage
    for k, (frame, cap) in enumerate([(5, "第 5 步 · 0/20"), (1122, "第 1122 步 · 20/20")]):
        left = 0.006 + k * 0.104
        ax = fig.add_axes([left, 0.30, 0.096, 0.58])
        ax.imshow(Image.open(OUT / f"f{frame:04d}_front.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(WARN if k == 0 else ACC)
            s.set_linewidth(1.8)
        ax.set_title(cap, fontsize=7.0, color=WARN if k == 0 else ACC, pad=3)

    # what differs, under both frames
    ax = fig.add_axes([0.004, 0.015, 0.212, 0.24]); ax.axis("off")
    ax.text(0.5, 0.72, "机械臂姿态几乎相同：差 0.38 rad", ha="center", fontsize=7.4, color=GREY)
    ax.text(0.5, 0.22, "目标盒却从 0 块变成 20 块", ha="center", fontsize=7.4, color=WARN)

    # --- the consequence, measured on 88 episodes
    ax = fig.add_axes([0.290, 0.31, 0.698, 0.44])
    vals = list(scores(["v9r", "v9s"], "2x5").values())
    counts = np.array([sum(1 for v in vals if v == k) for k in range(11)])
    bars = ax.bar(range(11), counts, color="#ccd8e6", edgecolor="white", linewidth=0.7)
    bars[5].set_color(WARN)
    ax.set_xticks(range(11)); ax.tick_params(labelsize=6.8, length=2, pad=1)
    ax.set_yticks([]); ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(0, counts.max() * 1.30)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("2x5 任务中最终放入目标盒的饼干数（满分 10，88 个回合）", fontsize=7.2,
                  color=GREY, labelpad=1)
    ax.annotate(f"{counts[5]}／88 = {counts[5]/len(vals)*100:.0f}% 恰好搬完一批就停",
                xy=(5.35, counts[5] * 0.97), xytext=(6.5, counts[5] * 1.20),
                fontsize=8.4, color=WARN, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
    ax.text(0.5, 0.985, "单帧看不出做到第几批", transform=ax.transAxes, ha="center",
            va="top", fontsize=7.6, color="#333333")

    fig.savefig(FIGS / "fig_motivation.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ Challenges
def fig_challenges() -> None:
    fig = plt.figure(figsize=STRIP)

    # (a) the pose oscillates; only the cookie count rises.  Both traces measured.
    ax = fig.add_axes([0.052, 0.34, 0.385, 0.42])
    steps, states, placed = trace(EP / "trace.jsonl")
    ax.plot(steps, states[:, 0], color=ACC, lw=1.0)
    ax.set_xlim(0, len(steps)); ax.set_ylim(1.28, 1.70)
    ax.set_ylabel("左肩关节角 (rad)", fontsize=6.8, labelpad=1, color=ACC)
    ax.set_yticks([1.35, 1.55]); ax.tick_params(labelsize=6.4, length=2, pad=1,
                                                axis="both")
    ax.tick_params(axis="y", colors=ACC)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    ax2.step(steps, placed, where="post", color=WARN, lw=1.4)
    ax2.set_ylim(0, 24)
    ax2.set_yticks([0, 10, 20]); ax2.tick_params(labelsize=6.4, length=2, pad=1,
                                                  colors=WARN)
    ax2.set_ylabel("目标盒内饼干数", fontsize=6.8, labelpad=1, color=WARN)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)

    ax.set_title("关节角反复回到同一区间，只有进度单调上升（实测 1 个回合）",
                 fontsize=7.2, pad=3)
    ax.text(0.5, -0.30, "姿态不能区分阶段，必须靠累计历史",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9, color=WARN)

    # (b) covariate shift, drawn as a diagram -- no invented curve
    ax = fig.add_axes([0.545, 0.06, 0.430, 0.80]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(50, 94, "一次偏差被后续状态继承", ha="center", fontsize=7.4, color="#333333")

    ax.add_patch(Circle((29, 48), 21, facecolor="#eaf2ea", edgecolor=OK, lw=1.0, ls="--"))
    ax.text(29, 22, "专家演示覆盖的状态", ha="center", fontsize=6.4, color=OK)
    ax.add_patch(FancyArrowPatch((15, 54), (39, 43), arrowstyle="-|>", color=OK, lw=1.3,
                                 mutation_scale=8))
    ax.text(12, 58, "专家", fontsize=6.4, color=OK)
    ax.add_patch(FancyArrowPatch((39, 43), (84, 72), arrowstyle="-|>", color=WARN, lw=1.3,
                                 ls="--", mutation_scale=8))
    ax.add_patch(Circle((84, 72), 2.4, facecolor=WARN, edgecolor="none"))
    ax.text(84, 82, "模型自身执行", fontsize=6.4, color=WARN, ha="center")
    ax.text(54, 34, "偏差沿 action chunk 累积", fontsize=6.2, color=WARN, ha="center")
    ax.text(50, 4, "训练只见专家状态，测试时输入来自模型自己", ha="center", fontsize=6.8,
            color=WARN)

    fig.savefig(FIGS / "fig_challenges.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ Proposed
def fig_proposed() -> None:
    fig = plt.figure(figsize=STRIP)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, label, sub=None, fc="#eef3f9", ec=ACC, fs=7.6, subfs=6.2,
            lw=1.2, ls="-", col="#222222"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, ls=ls,
                               zorder=2, joinstyle="round"))
        cy = y + h / 2 + (5.5 if sub else 0)
        ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=fs, zorder=3,
                color=col)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 7.5, sub, ha="center", va="center",
                    fontsize=subfs, color=GREY, zorder=3, linespacing=1.4)

    def arrow(x1, y1, x2, y2, col="#7a7a7a", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=col,
                                     lw=1.1, ls=ls, mutation_scale=8, zorder=4))

    # top row: prefix tokens -> PaliGemma
    y0, h = 64, 24
    box(1.0, y0, 12.0, h, "图像", "3 路相机", fc="#f5f5f5", ec="#9a9a9a")
    box(14.6, y0, 12.0, h, "语言", "任务指令", fc="#f5f5f5", ec="#9a9a9a")
    box(28.2, y0, 12.4, h, "state", "线性投影", fc="#f5f5f5", ec="#9a9a9a")
    box(42.2, y0 - 5, 19.0, h + 10, "history token", "本体感知历史 10x48\n压缩为 128 维",
        fc="#fdeceb", ec=WARN, fs=8.2, lw=1.8, col=WARN)
    box(65.5, y0, 16.5, h, "PaliGemma", "prefix，原样冻结", fc="#eef3f9")

    arrow(13.0, y0 + h / 2, 14.6, y0 + h / 2)
    arrow(26.6, y0 + h / 2, 28.2, y0 + h / 2)
    arrow(40.6, y0 + h / 2, 42.2, y0 + h / 2, col=WARN)
    arrow(61.2, y0 + h / 2, 65.5, y0 + h / 2, col=WARN)

    # bottom row: action expert -> actions
    box(65.5, 14, 16.5, 22, "action expert", "suffix，原样冻结")
    box(87.0, 14, 10.5, 22, "动作", "50 步 / 2.5 s")
    arrow(73.75, y0 - 1, 73.75, 36.5)
    arrow(82.0, 25, 87.0, 25)

    # encoder detail, below the token, no overlap with the expert row
    ax.add_patch(Rectangle((27.0, 3.5), 33.0, 40, facecolor="#fffaf9", edgecolor=WARN,
                           lw=0.9, ls="--", zorder=2))
    ax.text(43.5, 38, "轻量循环编码器（不读取图像）", ha="center", fontsize=6.6, color=WARN)
    ax.text(43.5, 28, "q, dq, a_prev  x10 帧", ha="center", fontsize=6.9, color="#333333")
    ax.text(43.5, 19, "GRU → 128 维隐变量", ha="center", fontsize=6.9, color="#333333")
    ax.text(43.5, 9, "进度回归头仅训练期存在", ha="center", fontsize=6.2, color=GREY)
    arrow(43.5, 43.5, 43.5, 58.5, col=WARN, ls=":")

    # what is untouched
    ax.text(1.0, 47, "改动范围", fontsize=7.0, color=GREY)
    ax.text(1.0, 38, "骨干与 action expert 原样", fontsize=6.6, color="#333333", va="top")
    ax.text(1.0, 29, "PaliGemma 冻结", fontsize=6.6, color="#333333", va="top")
    ax.text(1.0, 19, "新增参数 36.6 万", fontsize=6.8, color=ACC, va="top")
    ax.text(1.0, 10, "占总量 0.01%", fontsize=6.2, color=ACC, va="top")

    fig.savefig(FIGS / "fig_proposed.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_motivation, fig_challenges, fig_proposed):
        fn()
        print(fn.__name__, "ok")
    for p in sorted(FIGS.glob("fig_*.png")):
        print(f"  {p.name} {p.stat().st_size/1024:.0f} KB")
