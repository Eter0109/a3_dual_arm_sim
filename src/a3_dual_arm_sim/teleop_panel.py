from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

from .teleop import KeyboardTeleopPolicy

ResultT = TypeVar("ResultT")


class TeleopControlPanel:
    """Dedicated keyboard and button UI that leaves MuJoCo Viewer display-only."""

    def __init__(self, policy: KeyboardTeleopPolicy, finished: threading.Event) -> None:
        try:
            import tkinter as tk
            import tkinter.font as tkfont
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError("Tk 8.6 or newer is required for the teleop panel") from exc

        self._policy = policy
        self._finished = finished
        self.root = tk.Tk()
        self.root.title("A3 Teleoperation Control")
        self.root.geometry("760x600+30+40")
        self.root.minsize(700, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._save_and_quit)

        available_fonts = set(tkfont.families(self.root))
        cjk_candidates = (
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
        )
        self._cjk_font = next(
            (name for name in cjk_candidates if name in available_fonts), None
        )
        if self._cjk_font:
            for named_font in (
                "TkDefaultFont",
                "TkTextFont",
                "TkMenuFont",
                "TkHeadingFont",
                "TkCaptionFont",
            ):
                tkfont.nametofont(named_font).configure(family=self._cjk_font, size=10)

        style = ttk.Style(self.root)
        style.configure("Move.TButton", padding=(8, 9))
        style.configure("Exit.TButton", padding=(8, 7))

        container = ttk.Frame(self.root, padding=14)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text=self._text("A3 双臂遥操作", "A3 Dual-Arm Teleoperation"),
            font=(self._cjk_font or "TkDefaultFont", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            container,
            text=self._text(
                "键盘请在本窗口使用；MuJoCo 窗口只负责观察和鼠标调整视角。",
                "Keep keyboard focus here. Use MuJoCo only to watch or adjust the view.",
            ),
            anchor="w",
            padx=10,
            pady=7,
            bg="#fff3cd",
            fg="#5f4500",
        ).pack(fill="x", pady=(5, 10))

        status_frame = ttk.LabelFrame(
            container, text=self._text("当前状态", "Current status"), padding=8
        )
        status_frame.pack(fill="x", pady=(0, 8))
        self._status_var = tk.StringVar()
        self._gripper_status_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self._status_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self._gripper_status_var).pack(
            anchor="w", pady=(3, 0)
        )

        arm_frame = ttk.LabelFrame(
            container,
            text=self._text("第 1 步：选择要控制的手臂", "Step 1: Select an arm"),
            padding=8,
        )
        arm_frame.pack(fill="x", pady=4)
        self._arm_var = tk.StringVar(value=policy.selected)
        arm_choices = (
            (self._text("左臂  [1]", "Left arm  [1]"), "left"),
            (self._text("右臂  [2]", "Right arm  [2]"), "right"),
            (self._text("双臂  [3]", "Both arms  [3]"), "both"),
        )
        for column, (text, value) in enumerate(arm_choices):
            arm_frame.columnconfigure(column, weight=1)
            ttk.Radiobutton(
                arm_frame,
                text=text,
                value=value,
                variable=self._arm_var,
                command=lambda selected=value: self._policy.select(selected),
            ).grid(row=0, column=column, padx=12, sticky="w")

        speed_frame = ttk.LabelFrame(
            container,
            text=self._text("第 2 步：调节速度（建议先用 30%）", "Step 2: Set speed (start at 30%)"),
            padding=8,
        )
        speed_frame.pack(fill="x", pady=4)
        self._speed_var = tk.DoubleVar(value=policy.speed)
        ttk.Scale(
            speed_frame,
            from_=0.1,
            to=1.0,
            variable=self._speed_var,
            command=lambda value: self._policy.set_speed(float(value)),
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._speed_label = ttk.Label(speed_frame, width=8)
        self._speed_label.pack(side="right")

        control_frame = ttk.LabelFrame(
            container,
            text=self._text(
                "第 3 步：按住按钮或快捷键运动，松开即停止",
                "Step 3: Hold a button/key to move; release to stop",
            ),
            padding=8,
        )
        control_frame.pack(fill="x", pady=4)
        translation = ttk.LabelFrame(
            control_frame, text=self._text("移动夹爪位置", "Move gripper position"), padding=7
        )
        translation.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        rotation = ttk.LabelFrame(
            control_frame, text=self._text("旋转夹爪姿态", "Rotate gripper pose"), padding=7
        )
        rotation.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        control_frame.columnconfigure(0, weight=3)
        control_frame.columnconfigure(1, weight=2)

        self._axis_row(
            translation,
            0,
            self._text("前 / 后", "Forward / Back"),
            self._text("S  后退  -X", "S  Back  -X"),
            "s",
            self._text("W  前进  +X", "W  Forward  +X"),
            "w",
        )
        self._axis_row(
            translation,
            1,
            self._text("左 / 右", "Left / Right"),
            self._text("D  向右  -Y", "D  Right  -Y"),
            "d",
            self._text("A  向左  +Y", "A  Left  +Y"),
            "a",
        )
        self._axis_row(
            translation,
            2,
            self._text("上 / 下", "Up / Down"),
            self._text("F  下降  -Z", "F  Down  -Z"),
            "f",
            self._text("R  上升  +Z", "R  Up  +Z"),
            "r",
        )
        self._axis_row(rotation, 0, "Roll / X", "K  Roll -", "k", "I  Roll +", "i")
        self._axis_row(rotation, 1, "Pitch / Y", "L  Pitch -", "l", "J  Pitch +", "j")
        self._axis_row(rotation, 2, "Yaw / Z", "O  Yaw -", "o", "U  Yaw +", "u")

        lower = ttk.Frame(container)
        lower.pack(fill="x", pady=(8, 0))
        gripper = ttk.LabelFrame(
            lower, text=self._text("夹爪", "Gripper"), padding=8
        )
        gripper.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._hold_button(
            gripper, self._text("[  按住关闭", "[  Hold to close"), "["
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._hold_button(
            gripper, self._text("]  按住打开", "]  Hold to open"), "]"
        ).pack(side="right", fill="x", expand=True, padx=(3, 0))

        actions = ttk.LabelFrame(
            lower, text=self._text("结束操作", "Finish"), padding=8
        )
        actions.pack(side="right", fill="x", expand=True, padx=(5, 0))
        self._record_button = ttk.Button(
            actions,
            text=self._text("P  暂停/继续录制", "P  Pause/resume recording"),
            command=self._policy.toggle_recording,
            style="Exit.TButton",
        )
        self._record_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            actions,
            text=self._text("Q  正常退出", "Q  Quit"),
            command=self._save_and_quit,
            style="Exit.TButton",
        ).pack(side="right", fill="x", expand=True, padx=(3, 0))

        tk.Button(
            container,
            text=self._text("紧急停止（空格键）", "EMERGENCY STOP  [SPACE]"),
            command=self._emergency_stop,
            bg="#b71c1c",
            fg="white",
            activebackground="#d32f2f",
            activeforeground="white",
            font=(self._cjk_font or "TkDefaultFont", 13, "bold"),
            height=2,
        ).pack(fill="x", pady=(10, 0))

        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.bind_all("<KeyRelease>", self._on_key_release)
        self.root.after(50, self._refresh)
        self.root.after(400, self._take_focus)
        self.root.after(1200, self._take_focus)

    def _text(self, chinese: str, english: str) -> str:
        return chinese if self._cjk_font else english

    def _axis_row(
        self,
        parent: Any,
        row: int,
        axis: str,
        negative_text: str,
        negative_key: str,
        positive_text: str,
        positive_key: str,
    ) -> None:
        from tkinter import ttk

        ttk.Label(parent, text=axis, width=9).grid(row=row, column=0, padx=3, pady=4)
        self._hold_button(parent, negative_text, negative_key).grid(
            row=row, column=1, padx=3, pady=4, sticky="ew"
        )
        self._hold_button(parent, positive_text, positive_key).grid(
            row=row, column=2, padx=3, pady=4, sticky="ew"
        )
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)

    def _hold_button(self, parent: Any, text: str, key: str) -> Any:
        from tkinter import ttk

        button = ttk.Button(parent, text=text, style="Move.TButton")
        button.bind(
            "<ButtonPress-1>", lambda _event: self._policy.set_key_state(key, True)
        )
        button.bind(
            "<ButtonRelease-1>", lambda _event: self._policy.set_key_state(key, False)
        )
        button.bind("<Leave>", lambda _event: self._policy.set_key_state(key, False))
        return button

    def _event_key(self, event: Any) -> str:
        if event.keysym == "space":
            return " "
        return str(event.char).lower()

    def _on_key_press(self, event: Any) -> str:
        key = self._event_key(event)
        if key:
            self._policy.set_key_state(key, True)
        return "break"

    def _on_key_release(self, event: Any) -> str:
        key = self._event_key(event)
        if key:
            self._policy.set_key_state(key, False)
        return "break"

    def _take_focus(self) -> None:
        if self.root.winfo_exists():
            self.root.lift()
            self.root.focus_force()

    def _refresh(self) -> None:
        if self._finished.is_set():
            self.root.destroy()
            return
        status = self._policy.status()
        labels = {
            "left": self._text("左臂", "Left arm"),
            "right": self._text("右臂", "Right arm"),
            "both": self._text("双臂", "Both arms"),
        }
        self._arm_var.set(status["selected"])
        self._speed_label.configure(text=f"{status['speed']:.0%}")
        recording = self._text("录制中", "Recording") if status["recording"] else self._text(
            "录制已暂停", "Recording paused"
        )
        if not status["recording_available"]:
            recording = self._text("本次不录制", "Not recording")
            self._record_button.configure(state="disabled")
        emergency = self._text(" | 已触发急停", " | EMERGENCY STOP") if status[
            "emergency"
        ] else ""
        left_grip, right_grip = status["grippers"]
        self._status_var.set(
            self._text("控制：", "Control: ")
            + f"{labels[status['selected']]}   |   "
            + self._text("速度：", "Speed: ")
            + f"{status['speed']:.0%}   |   {recording}{emergency}"
        )
        self._gripper_status_var.set(
            self._text("夹爪开度：", "Gripper opening: ")
            + f"L {left_grip:.2f}   |   R {right_grip:.2f}   "
            + self._text("（1.0 全开，-1.0 全闭）", "(1.0 open, -1.0 closed)")
        )
        self.root.after(50, self._refresh)

    def _emergency_stop(self) -> None:
        self._policy.request_emergency_stop()

    def _save_and_quit(self) -> None:
        self._policy.request_stop(discard=False)

    def run(self) -> None:
        self.root.mainloop()


def run_teleop_control_panel(
    policy: KeyboardTeleopPolicy, rollout: Callable[[], ResultT]
) -> ResultT:
    """Run simulation in a worker while Tk owns input on the main thread."""
    finished = threading.Event()
    results: list[ResultT] = []
    errors: list[Exception] = []

    def run_rollout() -> None:
        try:
            results.append(rollout())
        except Exception as exc:  # noqa: BLE001 - re-raised on the controlling thread
            errors.append(exc)
        finally:
            finished.set()

    panel = TeleopControlPanel(policy, finished)
    worker = threading.Thread(target=run_rollout, name="a3-teleop-rollout", daemon=True)
    worker.start()
    try:
        panel.run()
    finally:
        if not finished.is_set():
            policy.request_stop()
        worker.join()
    if errors:
        raise errors[0]
    if not results:
        raise RuntimeError("teleoperation ended without a rollout result")
    return results[0]
