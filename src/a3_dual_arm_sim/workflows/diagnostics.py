"""Optional, privileged evaluation traces; never added to policy observations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class EpisodeDiagnostic:
    def __init__(self, root: Path, fps: int = 20):
        root.mkdir(parents=True, exist_ok=False)
        self.stream = (root / "trace.jsonl").open("w")
        self.root, self.fps = root, fps
        self.writers = {}
        self.step = 0
        self.first_count_fill_step = None
        self.first_exact_fill_step = None
        self.max_hold_count = 0
        self.final_conditions = None

    def add_frame(self, observation, action, info, *, raw_action=None, phase=""):
        for key, frame in observation.items():
            if not key.startswith("observation.images."):
                continue
            if key not in self.writers:
                writer = cv2.VideoWriter(
                    str(self.root / f"{key.rsplit('.', 1)[-1]}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video for {key}")
                self.writers[key] = writer
            self.writers[key].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        record = {
            "step": self.step,
            "observation_before": {k: v for k, v in observation.items() if "images" not in k},
            "raw_action": raw_action,
            "processed_action": action,
            "applied_action": info["applied_action"],
            "expert_phase": phase,
            "after": {
                k: info.get(k)
                for k in (
                    "cookie_positions",
                    "cookies_in_target",
                    "cookies_in_source",
                    "cookies_in_target_mask",
                    "cookies_in_source_mask",
                    "target_slot_occupancy",
                    "target_touches_all_walls",
                    "exact_2x5_fill",
                    "count_fill",
                    "success_criterion",
                    "success",
                    "success_hold_count",
                    "required_success_hold_steps",
                    "safety_reason",
                )
            },
        }
        after = record["after"]
        unmet = []
        if after["count_fill"] is False:
            unmet.append("count_fill")
        if after["success_criterion"] == "exact_slots":
            slots = after["target_slot_occupancy"]
            if slots is not None and any(index < 0 for index in slots):
                unmet.append("slot_occupancy")
            if after["target_touches_all_walls"] is False:
                unmet.append("boundary_coverage")
        hold = after["success_hold_count"]
        required = after["required_success_hold_steps"]
        if hold is not None and required is not None and hold < required:
            unmet.append("hold_steps")
        after["unmet_success_conditions"] = unmet
        if after["count_fill"] and self.first_count_fill_step is None:
            self.first_count_fill_step = self.step
        if after["exact_2x5_fill"] and self.first_exact_fill_step is None:
            self.first_exact_fill_step = self.step
        self.max_hold_count = max(self.max_hold_count, hold or 0)
        self.final_conditions = after
        self.stream.write(json.dumps(record, default=json_value) + "\n")
        self.step += 1

    def close(self):
        self.stream.close()
        for writer in self.writers.values():
            writer.release()
        summary = {
            "steps": self.step,
            "step_index_convention": "zero-based",
            "first_count_fill_step": self.first_count_fill_step,
            "first_exact_fill_step": self.first_exact_fill_step,
            "max_success_hold_count": self.max_hold_count,
            "final_conditions": self.final_conditions,
        }
        (self.root / "success_diagnostics.json").write_text(
            json.dumps(summary, default=json_value, indent=2) + "\n"
        )
