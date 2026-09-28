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
                    "success",
                    "success_hold_count",
                    "required_success_hold_steps",
                    "safety_reason",
                )
            },
        }
        self.stream.write(json.dumps(record, default=json_value) + "\n")
        self.step += 1

    def close(self):
        self.stream.close()
        for writer in self.writers.values():
            writer.release()
