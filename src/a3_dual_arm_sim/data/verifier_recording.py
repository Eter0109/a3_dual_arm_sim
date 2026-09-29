"""Write RGB/subgoal verifier windows separately from privileged evaluation."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from a3_dual_arm_sim.core.skills import SkillRequest

if TYPE_CHECKING:
    from a3_dual_arm_sim.evaluation.skill_truth import Diagnosis, SkillTruth


class TemporalVerificationRecorder:
    """Write image-window/subgoal/Yes-No samples, including incomplete attempts.

    This dataset is separate from successful action demonstrations. Only the
    images and subgoal are model inputs; labels and privileged audits never are.
    An explicit diagnosis can override the conservative stationary oracle label
    when supervised failure experiments establish a true stuck condition.
    """

    CAMERAS = ("front", "left_wrist", "right_wrist")

    def __init__(self, root: str | Path, *, max_frames: int = 8, frame_stride: int = 4) -> None:
        if max_frames < 2 or frame_stride < 1:
            raise ValueError("temporal samples require at least two frames and positive stride")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self._frames: deque[tuple[int, dict[str, np.ndarray]]] = deque(maxlen=max_frames)
        self._samples_path = self.root / "samples.jsonl"
        self._audit_path = self.root / "truth_audit.jsonl"
        if self._samples_path.exists() or self._audit_path.exists():
            raise FileExistsError("use a new verifier dataset root; implicit append is unsafe")
        self._sample_index = 0
        self.request: SkillRequest | None = None

    def start(self, request: SkillRequest, *, seed: int, attempt: int = 0) -> None:
        self.request = request
        self.seed = seed
        self.attempt = attempt
        self._frames.clear()
        self._last_observed_step: int | None = None

    def observe(self, observation: dict[str, Any], *, step: int) -> None:
        if self.request is None:
            raise RuntimeError("start a skill before observing its image window")
        if self._last_observed_step is not None and step <= self._last_observed_step:
            raise ValueError("temporal frame steps must increase")
        if self._last_observed_step is not None and step - self._last_observed_step < self.frame_stride:
            return
        frames: dict[str, np.ndarray] = {}
        for camera in self.CAMERAS:
            frame = np.asarray(observation[f"observation.images.{camera}"])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("verifier images must be HWC uint8 RGB")
            frames[camera] = frame.copy()
        self._frames.append((step, frames))
        self._last_observed_step = step

    def record(
        self, truth: SkillTruth, *, diagnosis: Diagnosis | None = None,
        prediction: str | None = None,
    ) -> int | None:
        if self.request is None:
            raise RuntimeError("start a skill before recording verifier labels")
        if len(self._frames) < 2:
            return None
        if diagnosis not in (None, "StillTrying", "Stuck"):
            raise ValueError("diagnosis must be StillTrying or Stuck")
        if truth.complete and diagnosis is not None:
            raise ValueError("successful verification has no stuck/trying diagnosis")
        from PIL import Image

        sample_index = self._sample_index
        window: list[dict[str, Any]] = []
        for frame_index, (step, cameras) in enumerate(self._frames):
            paths: dict[str, str] = {}
            for camera, frame in cameras.items():
                relative = Path("images") / f"{sample_index:08d}" / f"{frame_index:02d}_{camera}.jpg"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(path, quality=92)
                paths[camera] = relative.as_posix()
            window.append({"step": step, "images": paths})
        sample = {
            "sample_index": sample_index,
            "parent_seed": self.seed,
            "skill": self.request.skill.value,
            "batch_index": self.request.batch_index,
            "attempt": self.attempt,
            "request": asdict(self.request),
            "input": {"subgoal": self.request.instruction, "frames": window},
            "label": {
                "completion": "Yes" if truth.complete else "No",
                "diagnosis": None if truth.complete else (diagnosis or truth.diagnosis or "StillTrying"),
            },
        }
        with self._samples_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "sample_index": sample_index, "prediction": prediction, **asdict(truth)
            }) + "\n")
        self._sample_index += 1
        return sample_index

    def close(self) -> None:
        self._frames.clear()
        self.request = None
