from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    EpisodeContext,
)
from .paths import project_root

_RUNTIME = project_root() / ".runtime"
os.environ.setdefault("HF_HOME", str(_RUNTIME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_RUNTIME / "datasets"))

ACTION = "action"
TASK = "task"


class Recorder(Protocol):
    def start_episode(self, context: EpisodeContext, controller_type: str) -> None: ...
    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None: ...
    def finish_episode(self, success: bool | None = None) -> None: ...
    def discard_episode(self) -> None: ...
    def close(self) -> None: ...


class MemoryRecorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.episodes: list[list[dict[str, Any]]] = []
        self.context: EpisodeContext | None = None

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self.context = context
        self.frames = []

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        self.frames.append({**observation, ACTION: np.asarray(action).copy()})

    def finish_episode(self, success: bool | None = None) -> None:
        self.episodes.append(self.frames)
        self.frames = []

    def discard_episode(self) -> None:
        self.frames = []

    def close(self) -> None:
        return None


def lerobot_features(height: int, width: int) -> dict[str, Any]:
    image = {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        FRONT_IMAGE: image,
        LEFT_WRIST_IMAGE: image.copy(),
        RIGHT_WRIST_IMAGE: image.copy(),
        STATE: {"dtype": "float32", "shape": (16,), "names": [f"state_{i}" for i in range(16)]},
        VELOCITY: {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"velocity_{i}" for i in range(16)],
        },
        EEF_POSE: {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"eef_pose_{i}" for i in range(14)],
        },
        FORCE: {"dtype": "float32", "shape": (18,), "names": [f"force_{i}" for i in range(18)]},
        ACTION: {"dtype": "float32", "shape": (16,), "names": [f"action_{i}" for i in range(16)]},
    }


class LeRobotV3Recorder:
    """Write canonical A3 episodes using the LeRobot v3 dataset writer."""

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str,
        fps: int,
        image_height: int,
        image_width: int,
        use_videos: bool = False,
        resume: bool = False,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Install recording support with: pip install -e '.[dataset]'"
            ) from exc
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(f"refusing to overwrite non-empty dataset: {self.root}")
        if self.root.exists() and not resume:
            self.root.rmdir()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.root / "a3_episode_metadata.jsonl"
        resumed_episodes: list[dict[str, Any]] = []
        if resume:
            info_path = self.root / "meta" / "info.json"
            if not self._metadata_path.is_file() or not info_path.is_file():
                raise FileNotFoundError("resume requires finalized LeRobot and A3 metadata")
            resumed_episodes = [
                json.loads(line)
                for line in self._metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if len(resumed_episodes) != info.get("total_episodes"):
                raise RuntimeError("Incomplete save: LeRobot/A3 episode counts differ")
        self.dataset = (
            LeRobotDataset.resume(
                repo_id=repo_id,
                root=self.root,
                image_writer_threads=3,
            )
            if resume
            else LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=self.root,
                robot_type="A3_dual_arm",
                features=lerobot_features(image_height, image_width),
                use_videos=use_videos,
                image_writer_threads=3,
            )
        )
        self._context: EpisodeContext | None = None
        self._controller_type = "unknown"
        self._frames = 0
        self._episode_index = 0
        if resume:
            if len(resumed_episodes) != self.dataset.meta.total_episodes:
                raise RuntimeError("Incomplete save: LeRobot/A3 episode counts differ")
            self._episode_index = len(resumed_episodes)

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self._context = context
        self._controller_type = controller_type
        self._frames = 0

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        if self._context is None:
            raise RuntimeError("start_episode must be called before add_frame")
        frame = {
            FRONT_IMAGE: observation[FRONT_IMAGE],
            LEFT_WRIST_IMAGE: observation[LEFT_WRIST_IMAGE],
            RIGHT_WRIST_IMAGE: observation[RIGHT_WRIST_IMAGE],
            STATE: np.asarray(observation[STATE], dtype=np.float32),
            VELOCITY: np.asarray(observation[VELOCITY], dtype=np.float32),
            EEF_POSE: np.asarray(observation[EEF_POSE], dtype=np.float32),
            FORCE: np.asarray(observation[FORCE], dtype=np.float32),
            ACTION: np.asarray(action, dtype=np.float32),
            TASK: self._context.task,
        }
        self.dataset.add_frame(frame)
        self._frames += 1

    def finish_episode(self, success: bool | None = None) -> None:
        if self._context is None:
            raise RuntimeError("no episode is active")
        self.dataset.save_episode(parallel_encoding=True)
        metadata = {
            "episode_index": self._episode_index,
            "seed": self._context.seed,
            "task": self._context.task,
            "controller_type": self._controller_type,
            "source_action_mode": self._context.action_mode,
            "stored_action_mode": "joint_position",
            "frames": self._frames,
            "success": success,
        }
        with self._metadata_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self._episode_index += 1
        self._context = None
        self._frames = 0

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self._context = None
        self._frames = 0

    def close(self) -> None:
        if self._context is not None:
            self.discard_episode()
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()
