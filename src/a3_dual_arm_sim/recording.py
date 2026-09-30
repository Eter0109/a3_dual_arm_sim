from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
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
    jsonable,
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
    def finish_episode(
        self,
        success: bool | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None: ...
    def discard_episode(self) -> None: ...
    def close(self) -> None: ...


class MemoryRecorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.episodes: list[list[dict[str, Any]]] = []
        self.metrics: list[Mapping[str, Any]] = []
        self.context: EpisodeContext | None = None

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self.context = context
        self.frames = []

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        self.frames.append({**observation, ACTION: np.asarray(action).copy()})

    def finish_episode(
        self,
        success: bool | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        del success
        self.episodes.append(self.frames)
        self.metrics.append(dict(metrics) if metrics else {})
        self.frames = []

    def discard_episode(self) -> None:
        self.frames = []

    def close(self) -> None:
        return None


def lerobot_features(
    height: int, width: int, *, use_videos: bool = False
) -> dict[str, Any]:
    """Describe one canonical A3 frame to LeRobot.

    LeRobot selects video storage from the feature ``dtype`` rather than from the
    ``use_videos`` writer flag, so the camera keys must be declared as ``video``
    for episodes to be encoded into mp4 files instead of PNG rows.
    """
    image = {
        "dtype": "video" if use_videos else "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        FRONT_IMAGE: dict(image),
        LEFT_WRIST_IMAGE: dict(image),
        RIGHT_WRIST_IMAGE: dict(image),
        STATE: {"dtype": "float32", "shape": (16,), "names": [f"state_{i}" for i in range(16)]},
        VELOCITY: {"dtype": "float32", "shape": (16,), "names": [f"velocity_{i}" for i in range(16)]},
        EEF_POSE: {"dtype": "float32", "shape": (14,), "names": [f"eef_pose_{i}" for i in range(14)]},
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
        rollover_mb: float | None = None,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("Install recording support with: pip install -e '.[dataset]'") from exc
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty dataset: {self.root}")
        if self.root.exists():
            self.root.rmdir()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=self.root,
            robot_type="A3_dual_arm",
            features=lerobot_features(image_height, image_width, use_videos=use_videos),
            use_videos=use_videos,
            image_writer_threads=3,
        )
        if rollover_mb is not None:
            self._set_rollover(rollover_mb)
        self._context: EpisodeContext | None = None
        self._controller_type = "unknown"
        self._frames = 0
        self._episode_index = 0
        self._metadata_path = self.root / "a3_episode_metadata.jsonl"

    def _set_rollover(self, rollover_mb: float) -> None:
        """Cap how much one parquet file may hold before the writer seals it.

        A dataset written in one long run is only *readable* when the writer is
        closed: ``ParquetWriter`` appends row groups and writes the footer -- which
        holds the schema and the row-group offsets -- on ``close()`` alone.  LeRobot
        closes it when a file reaches ``data_files_size_in_mb``, and the default is
        100 MB, so a long collection produces files that stay sealed for the whole
        run.  Anything that wants to read the data part-way through -- training on
        episodes collected so far, or simply checking that the writer is producing
        sound data -- gets "Parquet magic bytes not found in footer" instead, which
        reads like corruption and is not.

        Lowering the threshold makes the writer seal each file as it fills, so
        completed files are readable while the run continues.  It is a threshold on
        one file, not on the dataset: every episode still lands in exactly one
        sealed file, and readers iterate the files in order.

        Both limits are set because they govern different writers -- the parquet
        file and the per-camera video files -- and both are stored in ``info.json``.
        The in-memory copy is what the writer consults on every episode, so it is
        patched first; ``info.json`` is rewritten so the dataset keeps declaring the
        setting it was actually written with.
        """

        meta = self.dataset.meta
        meta.info["data_files_size_in_mb"] = rollover_mb
        meta.info["video_files_size_in_mb"] = rollover_mb
        try:
            from lerobot.datasets.utils import write_info
        except ImportError as exc:  # pragma: no cover - a LeRobot without it
            warnings.warn(
                f"cannot persist the rollover setting to info.json ({exc}); the "
                "in-memory value still governs this writer, so the run is unaffected "
                "but the dataset will not record the threshold it was written with",
                stacklevel=2,
            )
        else:
            write_info(meta.info, meta.root)
        self.rollover_mb = rollover_mb

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

    def finish_episode(
        self,
        success: bool | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._context is None:
            raise RuntimeError("no episode is active")
        self.dataset.save_episode(parallel_encoding=True)
        metadata: dict[str, Any] = {
            "episode_index": self._episode_index,
            "seed": self._context.seed,
            "task": self._context.task,
            "controller_type": self._controller_type,
            "source_action_mode": self._context.action_mode,
            "stored_action_mode": "joint_position",
            "frames": self._frames,
            "success": success,
        }
        # Why an episode ended is part of the episode.  A scene that runs several
        # attempts per process writes both accepted and rejected episodes, and a
        # reader who has to open the collection summary to learn which is which
        # cannot filter a dataset by itself; nor can a rejected episode explain
        # itself, which is the whole point of keeping it.
        if metrics:
            metadata["metrics"] = jsonable(dict(metrics))
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
