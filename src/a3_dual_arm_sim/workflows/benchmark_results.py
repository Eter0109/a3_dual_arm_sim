"""Benchmark suite for single-box batch cookie transfer.

Provides a standardized 20-episode benchmark evaluating policies or expert models
on the 10-cookie transfer task under slight randomization of source/target boxes
and cookies. Scoring is based on the number of cookies settled in the target box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from a3_dual_arm_sim.paths import resource_root

DEFAULT_CONFIG_PATH = resource_root() / "configs" / "cookie_batch.yaml"


@dataclass
class EpisodeScore:
    """Result of a single benchmark episode."""

    episode: int
    seed: int
    score: int  # Number of cookies settled in small target box (0 - 10)
    max_score: int = 10
    success: bool = False
    steps: int = 0
    wall_seconds: float = 0.0
    cookies_in_target: int = 0
    cookies_in_source: int = 70
    target_bin_pos: list[float] = field(default_factory=list)
    source_bin_pos: list[float] = field(default_factory=list)
    phase: str = ""
    failure_reason: str | None = None
    randomization: dict[str, Any] = field(default_factory=dict)
    source_column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics across all episodes."""

    policy_name: str
    total_episodes: int
    total_score: int
    max_possible_score: int
    mean_score: float
    min_score: int
    max_score: int
    success_rate: float
    mean_steps: float
    mean_wall_seconds: float
    score_distribution: dict[int, int]
    episodes: list[EpisodeScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "policy_name": self.policy_name,
                "total_episodes": self.total_episodes,
                "total_score": self.total_score,
                "max_possible_score": self.max_possible_score,
                "mean_score": round(self.mean_score, 2),
                "min_score": self.min_score,
                "max_score": self.max_score,
                "success_rate": round(self.success_rate, 4),
                "mean_steps": round(self.mean_steps, 1),
                "mean_wall_seconds": round(self.mean_wall_seconds, 2),
                "score_distribution": self.score_distribution,
            },
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def summary_table(self) -> str:
        lines = [
            f"=== Benchmark Summary: {self.policy_name} ===",
            f"Total Episodes      : {self.total_episodes}",
            f"Total Score         : {self.total_score} / {self.max_possible_score} ({self.total_score / self.max_possible_score * 100:.1f}%)",
            f"Mean Score / Ep     : {self.mean_score:.2f} / 10.00",
            f"Min / Max Score     : {self.min_score} / {self.max_score}",
            f"Complete Task Success: {self.success_rate * 100:.1f}%",
            f"Mean Steps / Ep     : {self.mean_steps:.1f}",
            f"Mean Wall Time / Ep : {self.mean_wall_seconds:.2f}s",
            f"Score Distribution  : {dict(sorted(self.score_distribution.items(), reverse=True))}",
        ]
        return "\n".join(lines)
