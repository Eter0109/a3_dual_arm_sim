"""Reproducible four-grasp instructions, independent of simulator state."""

from dataclasses import dataclass
from random import Random


def parse_first_grasp(value: int | str) -> int | str:
    if value == "random":
        return value
    if isinstance(value, bool):
        raise TypeError("first_grasp must be an integer 0..9 or 'random'")
    if isinstance(value, str):
        if value not in {str(n) for n in range(10)}:
            raise ValueError("first_grasp must be an integer 0..9 or 'random'")
        return int(value)
    if isinstance(value, int) and 0 <= value <= 9:
        return value
    raise ValueError("first_grasp must be an integer 0..9 or 'random'")


@dataclass(frozen=True)
class Cookie2x10Plan:
    first_grasp: int
    mode: int | str

    @classmethod
    def from_seed(cls, mode: int | str, seed: int) -> "Cookie2x10Plan":
        mode = parse_first_grasp(mode)
        # A separate stream keeps sampling independent of scene randomization.
        count = Random(f"a3-cookie-2x10-first-grasp-v1:{seed}").randrange(10)
        return cls(count if mode == "random" else mode, mode)

    @property
    def batch_sizes(self) -> tuple[int, int, int, int]:
        n = self.first_grasp
        return n, 10 - n, n, 10 - n

    def target_rows(self, batch_index: int) -> tuple[int, ...]:
        if not 0 <= batch_index < 4:
            raise ValueError("batch_index must be 0..3")
        return (
            tuple(range(self.first_grasp))
            if batch_index % 2 == 0
            else tuple(range(self.first_grasp, 10))
        )

    def prompt(self, source_column: int) -> str:
        n, complement, _, _ = self.batch_sizes
        if n == 0:
            steps = (
                "Skip grasp 1 and transfer 10 cookies with grasp 2 into target column 1. "
                "Skip grasp 3 and transfer 10 cookies with grasp 4 into target column 2."
            )
        else:
            steps = (
                f"Grasp {n} cookies first, then {complement} to fill target column 1 with 10. "
                f"Repeat: grasp {n}, then {complement} to fill target column 2 with 10."
            )
        return (
            f"Transfer 20 cookies from source column {source_column} into the 2x10 target box. "
            f"{steps} Release all 20 cookies upright in the target box and leave 60 in the source box."
        )

    def metadata(self) -> dict:
        return {
            "target_layout": "2x10",
            "required_cookies": 20,
            "first_grasp_mode": self.mode,
            "first_grasp_count": self.first_grasp,
            "grasp_counts": list(self.batch_sizes),
            "zero_grasp_behavior": "skip",
            "first_grasp_sampling_version": 1,
        }
