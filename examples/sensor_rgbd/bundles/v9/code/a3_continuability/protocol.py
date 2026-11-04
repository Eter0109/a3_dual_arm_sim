"""Research boundaries are enforced by code, not just a report."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Protocol:
    version: str = "continuability-v1"
    phase: str = "zero_shot"
    batch_size: int = 5
    action_dim: int = 8
    prediction_steps: int = 50
    execution_steps: int = 8
    render_protocol: str = "native"
    anchors: int = 64
    candidates: int = 4
    probes: int = 2
    cycle_limit: int = 750
    stable_steps: int = 5
    train_seed_start: int = 26000
    validation_seed_start: int = 27000
    test_seed_start: int = 28000
    seed_block: int = 500
    original_fraction: float = 0.5
    loss_weight: float = 2.0
    continuation_weight: float = 1.0
    time_weight: float = 0.05
    train_rounds: int = 3
    steps_per_round: int = 5000
    train_micro_batch: int = 2
    train_seeds: tuple = (1001, 1002, 1003)
    evaluation_episodes: int = 50
    unfreeze_llm_layers: int = 18
    unfreeze_vision: bool = True
    adaptation_budgets: tuple = (25000, 100000)
    interaction_budget: int | None = None
    max_steps: dict = field(default_factory=lambda: {"2x5": 1500, "2x10": 3000})

    def __post_init__(self):
        if self.render_protocol not in {"native", "single_sample"}:
            raise ValueError("Unknown research rendering protocol")
        if self.phase not in {"zero_shot", "adaptation"}:
            raise ValueError("Unknown research phase")
        if self.action_dim != 8 or self.batch_size != 5:
            raise ValueError("v1 requires left8 and five-cookie batches")
        if not (0 < self.execution_steps <= self.prediction_steps):
            raise ValueError("Invalid action horizons")
        if self.train_micro_batch < 1 or 6 % self.train_micro_batch:
            raise ValueError("Micro batch must divide the unchanged effective batch of six")
        if min(self.anchors, self.candidates, self.probes, self.cycle_limit, self.stable_steps) < 1:
            raise ValueError("Counts must be positive")
        if self.original_fraction != 0.5:
            raise ValueError("Matched v1 training requires exactly 50% original demonstrations")
        if self.phase == "adaptation" and self.interaction_budget not in self.adaptation_budgets:
            raise ValueError("Adaptation must declare a 25k or 100k interaction budget")
        starts = sorted((self.train_seed_start, self.validation_seed_start, self.test_seed_start))
        if any(b-a < self.seed_block for a, b in zip(starts, starts[1:])):
            raise ValueError("Train, validation and test seed blocks overlap")

    def check_data(self, task, seed, *, split="train"):
        start = {"train": self.train_seed_start, "validation": self.validation_seed_start,
                 "test": self.test_seed_start}[split]
        if not start <= seed < start + self.seed_block:
            raise ValueError(f"Seed {seed} is outside {split} block")
        if split == "train" and self.phase == "zero_shot" and task != "2x5":
            raise ValueError("2x10 interaction data is forbidden in zero-shot training")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def prompt(task):
    count = {"2x5": 10, "2x10": 20}[task]
    return (f"Transfer {count} cookies from source column 1 into the target box, five at a time. "
            "Keep the cookies upright and stable, preserve cookies already placed, and fill the box.")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class InteractionBudget:
    """Persist a charge *before* every physical step, including branches/probes."""
    def __init__(self, root, limit=None):
        self.path = Path(root)/"interaction_ledger.json"
        self.limit = limit
        if self.path.exists():
            ledger = json.loads(self.path.read_text())
            if ledger["limit"] != limit:
                raise ValueError("Interaction budget changed while resuming")
            self.used = ledger["used"]
        else:
            self.used = 0

    def consume(self):
        if self.limit is not None and self.used >= self.limit:
            raise RuntimeError("Declared physical interaction budget exhausted")
        self.used += 1
        atomic_json(self.path, {"used": self.used, "limit": self.limit})
