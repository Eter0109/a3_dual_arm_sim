"""C51 value distribution and the head that predicts it.

Pure logic, no dataset and no model loading, so the arithmetic can be tested in milliseconds.

Why C51 rather than regression
------------------------------
The value is the expected final Cookie count, and that expectation is genuinely uncertain from
an early frame: the same frame at step 200 leads to 20 Cookies in some episodes and 6 in
others.  A regressor trained with MSE would output the conditional mean and be scored on it,
which is fine for ranking but throws away the shape.  A distribution lets the same head carry
"60 % finish, 40 % stall", and the expectation is then available without pretending the
variance does not exist.  RLinf and the community RECAP port both use 201 atoms; this follows
them so the numbers are comparable.

The atom range is `[-1, 0]`, not `[0, required]`
-----------------------------------------------
Both upstream implementations use `[-1, 0]` because their target is a normalised return.  Keeping
the same range here means the `two_hot` and loss code is interchangeable with theirs, and the
mapping `count/required - 1` is a single line.  The plan (§2.3) wrote `x/10 - 1`, which maps
`[0, 20]` onto `[-1, 1]` and would clip half the distribution; `x/required - 1` is the correct
form and is what this file implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_ATOMS = 201
V_MIN = -1.0
V_MAX = 0.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)


def atoms(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 201 support points, evenly spaced over `[V_MIN, V_MAX]`."""

    return torch.linspace(V_MIN, V_MAX, NUM_ATOMS, device=device, dtype=dtype)


def count_to_value(count: torch.Tensor, required: float) -> torch.Tensor:
    """Cookie count -> the `[-1, 0]` value scale.

    `0` Cookies maps to `-1` and `required` maps to `0`, so a perfectly executed episode has
    value exactly `0` -- the same convention both upstream implementations use for "task
    complete".
    """

    return count / required - 1.0


def value_to_count(value: torch.Tensor, required: float) -> torch.Tensor:
    """Inverse of `count_to_value`, for reporting in Cookie units."""

    return (value + 1.0) * required


def two_hot(target: torch.Tensor) -> torch.Tensor:
    """Project targets onto the atom grid with linear interpolation between neighbours.

    Two-hot rather than one-hot because a target almost never lands exactly on an atom: with
    201 atoms over a unit interval the spacing is 0.005, and `count/20 - 1` for integer counts
    hits every 0.05 -- which happens to align, but any future continuous target (a soft count,
    a discounted return) would not.  Two-hot is also what makes the loss a proper interpolation
    of neighbouring atoms instead of a hard nearest-bin assignment.

    Edge case handled explicitly: a target that lands exactly on a bin gives `b_left == b_right`,
    and the naive `weight_left = 1 - weight_right` then writes `1.0` to the same index twice,
    summing to 2 instead of 1.  The `same_bin` branch below assigns a clean one-hot instead.
    """

    target = torch.clamp(target.float(), V_MIN, V_MAX)
    b = (target - V_MIN) / DELTA_Z
    lower = torch.floor(b).long().clamp(0, NUM_ATOMS - 1)
    upper = torch.ceil(b).long().clamp(0, NUM_ATOMS - 1)

    weight_upper = b - lower.float()
    weight_lower = 1.0 - weight_upper

    same = lower == upper
    weight_upper = torch.where(same, torch.zeros_like(weight_upper), weight_upper)
    weight_lower = torch.where(same, torch.ones_like(weight_lower), weight_lower)

    out = torch.zeros(target.shape[0], NUM_ATOMS, device=target.device, dtype=torch.float32)
    rows = torch.arange(target.shape[0], device=target.device)
    out[rows, lower] += weight_lower
    out[rows, upper] += weight_upper
    return out


def distributional_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the two-hot target, per sample.

    Returned unreduced so a caller can weight samples or split by episode; the trainers in this
    repo have been bitten by a `reduction` argument that silently dropped an auxiliary term
    (§9.81), so this one has no such argument.
    """

    target_dist = two_hot(target)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)


@torch.no_grad()
def expected_value(logits: torch.Tensor) -> torch.Tensor:
    """`sum p_i * z_i` -- the value the head reports, in the `[-1, 0]` scale."""

    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * atoms(logits.device, probs.dtype)).sum(dim=-1)


@dataclass(frozen=True)
class HeadConfig:
    """Widths of the value head.

    Each modality gets its own projection before the shared trunk.  Concatenating them raw
    would let the image block (3 cameras x 1152) dominate a 32-wide state vector by three
    orders of magnitude in the first linear layer, and the trunk would spend its capacity
    re-learning a per-modality scale.
    """

    image_dim: int = 1152
    num_cameras: int = 3
    language_dim: int = 1152
    state_dim: int = 16
    history_dim: int = 128
    progress_dim: int = 2

    image_out: int = 256
    language_out: int = 64
    state_out: int = 32
    history_out: int = 64
    progress_out: int = 8

    trunk: int = 512
    depth: int = 2

    @property
    def fused(self) -> int:
        # `image_out * num_cameras`, not `image_out`: the image branch runs per camera and the
        # per-camera results are concatenated, so a three-camera input contributes three times
        # the projection width.  Getting this wrong is a LayerNorm shape error at the first
        # forward pass -- which is how it was found.
        return (
            self.image_out * self.num_cameras
            + self.language_out
            + self.state_out
            + self.history_out
            + self.progress_out
        )


class ValueHead(nn.Module):
    """Per-modality encoders, a shared trunk, and 201 logits.

    The image input is the **mean of the SigLIP patch embeddings per camera**, not a learned
    attention pool.  That is a deliberate compromise: the expensive part (SigLIP over 3 x 256
    patch views) is cached once, and a learned query would have to be trained inside the same
    pass that produced the cache.  Mean-pooling keeps the cached feature a pure function of the
    frozen encoder, so the head can be re-trained and re-shaped without re-extracting.  If the
    Spearman gate fails, an attention pool over cached *patch* features is the next thing to
    try -- that is a different cache shape, not a different idea.
    """

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        c = self.config

        self.image = nn.Sequential(
            nn.LayerNorm(c.image_dim), nn.Linear(c.image_dim, c.image_out), nn.SiLU()
        )
        self.camera_count = c.num_cameras
        self.language = nn.Sequential(
            nn.LayerNorm(c.language_dim), nn.Linear(c.language_dim, c.language_out), nn.SiLU()
        )
        self.state = nn.Sequential(nn.Linear(c.state_dim, c.state_out), nn.SiLU())
        self.history = nn.Sequential(
            nn.LayerNorm(c.history_dim), nn.Linear(c.history_dim, c.history_out), nn.SiLU()
        )
        self.progress = nn.Sequential(nn.Linear(c.progress_dim, c.progress_out), nn.SiLU())

        layers: list[nn.Module] = [nn.LayerNorm(c.fused), nn.Linear(c.fused, c.trunk), nn.SiLU()]
        for _ in range(c.depth - 1):
            layers += [nn.Linear(c.trunk, c.trunk), nn.SiLU()]
        layers += [nn.Linear(c.trunk, NUM_ATOMS)]
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        image_mean: torch.Tensor,
        language_mean: torch.Tensor,
        state: torch.Tensor,
        history: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        """`image_mean` is `[B, num_cameras, image_dim]`; the rest are `[B, dim]`."""

        batch = image_mean.shape[0]
        image = self.image(image_mean.reshape(batch * self.camera_count, -1))
        image = image.reshape(batch, self.camera_count * self.config.image_out)
        fused = torch.cat(
            [
                image,
                self.language(language_mean),
                self.state(state),
                self.history(history),
                self.progress(progress),
            ],
            dim=-1,
        )
        return self.trunk(fused)
