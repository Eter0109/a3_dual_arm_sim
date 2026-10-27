#!/usr/bin/env python3
"""The recurrent-state patch, kept separate from the launcher it is installed by.

The idea in one paragraph
------------------------
Every arm so far fails the same way: v4 places everything it picks up and stops after one
batch of five on a task that needs four (9.43).  pi0.5 is `n_obs_steps = 1`, so nothing in
a sample says how far along the task is.  The v5 probe showed the information exists -- one
seed moved 15 Cookies where every earlier arm stayed at or below 5 -- but it was privileged
state and its attribution was muddy because the state reaches the model by two paths and the
probe moved both.

This patch adds a proper hidden state, with two differences that address both problems:

*   **The encoder sees no images.**  Its input is the last few frames of
    ``[q, q_dot, a_previous]`` -- the robot's own trajectory.  That is an information
    bottleneck by modality rather than by loss: it *cannot* represent "the Cookies look
    messy", only "where the arm has been and what it did".  Task progress is derivable from
    that (visits to the target box are in the window) and the visual state of the pile is
    not.
*   **It is supervised and compressed.**  A progress head predicts how many batches have
    landed (auxiliary loss, discarded at inference), and a KL term pulls the latent toward
    N(0, I) so it cannot spend capacity on anything the flow-matching loss does not need.

No DAgger pass is required, and that is a property of the design rather than an omission: a
history *window* reads the actual past at both training and inference time, so there is no
predicted-past to drift.  What remains is the shift between "windows from expert rollouts"
and "windows from the model's own rollouts", and the training set already contains the
latter -- the grafted episodes' first phase is the 4.05 policy driving (9.40) -- so the mix
supplies on-policy windows for free.  `A3_HISTORY_DROPOUT` adds the cheap remaining piece:
truncating a window at random keeps the encoder from assuming it always has ten clean frames.

Where it sits in the network
---------------------------
An extra prefix token, next to the state token that `_attach_state_projector` already adds:

    prefix = [image tokens] + [language tokens] + [state token] + [history token]
    suffix = [noisy actions] + [timestep]                    (action expert)

The action expert reads the prefix through cross-attention, so no change to the expert is
needed.  Not in the suffix: that input is re-used by every step of the flow-matching
denoising loop, which would recompute the latent ten times per inference and interfere with
the timestep conditioning.

Magnitude is the trap that already cost one arm (see `_attach_state_projector`): a token
whose key is orders of magnitude smaller than its neighbours takes a softmax weight of
essentially zero.  The projection therefore uses the same `std=0.02` init and the same
`sqrt(hidden)` scaling as the state token.
"""

from __future__ import annotations

import math
import os

_SWITCH = "A3_HISTORY_ENCODER"
_STEPS = "A3_HISTORY_STEPS"
_PER_STEP = "A3_HISTORY_PER_STEP"
_WIDTH = "A3_HISTORY_WIDTH"
_PROGRESS_WEIGHT = "A3_HISTORY_PROGRESS_WEIGHT"
_KL_WEIGHT = "A3_HISTORY_KL_WEIGHT"
_DROPOUT = "A3_HISTORY_DROPOUT"
_SAMPLE_ENV = "A3_HISTORY_SAMPLE"


def _read(name: str, default):
    value = os.environ.get(name, "")
    if value.strip() == "":
        return default
    return type(default)(value)


def enabled() -> bool:
    return os.environ.get(_SWITCH, "1").strip().lower() not in ("0", "false", "")


def settings() -> dict:
    return {
        "steps": _read(_STEPS, 10),
        "per_step": _read(_PER_STEP, 48),
        "width": _read(_WIDTH, 128),
        "progress_weight": _read(_PROGRESS_WEIGHT, 0.1),
        "kl_weight": _read(_KL_WEIGHT, 0.05),
        "dropout": _read(_DROPOUT, 0.1),
        "sample": os.environ.get(_SAMPLE_ENV, "0").strip().lower() in ("1", "true"),
    }


def summary() -> str:
    s = settings()
    return (
        f"history encoder    : GRU({s['steps']}x{s['per_step']} -> {s['width']}), one prefix "
        f"token, progress={s['progress_weight']}, kl={s['kl_weight']}, "
        f"window-dropout={s['dropout']} (no images: proprioception + previous action only)"
    )


def build(config, hidden: int, *, steps: int | None = None):
    """The module itself, importable on its own so the launcher stays a wiring file."""

    import torch
    import torch.nn as nn

    s = settings()
    steps = s["steps"] if steps is None else steps

    class GRULite(nn.Module):
        """A GRU written out in `Linear` layers instead of `nn.GRU`.

        Not a preference.  `nn.GRU` allocates its parameters so that `weight_ih_l0` is a
        *view* of a larger storage, and safetensors refuses to save a state dict containing
        a tensor that does not cover its storage ("None is covering the entire storage").
        The run then trains for 60 steps, hits the first checkpoint, and dies -- which is how
        this was found.  Three `Linear`s and an explicit recurrence have no such shared
        storage, and for a ten-step window the arithmetic is identical and the cost is
        irrelevant (676k MACs per sample).
        """

        def __init__(self) -> None:
            super().__init__()
            self.per_step = s["per_step"]
            self.width = s["width"]
            self.to_gates = nn.Linear(self.per_step, 3 * self.width)
            self.from_hidden = nn.Linear(self.width, 3 * self.width)
            for module in (self.to_gates, self.from_hidden):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, steps, per_step); returns the final hidden state (B, width).
            width = self.width
            h = x.new_zeros(x.shape[0], width)
            for step in range(x.shape[1]):
                gates = self.to_gates(x[:, step]) + self.from_hidden(h)
                reset, update, candidate = gates.split(width, dim=-1)
                reset = torch.sigmoid(reset)
                update = torch.sigmoid(update)
                candidate = torch.tanh(candidate)
                h = (1.0 - update) * h + update * candidate
            return h

    class HistoryEncoder(nn.Module):
        """Window of ``[q, q_dot, a_previous]`` -> one latent, plus a progress prediction.

        A recurrence over ten frames rather than an MLP over the flattened window: the
        window's informative content is a *path* (the hand went to the box and came back),
        and a recurrent layer carries the order cheaply while an MLP would have to learn that
        the rows are sequential.  Width 128 is small on purpose -- the whole point is a
        narrow channel.
        """

        def __init__(self) -> None:
            super().__init__()
            self.steps = steps
            self.per_step = s["per_step"]
            self.width = s["width"]
            self.gru = GRULite()
            self.to_mu = nn.Linear(self.width, self.width)
            self.to_logvar = nn.Linear(self.width, self.width)
            self.out = nn.Linear(self.width, hidden)
            # Two outputs: batches delivered and Cookies placed, both on the state's
            # [-1, 1] scale.
            self.progress = nn.Linear(self.width, 2)
            for module in (self.to_mu, self.to_logvar, self.out, self.progress):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        def forward(self, window: "torch.Tensor", dropout: float = 0.0):
            expected = self.steps * self.per_step
            if window.shape[-1] != expected:
                raise ValueError(
                    f"history window is {window.shape[-1]} wide; the encoder expects "
                    f"{expected} ({self.steps} steps x {self.per_step}).  A dataset built with "
                    f"a different --steps needs a matching {_STEPS}."
                )
            x = window.view(-1, self.steps, self.per_step)
            if dropout > 0.0 and self.training:
                # Truncate the window at a random point by repeating an early frame over the
                # older part: the encoder then still sees "how far the hand has come" but not
                # a fixed ten-frame span, which is the cheap half of what an on-policy
                # distribution shift would otherwise break.
                keep = int(torch.randint(1, self.steps, (1,)).item())
                cut = self.steps - keep
                if cut > 0:
                    head = x[:, :1].expand(-1, cut, -1)
                    x = torch.cat([head, x[:, cut:]], dim=1)
            h = self.gru(x)
            mu = self.to_mu(h)
            logvar = torch.clamp(self.to_logvar(h), -10.0, 4.0)
            return mu, logvar

        def latent(self, mu, logvar, *, sample: bool):
            if not sample:
                return mu
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

    return HistoryEncoder()


def project(latent, hidden: int, *, std: float = 0.02):
    """`Linear(width -> hidden)` with the scaling the other prefix tokens get.

    Separate from the encoder so the projection's init can be tested on its own: the
    magnitude of a new prefix token is what decides whether attention can see it at all.
    """

    import torch.nn as nn

    layer = nn.Linear(latent.shape[-1], hidden)
    nn.init.normal_(layer.weight, std=std)
    nn.init.zeros_(layer.bias)
    return layer


def scale(latent):
    """`sqrt(hidden)` rescale, applied where the prefix is assembled."""

    return latent * math.sqrt(latent.shape[-1])
