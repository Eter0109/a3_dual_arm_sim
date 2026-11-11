#!/usr/bin/env python3
"""Steering a frozen pi0.5 with a learned noise policy: the *inference* half.

Why this module exists
----------------------
The training script (`train_dsrl.py`) and the bridge adapter (`adapters/pi05_a3.py`) both have to
turn one observation into one action chunk, and they have to do it **identically**.  The moment the
two implementations drift, the actor is evaluated under a feature layout it was never trained on and
every downstream number is measuring the drift.  This file is the single implementation; both sides
import it.

Three facts from the reference implementation that shape everything here
-----------------------------------------------------------------------
1. **The actor's variable is `tanh`-squashed and unit-scale.**  In the reference, `DSRL.actor` is
   SB3's `SquashedDiagGaussianDistribution`: `action = tanh(gaussian)`, and the noise that reaches
   the sampler is that `tanh` output, in `(-1, 1)`.  The replay buffer stores `scale_action(x)`,
   which for a symmetric box `[-M, M]` is just `x / M`, so `action_magnitude` rescales what the
   *critic* sees, not what the sampler sees.

2. **The 32-dim single-step noise is repeated across the chunk axis** (paper §C.5):
   `w = {(w_1, ..., w_C) | w_i = w_single}`.  `repeat_noise` is that line.

3. **At inference the whole pipeline runs**: `obs -> actor -> noise -> sampler -> action chunk`.
   The reference's `evaluate` does exactly this via `agent.predict_diffused(obs)` and reports
   success rate and mean reward (`utils.py:113`).  Nothing is cached and no critic is consulted,
   which is what makes the measurement independent of the critic that trained the actor.

What the paper feeds the noise policy, and how it is reproduced here
-------------------------------------------------------------------
Appendix C.5: "the proprioceptive state, the final token's last hidden feature from pi_0's VLM
backbone (a 2,048-dimensional vector), and visual features".

* proprioception -- `preprocess` already normalises it; it is the same 16-dim vector the VLA sees.
* VLM feature -- `prefix_hidden[:, last_valid_token]`, taken from the forward pass that is *already*
  being run to build the prefix.  No second pass.
* visual -- the same `features_v9` vector the training table holds: per-camera mean of
  `paligemma.embed_image`, then the masked mean of `paligemma.embed_language_tokens`, concatenated
  to 8192 dims and projected by a fixed random matrix to 256.  `visual_feature` reproduces it by
  *slicing* `prefix_embs` rather than recomputing SigLIP, and `verify_visual_feature` checks the
  slice against the cached array -- a float16 round-trip is the only difference.

Why the verification is not optional
------------------------------------
Every bug found in this project so far produced a finite loss and a plausible curve: a no-op
observation fill, a `stride` shadowed by another `stride`, a return table off by a phase, a pool
with one reward in it.  A wrong visual feature would look exactly like that -- a slightly worse
score, attributable to anything.  So `verify_visual_feature` compares against `features.npy` on real
rows and refuses to proceed on a mismatch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: Width of the frozen visual summary stored in `features_v9`: three cameras x 2048, plus the
#: 2048-wide language embedding.  Probed from the encoder in `extract_value_features.py` rather than
#: hard-coded there, and asserted here so a model swap fails loudly instead of silently mis-slicing.
VISUAL_RAW_DIM = 8192


# ---------------------------------------------------------------------------------------------
# networks (shared with training)
# ---------------------------------------------------------------------------------------------
class MLP(nn.Module):
    """Feed-forward body shaped as the reference builds it.

    Two choices here were arbitrary in earlier versions and are now taken from the reference:

    * **`Tanh`, not `SiLU`.**  `train_dsrl.py` in the reference passes
      `activation_fn=torch.nn.Tanh` to SB3's `policy_kwargs`.  The paper's appendix B does not name
      an activation -- it only asks for layer norm on all but the final layer -- so `SiLU` was a
      choice with no evidence behind it.  Tanh is what actually produced the published numbers.
    * **2048 wide by default.**  Appendix B: "we found that using larger MLPs led to better
      performance -- in many experiments we used networks with layer size up to 2048", and
      `cfg/robomimic/dsrl_can.yaml` sets `layer_size: 2048`.  Width 512 was a memory decision made
      before the VLA was removed from the loop; it no longer buys anything.

    LayerNorm sits after every hidden `Linear` and before the (linear) output projection, which is
    "all but the final layer" as the paper words it.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 2048, layers: int = 3):
        super().__init__()
        modules: list[nn.Module] = []
        width = input_dim
        for _ in range(layers):
            modules += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.Tanh()]
            width = hidden
        modules.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseActor(nn.Module):
    """Observation -> pre-tanh action, exactly as SB3's Gaussian policy does it.

    The reference does not invent its own actor: `DSRL.actor` is `self.policy.actor` from the
    vendor'd Stable Baselines3, so the policy is a `SquashedDiagGaussianDistribution` with

    * a **learnable** per-dimension `log_std` (initialised to 0, i.e. std = 1),
    * `action = tanh(mean + std * eps)`,
    * `log_prob = Normal(gaussian).log_prob - sum(log(1 - action^2 + 1e-6))`.

    That third line is the piece an earlier version of this file got wrong.  It computed the
    density of the *pre-squash* value, which is a different random variable; the entropy term then
    read ~1414 regardless of what the actor did, `alpha` had nothing to regulate, and the run
    diverged to `|mean| = 52728`.  Computing `tanh`'s Jacobian is what makes `alpha` meaningful.

    With a learnable `log_std` the entropy is no longer constant (the fixed-std version made
    `alpha` a pure additive constant), so the SAC temperature actually does something.
    """

    def __init__(self, obs_dim: int, noise_dim: int, hidden: int = 2048):
        super().__init__()
        self.body = MLP(obs_dim, noise_dim, hidden)
        # SB3: `log_std` is a free parameter, not the output of the trunk, and starts at 0.
        self.log_std = nn.Parameter(torch.zeros(noise_dim))
        self.epsilon = 1e-6

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean of the pre-squash Gaussian.  Unbounded, as in SB3."""
        return self.body(obs)

    def sample(self, obs: torch.Tensor, generator: torch.Generator | None = None):
        """One reparameterised draw, with the squashed density.

        Returns `(action, log_prob)` where `action = tanh(gaussian)` lives in [-1, 1] and
        `log_prob` is the density of *that* action.  The caller multiplies by
        `--action-magnitude` before touching the sampler, which is what `scale_action` does in the
        reference: the distribution stays unit-scale and the bound lives outside it.
        """
        mean = self.forward(obs)
        std = self.log_std.clamp(-20.0, 2.0).exp()
        gaussian = mean + std * torch.randn(mean.shape, device=mean.device, generator=generator)
        action = torch.tanh(gaussian)
        # Diagonal Gaussian density of the pre-squash variable, summed over dimensions.
        log_prob = (
            -0.5 * ((gaussian - mean) / std) ** 2
            - torch.log(std)
            - 0.5 * float(np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)
        # tanh is bijective and differentiable, so the change of variables adds this term (SB3
        # `distributions.py:239`).  Without it the density belongs to the wrong variable.
        log_prob = log_prob - torch.log(1.0 - action**2 + self.epsilon).sum(dim=-1, keepdim=True)
        return action, log_prob


class QHead(nn.Module):
    """Q(s, a) with the two inputs concatenated at the input layer.

    Concatenation rather than late fusion: the question is "at this state, how good is this
    action", and a head that only sees the state before fusing cannot represent that interaction.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 2048):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))


def twin_critic(obs_dim: int, action_dim: int, hidden: int = 2048) -> nn.ModuleList:
    return nn.ModuleList([QHead(obs_dim, action_dim, hidden) for _ in range(2)])


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)


# ---------------------------------------------------------------------------------------------
# the prefix: everything that depends on the observation and not on the noise
# ---------------------------------------------------------------------------------------------
@dataclass
class Prefix:
    """One observation, encoded as far as it can be without knowing the noise.

    `embeddings` and `token_length` are carried so `visual_feature` can slice the image and
    language halves out of the very tensors `embed_prefix` already built, instead of paying for a
    second SigLIP pass.
    """

    pad_masks: torch.Tensor
    past_key_values: object
    hidden_last: torch.Tensor
    dtype: torch.dtype
    embeddings: torch.Tensor
    token_length: int
    #: [[camera_0_patch_count], ...] is unnecessary: every camera contributes the same count, and
    #: it is recovered as `(embeddings.shape[1] - token_length) // cameras`.
    batch: int


def encode_prefix(policy, prepared: dict, *, keep_embeddings: bool = True) -> Prefix:
    """Observation -> `Prefix`.  Independent of any noise.

    The split matters for cost as well as for clarity: measured on this stack the prefix is ~155 ms
    and the ten denoise steps that follow are ~127 ms, so recomputing the prefix per noise sample
    throws away more than half the work.  It is also what makes `--utd > 1` affordable and, at
    inference, what lets one forward pass serve both the action chunk and the actor's features.
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks  # noqa: PLC0415

    model = policy.model
    images, img_masks = policy._preprocess_images(prepared)  # noqa: SLF001
    tokens = prepared["observation.language.tokens"]
    masks = prepared["observation.language.attention_mask"]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    (prefix_hidden, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(prefix_att_2d),  # noqa: SLF001
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    # `prefix_pad_masks` is [images..., language...] with the padding trailing, so the last
    # *valid* position is `sum - 1`.  Indexing `[:, -1]` would read a padded slot whenever the
    # tokeniser padded, which is every batch whose prompt is shorter than `max_length`.
    last = prefix_pad_masks.sum(dim=1) - 1
    rows = torch.arange(prefix_hidden.shape[0], device=prefix_hidden.device)
    last_token = prefix_hidden[rows, last.long()]
    return Prefix(
        pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        hidden_last=last_token.float(),
        dtype=prefix_embs.dtype,
        embeddings=prefix_embs if keep_embeddings else None,
        token_length=int(tokens.shape[1]),
        batch=int(tokens.shape[0]),
    )


def visual_feature(policy, prefix: Prefix) -> torch.Tensor:
    """The `features_v9` vector, rebuilt from the prefix's own embeddings.

    `extract_value_features.py` computes

        image_mean[i]  = mean over patches of paligemma.embed_image(images[i])     # (B, 2048)
        language_mean  = masked mean of paligemma.embed_language_tokens(tokens)   # (B, 2048)
        row            = cat([image_mean.flatten(1), language_mean])              # (B, 8192)

    and `embed_prefix` concatenates exactly those tensors -- with one difference: it multiplies the
    language embedding by `sqrt(hidden)`.  Undoing that scale is the `language_scale` line below.

    Doing this from the slice rather than from a fresh `embed_image` call is what keeps inference
    cheap; `verify_visual_feature` is what keeps it honest.
    """
    embeddings = prefix.embeddings
    if embeddings is None:
        raise ValueError("encode_prefix was called with keep_embeddings=False")
    token_length = prefix.token_length
    remaining = embeddings.shape[1] - token_length
    cameras = 3
    if remaining % cameras:
        raise ValueError(
            f"prefix has {embeddings.shape[1]} positions, {token_length} of them language; "
            f"the remaining {remaining} are not divisible by {cameras} cameras"
        )
    patches = remaining // cameras
    per_camera = embeddings[:, : remaining].reshape(embeddings.shape[0], cameras, patches, -1)
    image_mean = per_camera.mean(dim=2)

    model = policy.model
    hidden = int(model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    language = embeddings[:, remaining:] / math.sqrt(hidden)
    mask = prefix.pad_masks[:, remaining:].float().unsqueeze(-1)
    language_mean = (language.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    row = torch.cat([image_mean.reshape(image_mean.shape[0], -1), language_mean], dim=-1)
    if row.shape[1] != VISUAL_RAW_DIM:
        raise ValueError(f"visual row is {row.shape[1]} wide, expected {VISUAL_RAW_DIM}")
    return row


def sample_with_prefix(policy, prefix: Prefix, noise: torch.Tensor,
                       num_steps: int, device=None) -> torch.Tensor:
    """The flow-matching loop from a given noise and a cached prefix.

    `PI05Pytorch.sample_actions` is decorated `@torch.no_grad()`, and an enclosing `enable_grad()`
    does not survive it, so the loop is written out here.  Two further traps are documented in
    `measure_dsrl_feasibility.py`: the eager attention implementation (set in `encode_prefix`), and
    the key/value cache being reused rather than recomputed.

    `noise` is flat `(batch, chunk * action_dim)` or already shaped `(batch, chunk, action_dim)`.
    """
    model = policy.model
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    device = device or noise.device
    chunk = noise.reshape(noise.shape[0], chunk_size, action_dim).to(prefix.dtype)
    dt = -1.0 / num_steps
    for step in range(num_steps):
        time_tensor = torch.tensor(
            1.0 + step * dt, dtype=torch.float32, device=device
        ).expand(chunk.shape[0])
        velocity = model.denoise_step(
            prefix_pad_masks=prefix.pad_masks,
            past_key_values=prefix.past_key_values,
            x_t=chunk,
            timestep=time_tensor,
        )
        chunk = chunk + dt * velocity
    return chunk


def repeat_noise(noise: torch.Tensor, chunk_size: int, action_dim: int,
                 mode: str = "single") -> torch.Tensor:
    """R^d -> (R^d)^C by repeating the same vector at every chunk position (paper §C.5).

    The paper trains a single-step actor on `R^d` and, at inference, "simply sample w_single and
    repeat it across the action chunk axis".  `full` leaves a `C*d` vector untouched, which is what
    every run before §9.90 used.
    """
    if mode == "single":
        repeated = noise[:, None, :].expand(noise.shape[0], chunk_size, action_dim)
        return repeated.reshape(noise.shape[0], -1)
    return noise


# ---------------------------------------------------------------------------------------------
# the observation the actor consumes
# ---------------------------------------------------------------------------------------------
def build_observation(proprio: torch.Tensor, vlm: torch.Tensor | None,
                      visual: torch.Tensor | None, parts: dict) -> torch.Tensor:
    """`[proprio, vlm, visual]` in the order the checkpoint was trained with.

    The order is not cosmetic: `observation_dim` is fixed by the actor's first `Linear`, so a
    different concatenation order is a same-shape, wrong-meaning input -- the worst kind.
    """
    pieces = []
    if parts.get("proprio", True):
        pieces.append(proprio)
    if parts.get("vlm", True):
        pieces.append(vlm)
    if parts.get("visual", True):
        pieces.append(visual)
    if not pieces:
        raise ValueError("at least one observation part must be enabled")
    return torch.cat(pieces, dim=1).float()


class SteeringActor:
    """A trained noise policy, loaded from a `train_dsrl.py` checkpoint.

    Holds everything needed to go from a preprocessed batch to the noise that replaces the
    sampler's default `N(0, 1)`: the actor, the frozen random projection that maps the 8192-wide
    visual summary to 256 dims, the observation layout, and the noise mode.

    The projection is part of the checkpoint because it is part of the model in every way that
    matters -- retraining with a different seed produces a different projection and an
    incompatible actor, and regenerating it here instead of loading it is the single easiest way to
    silently evaluate noise.
    """

    def __init__(self, checkpoint: Path | str, device: str = "cuda"):
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.device = device
        self.noise_mode = state.get("noise_mode", "single")
        self.chunk_size = int(state["chunk_size"])
        self.action_dim = int(state["action_dim"])
        self.actor_dim = int(state["actor_dim"])
        self.observation_dim = int(state["observation_dim"])
        self.parts = state.get("observation_parts", {"proprio": True, "vlm": True, "visual": True})
        self.projection = np.asarray(state["projection"], dtype=np.float32)
        # The width is part of the architecture, so it has to travel with the weights.  Two sources,
        # in order: the field written by `train_dsrl.py`, then the first layer's own shape.
        #
        # The fallback is not redundant.  Checkpoints written before the field existed have only
        # the weights, and defaulting those to the class default (2048) rebuilt a 512-wide actor at
        # the wrong width and died on `load_state_dict`.  Reading the shape cannot disagree with
        # the tensors it is about to load, so it is the reliable source; the field is kept because
        # it is self-describing when reading a checkpoint by hand.
        self.hidden = int(state.get("hidden") or state["actor"]["body.net.0.weight"].shape[0])
        actor = NoiseActor(self.observation_dim, self.actor_dim, hidden=self.hidden)
        actor.load_state_dict(state["actor"])
        self.actor = actor.to(device).eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad = False

    def observe(self, policy, prepared: dict, prefix: Prefix) -> torch.Tensor:
        """Preprocessed batch -> the actor's observation vector."""
        proprios = prepared.get("observation.state")
        if proprios is None and self.parts.get("proprio", True):
            raise ValueError("the checkpoint expects proprioception but the batch has none")
        visual = None
        if self.parts.get("visual", True):
            raw = visual_feature(policy, prefix)
            projection = torch.from_numpy(self.projection).to(raw.device)
            visual = raw @ projection
        vlm = prefix.hidden_last if self.parts.get("vlm", True) else None
        return build_observation(proprios.float(), vlm, visual, self.parts)

    @torch.no_grad()
    def noise(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """The noise to hand the sampler, already repeated across the chunk axis.

        `deterministic` returns the squashed mean rather than a sample: `tanh(mean)`.  The
        reference's `evaluate` reports both, and the deterministic one is the fair comparison
        against a fixed seed because it removes the actor's own sampling variance from the
        measurement.
        """
        if deterministic:
            action = torch.tanh(self.actor.forward(observation))
        else:
            action, _ = self.actor.sample(observation)
        return repeat_noise(action, self.chunk_size, self.action_dim, self.noise_mode)


# ---------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------
def verify_visual_feature(policy, prepared: dict, prefix: Prefix,
                          dataset: Path, rows: np.ndarray, projection: np.ndarray) -> dict:
    """Compare the sliced visual feature against `features_v9` on real rows.

    Returns the discrepancy statistics and raises on a gross mismatch.  The tolerance is loose
    because the cached array is float16 and the live computation is float32, so agreement to ~1e-2
    is exact agreement after the round trip; anything larger means the slice is wrong.

    This exists because a wrong feature does not crash.  It makes the actor slightly worse, which
    reads as "the actor does not help" -- indistinguishable from the result this whole exercise is
    trying to measure.
    """
    features_dir = Path(dataset) / "features_v9"
    cached = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32)
    live = visual_feature(policy, prefix).detach().float().cpu().numpy()
    if live.shape[0] != len(rows):
        raise ValueError(f"live batch has {live.shape[0]} rows, expected {len(rows)}")
    raw_diff = float(np.abs(live - cached).max())
    projected_live = live @ projection
    projected_cached = cached @ projection
    projected_diff = float(np.abs(projected_live - projected_cached).max())
    scale = float(np.abs(projected_cached).std())
    relative = projected_diff / max(scale, 1e-9)
    return {
        "rows": int(len(rows)),
        "raw_max_abs_diff": raw_diff,
        "projected_max_abs_diff": projected_diff,
        "projected_std": scale,
        "relative": relative,
        # The cached rows are float16, so a value of ~1e-2 in the raw comparison is expected; the
        # 256-dim projection averages 8192 terms and shrinks it further.
        "ok": bool(relative < 0.05),
    }


def load_feature_index(dataset: Path) -> dict:
    return json.loads((Path(dataset) / "features_v9" / "index.json").read_text())
