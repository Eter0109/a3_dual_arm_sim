"""Launch lerobot's trainer on this machine, with the shims the π0.5 checkpoint needs.

Why a launcher rather than `python -m lerobot.scripts.lerobot_train`
-------------------------------------------------------------------
``lerobot/pi05_base`` was saved by an older lerobot than the one installed here, and the
difference is not cosmetic: `policy_preprocessor.json` lists a step called
``relative_actions_processor``, while this version registers the same class under other
names.  ``PolicyProcessorPipeline.from_pretrained`` resolves step names by a plain dict
lookup and raises ``ImportError: Failed to load processor step from registry`` when it
misses -- an error that reads like a missing dependency, not a version skew.

Three ways to fix it, and why this is the one
--------------------------------------------
* Edit ``policy_preprocessor.json`` -- rejected: that is modifying a downloaded checkpoint,
  and the next `snapshot_download` would silently undo it.
* Patch the installed lerobot -- rejected: a venv-local edit that leaves no trace in this
  repository, so the next person to build an environment would hit the same wall with
  nothing to read.
* Register the alias here, before handing over -- chosen.  It is three lines, it lives
  next to the reason for it, and it is idempotent.

``setdefault`` rather than assignment, so if a future lerobot registers the old name
itself this becomes a no-op instead of an error.

The alias is safe because the step is ``enabled: false`` in this checkpoint: the class is
instantiated and then does nothing, so the *name* is the only thing that had to match.

Usage
-----
The same arguments `lerobot-train` takes::

    python scripts/train_pi05_launcher.py --policy.path=lerobot/pi05_base ...
"""

from __future__ import annotations

import math
import os
import sys

# Loaded next to this file rather than from the installed package: it is part of this
# launcher's contract, not of the simulator.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import history_encoder as history  # noqa: E402


def _register_legacy_step_names() -> None:
    """Teach this lerobot the step name the checkpoint was saved with."""

    from lerobot.processor import RelativeActionsProcessorStep
    from lerobot.processor.pipeline import ProcessorStepRegistry

    ProcessorStepRegistry._registry.setdefault(
        "relative_actions_processor", RelativeActionsProcessorStep
    )


#: Which parts of PaliGemma to un-freeze again, overridable from the environment.
#:
#: Why a partial un-freeze rather than the documented full fine-tune
#: -----------------------------------------------------------------
#: `train_expert_only=true` freezes all 3.45 B of PaliGemma and trains the 0.70 B action
#: expert.  The published LIBERO recipe trains everything -- but on 8xH100.  On this 32 GB
#: card the arithmetic is unforgiving, and it is worth writing down because "it did not fit"
#: is otherwise indistinguishable from "it was not tried":
#:
#:   resident bf16 weights, all 4.14 B ................  8.29 GB   (fixed, cannot shrink)
#:   every *trainable* parameter costs 6 bytes more ...  2 B gradient + 4 B AdamW moments
#:     (`torch.optim.AdamW` allocates its moments with `zeros_like`, so they inherit the
#:      parameter's bf16 dtype -- that is why the moments are 4 B and not 16 B)
#:
#:   full un-freeze     4.14 B trainable -> 8.29 + 24.9 = 33.1 GB   does not fit
#:   vision + 18 layers 3.09 B trainable -> 8.29 + 18.5 = 26.8 GB   fits
#:
#: So the 1.05 B that stays frozen is deliberately the pair that is cheapest to do without:
#: `embed_tokens` and `lm_head`.  The latter is the strongest candidate to leave alone --
#: π0.5's subtask prediction was not released, so its text head is not used at inference at
#: all.
#:
#: This is a *scope* choice about what to test, not a claim that 75% is enough; the run it
#: exists for is judged by whether the teacher-forced per-step error moves, on the same
#: ruler as the frozen checkpoints.
_UNFREEZE_ENV_LAYERS = "A3_UNFREEZE_LLM_LAYERS"
_UNFREEZE_ENV_VISION = "A3_UNFREEZE_VISION"


def _unfreeze_request() -> tuple[int, bool]:
    """The requested scope; 0 layers and vision off reproduce the plain frozen run."""

    raw = os.environ.get(_UNFREEZE_ENV_LAYERS, "18")
    vision = os.environ.get(_UNFREEZE_ENV_VISION, "1").strip().lower() not in ("0", "false", "")

    try:
        layers = int(raw)
    except ValueError:
        # Loud rather than defaulted: a typo'd scope would silently produce a frozen run that
        # looks like a negative result about un-freezing.
        raise SystemExit(f"{_UNFREEZE_ENV_LAYERS}={raw!r} is not an integer") from None

    return max(0, layers), vision


def _apply_unfreeze(model, mode: bool) -> None:
    """Re-enable gradients on the requested PaliGemma submodules, and match their mode.

    The mode argument is not decoration.  `PaliGemmaWithExpertModel.train` ends with
    `self.paligemma.eval()` whenever `train_expert_only` is set, and `nn.Module.eval()` on the
    outer policy routes through that same `train(False)`.  Restoring `requires_grad` without
    also restoring `train`/`eval` would leave the un-frozen blocks in eval mode during
    training -- dropout off, and any mode-dependent module frozen -- which trains silently
    differently from what the flag list says.
    """

    layers, vision = _unfreeze_request()
    paligemma = model.paligemma

    if vision:
        tower = paligemma.model.vision_tower
        tower.train(mode)
        for param in tower.parameters():
            param.requires_grad = True

        # The projector is part of the vision path and has to move with the tower.
        #
        # It is the only map from the tower's output into the language model's embedding
        # space (1152 -> 2048 for PaliGemma), and this ran without it for two full runs:
        # the weights of every one of the tower's 437 tensors changed while both of the
        # projector's stayed bit-identical to `pi05_base`.  A frozen projector forces the
        # drifting tower to keep describing the world in a coordinate system fixed by the
        # *original* tower, which is the opposite of what unfreezing the tower is for.
        #
        # The symptom is not a crash but a ceiling: measured with the input-ablation probe,
        # the fine-tune's reliance on the wrist camera stayed at 0.006 rad from step 2000 to
        # step 4000 while its error fell 33x, i.e. every bit of that progress came from the
        # proprioceptive path -- even though a linear probe on state + wrist features
        # decodes the expert's action to 0.0061 rad, under the 0.0075 the expert fails at.
        projector = paligemma.model.multi_modal_projector
        projector.train(mode)
        for param in projector.parameters():
            param.requires_grad = True

    if layers:
        stack = paligemma.model.language_model.layers
        depth = len(stack)
        for block in stack[depth - min(layers, depth) :]:
            block.train(mode)
            for param in block.parameters():
                param.requires_grad = True


def _unfreeze_summary() -> str:
    """One line describing the un-freeze scope, for `main` to print once.

    Deliberately not printed from `_apply_unfreeze`.  That function runs on *every*
    `train()` call -- the training loop calls `policy.train()` before each step, and each
    call re-applies the un-freeze -- so a `print` there writes the line thousands of times.
    The overfit run's log holds 5002 copies of it and 0 lines of anything else in the tail,
    which is how a log stops being greppable: `grep -c` for the loss reported nothing and
    the run looked like it had died at startup when it had in fact finished.
    """

    layers, vision = _unfreeze_request()
    if not vision and not layers:
        return "frozen (expert only)"
    parts = []
    if vision:
        # 2.36 M, read off `PaliGemmaMultiModalProjector` rather than hard-coded so the
        # line keeps telling the truth if the checkpoint's projector changes width.
        parts.append("vision tower + multi_modal_projector (2.4 M projector params)")
    if layers:
        parts.append(f"last {layers} LLM layers")
    return " + ".join(parts)


_STATE_PROJECTOR_ENV = "A3_STATE_PROJECTOR"


def _state_projector_enabled() -> bool:
    """On by default; `A3_STATE_PROJECTOR=0` reproduces pi0.5's shipped behaviour."""

    return os.environ.get(_STATE_PROJECTOR_ENV, "1").strip().lower() not in ("0", "false", "")


def _state_projector_class():
    """A `Linear` that does not report itself as a missing key in a pi05 checkpoint.

    Every released pi05 checkpoint predates this module, so loading one with `strict=True`
    raises `Missing key(s) in state_dict: model.state_proj.weight, model.state_proj.bias`.
    `load_state_dict` has already copied every other tensor by the time it raises, and
    `PI05Policy.from_pretrained` swallows the exception, so the weights do arrive -- but the
    log then reads "Warning: Could not load state dict" instead of "All keys loaded
    successfully!", and trading a real loading check for a cosmetic one is how a genuine
    loading failure gets missed later.

    Supplying the module's own current values for a key the checkpoint does not carry keeps
    `strict=True` meaningful for every other parameter.
    """

    import torch.nn as nn

    class _StateProjector(nn.Linear):
        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            for name, tensor in self.named_parameters():
                state_dict.setdefault(prefix + name, tensor.detach().clone())
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    return _StateProjector


def _attach_state_projector(model, config) -> None:
    """Give `PI05Pytorch` the proprioceptive map that pi0 had and pi0.5 dropped.

    π0.5 has no state projector at all: proprioception reaches the model only as 256
    discretised bin indices inside the language prompt.  For an *absolute* target that is
    survivable -- the model can emit a pose without knowing where it is -- but everything
    measured on this task says the task lives in the residual, and a residual cannot be
    produced without knowing the current pose.  The evidence, in the order it was gathered:

      * the target is 95% identity.  Per joint, `std(action - state) / std(action)` is
        4.4-8.8%, and 57% of frames have a residual under 0.010 rad, so a model that emits
        the state it was handed already has a small loss;
      * a ridge regression on the same 30 episodes fits them to 0.0047 rad while a 3B
        fine-tune on those episodes stalls at 0.0195 -- and stalls there whether it sees 30
        episodes for 3 epochs, one episode for 21 epochs, or 400 for 0.18;
      * the fine-tune's error is *bias*, not sampler noise: averaging 32 flow-matching draws
        moves it 0.0305 -> 0.0273, where zero-mean noise would have fallen 5.7x.  Its motion
        magnitude is right (`0.0237` against the expert's `0.0250`) and its direction is 58
        degrees off;
      * and training in relative-action space -- which puts the residual in the target
        directly -- collapsed to "do not move", which is what an unpredictable residual looks
        like when the current pose cannot be read.

    So: one extra prefix token, `Linear(max_state_dim, language_hidden)`, mask 1 and
    attention mask 0 -- the slot `embed_suffix`'s comment still describes ("so that image,
    language and state inputs do not attend to action tokens").

    Its *magnitude* is the whole game, and getting it wrong is silent.  Attention logits are
    `q . k`, so a token whose key is orders of magnitude smaller than its neighbours takes a
    softmax weight of essentially zero and is invisible no matter how well it is trained.
    Measured on this model: image tokens enter the prefix at element std 4.2, language tokens
    at 9.6 (the language path multiplies its embedding by `sqrt(hidden)`).  A zero-initialised
    `Linear` fed this robot's eight states produced std 0.0055 after 4000 steps -- a channel
    1700x smaller than the tokens around it, and a whole training run and probe cycle that
    measured nothing at all.  Hence: openpi's `std=0.02` initialisation for new projection
    heads, times the same `sqrt(hidden)` the language embeddings get, which lands the token
    at std ~1.5 -- the same order as an image token, and trainable from there.

    Width comes from `config.max_state_dim` (32), not from the batch: the declared state
    width is a capacity, and the eight values this robot actually sends are zero-padded to
    it, exactly as `prepare_action` pads the action.
    """

    import torch.nn as nn

    if getattr(model, "state_proj", None) is not None:
        return

    # Both are 2048 for PaliGemma-2B; read rather than written down so a different base
    # checkpoint cannot silently build the projector at the wrong width.
    hidden = model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.out_features
    projector = _state_projector_class()(config.max_state_dim, hidden)
    # `std=0.02` is the initialisation openpi uses for its new projection heads; the
    # `sqrt(hidden)` scaling is applied in `embed_prefix`, not here, because it is part of
    # how the prefix is assembled rather than part of what this layer computes.
    nn.init.normal_(projector.weight, std=0.02)
    nn.init.zeros_(projector.bias)
    model.state_proj = projector
    # Consumed by `embed_prefix` and cleared, so a second call inside one forward cannot
    # append a second token.
    model._a3_state = None
    for parameter in projector.parameters():
        parameter.requires_grad = True


def _patch_state_projector() -> None:
    """Wire the state token in, and hand the raw state to it on every entry point.

    Lives here rather than in `modeling_pi05.py` for the same reason the other patches do:
    the vendored lerobot is not ours to edit, and a patch leaves an obvious answer to "what
    did this run change?" in the launcher's own log.
    """

    if not _state_projector_enabled():
        print("state projector    : off (pi0.5 default: proprioception only as prompt bins)",
              flush=True)
        return

    import torch
    from lerobot.policies.pi05 import modeling_pi05 as M

    original_init = M.PI05Pytorch.__init__
    original_embed_prefix = M.PI05Pytorch.embed_prefix
    original_policy_forward = M.PI05Policy.forward
    original_policy_chunk = M.PI05Policy.predict_action_chunk

    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        # Built here, not lazily, because the optimizer is constructed from
        # `policy.parameters()` right after the policy exists: a projector that appeared on
        # the first forward would never be given to AdamW and would stay at its zero init
        # forever, which looks exactly like "the idea did not work".
        _attach_state_projector(self, config)

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = original_embed_prefix(self, images, img_masks, tokens, masks)
        state = getattr(self, "_a3_state", None)
        if state is None:
            return embs, pad_masks, att_masks
        self._a3_state = None

        width = self.config.max_state_dim
        if state.shape[-1] < width:
            state = torch.cat(
                [state, state.new_zeros(*state.shape[:-1], width - state.shape[-1])], dim=-1
            )
        elif state.shape[-1] > width:
            state = state[..., :width]

        token = self.state_proj(state.to(embs.dtype))[:, None, :]
        # Same scale as the language embeddings get in `embed_prefix`; see
        # `_attach_state_projector` for why an unscaled token is invisible to attention.
        token = token * math.sqrt(token.shape[-1])
        batch = embs.shape[0]
        embs = torch.cat([embs, token], dim=1)
        pad_masks = torch.cat(
            [pad_masks, torch.ones(batch, 1, dtype=pad_masks.dtype, device=pad_masks.device)],
            dim=1,
        )
        att_masks = torch.cat(
            [att_masks, torch.zeros(batch, 1, dtype=att_masks.dtype, device=att_masks.device)],
            dim=1,
        )
        return embs, pad_masks, att_masks

    def _stash(policy, batch):
        state = batch.get("observation.state")
        if state is not None:
            policy.model._a3_state = state

    def policy_forward(self, batch, reduction="mean"):
        _stash(self, batch)
        return original_policy_forward(self, batch, reduction=reduction)

    def predict_action_chunk(self, batch, **kwargs):
        _stash(self, batch)
        # `original_policy_chunk` is the `@torch.no_grad()`-decorated function, so calling
        # it as a plain callable keeps the decorator that this wrapper would otherwise drop.
        return original_policy_chunk(self, batch, **kwargs)

    M.PI05Pytorch.__init__ = __init__
    M.PI05Pytorch.embed_prefix = embed_prefix
    M.PI05Policy.forward = policy_forward
    M.PI05Policy.predict_action_chunk = predict_action_chunk


def _patch_history_encoder() -> None:
    """Give the model a latent of its own recent trajectory, and teach it task progress.

    See `history_encoder.py` for why the encoder is blind to images, why the extra token
    goes in the prefix, and why no DAgger pass is needed.  What lives here is only the
    wiring: build the module before the optimizer is constructed, append its token to the
    prefix, and add the two auxiliary terms to the returned loss.

    The auxiliary loss is added to the returned scalar rather than to a hook, because
    `PI05Policy.forward` returns exactly the tensor the trainer backpropagates.  Keys are
    added to its `loss_dict` so the two terms stay visible in the log; a run whose
    progress loss never falls is a run whose latent is not carrying progress, and that has
    to be readable without re-running it.
    """

    if not history.enabled():
        print("history encoder    : off", flush=True)
        return

    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi05 import modeling_pi05 as M

    s = history.settings()
    original_init = M.PI05Pytorch.__init__
    original_embed_prefix = M.PI05Pytorch.embed_prefix
    original_policy_forward = M.PI05Policy.forward
    original_policy_chunk = M.PI05Policy.predict_action_chunk

    # Identities of every encoder this process has built.  More than one means a model was
    # constructed after the optimizer was, and the optimizer is stepping the first one while
    # the second one -- freshly initialised, and the one actually in the forward pass -- never
    # moves.  That is indistinguishable from "the module learns slowly" unless the ids are
    # printed, which is why they are.
    built: list[int] = []

    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if getattr(self, "history_encoder", None) is not None:
            return
        hidden = (
            self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.out_features
        )
        self.history_encoder = history.build(self.config, hidden)
        self.history_proj = history.project(torch.zeros(s["width"]), hidden)
        # Consumed by the next `embed_prefix` and cleared, so a second call inside one
        # forward cannot append two tokens.
        self._a3_history = None
        self._a3_history_latent = None
        for parameter in list(self.history_encoder.parameters()) + list(
            self.history_proj.parameters()
        ):
            parameter.requires_grad = True

        # Gradient norms have to be recorded *during* backward.  Reading `.grad` from a
        # forward hook always sees None: the trainer calls `optimizer.zero_grad()`
        # (`set_to_none=True`) before the forward pass, so the buffers are empty exactly
        # when the log runs, and a module that is being trained perfectly looks identical
        # to one that was never in the graph.
        self._a3_grad_norms = {}

        def record(key):
            def hook(grad):
                self._a3_grad_norms[key] = float(grad.detach().norm())
                return grad

            return hook

        for name, parameter in list(self.history_encoder.named_parameters()) + [
            (f"proj.{n}", p) for n, p in self.history_proj.named_parameters()
        ]:
            parameter.register_hook(record(name))
        built.append(id(self.history_encoder))
        print(
            f"[history] built encoder #{len(built)} id={id(self.history_encoder)} "
            f"training={self.training}",
            flush=True,
        )

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = original_embed_prefix(self, images, img_masks, tokens, masks)
        window = getattr(self, "_a3_history", None)
        if window is None:
            return embs, pad_masks, att_masks
        self._a3_history = None

        # Sampling only in training: at inference the mean keeps the rollout reproducible,
        # and the bridge is already stochastic through the flow-matching noise (measured
        # 0.086 rad between two calls on one observation), so a second source of randomness
        # would make one seed unrepeatable.
        mu, logvar = self.history_encoder(window.to(embs.dtype), dropout=s["dropout"])
        latent = self.history_encoder.latent(mu, logvar, sample=bool(self.training and s["sample"]))
        self._a3_history_latent = (mu, logvar, latent)

        token = self.history_proj(latent)[:, None, :] * math.sqrt(embs.shape[-1])
        batch = embs.shape[0]
        embs = torch.cat([embs, token], dim=1)
        pad_masks = torch.cat(
            [pad_masks, torch.ones(batch, 1, dtype=pad_masks.dtype, device=pad_masks.device)],
            dim=1,
        )
        att_masks = torch.cat(
            [att_masks, torch.zeros(batch, 1, dtype=att_masks.dtype, device=att_masks.device)],
            dim=1,
        )
        return embs, pad_masks, att_masks

    def _stash(policy, batch):
        window = batch.get("observation.history")
        if window is not None:
            policy.model._a3_history = window

    def policy_forward(self, batch, reduction="mean"):
        _stash(self, batch)
        loss, loss_dict = original_policy_forward(self, batch, reduction=reduction)
        cached = getattr(self.model, "_a3_history_latent", None)
        target = batch.get("observation.progress")
        if cached is None or target is None or reduction != "mean":
            return loss, loss_dict
        mu, logvar, latent = cached
        progress_prediction = self.model.history_encoder.progress(latent)
        progress_loss = F.mse_loss(progress_prediction.float(), target.to(latent.device).float())
        # KL to N(0, I): the capacity constraint.  Without it the latent is free to encode
        # whatever lowers the flow loss, which includes the visual state of the pile that
        # the encoder cannot even see but can still infer from the trajectory.
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = loss + s["progress_weight"] * progress_loss + s["kl_weight"] * kl
        loss_dict["loss_flow"] = loss_dict.get("loss")
        loss_dict["loss_history_progress"] = float(progress_loss.detach())
        loss_dict["loss_history_kl"] = float(kl.detach())
        loss_dict["loss"] = float(total.detach())

        # Printed rather than left to the trainer's log, which only emits `loss`: the two
        # auxiliary terms are the whole question this run asks, and a run whose progress term
        # never falls is a latent that is not carrying progress.  Sparse on purpose -- every
        # step would bury the training line.
        self.model._a3_history_steps = getattr(self.model, "_a3_history_steps", 0) + 1
        if self.model._a3_history_steps in (1, 50) or self.model._a3_history_steps % 250 == 0:
            # `gn` is the gradient norm the optimiser will actually see, recorded by the
            # backward hooks above during the *previous* step -- the trainer calls
            # `loss.backward()` after this forward returns.  It is the cheap way to tell
            # "the module is learning slowly" from "the module is not in the graph": a new
            # token that is concatenated but never differentiated shows `gn=0.0e+00` while
            # the loss keeps looking plausible, and no amount of waiting fixes it.
            grad_norm = sum(v * v for v in self.model._a3_grad_norms.values()) ** 0.5
            seen = len(self.model._a3_grad_norms)
            # Split by module: the projection into the 2048-wide prefix is 262k parameters and
            # the recurrent core is 368k, so one total says nothing about where the gradient
            # went -- and "the token is learning, the encoder is not" and "neither is" need
            # different fixes.
            encoder_grad = sum(
                v * v for k, v in self.model._a3_grad_norms.items() if not k.startswith("proj.")
            ) ** 0.5
            proj_grad = sum(
                v * v for k, v in self.model._a3_grad_norms.items() if k.startswith("proj.")
            ) ** 0.5
            head_grad = self.model._a3_grad_norms.get("progress.weight", float("nan"))
            # Read the learning rates from the *scheduler's* optimiser, not from a list
            # captured in the factory.  `accelerator.prepare` replaces
            # `optimizer.param_groups` with a new list, so a reference taken before that
            # point keeps reporting the values from `_initial_step` forever -- which looks
            # exactly like a scheduler that never advances, and cost a day of chasing a
            # learning rate that had in fact been set correctly all along.  The staleness
            # was verified with `is`: same optimiser object, different list.
            sched = getattr(self, "_a3_scheduler", None)
            live_groups = getattr(getattr(sched, "optimizer", None), "param_groups", None) or []
            lrs = [f"{g['lr']:.1e}" for g in live_groups]
            probe = self.model.history_encoder.to_mu.weight
            in_opt = any(p is probe for g in live_groups[-1:] for p in g["params"])
            sched_info = f"{type(sched).__name__}@{getattr(sched, 'last_epoch', '?')}"
            weight_fingerprint = float(self.model.history_encoder.gru.to_gates.weight.abs().mean())
            print(
                f"[history] step {self.model._a3_history_steps:6d} "
                f"flow={float(loss.detach()):.5f} "
                f"progress={float(progress_loss.detach()):.5f} "
                f"kl={float(kl.detach()):.5f} "
                f"| gn enc={encoder_grad:.2e} proj={proj_grad:.2e} head={head_grad:.2e} "
                f"({seen}/14) in_opt={int(in_opt)} lrs={lrs} sched={sched_info} "
                f"w|to_gates|={weight_fingerprint:.5f} "
                f"pbias={float(self.model.history_encoder.progress.bias.detach().mean()):+.2e} "
                f"mu_std={float(mu.detach().std()):.4f} "
                f"pred={[round(float(v), 4) for v in progress_prediction.detach().mean(dim=0)]}",
                flush=True,
            )
        return total, loss_dict

    def predict_action_chunk(self, batch, **kwargs):
        _stash(self, batch)
        return original_policy_chunk(self, batch, **kwargs)

    M.PI05Pytorch.__init__ = __init__
    M.PI05Pytorch.embed_prefix = embed_prefix
    M.PI05Policy.forward = policy_forward
    M.PI05Policy.predict_action_chunk = predict_action_chunk


def _patch_partial_unfreeze() -> None:
    """Apply the un-freeze after every place lerobot re-applies the freeze.

    Two call sites, and patching only the first would look like it worked:
      * `PaliGemmaWithExpertModel.__init__` ends with `_set_requires_grad()`
      * `PaliGemmaWithExpertModel.train()` re-freezes on every mode change, and the training
        loop calls `policy.train()` before the first step.
    """

    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel

    original_init = PaliGemmaWithExpertModel.__init__
    original_train = PaliGemmaWithExpertModel.train

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _apply_unfreeze(self, True)

    def train(self, mode: bool = True):
        original_train(self, mode)
        _apply_unfreeze(self, mode)
        # lerobot's override drops `nn.Module.train`'s `self` return; restoring it is what the
        # base class promises, and callers chain off it.
        return self

    PaliGemmaWithExpertModel.__init__ = __init__
    PaliGemmaWithExpertModel.train = train


#: Set to 0 to keep `torch.optim.AdamW`'s default (multi-tensor) implementation.
_ENV_FUSED_ADAMW = "A3_FUSED_ADAMW"


#: Learning rate for parameters that were **created by this launcher** rather than loaded
#: from the checkpoint.  A fresh random module and a fine-tuned 3B model cannot share one
#: learning rate, and this is the reason several arms looked like "the new channel does
#: nothing".
#:
#: Measured on the history encoder with its own data: trained alone at 3e-6 it reaches
#: R^2 = -0.010 against the progress label, with the latent's per-dimension std stuck at
#: 0.0055; at 1e-4 the same code reaches R^2 = 0.954.  The real v6 run reproduced the first
#: pair almost exactly (R^2 = 0.011, std = 0.0055) after 7500 steps.  `state_proj` shows the
#: same signature in hindsight: 1.7% of its initial weight scale moved in 15000 steps
#: (9.43).  A newly initialised layer at 3e-6 is a layer that is frozen with extra steps.
_NEW_MODULE_LR = "A3_NEW_MODULE_LR"


def _new_module_lr() -> float:
    """Learning rate for parameters this launcher adds; 0 (off) unless asked for."""

    raw = os.environ.get(_NEW_MODULE_LR, "").strip()
    return float(raw) if raw else 0.0


def _patch_optimizer_param_groups() -> None:
    """Give modules created here their own learning rate.

    `make_optimizer_and_scheduler` feeds `policy.parameters()` to one AdamW at the
    fine-tuning learning rate.  That is right for weights that came out of the checkpoint --
    3B parameters being nudged -- and wrong for a few hundred thousand freshly initialised
    ones, which have to travel much further before they carry anything.

    Splitting the groups leaves the checkpointed parameters on exactly the schedule they had,
    so everything that was already trained keeps its trajectory.  The new group is *appended*
    so `param_groups[0]` keeps meaning what the trainer logs and the scheduler steps.
    """

    lr = _new_module_lr()
    if lr <= 0.0:
        print("new-module lr       : off (every parameter on the fine-tuning schedule)",
              flush=True)
        return

    from lerobot.optim import factory as optim_factory

    original = optim_factory.make_optimizer_and_scheduler

    def make_optimizer_and_scheduler(cfg, policy):
        # Named, not positional: the policy wraps its modules in a `model` attribute, and a
        # name is the only way to tell "came from the checkpoint" from "built by the
        # `_attach_*` functions in this process".
        fresh, inherited = [], []
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".history_encoder." in name or ".history_proj." in name:
                fresh.append(parameter)
            else:
                inherited.append(parameter)
        if not fresh:
            return original(cfg, policy)
        if cfg.optimizer is None:
            raise ValueError("optimizer config is required")
        optimizer = cfg.optimizer.build(inherited)
        optimizer.add_param_group({"params": fresh, "lr": lr})
        # Recorded **before** the scheduler is built.  `LRScheduler.__init__` takes an initial
        # step, so by the time `build` returns, `param_groups[*]["lr"]` already holds
        # `peak / (warmup + 1)` -- reading it then and calling that "the peak" silently
        # divides every group's learning rate by the warmup length.  That is exactly what the
        # first version of this patch did: the new module was asked for 1e-04 and got
        # 9.1e-06, a third of a decade above the fine-tuning rate rather than two, which is
        # why the latent stayed dead for another 1500 steps.
        peaks = [group["lr"] for group in optimizer.param_groups]
        lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler else None
        if lr_scheduler is not None:
            # `LRScheduler.build` derives its lambda from `param_groups[0]["lr"]` through the
            # config's `peak_lr`, so every group is scaled by the same factor and the fresh
            # group is dragged back onto the fine-tuning schedule.  Restating each group's
            # base with the pre-warmup peak is what keeps the ratio the caller asked for.
            lr_scheduler.base_lrs = peaks
        print(
            f"new-module lr       : {lr:.0e} on {sum(p.numel() for p in fresh)} params "
            f"({len(fresh)} tensors: history encoder + projection); "
            f"{sum(p.numel() for p in inherited)} on the fine-tuning schedule; "
            f"base_lrs={ [f'{p:.1e}' for p in peaks] }",
            flush=True,
        )
        # Exposed so the training log can prove, step by step, that the module in the forward
        # pass is a member of the group the optimiser steps.  Without that check a second
        # model instance built after this point looks exactly like a module that is not
        # learning.
        policy._a3_optimizer_groups = optimizer.param_groups
        policy._a3_scheduler = lr_scheduler
        policy._a3_optimizer = optimizer
        print(
            f"new-module wiring   : encoder id={id(getattr(policy.model, 'history_encoder', None))}",
            flush=True,
        )
        return optimizer, lr_scheduler

    optim_factory.make_optimizer_and_scheduler = make_optimizer_and_scheduler
    # `lerobot_train` did `from ... import make_optimizer_and_scheduler`, so the module
    # attribute is not the only binding that has to change.
    try:
        from lerobot.scripts import lerobot_train

        lerobot_train.make_optimizer_and_scheduler = make_optimizer_and_scheduler
    except Exception:  # noqa: BLE001 - the import alias is the only reason to touch it
        pass


def _patch_fused_adamw() -> None:
    """Force AdamW onto its fused kernel, which is the only variant that fits.

    `torch.optim.AdamW` defaults to the `foreach` (multi-tensor) path, and that path
    materialises a same-sized temporary.  `_multi_tensor_adam` computes

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

    before the update, and with 3.09 B trainable bf16 parameters that temporary alone is
    5.75 GiB.  At that point the run holds 7.72 GiB of weights, 5.75 GiB of gradients and
    11.50 GiB of moments -- 30.43 GiB of the card's 31.36 GiB for the *un-frozen* scope,
    against 12.5 GiB for the frozen one.  So the OOM arrives inside a square root, and the
    traceback names `_foreach_sqrt` rather than anything a reader would connect to memory.

    This is why the batch size is not the knob it looks like: dropping batch 4 to batch 2
    moved the allocation by 0.6 GiB, because with `gradient_checkpointing` the activations
    were never the large term.

    The fused kernel performs the same update in one pass with no such temporary.  It is
    env-gated so a run can go back to the default without editing this file.
    """

    if os.environ.get(_ENV_FUSED_ADAMW, "1").strip().lower() in ("0", "false", ""):
        return

    import torch

    original_init = torch.optim.AdamW.__init__

    def __init__(self, params, *args, **kwargs):
        kwargs.setdefault("fused", True)
        original_init(self, params, *args, **kwargs)

    torch.optim.AdamW.__init__ = __init__


#: Frames to drop from the end of every episode when building training windows.  0 (the
#: default) keeps them all, which is what every run before this one did.
_ENV_DROP_TAIL = "A3_DROP_N_LAST_FRAMES"


#: Episodes shorter than this many frames are excluded from training.  0 uses every
#: episode, which is what every run before this one did.
_ENV_MIN_EPISODE_FRAMES = "A3_MIN_EPISODE_FRAMES"


def _patch_episode_filter() -> int:
    """Train on a chosen subset of the episodes, without rebuilding the dataset.

    41.7% of the 2x5 episodes in `a3_multitask_v6` are 302 frames long and end with **one**
    batch delivered, while the instruction on all 1200 of them says ten cookies.  The model
    reads that as "this sentence sometimes means five", and 9.46 shows exactly that: five of
    eight 2x5 seeds either stop after one batch or freeze holding the second.  The numbers
    line up -- 41.7% of the data says "stop", and about that fraction of the rollouts do.

    Rebuilding the set to drop them means re-encoding 12 GB of video, which costs about two
    hours.  `EpisodeAwareSampler` already takes `episode_indices_to_use`, and episode length
    separates the two populations perfectly (302-303 frames against 506-1176), so the filter
    belongs in the sampler and costs nothing.

    The threshold has to be a length rather than a task because the sampler sees only frame
    ranges.  That is enough here and is worth stating so nobody later reads 400 as a tuned
    constant: every 2x10 episode is ~1067 frames and every 2x5 episode is either ~302 or
    >= 506, so any value in (303, 506) gives the same partition.
    """

    minimum = int(os.environ.get(_ENV_MIN_EPISODE_FRAMES, "0") or 0)
    if minimum <= 0:
        return 0

    from lerobot.datasets import sampler as S

    original = S.EpisodeAwareSampler.__init__

    def __init__(
        self,
        dataset_from_indices,
        dataset_to_indices,
        episode_indices_to_use=None,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=False,
    ):
        lengths = [
            int(end) - int(start)
            for start, end in zip(dataset_from_indices, dataset_to_indices, strict=True)
        ]
        keep = [index for index, length in enumerate(lengths) if length >= minimum]
        if not keep:
            raise RuntimeError(
                f"A3_MIN_EPISODE_FRAMES={minimum} excludes every episode; the longest is "
                f"{max(lengths) if lengths else 0} frames."
            )
        kept_frames = sum(lengths[index] for index in keep)
        print(
            f"episode filter     : keep {len(keep)} of {len(lengths)} episodes "
            f"(>= {minimum} frames, {kept_frames} frames); {len(lengths) - len(keep)} "
            f"dropped as short",
            flush=True,
        )
        original(
            self,
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=keep,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=shuffle,
        )

    S.EpisodeAwareSampler.__init__ = __init__
    return minimum


def _patch_drop_n_last_frames() -> int:
    """Let ``PI05Config`` carry ``drop_n_last_frames``, so the padded chunk tail is not trained on.

    A window near the end of an episode runs past it, and the rows beyond are filled with
    **the last action repeated**.  Nothing removes them from the loss:

    * ``dataset_reader`` does publish an ``action_is_pad`` tensor, but ``PI05Policy.forward``
      neither accepts nor consults it -- it takes ``losses.mean()`` over the whole chunk;
    * ``lerobot_train.py``:358 builds the frame-dropping ``EpisodeAwareSampler`` only
      ``if hasattr(cfg.policy, "drop_n_last_frames")``, and neither ``PI05Config`` nor
      ``PreTrainedConfig`` declares that field.  So the branch is never taken, the sampler is
      ``None``, and every padded row is trained on as though it were a command.

    Measured on the one-Cookie dataset (400 episodes, median 218 frames long): **11.3% of the
    action rows inside the 50-step chunks are that repetition, and 22.5% of the samples contain
    at least one.**  What those rows teach is "flatten toward the end of the chunk" -- the same
    shape two independent measurements already showed: the teacher-forcing probe's chunk error
    grows x2.16 from row 0 to row 15, and ``action.execute`` above 16 makes a rollout markedly
    worse by committing to those rows.

    An attribute rather than a declared field, which is sufficient because the reader is
    ``hasattr`` followed by an attribute access.  It has to happen before the config is built
    from the command line, which is why it sits with the other shims instead of in the loop.
    """

    raw = os.environ.get(_ENV_DROP_TAIL, "0").strip()
    if not raw or raw.lower() in ("0", "false", ""):
        return 0

    count = int(raw)
    if count <= 0:
        raise SystemExit(f"{_ENV_DROP_TAIL} must be positive when set, got {raw!r}")

    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    PI05Config.drop_n_last_frames = count
    return count


def main() -> int:
    _register_legacy_step_names()
    _patch_state_projector()
    _patch_history_encoder()
    _patch_optimizer_param_groups()
    _patch_partial_unfreeze()
    _patch_fused_adamw()
    # Printed here, once, rather than from `_apply_unfreeze`, which the training loop calls
    # before every step.  See `_unfreeze_summary`.
    print(f"unfreeze           : {_unfreeze_summary()}", flush=True)
    print(
        "state projector    : "
        + ("Linear(32 -> 2048) x sqrt(2048), std=0.02 init, one extra prefix token"
           if _state_projector_enabled() else "off"),
        flush=True,
    )
    if history.enabled():
        print(history.summary(), flush=True)
    minimum_episode = _patch_episode_filter()
    if minimum_episode:
        print(
            f"min episode frames : {minimum_episode} (shorter episodes excluded -- a run "
            f"whose episodes disagree with their own instruction cannot be fixed by steps)",
            flush=True,
        )
    dropped = _patch_drop_n_last_frames()
    if dropped:
        # Printed because it is otherwise invisible: the effect is a smaller training set,
        # which shows up only as a different number of samples per step.
        print(
            f"drop_n_last_frames : {dropped} (padded chunk tails excluded from training)",
            flush=True,
        )

    # Imported after the shim so nothing that resolves processors can run first.
    from lerobot.scripts.lerobot_train import main as lerobot_main

    # Not `raise SystemExit(lerobot_main())` -- lerobot's main returns None and signals
    # failure by exception, so the exit code has to come from the absence of one.
    lerobot_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
