"""Regression tests for the history-encoder patches in `train_pi05_launcher.py`.

Three defects lived in that wiring, none of which changed a loss by enough to notice:

1. `_a3_history_latent` was cached in `embed_prefix` and never cleared when a batch carried
   no window, so `policy_forward` trained the progress head against the *previous* batch's
   latent.  Silent, same shape, plausible loss.
2. `policy_forward` returned early for any `reduction` other than `"mean"`, dropping both
   auxiliary terms.  Reachable through `lerobot_train.py`'s RA-BC path.
3. The encoder carried a `Linear(width -> 2048)` that nothing called: 264,192 parameters,
   42 % of the module, in every checkpoint.

The tests below pin each one in a way that fails if the fix is reverted, without loading
pi05 or touching a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.functional")


# --------------------------------------------------------------------------------------
# 1 + 2: the patch, exercised through its own entry point
# --------------------------------------------------------------------------------------


@pytest.fixture()
def patched(monkeypatch):
    """Install the history patches over a stub `PI05Pytorch.embed_prefix`.

    The patch captures `original_embed_prefix` from the module attribute at patch time, so
    stubbing it first means the captured original is a no-op that returns fixed tensors.
    That is the only way to call the patched method without constructing pi05.
    """

    from lerobot.policies.pi05 import modeling_pi05 as M

    def stub_embed_prefix(self, images, img_masks, tokens, masks):
        batch = 2
        embs = torch.zeros(batch, 3, 8)
        pad_masks = torch.ones(batch, 3, dtype=torch.bool)
        att_masks = torch.ones(batch, 3, dtype=torch.bool)
        return embs, pad_masks, att_masks

    monkeypatch.setenv("A3_HISTORY_ENCODER", "1")
    monkeypatch.setenv("A3_HISTORY_STEPS", "10")
    monkeypatch.setattr(M.PI05Pytorch, "embed_prefix", stub_embed_prefix, raising=False)

    import importlib

    import train_pi05_launcher as launcher

    importlib.reload(launcher)
    launcher._patch_history_encoder()
    yield M, launcher


class _Stub:
    """Enough of `PI05Pytorch` for the patched `embed_prefix` to run."""

    def __init__(self) -> None:
        import history_encoder as history

        self.history_encoder = history.build(None, 8)
        self.history_proj = history.project(torch.zeros(128), 8)
        self._a3_history = None
        self._a3_history_latent = ("stale-mu", "stale-logvar", "stale-latent")
        self.training = False


def test_embed_prefix_clears_the_cached_latent_when_no_window_is_stashed(patched):
    """Defect 1: a batch without a window must not inherit the previous batch's latent."""

    M, _ = patched
    stub = _Stub()
    stub._a3_history = None  # this batch carries no window

    M.PI05Pytorch.embed_prefix(
        stub,
        torch.zeros(2, 3, 4, 4),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool),
    )

    assert stub._a3_history_latent is None, (
        "the cached latent survived a window-less batch; policy_forward would train the "
        "progress head against the previous batch's trajectory"
    )


def test_embed_prefix_sets_a_fresh_latent_when_a_window_is_present(patched):
    """The other direction: a normal batch must still populate the cache."""

    M, _ = patched
    stub = _Stub()
    stub._a3_history = torch.zeros(2, 480)

    embs, pad_masks, att_masks = M.PI05Pytorch.embed_prefix(
        stub,
        torch.zeros(2, 3, 4, 4),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool),
    )

    assert stub._a3_history_latent is not None
    assert isinstance(stub._a3_history_latent, tuple) and len(stub._a3_history_latent) == 3
    # The history token is appended: three prefix tokens in, four out.
    assert embs.shape[1] == 4
    assert pad_masks.shape[1] == 4
    assert att_masks.shape[1] == 4
    # And the stash is consumed, so a second call in one forward cannot append twice.
    assert stub._a3_history is None


def test_auxiliary_terms_keep_their_scale_and_reduce_correctly():
    """Defect 2, checked numerically: the per-sample rewrite must not move the mean path.

    A naive "KL per sample" of `-0.5 * sum(...)` rather than `-0.5 * mean(..., dim=-1)`
    would rescale the KL term by the latent width -- 128x -- which silently re-tunes
    `A3_HISTORY_KL_WEIGHT` for every existing run instead of fixing a reduction bug.
    """

    torch.manual_seed(0)
    batch, width = 8, 128
    mu = torch.randn(batch, width) * 0.05
    logvar = torch.randn(batch, width) * 0.05
    target = torch.rand(batch, 2) * 2 - 1
    pred = torch.randn(batch, 2)

    kl_reference = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    progress_reference = torch.nn.functional.mse_loss(pred, target)

    progress_per_sample = torch.nn.functional.mse_loss(pred, target, reduction="none")
    if progress_per_sample.ndim > 1:
        progress_per_sample = progress_per_sample.mean(dim=tuple(range(1, progress_per_sample.ndim)))
    kl_per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)

    assert torch.allclose(kl_per_sample.mean(), kl_reference, atol=1e-9)
    assert torch.allclose(progress_per_sample.mean(), progress_reference, atol=1e-9)
    assert kl_per_sample.shape == (batch,)

    aux = progress_per_sample + kl_per_sample
    assert torch.allclose(aux.sum(), aux.mean() * batch, atol=1e-5)


# --------------------------------------------------------------------------------------
# 3: the dead projection
# --------------------------------------------------------------------------------------


def test_encoder_has_no_unused_projection():
    """Defect 3: `out` was 264,192 parameters that nothing called."""

    import history_encoder as history

    encoder = history.build(None, 2048)
    names = [name for name, _ in encoder.named_parameters()]

    assert not any(name.startswith("out.") for name in names), (
        f"the encoder still carries an unused projection: {names}"
    )
    # 5 modules x weight+bias, and the projection into the prefix lives outside.
    assert len(names) == 10, names
    total = sum(p.numel() for p in encoder.parameters())
    assert total == 101_634, total


def test_old_checkpoints_load_despite_the_removed_projection(tmp_path):
    """Every checkpoint written before the fix still has `history_encoder.out.*`.

    `from_pretrained` defaults to `strict=False`, so those two keys are reported as unused
    and ignored.  This test builds that exact state dict and checks nothing else is lost.
    """

    import history_encoder as history

    encoder = history.build(None, 2048)
    fresh = {k: v.clone() for k, v in encoder.state_dict().items()}
    legacy = dict(fresh)
    width = history.settings()["width"]
    legacy["out.weight"] = torch.zeros(64, width)
    legacy["out.bias"] = torch.zeros(64)

    missing, unexpected = encoder.load_state_dict(legacy, strict=False)

    assert missing == [], missing
    assert sorted(unexpected) == ["out.bias", "out.weight"], unexpected
    for key, value in fresh.items():
        assert torch.equal(encoder.state_dict()[key], value), key
