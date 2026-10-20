from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from a3_dual_arm_sim.learning.smolvla_attention import (
    install_sdpa_attention,
    sdpa_attention_forward,
)


def test_sdpa_attention_matches_eager_on_gqa_tensors() -> None:
    pytest.importorskip("lerobot")
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    heads, kv_heads, head_dim, batch, seq = 15, 5, 8, 2, 6
    stub = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv_heads)
    torch.manual_seed(0)
    q = torch.randn(batch, seq, heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch, seq, kv_heads, head_dim, dtype=torch.float32)
    # The real caller passes a 3-D (B, T, S) boolean mask.
    mask = torch.rand(batch, seq, seq) > 0.2

    eager = SmolVLMWithExpertModel.eager_attention_forward.__get__(stub)
    expected = eager(mask, batch, head_dim, q, k, v)
    actual = sdpa_attention_forward(stub, mask, batch, head_dim, q, k, v)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_install_sdpa_attention_replaces_interface() -> None:
    pytest.importorskip("lerobot")
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    original = SmolVLMWithExpertModel.get_attention_interface
    try:
        assert install_sdpa_attention() is True
        assert SmolVLMWithExpertModel.get_attention_interface is not original
    finally:
        SmolVLMWithExpertModel.get_attention_interface = original