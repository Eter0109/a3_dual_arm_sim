"""Check that pi05 checkpoint verification works with safetensors handles."""

import pytest


def test_pi05_weight_verifier_checks_matching_and_mismatched_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from a3_dual_arm_sim.policies.pi05 import _assert_weights_loaded

    policy = torch.nn.Module()
    policy.model = torch.nn.Linear(2, 2)
    safetensors.save_file(policy.state_dict(), str(tmp_path / "model.safetensors"))
    _assert_weights_loaded(policy, tmp_path)
    with torch.no_grad():
        policy.model.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="does not match the checkpoint"):
        _assert_weights_loaded(policy, tmp_path)
