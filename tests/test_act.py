from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a3_dual_arm_sim.cli import parser
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.learning.act_training import train_act, validate_act_horizon
from a3_dual_arm_sim.policies.act import ACTPolicyPlugin
from a3_dual_arm_sim.workflows import policy_adapters


def test_act_left_contract_and_episode_reset():
    p = ACTPolicyPlugin.__new__(ACTPolicyPlugin)
    p._torch = torch
    p._device = torch.device("cpu")
    p._action_dim = 8
    p._input_keys = ("observation.state",)
    p._prepare_observation = lambda obs, *args: obs
    p._preprocessor = p._postprocessor = lambda x: x
    seen, resets = [], []

    def select(batch):
        seen.append(batch["observation.state"].copy())
        return torch.full((1, 8), 0.3)

    p._policy = SimpleNamespace(select_action=select, reset=lambda: resets.append(True))
    p.inference_seed = 17
    ctx = EpisodeContext(seed=1, task="transfer", action_mode="joint_position")
    p.reset(ctx)
    state = np.arange(16, dtype=np.float32)
    action = p.act({"observation.state": state}, "task")
    assert np.array_equal(seen[0], state[:8])
    assert np.array_equal(action[8:], state[8:])
    changed = state + 10
    assert np.array_equal(p.act({"observation.state": changed})[8:], state[8:])
    assert np.array_equal(state, np.arange(16))
    p.reset(ctx)
    assert len(resets) == 2 and p.last_raw_action is None
    assert np.array_equal(p.act({"observation.state": changed})[8:], changed[8:])
    with pytest.raises(ValueError, match="16-D"):
        p.act({"observation.state": np.zeros(8)})
    p._policy.select_action = lambda _: torch.full((1, 8), float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        p.act({"observation.state": state})


@pytest.mark.parametrize(
    "chunk,steps,coeff",
    [(50, 0, None), (50, 51, None), (50, 8, 0.01), (50, 1, float("nan"))],
)
def test_invalid_act_horizon(chunk, steps, coeff):
    with pytest.raises(ValueError):
        validate_act_horizon(chunk, steps, coeff)


def test_act_defaults_and_temporal_ensemble():
    args = parser().parse_args(["train-act", "--root", "data", "--output", "out"])
    assert (args.chunk_size, args.n_action_steps, args.steps) == (50, 8, 20000)
    validate_act_horizon(50, 1, 0.01)


def test_act_dry_run_public_config(tmp_path, monkeypatch):
    import json

    from a3_dual_arm_sim.learning import act_training

    monkeypatch.setattr(act_training, "audit_training_dataset", lambda *a, **kw: {"action_dim": 8})

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not train")

    monkeypatch.setattr(act_training.subprocess, "run", forbidden)
    report = train_act(
        dataset_root=tmp_path, output_dir=tmp_path / "run", device="cpu", dry_run=True
    )
    config = json.loads((tmp_path / ".run_act_config/train_config.json").read_text())
    assert config["policy"]["type"] == "act"
    assert config["policy"]["input_features"]["observation.state"]["shape"] == [8]
    assert len(config["policy"]["input_features"]) == 4
    assert config["policy"]["normalization_mapping"]["ACTION"] == "MEAN_STD"
    assert config["policy"]["push_to_hub"] is False
    assert report["command"][-1].startswith("--config_path=")
    assert not (tmp_path / "run").exists()
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        train_act(dataset_root=tmp_path, output_dir=tmp_path / "run")


def test_act_spec_preserves_case(monkeypatch):
    monkeypatch.setattr(policy_adapters, "ACTPolicyAdapter", lambda path: path)
    assert policy_adapters.make_policy_adapter("act:/tmp/MyCheckpoint") == "/tmp/MyCheckpoint"


def test_act_benchmark_passes_overrides_and_closes(monkeypatch, tmp_path):
    import sys

    from a3_dual_arm_sim.workflows import benchmark_cookie_batch_cli as cli

    seen = {}

    class Adapter:
        def __init__(self, checkpoint, **kwargs):
            seen.update(kwargs)

        def close(self):
            seen["closed"] = True

    class Benchmark:
        def __init__(self, **kwargs):
            pass

        def evaluate(self, **kwargs):
            seen["workers"] = kwargs["workers"]

    monkeypatch.setattr(cli, "ACTPolicyAdapter", Adapter)
    monkeypatch.setattr(cli, "DiagnosticBenchmark", Benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--policy",
            "act:run",
            "--device",
            "cpu",
            "--dataset-root",
            str(tmp_path),
            "--n-action-steps",
            "1",
        ],
    )
    assert cli.main() == 0
    assert seen["closed"] and seen["workers"] == 1 and seen["n_action_steps"] == 1


@pytest.mark.parametrize("ensemble", [False, True])
def test_public_act_reset_discards_previous_episode_predictions(ensemble):
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

    policy = ACTPolicy.__new__(ACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        temporal_ensemble_coeff=0.01 if ensemble else None, n_action_steps=1 if ensemble else 8
    )
    if ensemble:
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, 50)
    value = [1.0]
    policy.predict_action_chunk = lambda batch: torch.full((1, 50, 8), value[0])
    plugin = ACTPolicyPlugin.__new__(ACTPolicyPlugin)
    plugin._policy = policy
    plugin.inference_seed = None
    context = EpisodeContext(seed=0, task="transfer", action_mode="joint_position")
    plugin.reset(context)
    assert torch.all(policy.select_action({}) == 1)
    value[0] = 9.0
    plugin.reset(context)
    assert torch.all(policy.select_action({}) == 9)
