"""Research integrity and controller tests; synthetic metrics are never model results."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autovla import agent, cli
from autovla.config import apply_proposal, read_json, validate_config, validate_protocol, write_json
from autovla.metrics import is_better, summarize
from autovla.process import run_job, study_lock
from autovla.study import init_study, refresh, verify

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    value = read_json(REPO / "autovla/protocol.json")
    value.update(steps=20, max_experiments=5, trace_episodes=0)
    value["development"] = {"seed_start": 10000, "episodes": 4}
    value["acceptance"] = {"seed_start": 30000, "episodes": 4}
    return value


@pytest.fixture
def study(tmp_path, protocol):
    config = read_json(REPO / "autovla/experiment.json")
    config["warmup_steps"] = 5
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "config.json", config)
    root = tmp_path / "outputs/study"
    init_study(
        root,
        repo=REPO,
        protocol_path=tmp_path / "protocol.json",
        config_path=tmp_path / "config.json",
        demo=True,
    )
    return root


def proposal(parameter="n_action_steps", value=4):
    return {
        "hypothesis": "Test a shorter inference chunk",
        "change": {
            "parameter": parameter,
            "value": value,
        },
        "expected_effect": "More frequent corrections",
        "finding": "A hypothesis",
        "stop": False,
    }


def test_keep_discard_reuse_restart_and_holdout(study):
    assert cli.main(["loop", "--study", str(study), "--agent", "mock", "--rounds", "2"]) == 0
    state = refresh(study)
    assert [row["status"] for row in state["recent"]] == ["keep", "keep", "discard"]
    assert state["best"]["id"] == "exp_0002"
    assert state["best"]["config"]["n_action_steps"] == 4
    assert read_json(study / "experiments/exp_0003/train.json")["reused_from"] == "exp_0002"
    assert cli.main(["loop", "--study", str(study), "--agent", "mock", "--rounds", "1"]) == 0
    assert refresh(study)["experiments"] == 4
    assert refresh(study)["best"]["id"] == "exp_0002"  # Exact tie discarded.
    assert cli.main(["accept", "--study", str(study)]) == 0
    assert read_json(study / "acceptance.json")["metrics"]["synthetic"] is True
    assert cli.main(["accept", "--study", str(study)]) == 1
    assert cli.main(["loop", "--study", str(study), "--agent", "mock"]) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"lr": float("nan")},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"n_action_steps": 51},
        {"warmup_steps": 20},
        {"decay_lr": 0.001},
    ],
)
def test_invalid_config_rejected(study, change):
    with pytest.raises(ValueError):
        validate_config({**read_json(study / "baseline_config.json"), **change}, 20)


def test_one_parameter_allowlist(study):
    config = read_json(study / "baseline_config.json")
    with pytest.raises(ValueError, match="allowlist"):
        apply_proposal(config, proposal("eval_seed", 9000), 20)
    value = proposal()
    value["change"]["other"] = "lr"
    with pytest.raises(ValueError):
        apply_proposal(config, value, 20)


def test_duplicate_never_trains(study, tmp_path):
    assert cli.main(["loop", "--study", str(study), "--rounds", "0"]) == 0
    file = tmp_path / "proposal.json"
    write_json(file, proposal())
    assert cli.main(["submit", "--study", str(study), "--proposal", str(file)]) == 0
    assert cli.main(["submit", "--study", str(study), "--proposal", str(file)]) == 1
    directory = study / "experiments/exp_0003"
    assert not (directory / "train.json").exists()
    assert read_json(directory / "result.json")["status"] == "error"
    assert refresh(study)["best"]["id"] == "exp_0002"


def test_frozen_inputs_block_before_gpu(study):
    (study / "scene.yaml").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen"):
        verify(study)
    assert cli.main(["loop", "--study", str(study), "--rounds", "0"]) == 1
    assert not (study / "experiments").exists()


def test_partial_and_duplicate_evaluations_rejected(protocol):
    config = {"n_action_steps": 8}
    report = cli.synthetic_eval(protocol, config, "development")
    summarize(report, protocol)
    report["episodes"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(report, protocol)
    report["episodes"].append(report["episodes"][0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(report, protocol)


def test_success_contract_and_noncausal_failure_report(protocol):
    report = cli.synthetic_eval(protocol, {"n_action_steps": 8}, "development")
    metrics = summarize(report, protocol)
    assert metrics["successes"] == 6
    assert "pose_error_mm" in metrics["unavailable_metrics"]
    report["episodes"][0]["cookies_in_source"] = 71
    with pytest.raises(ValueError, match="contract"):
        summarize(report, protocol)


def test_predeclared_tiebreaker():
    best = {"successes": 0, "mean_cookies": 3}
    assert is_better({"successes": 1, "mean_cookies": 1}, best)
    assert is_better({"successes": 0, "mean_cookies": 4}, best)
    assert not is_better(best, best)


def test_split_overlap_rejected(protocol):
    protocol["acceptance"] = copy.deepcopy(protocol["development"])
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(protocol)


def test_training_seed_overlap_rejected(tmp_path, protocol):
    dataset, model = tmp_path / "dataset", tmp_path / "model"
    write_json(dataset / "meta/info.json", {})
    (dataset / "a3_episode_metadata.jsonl").write_text('{"seed": 10000}\n', encoding="utf-8")
    write_json(model / "config.json", {"type": "smolvla"})
    (model / "model.safetensors").write_bytes(b"test only")
    protocol.update(dataset_root=str(dataset), base_model=str(model))
    write_json(tmp_path / "protocol.json", protocol)
    config = {**read_json(REPO / "autovla/experiment.json"), "warmup_steps": 5}
    write_json(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="seeds overlap"):
        init_study(
            tmp_path / "outputs/live",
            repo=REPO,
            protocol_path=tmp_path / "protocol.json",
            config_path=tmp_path / "config.json",
        )


def test_failure_and_restart_keep_previous_best(study, tmp_path, monkeypatch):
    assert cli.main(["loop", "--study", str(study), "--rounds", "0"]) == 0

    def fail(*args):
        raise RuntimeError("Simulated GPU OOM")

    monkeypatch.setattr(cli, "execute", fail)
    file = tmp_path / "proposal.json"
    write_json(file, proposal("batch_size", 4))
    assert cli.main(["submit", "--study", str(study), "--proposal", str(file)]) == 1
    state = refresh(study)
    assert state["best"]["id"] == "exp_0001"
    assert state["recent"][-1]["status"] == "error"
    row = read_json(study / "experiments/exp_0002/result.json")
    row["status"] = "training"
    write_json(study / "experiments/exp_0002/result.json", row)
    assert cli.main(["status", "--study", str(study)]) == 0
    assert refresh(study)["recent"][-1]["status"] == "interrupted"


def test_budget_survives_restart(study):
    assert cli.main(["loop", "--study", str(study), "--agent", "mock", "--rounds", "20"]) == 0
    assert refresh(study)["experiments"] == 5
    assert cli.main(["loop", "--study", str(study), "--agent", "mock", "--rounds", "20"]) == 0
    assert refresh(study)["experiments"] == 5


def test_api_backed_loop_and_compact_request(study, monkeypatch):
    monkeypatch.setenv("AUTOVLA_API_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("AUTOVLA_API_MODEL", "user-selected-model")
    monkeypatch.setenv("AUTOVLA_API_KEY", "test-secret-never-log")
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(proposal())}}],
                    "usage": {"total_tokens": 100},
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            return Response()

    monkeypatch.setattr(agent.urllib.request, "build_opener", lambda *args: Opener())
    assert cli.main(["loop", "--study", str(study), "--agent", "api", "--rounds", "1"]) == 0
    assert len(captured) == 1
    request, timeout = captured[0]
    body = json.loads(request.data)
    assert request.full_url == "https://provider.example/v1/chat/completions"
    assert body["model"] == "user-selected-model"
    assert len(body["messages"]) == 2
    assert timeout <= 180
    assert body["response_format"] == {"type": "json_object"}
    prompt = body["messages"][1]["content"]
    assert "test-secret" not in prompt
    assert "30000" not in prompt  # No holdout seeds sent to the proposing agent.
    assert str(study) not in prompt
    assert refresh(study)["best"]["id"] == "exp_0002"
    for path in study.rglob("*"):
        if path.is_file():
            assert b"test-secret-never-log" not in path.read_bytes()


def test_api_invalid_response_never_launches_candidate(study, monkeypatch):
    assert cli.main(["loop", "--study", str(study), "--rounds", "0"]) == 0
    monkeypatch.setenv("AUTOVLA_API_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("AUTOVLA_API_MODEL", "test")
    monkeypatch.setenv("AUTOVLA_API_KEY", "test")
    monkeypatch.setattr(cli, "propose", lambda *args, **kwargs: {"command": "change evaluator"})
    assert cli.main(["loop", "--study", str(study), "--rounds", "1"]) == 1
    assert not (study / "experiments/exp_0002/train.json").exists()
    assert refresh(study)["best"]["id"] == "exp_0001"


def test_exclusive_controller_lock(study):
    with (
        study_lock(study),
        pytest.raises(RuntimeError, match="Another controller"),
        study_lock(study),
    ):
        pass


def test_worker_uses_existing_native_benchmark(study, tmp_path, monkeypatch):
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
    from a3_dual_arm_sim.workflows import benchmark
    from a3_dual_arm_sim.workflows.policy_adapters import ExpertPolicyAdapter
    from autovla import worker

    protocol = read_json(study / "contract.json")["protocol"]
    protocol.update(max_steps=2, trace_episodes=1, cases=[protocol["cases"][0]])
    protocol["development"]["episodes"] = 1
    checkpoint = tmp_path / "checkpoint"
    write_json(checkpoint / "config.json", {"type": "smolvla", "test_only": True})
    (checkpoint / "model.safetensors").write_bytes(b"test fixture, not model weights")
    # Exercise real MuJoCo and the observer-only trace, with an expert adapter in
    # place of the unavailable neural model. This is an integration smoke, not VLA evaluation.
    adapter = ExpertPolicyAdapter(A3SameColumnBatchExpert, name="test-expert")
    adapter.close = lambda: None
    monkeypatch.setattr(
        benchmark,
        "SmolVLAPolicyAdapter",
        lambda *args, **kwargs: adapter,
    )
    output = tmp_path / "evaluation/report.json"
    output.parent.mkdir()
    worker.evaluate(
        {
            "protocol": protocol,
            "config": {"n_action_steps": 8},
            "checkpoint": str(checkpoint),
            "split": "development",
        },
        output,
    )
    report = read_json(output)
    assert len(report["episodes"]) == 1
    assert report["episodes"][0]["steps"] == 2
    assert Path(report["episodes"][0]["trace"], "trace.jsonl").is_file()
    assert report["checkpoint_hashes"] == worker.checkpoint_hashes(checkpoint)
    summarize(report, protocol)


def test_worker_requires_final_training_budget(tmp_path, monkeypatch, protocol):
    from a3_dual_arm_sim.learning import training
    from autovla import worker

    checkpoint = tmp_path / "000010/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"test only")
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"checkpoint": str(checkpoint)}

    monkeypatch.setattr(training, "train_smolvla", fake_train)
    config = {**read_json(REPO / "autovla/experiment.json"), "warmup_steps": 5}
    with pytest.raises(RuntimeError, match="fixed update budget"):
        worker.train(
            {"protocol": protocol, "config": config, "model_output": str(tmp_path / "model")},
            tmp_path / "result.json",
        )
    assert captured["lr"] == config["lr"]
    assert captured["steps"] == protocol["steps"]
    assert captured["base_model"] == Path(protocol["base_model"])
    assert not (tmp_path / "result.json").exists()


def test_child_timeout_is_bounded(tmp_path):
    with pytest.raises(TimeoutError):
        run_job(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            log=tmp_path / "job.log",
            timeout=0.2,
        )


def test_module_help_requires_no_training_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "autovla", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "accept" in result.stdout
