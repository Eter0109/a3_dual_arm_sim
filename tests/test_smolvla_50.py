def load(name):
    import importlib

    return importlib.import_module("a3_dual_arm_sim.workflows." + name + "_cli")


def test_ranking_prefers_success_over_cookie_average():
    module = load("improve_smolvla_50")
    good = [{"success": True, "cookies_in_target": 10, "steps": 800}]
    bad = [{"success": False, "cookies_in_target": 10, "steps": 1000}]
    assert module.rank(good) < module.rank(bad)
    lo, hi = module.wilson(10, 20)
    assert 0 < lo < 0.5 < hi < 1


def test_cli_passes_overrides_and_closes(monkeypatch):
    module = load("benchmark_cookie_batch")
    captured = {}

    class Policy:
        def __init__(self, checkpoint, **kwargs):
            captured.update(kwargs)

        def close(self):
            captured["closed"] = True

    class Benchmark:
        def __init__(self, **kwargs):
            pass

        def evaluate(self, **kwargs):
            captured["workers"] = kwargs["workers"]

    monkeypatch.setattr(module, "SmolVLAPolicyAdapter", Policy)
    monkeypatch.setattr(module, "DiagnosticBenchmark", Benchmark)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--policy",
            "smolvla:checkpoint",
            "--inference-seed",
            "1001",
            "--n-action-steps",
            "4",
            "--dataset-root",
            "datasets/test",
        ],
    )
    assert module.main() == 0
    assert captured["inference_seed"] == 1001
    assert captured["n_action_steps"] == 4
    assert captured["dataset_root"].is_absolute()
    assert captured["workers"] == 1 and captured["closed"]
