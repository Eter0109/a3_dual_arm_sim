from a3_dual_arm_sim.workflows.benchmark import resolve_smolvla_checkpoint


def test_latest_step_precedes_older_ema_and_stale_last(tmp_path):
    old = tmp_path / "checkpoints/002000/pretrained_model_ema"
    latest = tmp_path / "checkpoints/020000/pretrained_model"
    for path in (old, latest):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
    try:
        (tmp_path / "checkpoints/last").symlink_to(old.parent)
    except OSError:
        pass
    assert resolve_smolvla_checkpoint(tmp_path) == latest
    ema = latest.parent / "pretrained_model_ema"
    ema.mkdir()
    (ema / "config.json").write_text("{}")
    assert resolve_smolvla_checkpoint(tmp_path) == ema
    assert resolve_smolvla_checkpoint(latest) == latest
