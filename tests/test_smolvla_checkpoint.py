from shutil import copytree

from a3_dual_arm_sim.workflows.benchmark import resolve_smolvla_checkpoint


def test_latest_step_precedes_older_ema_and_stale_last(tmp_path):
    old = tmp_path / "checkpoints/002000/pretrained_model_ema"
    latest = tmp_path / "checkpoints/020000/pretrained_model"
    for path in (old, latest):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
    last = tmp_path / "checkpoints/last"
    try:
        last.symlink_to(old.parent, target_is_directory=True)
    except OSError:
        # Preserve the stale-last scenario when Windows denies symlink creation.
        copytree(old.parent, last)
    assert resolve_smolvla_checkpoint(tmp_path) == latest
    ema = latest.parent / "pretrained_model_ema"
    ema.mkdir()
    (ema / "config.json").write_text("{}")
    assert resolve_smolvla_checkpoint(tmp_path) == ema
    assert resolve_smolvla_checkpoint(latest) == latest


def test_incomplete_new_step_does_not_hide_saved_checkpoint(tmp_path):
    saved = tmp_path / "checkpoints/000010/pretrained_model"
    saved.mkdir(parents=True)
    (saved / "config.json").write_text("{}")
    (tmp_path / "checkpoints/000020").mkdir()
    assert resolve_smolvla_checkpoint(tmp_path) == saved
