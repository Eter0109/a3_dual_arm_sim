"""What a merged dataset's summary still says about the collection behind it.

A `collection_summary.json` is the only place a dataset states how much its scene was allowed to
vary -- the randomization envelope, and the two flags saying whether the props moved and whether
the Cookies were jittered.  A training run's provenance depends on those numbers, and a merged
dataset is the artifact people actually train on, so the merge must carry them.

It did not.  `merge_cookie_shards.py` rebuilt the summary from an enumerated list of fields, and
the list predated the randomization block: the merged summary of a fully randomized collection
named `position_noise_m` and `yaw_noise_rad` (the Cookie jitter) and nothing about the boxes,
while every shard's own summary had them.  A reader could not tell that dataset from a
fixed-layout one.  The list is gone; `combine_summaries` takes everything and fixes up only what
is per shard, and these tests pin both halves of that.

Loaded by path rather than imported: `scripts/` is not a package and is not on `sys.path` (see
`conftest.py`, which adds `src`), and the function under test is the script's own summary
combiner, so bringing the script in is the point rather than a workaround.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: `combine_summaries` is deliberately pure -- no LeRobot, no filesystem -- but the script it
#: lives in imports `aggregate_datasets` at module level, so loading it needs the dependency the
#: collection runs already need.
pytest.importorskip("lerobot")


def _load_merge_module():
    spec = importlib.util.spec_from_file_location(
        "merge_cookie_shards", ROOT / "scripts" / "merge_cookie_shards.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_cookie_shards"] = module
    spec.loader.exec_module(module)
    return module


def _shard_summary(index: int, *, requested: int, attempts: int) -> dict:
    """A shard summary shaped like the one `collect_dataset` writes."""

    return {
        "schema_version": 2,
        "task": "a3_libero_panda_two_cookies",
        "success_contract": "released_2_upright_settled_contained_0_remain",
        "prompt": "pick up the two yellow blocks from the source bin and put them in the target box",
        "repo_id": f"local/run-shard-{index:02d}",
        "shard_index": index,
        "shard_count": 2,
        "placements_randomized": True,
        "cookies_jittered": True,
        "scene_randomization_m": {
            "source_bin_x_m": [-0.01, 0.01],
            "target_bin_yaw_rad": [-0.03, 0.03],
            "min_box_clearance_m": 0.02,
        },
        "position_noise_m": 0.002,
        "yaw_noise_rad": 0.05,
        "requested_episodes": requested,
        "accepted_episodes": requested,
        "attempts": attempts,
        # The one field that is large and genuinely per shard: it is a row per attempt.
        "results": [{"seed": 1000 + index, "success": True}],
    }


def test_the_randomization_envelope_survives_the_merge() -> None:
    """The scene's own fields are carried, not re-listed.

    This is the defect, stated as a test: assert on the *scene's* fields rather than on a copy of
    them, because the failure was that the copy was incomplete and nothing compared it back to
    its source.
    """

    module = _load_merge_module()
    summaries = [_shard_summary(0, requested=50, attempts=51), _shard_summary(1, requested=50, attempts=53)]

    combined = module.combine_summaries(
        summaries, repo_id="local/run-merged", shard_names=["shard_00", "shard_01"], accepted_episodes=100
    )

    assert combined["scene_randomization_m"] == summaries[0]["scene_randomization_m"]
    assert combined["placements_randomized"] is True
    assert combined["cookies_jittered"] is True
    assert combined["position_noise_m"] == 0.002
    assert combined["yaw_noise_rad"] == 0.05
    # The task triple, for the same reason: it is the collection's, not a shard's.
    assert combined["task"] == "a3_libero_panda_two_cookies"
    assert combined["success_contract"] == "released_2_upright_settled_contained_0_remain"
    assert combined["prompt"] == summaries[0]["prompt"]


def test_the_per_shard_fields_are_replaced_and_the_run_totals_summed() -> None:
    """The other half: what *is* per shard must not survive from the first shard.

    A merged summary carrying `shard_index = 0` and one shard's `results` would describe a shard
    while claiming to describe the collection.
    """

    module = _load_merge_module()
    summaries = [_shard_summary(0, requested=50, attempts=51), _shard_summary(1, requested=50, attempts=53)]

    combined = module.combine_summaries(
        summaries, repo_id="local/run-merged", shard_names=["shard_00", "shard_01"], accepted_episodes=100
    )

    assert combined["repo_id"] == "local/run-merged"
    assert "shard_index" not in combined
    assert "results" not in combined
    assert combined["shards"] == ["shard_00", "shard_01"]
    assert combined["shard_count"] == 2
    assert combined["requested_episodes"] == 100
    assert combined["attempts"] == 104
    # 100 accepted of 104 attempts, and the accepted count is the merge's, not the shards'.
    assert combined["success_rate"] == pytest.approx(100 / 104)


def test_the_episode_count_comes_from_the_merge_not_from_the_shards() -> None:
    """A shard's `accepted_episodes` is a claim about itself; the merge knows what it wrote.

    A shard can fail partway and still have written a summary, so summing the shards' own counts
    can produce a summary that disagrees with the dataset beside it -- the exact class of silent
    disagreement the collection path has already produced once.
    """

    module = _load_merge_module()
    summaries = [_shard_summary(0, requested=50, attempts=51), _shard_summary(1, requested=50, attempts=53)]
    summaries[1]["accepted_episodes"] = 50  # what it asked for
    summaries[1]["attempts"] = 80  # ... and it kept trying

    combined = module.combine_summaries(
        summaries, repo_id="local/run-merged", shard_names=["shard_00", "shard_01"], accepted_episodes=87
    )

    assert combined["accepted_episodes"] == 87
    assert combined["attempts"] == 131


def test_mismatched_shard_names_are_refused() -> None:
    """The names are the merge's own order, so a caller that miscounts has a bug to fix."""

    module = _load_merge_module()
    with pytest.raises(ValueError, match="shard names"):
        module.combine_summaries(
            [_shard_summary(0, requested=1, attempts=1)],
            repo_id="local/x",
            shard_names=["shard_00", "shard_01"],
            accepted_episodes=1,
        )
    with pytest.raises(ValueError, match="no shard summaries"):
        module.combine_summaries([], repo_id="local/x", shard_names=[], accepted_episodes=0)
