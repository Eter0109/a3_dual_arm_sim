"""Cross-checks on the shipped manifests.

A manifest is data, so its mistakes are not caught by any type checker -- they are
caught by auditing the relationships *between* its fields.  The most damaging of
those is the joint mapping: a model can receive one joint order in its observation
and command another in its action, and every number would still be in range.

These tests run against the manifests committed in ``configs/policies``, so they
fail when someone edits a mapping rather than when someone runs a rollout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from a3_dual_arm_sim.bridge.manifest import load_manifest, validate_manifest
from a3_dual_arm_sim.contracts import JOINT_ACTION_DIM

POLICY_DIR = Path(__file__).resolve().parents[1] / "configs" / "policies"

ALL_MANIFESTS = sorted(path for path in POLICY_DIR.glob("*.yaml"))
#: Files ending in ``_template`` are starting points to copy, so their paths point
#: at a checkout that does not exist here.  They must still parse into a valid
#: manifest -- a template that does not is worse than none -- but they are not
#: expected to be runnable as shipped.
TEMPLATES = [path for path in ALL_MANIFESTS if path.stem.endswith("_template")]
RUNNABLE = [path for path in ALL_MANIFESTS if path not in TEMPLATES]


def test_there_is_at_least_one_manifest_to_check() -> None:
    # A silently empty glob would turn every test below into a no-op.
    assert ALL_MANIFESTS, f"no manifests found in {POLICY_DIR}"
    assert RUNNABLE, "every manifest is a template, so nothing runnable is covered"


@pytest.mark.parametrize("path", RUNNABLE, ids=lambda p: p.name)
def test_every_runnable_manifest_audits_clean(path: Path) -> None:
    problems = validate_manifest(load_manifest(path))

    assert problems == [], f"{path.name}:\n  - " + "\n  - ".join(problems)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_a_template_parses_into_a_valid_manifest(path: Path) -> None:
    """A template must be structurally correct even though its paths are placeholders.

    Only the filesystem checks are expected to fail -- an author copying a template
    should start from something the parser already accepts.
    """

    manifest = load_manifest(path)
    problems = validate_manifest(manifest)

    path_only = ("not found", "not a directory", "no environment name", "no image is set")
    assert manifest.name, f"{path.name} needs a name"
    for problem in problems:
        assert any(token in problem for token in path_only), (
            f"{path.name} has a structural problem, not just a placeholder path: {problem}"
        )


@pytest.mark.parametrize("path", ALL_MANIFESTS, ids=lambda p: p.name)
def test_every_shipped_manifest_is_valid_yaml_with_a_name(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(raw, dict), f"{path.name} must contain a mapping"
    assert raw.get("name"), f"{path.name} needs a name"


@pytest.mark.parametrize("path", ALL_MANIFESTS, ids=lambda p: p.name)
def test_observation_and_action_joint_mappings_agree(path: Path) -> None:
    """The proprio selection and the action map must be the same mapping.

    A model trained on a robot whose joint layout differs from A3's needs two
    conversions: A3's state projected onto the model's joint order, and the
    model's actions projected back onto A3's.  If those disagree -- say because
    someone adds a joint to one and forgets the other -- the policy receives its
    inputs in one order and issues commands in another.  Nothing raises: every
    value is finite and in range, and the arm simply moves wrongly.

    So the test asserts they are inverse, per joint, rather than merely that both
    audit clean.
    """

    manifest = load_manifest(path)
    joint_map = manifest.action.joint_map
    if joint_map is None:
        pytest.skip(f"{path.name} declares no joint_map; the model uses A3's own order")

    # A narrower model may have several observation vectors; the one shaped like
    # the action vector is the proprioception that has to agree with it.
    candidates = [
        spec
        for spec in manifest.observation.vectors.values()
        if spec.indices is not None and len(spec.indices) == len(joint_map)
    ]
    if not candidates:
        pytest.skip(
            f"{path.name} has no observation vector whose indices match the "
            f"{len(joint_map)} action dimensions"
        )

    for spec in candidates:
        assert tuple(spec.indices) == tuple(joint_map), (
            f"{path.name}: {spec.source} selects A3 joints {list(spec.indices)} but the "
            f"action joint_map sends model dimensions to {list(joint_map)}. Feeding one "
            "order and commanding another moves the arm wrongly without raising."
        )


@pytest.mark.parametrize("path", ALL_MANIFESTS, ids=lambda p: p.name)
def test_a_hold_default_only_covers_joints_the_model_cannot_drive(path: Path) -> None:
    """Every joint left to ``hold`` must be one the model genuinely lacks.

    ``hold`` is the safe default for a joint the model was never trained to move.
    It is the wrong default for a joint the model *does* predict, because it would
    silently discard a predicted value -- and the arm would hold still where the
    policy asked it to move.
    """

    manifest = load_manifest(path)
    if manifest.action.default != "hold" or manifest.action.joint_map is None:
        pytest.skip(f"{path.name} does not use a hold default")

    uncovered = sorted(set(range(JOINT_ACTION_DIM)) - set(manifest.action.joint_map))

    assert uncovered, (
        f"{path.name} claims a hold default but its joint_map covers all "
        f"{JOINT_ACTION_DIM} dimensions, so no joint is actually held"
    )
    # The uncovered joints must not appear in the observation selection either, or
    # the model would be fed a joint it is not allowed to command.
    for spec in manifest.observation.vectors.values():
        if spec.indices is not None:
            shared = sorted(set(spec.indices) & set(uncovered))
            assert not shared, (
                f"{path.name}: {spec.source} feeds A3 joints {shared} to the model, but "
                "the action map leaves them uncommanded, so the model sees inputs it "
                "cannot act on"
            )
