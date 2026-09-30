"""The prompt the Cosmos adapter sends, and whether it says what it discarded.

This module exists because of a misreading that reached the project's own
documentation.  The adapter replaces the caller's task string with a label from the
checkpoint's cached six, and it did so silently: ``cosmos_task`` reported the label
actually used, but nothing said the request's instruction had been dropped.  A
rollout that asked for one thing and conditioned on another therefore looked exactly
like a model that refuses to accept a prompt, and "Cosmos Policy cannot read arbitrary
language" was written down as a property of the model.  It is a property of this
configuration.

The behaviour itself is correct -- a label outside the cached six would make the model
build a 45 GB T5-11B encoder mid-rollout, and the adapter is right to refuse that.  It
is the silence that was wrong, so these tests pin two things: which string is chosen,
and that a discarded string is always reported.

The adapter lives in the model's checkout rather than in this repository, so the copy
under ``integrations/`` is the one these tests load.  That copy imports nothing but
numpy at module level, which is what makes the test possible without the model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from a3_dual_arm_sim.bridge.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "integrations" / "cosmos_policy" / "adapter_cosmos_policy_aloha.py"
COSMOS_MANIFEST = REPO / "configs" / "policies" / "cosmos_policy_aloha.yaml"

#: The six labels the released ALOHA cache carries.  Repeated here as the fixture the
#: adapter is tested against; the manifest's own declaration is checked separately.
CACHED = (
    "fold shirt",
    "put candy in bag",
    "put candies in bowl",
    "put purple eggplant on plate",
    "put brown chicken wing on plate",
    "pick up the cube and place it in the bowl",
)


@pytest.fixture(scope="module")
def adapter():
    spec = importlib.util.spec_from_file_location("cosmos_aloha_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cached(adapter, monkeypatch):
    """Pretend the T5 cache is loaded, without any of Cosmos Policy present."""

    monkeypatch.setattr(adapter, "_available_tasks", lambda: CACHED)
    return adapter


def test_a_matching_configured_label_needs_no_note(cached) -> None:
    """Nothing was discarded, so nothing needs explaining."""

    label, note = cached._resolve_task("pick up the cube and place it in the bowl",
                                       "pick up the cube and place it in the bowl")
    assert label == "pick up the cube and place it in the bowl"
    assert note is None


def test_an_overriding_configured_label_says_the_request_string_was_dropped(cached) -> None:
    """The case that produced the misreading: a caller's instruction is discarded.

    The note has to name both strings and say that the cause is this configuration
    rather than the model, because "the model substituted my prompt" and "the
    manifest overrode my prompt" are different facts with the same symptom.
    """

    request = "grasp five yellow blocks together and fill all ten slots"
    label, note = cached._resolve_task(request, "pick up the cube and place it in the bowl")

    assert label == "pick up the cube and place it in the bowl"
    assert note is not None, "an overridden request string must be reported"
    assert request in note, "the note must quote the string that was dropped"
    assert "pick up the cube and place it in the bowl" in note[:80]
    # And it must not read as a limitation of the model.
    assert "not a limit of the model" in note
    # It must state the real blocker, so nobody re-derives "it cannot".
    assert "45 GB" in note


def test_an_empty_request_string_needs_no_note(cached) -> None:
    """A caller who sent no instruction discarded nothing."""

    label, note = cached._resolve_task("", "fold shirt")
    assert label == "fold shirt"
    assert note is None


def test_a_configured_label_outside_the_cache_is_replaced_and_reported(cached) -> None:
    label, note = cached._resolve_task("anything", "put the kettle on")
    assert label == CACHED[0]
    assert note is not None
    assert "put the kettle on" in note


# ---------------------------------------------------------------------------
# allow_t5_encode: the cost guard has an off switch
# ---------------------------------------------------------------------------


def test_by_default_an_uncached_instruction_is_still_replaced(cached) -> None:
    """The guard stays on by default; nothing about it changed.

    Building T5-XXL inside a rollout is a 45 GB surprise, so the default has to
    keep refusing it.  This test exists so a later edit that flips the default
    cannot pass unnoticed.
    """

    label, note = cached._resolve_task("grasp five yellow blocks", "")
    assert label in CACHED
    assert label != "grasp five yellow blocks"
    assert note is not None
    # And the note must now name the way out.
    assert "allow_t5_encode" in note


def test_allow_encode_lets_an_uncached_instruction_through(cached) -> None:
    """The whole point: the pkl is a cache, so an unseen string is a valid input."""

    instruction = "grasp five yellow blocks together and fill all ten slots"
    label, note = cached._resolve_task(instruction, "", allow_encode=True)
    assert label == instruction, "the caller's own words must reach the model"
    assert note is not None
    assert instruction in note


def test_allow_encode_overrides_a_manifest_pin(cached) -> None:
    """A pin plus an un-cached request is contradictory; the flag wins and says so.

    Silently preferring the pin is exactly the confusion this module was written to
    stop repeating, so the precedence is asserted rather than left to whoever reads
    the code next.
    """

    instruction = "grasp five yellow blocks together and fill all ten slots"
    label, note = cached._resolve_task(
        instruction, "pick up the cube and place it in the bowl", allow_encode=True
    )
    assert label == instruction
    assert note is not None
    assert "allow_t5_encode" in note


def test_allow_encode_leaves_cached_labels_alone(cached) -> None:
    """With the flag on, a string the cache already holds must not be re-encoded.

    Otherwise a rollout would pay for T5-XXL while an identical embedding was
    already on disk -- and worse, the two would not be bit-identical.
    """

    label, note = cached._resolve_task("fold shirt", "", allow_encode=True)
    assert label == "fold shirt"
    assert note is None


def test_allow_encode_does_not_fire_on_an_empty_instruction(cached) -> None:
    """An empty request has nothing to encode, so the pin still applies."""

    label, note = cached._resolve_task("", "fold shirt", allow_encode=True)
    assert label == "fold shirt"
    assert note is None


def test_a_task_already_in_the_cache_is_used_as_is(cached) -> None:
    label, note = cached._resolve_task("fold shirt", "")
    assert label == "fold shirt"
    assert note is None


def test_case_and_spacing_are_matched_without_paying_for_an_encoder(cached) -> None:
    """A near-duplicate string must not trigger the encoder it does not need."""

    label, note = cached._resolve_task("Fold  Shirt", "")
    assert label == "fold shirt"
    assert note is not None
    assert "matched" in note


def test_an_unknown_task_is_replaced_and_the_reason_given(cached) -> None:
    label, note = cached._resolve_task("do the thing", "")
    assert label in CACHED
    assert note is not None
    assert "do the thing" in note
    # The note must point at the deliberate option rather than leaving the reader
    # to guess how to choose another label.
    assert "task=" in note


def test_without_a_cache_the_label_passes_through(cached, monkeypatch) -> None:
    """Before the cache loads there is nothing to substitute against."""

    monkeypatch.setattr(cached, "_available_tasks", lambda: ())
    label, note = cached._resolve_task("any string at all", "")
    assert label == "any string at all"
    assert note is not None


def test_the_manifest_pins_a_label_it_also_declares() -> None:
    """The configured label must be one of the labels the manifest advertises.

    If it were not, the adapter would silently fall back to ``CACHED[0]`` and the
    rollout would condition on "fold shirt" while the manifest claimed otherwise --
    the same class of failure as the note above, one level up.
    """

    manifest = load_manifest(COSMOS_MANIFEST)
    declared = list(manifest.evaluation.prompt_labels)
    assert declared, "the manifest no longer declares its label set"

    pinned = [
        item.split("=", 1)[1]
        for item in manifest.transport.command
        if isinstance(item, str) and item.startswith("task=")
    ]
    assert len(pinned) == 1, f"expected exactly one pinned label, found {pinned}"
    assert pinned[0] in declared, (
        f"the manifest pins {pinned[0]!r}, which is not among the labels it declares, "
        "so the adapter would substitute silently"
    )


def test_the_declared_labels_are_the_six_the_cache_holds() -> None:
    """Keep the declaration honest about which cache it was written for.

    The point is not that six is special.  It is that this manifest was measured
    against a specific set, and the label ranking recorded in it is only meaningful
    for that set -- so if the cache changes, the ranking has to be revisited.
    """

    declared = set(load_manifest(COSMOS_MANIFEST).evaluation.prompt_labels)
    assert declared == set(CACHED), (
        "the manifest's declared labels no longer match the cache this was measured "
        "against; the recorded ranking in the manifest's comments is about the old set"
    )
