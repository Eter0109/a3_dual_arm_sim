"""Sharing one model process across a comparison's episodes.

Building a gateway per (manifest, seed) pair was the comparison loop's shape until it
was measured: the pi0.5 bridge spends about 110 s loading torch and four gigabytes of
weights before its first answer, which is most of a 400-step episode, and that cost was
paid again for every seed.  The transport -- the model process -- is now kept alive
across episodes while the environment is rebuilt, because scenes randomise on reset and
reusing one environment would carry contact state between realisations.

Two properties make that safe, and both are easy to get quietly wrong: closing an
episode must not close the shared process, and rebinding must rebuild everything the
gateway derived from the old environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway, GatewayCache
from a3_dual_arm_sim.bridge.manifest import load_manifest
from a3_dual_arm_sim.bridge.protocol import BridgeError
from a3_dual_arm_sim.bridge.transport import Transport, TransportSpec
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.env import A3DualArmEnv


class _SpyTransport(Transport):
    """A transport that records its lifecycle and never loads anything."""

    #: Pretend the model takes this long to load, so the attribution of that cost to
    #: an episode can be checked without loading four gigabytes of weights.
    LOAD_SECONDS = 12.5

    def __init__(self) -> None:
        super().__init__(TransportSpec(kind="spy"))
        self.connects = 0
        self.closes = 0
        self.resets: list[str] = []
        self._open = False

    @property
    def kind(self) -> str:
        return "spy"

    def connect(self) -> None:
        self.connects += 1
        self._open = True
        self.report.startup_s = self.LOAD_SECONDS

    def infer(self, request):  # pragma: no cover - never reached in these tests
        raise AssertionError("these tests do not ask the model for actions")

    def reset(self, episode_id: str) -> None:
        self.resets.append(episode_id)

    def close(self) -> None:
        self.closes += 1
        self._open = False


def _manifest(tmp_path: Path):
    payload = {
        "name": "spy-test",
        "transport": {"kind": "replay", "replay_path": "unused.json"},
        "observation": {"images": {"front": {"source": "observation.images.front", "resize": [32, 32]}},
                        "vectors": {"state": {"source": "observation.state"}}},
        "action": {"space": "joint_position", "dim": 16, "chunk": 8, "execute": 8},
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_manifest(path)


def test_a_shared_transport_survives_the_episode_it_belongs_to(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    env = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=env, transport=transport, keep_transport=True)
    try:
        gateway.reset(_pin(0))
        gateway.close()
        assert transport.closes == 0, (
            "closing an episode's gateway must not take the shared model process with it"
        )
    finally:
        gateway.shutdown()
        env.close()

    assert transport.closes == 1


def test_a_borrowed_transport_is_never_closed_by_the_gateway(tmp_path: Path) -> None:
    # No ``keep_transport``: the transport was handed in, so it belongs to the caller
    # and the gateway has no business shutting it down.
    manifest = _manifest(tmp_path)
    env = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=env, transport=transport)
    try:
        gateway.reset(_pin(0))
        gateway.close()
        gateway.shutdown()
    finally:
        env.close()

    assert transport.closes == 1, "shutdown is the only thing that closes it"


def test_only_the_first_reset_opens_the_process(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    env = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=env, transport=transport, keep_transport=True)
    try:
        for seed in (0, 1, 2):
            gateway.rebind(env)
            gateway.reset(_pin(seed))
        assert transport.connects == 1, "the process is opened once and then reused"
        assert len(transport.resets) == 3, "but every episode gets its own reset"
        assert len(set(transport.resets)) == 3, "with a fresh episode id each time"
    finally:
        gateway.shutdown()
        env.close()


def test_rebinding_swaps_the_environment_and_rebuilds_what_derived_from_it(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    first = A3DualArmEnv(render_cameras=False)
    second = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=first, transport=transport, keep_transport=True)
    try:
        gateway.rebind(second)
        assert gateway._env is second
        assert gateway._bridge._env is second, (
            "the action bridge caches values read off the environment -- the cartesian "
            "adapter's model among them -- so it has to be rebuilt, not just repointed"
        )
    finally:
        gateway.shutdown()
        first.close()
        second.close()


def test_rebinding_refuses_to_change_the_action_space(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    env = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=env, transport=transport, keep_transport=True)
    try:
        with pytest.raises(BridgeError, match="action space"):
            gateway.rebind(_DifferentActionSpace())
    finally:
        gateway.shutdown()
        env.close()


class _DifferentActionSpace:
    action_mode = "cartesian_delta"


def test_the_load_cost_is_charged_to_the_episode_that_paid_it(tmp_path: Path) -> None:
    """The transport times its own connect, once, and that total is cumulative.

    Read straight out of the transport it lands on every episode, so a per-episode
    table averages one load across all of them and reports the result as the cost of
    one -- which is exactly what happened, and it made the reuse look like it had not
    occurred because every episode showed the full 110 s.
    """

    manifest = _manifest(tmp_path)
    env = A3DualArmEnv(render_cameras=False)
    transport = _SpyTransport()
    gateway = ExternalPolicyGateway(manifest, env=env, transport=transport, keep_transport=True)
    try:
        # ``close`` stands in for the end of an episode: the comparison loop closes
        # the runner, then rebinds the gateway to the next environment.
        gateway.rebind(env)
        gateway.reset(_pin(0))
        gateway.close()
        first = gateway.metrics["bridge"]

        gateway.rebind(env)
        gateway.reset(_pin(1))
        gateway.close()
        second = gateway.metrics["bridge"]
    finally:
        gateway.shutdown()
        env.close()

    assert first["startup_s"] == _SpyTransport.LOAD_SECONDS
    assert second["startup_s"] == 0.0, "the second episode did not load anything"
    assert first["startup_total_s"] == second["startup_total_s"] == _SpyTransport.LOAD_SECONDS


def _pin(seed: int) -> EpisodeContext:
    return EpisodeContext(seed=seed, task="hold", action_mode="joint_position")


class _FakeGateway:
    """Stands in for a loaded model so the cache's lifecycle can be checked alone."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rebinds: list[object] = []
        self.shutdowns = 0

    def rebind(self, env: object) -> None:
        self.rebinds.append(env)

    def shutdown(self) -> None:
        self.shutdowns += 1


def _cache() -> tuple[GatewayCache, list[_FakeGateway]]:
    built: list[_FakeGateway] = []

    def factory(manifest, env):
        gateway = _FakeGateway(manifest.name)
        built.append(gateway)
        return gateway

    return GatewayCache(factory), built


class _NamedManifest:
    def __init__(self, name: str) -> None:
        self.name = name


def test_episodes_of_one_manifest_reuse_a_single_gateway() -> None:
    cache, built = _cache()
    manifest = _NamedManifest("model-a")

    first = cache.gateway_for(manifest, env="env-0")
    for env in ("env-1", "env-2"):
        assert cache.gateway_for(manifest, env=env) is first

    assert len(built) == 1, "the model is loaded once for all three episodes"
    assert first.rebinds == ["env-1", "env-2"], "each new episode gets the new environment"


def test_starting_a_second_manifest_releases_the_first_model() -> None:
    """The measurement this exists for: two loaded models do not fit side by side.

    A pi0.5 bridge holds 10 GB of a 32 GB card.  Holding the first manifest's model
    while the second runs put 20 GB on the device for the whole second half of every
    comparison -- which is survivable here and an out-of-memory error on a smaller
    card.  Releasing it restores the peak to one model.
    """

    cache, _built = _cache()
    first = cache.gateway_for(_NamedManifest("model-a"), env="env")
    cache.gateway_for(_NamedManifest("model-b"), env="env")

    assert first.shutdowns == 1, "the finished manifest's model must be released"
    assert cache.names == ("model-b",), "and only the running one is held"


def test_going_back_to_a_finished_manifest_loads_it_again() -> None:
    # Correct rather than wasteful: the model was released, so it has to be reloaded.
    # Nothing in this loop returns to a manifest after finishing it, so the reload is
    # the honest consequence of the release rather than a cost anyone actually pays.
    cache, built = _cache()
    cache.gateway_for(_NamedManifest("model-a"), env="env")
    cache.gateway_for(_NamedManifest("model-b"), env="env")
    again = cache.gateway_for(_NamedManifest("model-a"), env="env")

    assert len(built) == 3
    assert again.name == "model-a"
    assert built[-1] is again


def test_shutdown_releases_everything_and_is_idempotent() -> None:
    cache, built = _cache()
    cache.gateway_for(_NamedManifest("model-a"), env="env")

    cache.shutdown()
    cache.shutdown()

    assert built[0].shutdowns == 1, "a second shutdown has nothing left to release"
    assert cache.names == ()
