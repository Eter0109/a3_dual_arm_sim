"""One adapter for any external VLA or world-model project.

Everything that differs between models is data in a manifest; this class is the
only code, and it does four mechanical things:

1. **project** the observation onto the keys the manifest declares, failing loudly
   on a missing one rather than letting a model run blind;
2. **translate** the returned actions into the A3 contract, in the order
   denormalise -> rearrange -> translate;
3. **pace** the chunk against the environment's control rate;
4. **own the lifecycle** -- connect on the first episode, reset per episode,
   release at the end.

It never imports a model's framework and never names one, which is what lets the
same code drive a MuJoCo-adjacent LeRobot checkpoint and a 14-billion-parameter
world model on another host.

The runner reads two optional duck-typed attributes from a policy and both are
provided here: ``metrics`` (merged into the episode info, and therefore into the
dataset's own metadata) and ``success_override`` (a model evaluating the task
itself, when the manifest asks for it).
"""

from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..contracts import ActionMode, EpisodeContext
from .action_bridge import ActionBridge
from .chunk import ChunkBuffer
from .manifest import (
    ManifestError,
    PolicyManifest,
    dropped_cameras,
    load_manifest,
    validate_manifest,
)
from .observation import ObservationView
from .protocol import (
    ActionChunk,
    BridgeError,
    Request,
)
from .transport import InProcessHandler, Transport, build_transport


def resolve_handler(spec: str, env: Any = None) -> InProcessHandler:
    """Import ``module:factory`` and call it to obtain an in-process handler.

    Mirrors :func:`a3_dual_arm_sim.policy.load_policy`, so an in-process model is
    loaded the same way a local policy is.
    """

    if ":" not in spec:
        raise BridgeError(
            f"in-process handler must use module:factory syntax, got {spec!r}"
        )
    module_name, factory_name = spec.rsplit(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
        raise BridgeError(f"could not import handler {spec!r}: {exc}") from exc
    try:
        return factory(env) if _accepts_env(factory) else factory()  # type: ignore[no-any-return]
    except TypeError:
        return factory()  # type: ignore[no-any-return]


def _accepts_env(factory: Any) -> bool:
    import inspect

    try:
        return "env" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


class ExternalPolicyGateway:
    """A3 policy protocol over any manifest-described external model."""

    def __init__(
        self,
        manifest: PolicyManifest,
        *,
        env: Any = None,
        transport: Transport | None = None,
        handler: InProcessHandler | None = None,
        keep_transport: bool = False,
    ) -> None:
        self.manifest = manifest
        self._env = env
        target_mode: ActionMode = getattr(env, "action_mode", "joint_position")
        self.action_mode: ActionMode = target_mode

        self._view = ObservationView(manifest.observation)
        self._bridge = ActionBridge(
            manifest.action, target_mode=target_mode, env=env
        )
        self._chunk = ChunkBuffer(manifest.action.chunk, manifest.action.execute)
        self._chunk.expected_dim = manifest.action.dim

        built_here = transport is None
        if built_here:
            if manifest.transport.kind == "in_process" and handler is None:
                if not manifest.transport.command:
                    raise BridgeError(
                        "transport.kind is in_process but transport.command names no "
                        "handler; use a single module:factory entry"
                    )
                handler = resolve_handler(manifest.transport.command[0], env)
            transport = build_transport(manifest.transport, handler=handler)
        self._transport = transport
        #: Whether :meth:`close` may shut the transport down.  A transport handed in
        #: from outside belongs to its caller, and one built here belongs to this
        #: gateway -- unless the caller says it is keeping it, which the comparison
        #: loop does: it reuses one gateway across episodes so the model process
        #: survives them, and a transport closed at the end of the first episode
        #: would take the model down with it.  The last user calls :meth:`shutdown`.
        self._owns_transport = built_here and not keep_transport
        self._dropped = dropped_cameras(manifest)

        self._episode_id = ""
        self._step = 0
        self._last_info: dict[str, Any] = {}
        self._last_chunk: ActionChunk | None = None
        self._stop_requested = False
        self._connected = False
        self._predicted_rows = 0
        #: Model-load cost this episode actually paid.  Zero for every episode after
        #: the first, because the process is started once and reused.  Reported
        #: separately from the transport's cumulative total so a per-episode table
        #: does not average one 110 s load across every episode and then present the
        #: result as the cost of one -- which is what it did while the value was
        #: cumulative, and it made the reuse look like it had not happened.
        self._startup_s = 0.0

    # -- Policy protocol ----------------------------------------------------

    def reset(self, context: EpisodeContext) -> None:
        first = not self._connected
        if first:
            self._transport.connect()
            self._connected = True
        # The transport times its own connect, and that timer is cumulative: it holds
        # the one load this whole comparison paid.  Which episode paid it is what a
        # per-episode reading needs, so it is attributed here.
        self._startup_s = float(self._transport.report.startup_s) if first else 0.0
        self._episode_id = uuid.uuid4().hex[:12]
        self._step = 0
        self._last_info = {}
        self._last_chunk = None
        self._stop_requested = False
        self._predicted_rows = 0
        self._chunk.reset()
        # Freeze any joint the model cannot command at the pose this episode starts
        # from, so those dimensions stay inert instead of following dynamic coupling.
        self._bridge.latch_hold_values()
        self._transport.reset(self._episode_id)

    def act(self, observation: Mapping[str, Any], task: str) -> np.ndarray:
        action = self._next_action()
        if action is None:
            self._replan(observation, task, reason="queue empty")
            action = self._next_action()
            if action is None:
                raise BridgeError(
                    "the model returned a chunk but no action could be taken from it"
                )
        self._step += 1
        return action

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()
            # Only a transport this gateway closed has stopped being connected.  A
            # shared one is still up, and saying otherwise would make the next reset
            # believe it was starting the model -- which restarts nothing, because
            # ``connect`` is idempotent, but does misattribute the load cost to an
            # episode that did not pay it.
            self._connected = False

    def shutdown(self) -> None:
        """Release the transport even when it is shared; the last owner calls this."""

        self._transport.close()
        self._connected = False

    def rebind(self, env: Any) -> None:
        """Point this gateway at a new environment for the next episode.

        The comparison loop builds one environment per episode -- scenes randomise on
        reset, and reusing one would carry contact state between realisations -- while
        wanting a single model process for all of them.  Everything else the gateway
        holds per episode is cleared by :meth:`reset`, so only the environment
        reference is replaced here.  The bridge is rebuilt rather than patched because
        it caches values derived from the environment, the cartesian adapter's model
        among them, and those have to come from the new one.
        """

        target_mode: ActionMode = getattr(env, "action_mode", "joint_position")
        if target_mode != self.action_mode:
            raise BridgeError(
                f"{self.manifest.name} was built for a {self.action_mode!r} environment "
                f"and was handed a {target_mode!r} one; rebinding cannot change the "
                "action space the model's commands are interpreted in"
            )
        self._env = env
        self._bridge = ActionBridge(self.manifest.action, target_mode=target_mode, env=env)

    # -- duck-typed attributes the runner reads ------------------------------

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "bridge": {
                "manifest": self.manifest.name,
                "transport": self._transport.kind,
                **self._transport.report.to_dict(),
                # After the spread, so this episode's share of the load cost replaces
                # the cumulative figure rather than sitting beside it.
                "startup_s": round(self._startup_s, 3),
                "startup_total_s": round(float(self._transport.report.startup_s), 3),
                "dropped_cameras": list(self._dropped),
                "action_space": self.manifest.action.space,
                "action_dim": self.manifest.action.dim,
                "predicted_rows": self._predicted_rows,
            },
            **self._chunk.metrics(),
            **{key: value for key, value in self._last_info.items() if key != "success"},
        }

    @property
    def success_override(self) -> bool | None:
        """A model's own verdict, only when the manifest asks for it."""

        if self.manifest.success.probe != "policy":
            return None
        if "success" not in self._last_info:
            return None
        return bool(self._last_info["success"])

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    # -- internals ----------------------------------------------------------

    def _next_action(self) -> np.ndarray | None:
        row = self._chunk.pop()
        if row is None:
            return None
        return self._bridge.convert_row(row, self._simulation_data())

    def _simulation_data(self) -> Any:
        """The live ``MjData``, which only a cross-space translation needs."""

        if self._bridge.needs_env:
            if self._env is None:
                raise BridgeError(
                    "this manifest translates a model's cartesian actions into joint "
                    "positions, which is solved against the live model; construct the "
                    "gateway with the environment"
                )
            return self._env.data
        return None

    def _replan(self, observation: Mapping[str, Any], task: str, *, reason: str) -> None:
        arrays, _ = self._view.project(observation, task)
        request = Request(
            episode_id=self._episode_id,
            step=self._step,
            task=task,
            arrays=arrays,
            timestamp=_timestamp(observation),
            dropped_images=self._dropped,
            chunk=self.manifest.action.chunk,
        )
        chunk = self._transport.infer(request)
        self._accept(chunk, reason=reason)
        self._chunk.load(chunk.actions)
        self._predicted_rows += chunk.horizon

    def _accept(self, chunk: ActionChunk, *, reason: str) -> None:
        """Check a chunk's self-description against the manifest before using it.

        A model that reports a different space or a normalisation the manifest did
        not declare would otherwise be executed as if it matched, which is exactly
        the class of silent error this design exists to prevent.
        """

        if chunk.space != self.manifest.action.space:
            raise BridgeError(
                f"the model returned {chunk.space!r} actions but the manifest declares "
                f"{self.manifest.action.space!r}; fix the manifest, not the model"
            )
        if chunk.units != "raw" and self.manifest.action.normalize.mode == "none":
            raise BridgeError(
                f"the model reports {chunk.units!r}-scaled actions but the manifest "
                "declares no normalisation, so they would be executed as physical "
                "values. Add action.normalize with the statistics the model used."
            )
        self._last_chunk = chunk
        self._last_info = dict(chunk.info)
        if bool(chunk.info.get("stop", False)):
            self._stop_requested = True


def _timestamp(observation: Mapping[str, Any]) -> float | None:
    value = observation.get("time") if hasattr(observation, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI factories
# ---------------------------------------------------------------------------


def make_policy(env: Any = None) -> ExternalPolicyGateway:
    """CLI factory: ``--policy a3_dual_arm_sim.bridge.gateway:make_policy``.

    Configured entirely through ``A3_POLICY_MANIFEST``, so switching models is
    changing one path.  Declaring the ``env`` parameter means the CLI builds the
    environment first, which is what a cartesian-delta bridge needs.
    """

    path = os.environ.get("A3_POLICY_MANIFEST")
    if not path:
        raise RuntimeError(
            "Set A3_POLICY_MANIFEST to a policy manifest path, for example "
            "configs/policies/cosmos_policy.yaml"
        )
    manifest = load_manifest(path)
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError(
            f"{path} is not usable:\n  - " + "\n  - ".join(problems)
        )
    return ExternalPolicyGateway(manifest, env=env)


def make_handler(env: Any = None) -> InProcessHandler:
    """Factory for an in-process bridge server.

    A model that shares this interpreter registers itself here; the manifest then
    names this factory in ``transport.command``.  Kept declarative so no manifest
    ever needs a module path that only exists on one machine.
    """

    raise BridgeError(
        "no in-process models are registered in this repository yet. Use "
        "transport.kind subprocess for an external project, or transport.kind replay "
        "to exercise the pipeline without a model."
    )


class GatewayCache:
    """Keep one gateway -- and so one loaded model -- alive across episodes.

    A comparison runs several seeds per manifest and several manifests per command.
    Starting a model process costs about 110 s for the pi0.5 bridge -- most of a
    400-step episode -- so the process is started once and reused, with
    :meth:`ExternalPolicyGateway.rebind` pointing it at each new episode's
    environment.

    **One model at a time, deliberately.**  ``compare`` runs each manifest to
    completion before starting the next, so a manifest the cache has not seen means
    the previous one is finished, and holding it would keep a second loaded model on
    the device for the whole rest of the run.  Measured at 10 GB of a 32 GB card for
    one pi0.5 bridge, which turns a two-manifest comparison into an out-of-memory
    error on a smaller card.  Releasing it restores the peak to one model.

    The teardown is explicit rather than left to garbage collection because the
    process is the thing being managed: a leaked handle keeps a child process alive,
    and the child holds GPU memory whether or not anyone still refers to it.
    """

    def __init__(self, factory: Any) -> None:
        #: Called as ``factory(manifest, env)`` to build a gateway.  Injected so the
        #: lifecycle can be exercised without loading a model.
        self._factory = factory
        self._gateways: dict[str, ExternalPolicyGateway] = {}

    def gateway_for(self, manifest: PolicyManifest, env: Any) -> ExternalPolicyGateway:
        """The gateway for this manifest, built on first use and reused after."""

        if manifest.name not in self._gateways and self._gateways:
            self.shutdown()
        gateway = self._gateways.get(manifest.name)
        if gateway is None:
            gateway = self._factory(manifest, env)
            self._gateways[manifest.name] = gateway
        else:
            gateway.rebind(env)
        return gateway

    @property
    def names(self) -> tuple[str, ...]:
        """Which manifests currently hold a model, for assertions and diagnostics."""

        return tuple(self._gateways)

    def shutdown(self) -> None:
        """Release every held gateway and the process behind it."""

        for gateway in self._gateways.values():
            gateway.shutdown()
        self._gateways.clear()


__all__ = [
    "ExternalPolicyGateway",
    "GatewayCache",
    "make_handler",
    "make_policy",
    "resolve_handler",
]
