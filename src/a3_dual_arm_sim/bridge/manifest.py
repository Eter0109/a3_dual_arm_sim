"""Manifest parsing and audit for external policies.

A manifest is the only thing that should have to change when a new model is
added.  Everything a model needs -- which interpreter runs it, which cameras it
sees, what its action space is -- is data here, so the gateway never learns a
model's name.

``load_manifest`` parses; ``validate_manifest`` audits.  They are separate
because the audit has to be usable on a manifest whose model is not installed:
``a3-sim policy check`` answers "is this wiring coherent" without loading
anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..contracts import (
    CARTESIAN_ACTION_DIM,
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    JOINT_ACTION_DIM,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
)
from .protocol import ACTION_SPACES, NORMALIZE_MODES

#: Observation keys an A3 environment can actually supply.  A manifest naming
#: anything else is a wiring error, not a model limitation.
A3_OBSERVATION_KEYS = frozenset(
    {
        FRONT_IMAGE,
        LEFT_WRIST_IMAGE,
        RIGHT_WRIST_IMAGE,
        STATE,
        VELOCITY,
        EEF_POSE,
        FORCE,
        "time",
        "safety_stop",
    }
)

#: A3 supplies three cameras; a manifest that wants fewer gets them dropped, and
#: the drop is recorded so a comparison can report which models saw what.
A3_CAMERA_KEYS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)

TRANSPORT_KINDS = ("in_process", "subprocess", "socket", "replay")
RUNTIME_KINDS = ("local", "conda", "docker")

#: Action-space widths the A3 side speaks natively.
NATIVE_WIDTH = {"joint_position": JOINT_ACTION_DIM, "cartesian_delta": CARTESIAN_ACTION_DIM}


class ManifestError(ValueError):
    """A manifest is malformed or cannot be used."""


@dataclass(frozen=True)
class VectorSpec:
    """One non-image observation vector: which A3 key feeds it, and how much.

    ``slice`` and ``indices`` are alternatives, and ``indices`` is the one that
    matters for real checkpoints.  A model trained on a robot with a different
    joint count needs a non-contiguous selection, not a window: ALOHA has six
    joints plus a gripper per arm where A3 has seven plus a gripper, so feeding
    an ALOHA checkpoint means dropping A3's seventh wrist joint from each arm
    (indices 6 and 14) rather than taking a range.
    """

    source: str
    #: Inclusive-exclusive index range applied to the source vector.
    slice: tuple[int, int] | None = None
    #: Explicit index selection, for sources whose layout differs from the
    #: model's. Mutually exclusive with ``slice``.
    indices: tuple[int, ...] | None = None
    dtype: str = "float32"


@dataclass(frozen=True)
class ImageSpec:
    """One camera: which A3 key feeds it, and the size the model expects."""

    source: str
    #: ``(height, width)``; ``None`` keeps the environment's own resolution.
    resize: tuple[int, int] | None = None


@dataclass(frozen=True)
class ObservationSpec:
    #: Model-visible name -> camera.  Order is the order arrays are carried in.
    images: Mapping[str, ImageSpec] = field(default_factory=dict)
    #: Model-visible name -> vector.
    vectors: Mapping[str, VectorSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class AffineSpec:
    """A per-dimension ``scale * x + offset`` applied to the model's own values.

    This is the unit conversion, and it is declared rather than coded because it
    is the quietest error in the whole adapter: a model whose actions are in
    different units produces motion of the right *shape* at the wrong *size*, and
    nothing about it looks like a bug.

    The motivating case is GR00T.  Its bimanual actions are normalized deltas in
    robosuite's OSC units -- ``±1`` means ``±0.05 m`` and ``±0.5 rad`` per step --
    while A3's ``cartesian_delta`` is normalized against its own
    ``cartesian_translation_scale_m`` (0.025) and ``cartesian_rotation_scale_rad``
    (0.10).  So one A3 unit is twice GR00T's translation and five times its
    rotation.  Its gripper is also inverted in sense: GR00T's ``+1`` closes and
    A3's ``+1`` opens, which is a negative scale plus an offset, not a rescale.
    """

    #: Multipliers, one per model dimension.  ``None`` means all ones.
    scale: tuple[float, ...] | None = None
    #: Addends applied after the scale.  ``None`` means all zeros.
    offset: tuple[float, ...] | None = None

    @property
    def is_identity(self) -> bool:
        """Whether this transform would change anything."""

        scaled = self.scale is None or all(value == 1.0 for value in self.scale)
        shifted = self.offset is None or all(value == 0.0 for value in self.offset)
        return scaled and shifted

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Transform a row of model values."""

        out = np.asarray(values, dtype=np.float64)
        if self.scale is not None:
            out = out * np.asarray(self.scale, dtype=np.float64)
        if self.offset is not None:
            out = out + np.asarray(self.offset, dtype=np.float64)
        return out


@dataclass(frozen=True)
class NormalizeSpec:
    """How to undo a model's own preprocessing.

    ``stats`` names an ``.npz``/``.json`` file holding per-dimension statistics
    under the model-visible action key.  A mode other than ``none`` without a
    stats file is a manifest error, never a silent pass-through.
    """

    mode: str = "none"
    stats: Path | None = None
    #: Key to read inside the stats file; defaults to ``"action"``.
    key: str = "action"


@dataclass(frozen=True)
class ActionSpec:
    #: The space the model emits.
    space: str = "joint_position"
    #: The width the model emits.  Equal to the native width unless ``joint_map``
    #: rearranges it.
    dim: int = JOINT_ACTION_DIM
    #: Actions returned per inference call.
    chunk: int = 1
    #: Steps executed before the next inference call; ``chunk`` means open loop.
    execute: int = 1
    #: Physical interval between consecutive actions, when the model's training
    #: frequency differs from the environment's control rate.
    dt: float | None = None
    #: ``joint_map[i]`` is the destination index in the native vector of model
    #: dimension ``i``.  Only same-semantics rearrangements belong here; a model
    #: with a different joint count needs an adapter, not a map.
    joint_map: tuple[int, ...] | None = None
    #: Value written to native dimensions a model does not control.  A float is a
    #: constant; the string ``"hold"`` keeps whatever the joint currently reads,
    #: which is the only safe choice when the model was never trained to move that
    #: joint and the constant would be an arbitrary command.
    default: float | str = 0.0
    #: Inclusive bounds applied after normalisation, when the model is known to
    #: overshoot.
    clip: tuple[float, float] | None = None
    #: Unit conversion applied to the model's own values, before ``clip`` and
    #: before ``joint_map``.
    affine: AffineSpec = field(default_factory=AffineSpec)
    normalize: NormalizeSpec = field(default_factory=NormalizeSpec)


@dataclass(frozen=True)
class SuccessSpec:
    """Where the episode verdict comes from.

    ``env`` is the default because it keeps every model on one ruler.  ``policy``
    lets a model that carries its own criterion report that instead, and the
    runner already merges both into ``final_info`` so a disagreement is visible.
    """

    probe: str = "env"


#: How the model was obtained, which decides what a comparison may claim.
EVALUATION_REGIMES = ("zero_shot", "finetuned")
#: Which measurement the comparison should lead with.
PRIMARY_METRICS = ("behavior", "task", "both")
#: How much freedom the model has over the language instruction.
PROMPT_KINDS = ("free_text", "fixed_label", "none")


@dataclass(frozen=True)
class EvaluationSpec:
    """What a rollout of this model is allowed to be compared on.

    Present because the models on the sizing guide's list were trained on other
    robots and other tasks -- none on A3 -- so a bare success rate cannot separate
    "this model cannot do it" from "this model was never taught it".  Declaring the
    regime is what keeps those two apart, and it is a property of the weights rather
    than of any rollout, so it belongs in the manifest.
    """

    #: ``zero_shot`` for released weights, ``finetuned`` for weights trained on A3.
    regime: str = "zero_shot"
    #: Which family of numbers the comparison should lead with.  Under
    #: ``zero_shot`` the task metrics are expected to be near zero and ``behavior``
    #: is the informative one; ``task`` is for weights that were taught the task.
    primary_metric: str = "behavior"
    #: ``free_text`` when an arbitrary instruction reaches the model, ``fixed_label``
    #: when it can only look the instruction up in a table, ``none`` when language
    #: is not an input at all.
    prompt_kind: str = "free_text"
    #: The labels a ``fixed_label`` model was trained on.  Recorded because the
    #: choice of label measurably changes the output (Cosmos Policy's self-reported
    #: value moved 0.076 between two of its own labels), so a rollout that does not
    #: say which label it used cannot be reproduced.
    prompt_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeSpec:
    kind: str = "local"
    #: conda environment name.
    env: str | None = None
    #: docker image reference.
    image: str | None = None
    #: Extra mounts or flags are the caller's business; kept out on purpose so a
    #: manifest cannot become a shell script.


@dataclass(frozen=True)
class TransportSpec:
    kind: str = "subprocess"
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    #: Working directory for a subprocess bridge; relative paths resolve against
    #: the manifest's own directory.
    cwd: Path | None = None
    command: tuple[str, ...] = ()
    #: ``tcp://host:port`` or ``unix:///path`` for a socket bridge.
    url: str | None = None
    startup_timeout_s: float = 300.0
    #: Per-request deadline; a slower model needs a larger value, not a retry.
    request_timeout_s: float = 30.0
    #: Environment variables for the child process.
    env: Mapping[str, str] = field(default_factory=dict)
    #: ``replay`` transport source: a JSON file of recorded actions.
    replay_path: Path | None = None


@dataclass(frozen=True)
class PolicyManifest:
    name: str
    transport: TransportSpec
    observation: ObservationSpec
    action: ActionSpec
    success: SuccessSpec = field(default_factory=SuccessSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    #: Notes carried into the rollout report, so an episode can be read without
    #: the manifest file.
    notes: str = ""
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable summary, used by ``policy check`` and the report."""

        return {
            "name": self.name,
            "transport": {
                "kind": self.transport.kind,
                "runtime": self.transport.runtime.kind,
                "env": self.transport.runtime.env,
                "command": list(self.transport.command),
                "cwd": str(self.transport.cwd) if self.transport.cwd else None,
                "url": self.transport.url,
            },
            "observation": {
                "images": {
                    name: {
                        "source": spec.source,
                        "resize": list(spec.resize) if spec.resize else None,
                    }
                    for name, spec in self.observation.images.items()
                },
                "vectors": {
                    name: {
                        "source": spec.source,
                        "slice": list(spec.slice) if spec.slice else None,
                        "indices": list(spec.indices) if spec.indices else None,
                        "dtype": spec.dtype,
                    }
                    for name, spec in self.observation.vectors.items()
                },
            },
            "action": {
                "space": self.action.space,
                "dim": self.action.dim,
                "chunk": self.action.chunk,
                "execute": self.action.execute,
                "dt": self.action.dt,
                "joint_map": list(self.action.joint_map) if self.action.joint_map else None,
                "default": self.action.default,
                "affine": {
                    "scale": list(self.action.affine.scale) if self.action.affine.scale else None,
                    "offset": (
                        list(self.action.affine.offset) if self.action.affine.offset else None
                    ),
                },
                "normalize": {
                    "mode": self.action.normalize.mode,
                    "stats": (
                        str(self.action.normalize.stats) if self.action.normalize.stats else None
                    ),
                },
            },
            "success": {"probe": self.success.probe},
            "evaluation": {
                "regime": self.evaluation.regime,
                "primary_metric": self.evaluation.primary_metric,
                "prompt_kind": self.evaluation.prompt_kind,
                "prompt_labels": list(self.evaluation.prompt_labels),
            },
            "notes": self.notes,
            "source_path": str(self.source_path) if self.source_path else None,
        }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _relative_to(path: Path, base: Path | None) -> Path:
    """Expand a manifest path.  Bare relative paths resolve against ``base``."""

    expanded = Path(path).expanduser()
    if expanded.is_absolute() or base is None:
        return expanded
    return (base / expanded).resolve()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _float_pair(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ManifestError(f"{where} must be a two-element list")
    return (float(value[0]), float(value[1]))


def _int_pair(value: Any, where: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ManifestError(f"{where} must be a two-element list")
    return (int(value[0]), int(value[1]))


def _parse_observation(raw: Mapping[str, Any], base: Path | None) -> ObservationSpec:
    images: dict[str, ImageSpec] = {}
    for name, entry in _mapping(raw.get("images"), "observation.images").items():
        entry = _mapping(entry, f"observation.images.{name}")
        source = entry.get("source")
        if not source:
            raise ManifestError(f"observation.images.{name} needs a source key")
        resize = entry.get("resize")
        images[str(name)] = ImageSpec(
            source=str(source),
            resize=_int_pair(resize, f"observation.images.{name}.resize") if resize else None,
        )

    vectors: dict[str, VectorSpec] = {}
    for name, entry in _mapping(raw.get("vectors"), "observation.vectors").items():
        entry = _mapping(entry, f"observation.vectors.{name}")
        source = entry.get("source")
        if not source:
            raise ManifestError(f"observation.vectors.{name} needs a source key")
        window = entry.get("slice")
        chosen = entry.get("indices")
        if window and chosen:
            raise ManifestError(
                f"observation.vectors.{name} declares both slice and indices; they are "
                "alternatives, and allowing both would make the order of operations a "
                "convention rather than a statement"
            )
        vectors[str(name)] = VectorSpec(
            source=str(source),
            slice=_int_pair(window, f"observation.vectors.{name}.slice") if window else None,
            indices=(
                tuple(int(index) for index in chosen)
                if chosen
                else None
            ),
            dtype=str(entry.get("dtype", "float32")),
        )
    if not images and not vectors:
        raise ManifestError("observation declares neither images nor vectors")
    return ObservationSpec(images=images, vectors=vectors)


def _parse_normalize(raw: Mapping[str, Any], base: Path | None) -> NormalizeSpec:
    mode = str(raw.get("mode", "none"))
    if mode not in NORMALIZE_MODES:
        raise ManifestError(
            f"action.normalize.mode must be one of {', '.join(NORMALIZE_MODES)}, got {mode!r}"
        )
    stats = raw.get("stats")
    return NormalizeSpec(
        mode=mode,
        stats=_relative_to(stats, base) if stats else None,
        key=str(raw.get("key", "action")),
    )


def _parse_default(raw: Any, where: str) -> float | str:
    """A constant, or the literal ``hold`` for "leave this joint alone"."""

    if raw is None:
        return 0.0
    if isinstance(raw, str):
        if raw != "hold":
            raise ManifestError(
                f"{where} must be a number or the string 'hold', got {raw!r}"
            )
        return raw
    return float(raw)


def _float_list(value: Any, where: str) -> tuple[float, ...] | None:
    """A list of floats, or ``None`` when absent.

    A bare scalar is accepted and broadcast by the caller, because "scale
    everything by 2" is a real and common statement.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{where} must be a number or a list of numbers")
    return tuple(float(item) for item in value)


def _parse_affine(raw: Any, where: str, dim: int) -> AffineSpec:
    """Parse the per-dimension unit conversion, checking its width here.

    The width matters: a short ``scale`` list would broadcast against the action
    row in numpy and silently rescale the wrong dimensions, or raise from inside
    the bridge mid-rollout.
    """

    if raw is None:
        return AffineSpec()
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where} must be a mapping with 'scale' and/or 'offset'")
    scale = _float_list(raw.get("scale"), f"{where}.scale")
    offset = _float_list(raw.get("offset"), f"{where}.offset")
    # A single value stands for "the same for every dimension".
    if scale is not None and len(scale) == 1:
        scale = scale * dim
    if offset is not None and len(offset) == 1:
        offset = offset * dim
    if scale is not None and len(scale) != dim:
        raise ManifestError(
            f"{where}.scale has {len(scale)} values but action.dim is {dim}; either "
            "give one value per dimension or a single value to apply to all"
        )
    if offset is not None and len(offset) != dim:
        raise ManifestError(
            f"{where}.offset has {len(offset)} values but action.dim is {dim}; either "
            "give one value per dimension or a single value to apply to all"
        )
    return AffineSpec(scale=scale, offset=offset)


def _parse_action(raw: Mapping[str, Any], base: Path | None) -> ActionSpec:
    space = str(raw.get("space", "joint_position"))
    if space not in ACTION_SPACES:
        raise ManifestError(
            f"action.space must be one of {', '.join(ACTION_SPACES)}, got {space!r}. "
            "A model emitting another space needs an adapter on its own side."
        )
    joint_map = raw.get("joint_map")
    clip = raw.get("clip")
    chunk = int(raw.get("chunk", 1))
    if chunk < 1:
        raise ManifestError("action.chunk must be at least 1")
    dim = int(raw.get("dim", NATIVE_WIDTH[space]))
    return ActionSpec(
        space=space,
        dim=dim,
        chunk=chunk,
        execute=int(raw.get("execute", chunk)),
        dt=None if raw.get("dt") is None else float(raw["dt"]),
        joint_map=tuple(int(index) for index in joint_map) if joint_map else None,
        default=_parse_default(raw.get("default"), "action.default"),
        clip=_float_pair(clip, "action.clip") if clip else None,
        affine=_parse_affine(raw.get("affine"), "action.affine", dim),
        normalize=_parse_normalize(_mapping(raw.get("normalize"), "action.normalize"), base),
    )


def _parse_transport(raw: Mapping[str, Any], base: Path | None) -> TransportSpec:
    kind = str(raw.get("kind", "subprocess"))
    if kind not in TRANSPORT_KINDS:
        raise ManifestError(
            f"transport.kind must be one of {', '.join(TRANSPORT_KINDS)}, got {kind!r}"
        )
    runtime_raw = _mapping(raw.get("runtime"), "transport.runtime")
    runtime = RuntimeSpec(
        kind=str(runtime_raw.get("kind", "local")),
        env=runtime_raw.get("env"),
        image=runtime_raw.get("image"),
    )
    if runtime.kind not in RUNTIME_KINDS:
        raise ManifestError(
            f"transport.runtime.kind must be one of {', '.join(RUNTIME_KINDS)}, "
            f"got {runtime.kind!r}"
        )
    raw_command = raw.get("command")
    if isinstance(raw_command, str):
        raise ManifestError(
            "transport.command must be a list, not a string: a shell line would make "
            "the manifest a script and lose argument boundaries"
        )
    command = tuple(str(part) for part in (raw_command or ()))
    cwd = raw.get("cwd")
    replay = raw.get("replay_path")
    return TransportSpec(
        kind=kind,
        runtime=runtime,
        cwd=_relative_to(cwd, base) if cwd else None,
        command=command,
        url=raw.get("url"),
        startup_timeout_s=float(raw.get("startup_timeout_s", 300.0)),
        request_timeout_s=float(raw.get("request_timeout_s", 30.0)),
        env={
            str(key): str(value)
            for key, value in _mapping(raw.get("env"), "transport.env").items()
        },
        replay_path=_relative_to(replay, base) if replay else None,
    )


def _parse_evaluation(raw: Mapping[str, Any]) -> EvaluationSpec:
    """Parse the ``evaluation`` block, refusing values outside the vocabularies.

    Each field is a vocabulary rather than a free string because the comparison
    report switches on them: an unrecognised regime would silently fall into
    whichever branch the formatter happens to have as a default, and mixing
    zero-shot with fine-tuned results is exactly the mistake the block exists to
    prevent.
    """

    regime = str(raw.get("regime", "zero_shot"))
    if regime not in EVALUATION_REGIMES:
        raise ManifestError(
            f"evaluation.regime must be one of {', '.join(EVALUATION_REGIMES)}, got {regime!r}"
        )
    primary = str(raw.get("primary_metric", "behavior"))
    if primary not in PRIMARY_METRICS:
        raise ManifestError(
            f"evaluation.primary_metric must be one of {', '.join(PRIMARY_METRICS)}, "
            f"got {primary!r}"
        )
    prompt = str(raw.get("prompt_kind", "free_text"))
    if prompt not in PROMPT_KINDS:
        raise ManifestError(
            f"evaluation.prompt_kind must be one of {', '.join(PROMPT_KINDS)}, got {prompt!r}"
        )
    labels = raw.get("prompt_labels") or ()
    if isinstance(labels, str):
        raise ManifestError(
            "evaluation.prompt_labels must be a list, not a string: a single label is "
            "written as a one-element list"
        )
    return EvaluationSpec(
        regime=regime,
        primary_metric=primary,
        prompt_kind=prompt,
        prompt_labels=tuple(str(label) for label in labels),
    )


def load_manifest(path: str | Path) -> PolicyManifest:
    """Parse a manifest file.  Raises :class:`ManifestError` on malformed input."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{manifest_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{manifest_path} must contain a mapping at the top level")

    base = manifest_path.parent.resolve()
    name = raw.get("name")
    if not name:
        raise ManifestError("manifest needs a name")
    success_raw = _mapping(raw.get("success"), "success")
    probe = str(success_raw.get("probe", "env"))
    if probe not in ("env", "policy", "none"):
        raise ManifestError(f"success.probe must be env, policy or none, got {probe!r}")
    return PolicyManifest(
        name=str(name),
        transport=_parse_transport(_mapping(raw.get("transport"), "transport"), base),
        observation=_parse_observation(_mapping(raw.get("observation"), "observation"), base),
        action=_parse_action(_mapping(raw.get("action"), "action"), base),
        success=SuccessSpec(probe=probe),
        evaluation=_parse_evaluation(_mapping(raw.get("evaluation"), "evaluation")),
        notes=str(raw.get("notes", "")),
        source_path=manifest_path.resolve(),
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def validate_manifest(manifest: PolicyManifest) -> list[str]:
    """Return the problems that stop this manifest from running.

    The audit is deliberately split from parsing so it can run without the model:
    every check here is answerable from the manifest and the A3 side alone, and
    each one is a failure that would otherwise surface only mid-rollout.
    """

    problems: list[str] = []
    action = manifest.action
    native = NATIVE_WIDTH[action.space]

    if action.joint_map is not None:
        if len(action.joint_map) != action.dim:
            problems.append(
                f"action.joint_map has {len(action.joint_map)} entries but action.dim is "
                f"{action.dim}; the map must describe every dimension the model emits"
            )
        elif min(action.joint_map) < 0 or max(action.joint_map) >= native:
            problems.append(
                f"action.joint_map entries must index the {native}-dimensional "
                f"{action.space} vector, got {sorted(action.joint_map)}. A model that only "
                "drives part of the robot maps the dimensions it does control; the rest "
                "are filled with action.default."
            )
        elif len(set(action.joint_map)) != len(action.joint_map):
            problems.append(
                f"action.joint_map repeats a destination index: {sorted(action.joint_map)}; "
                "two model outputs cannot both own the same native dimension"
            )
    elif action.dim != native:
        problems.append(
            f"action.dim is {action.dim} but {action.space} is {native}-dimensional. "
            "Either set dim to the native width, or declare joint_map to rearrange "
            "the model's output onto it. A model with a different joint count needs "
            "an adapter on its own side, because index remapping cannot change semantics."
        )

    if action.execute > action.chunk:
        problems.append(
            f"action.execute ({action.execute}) exceeds action.chunk ({action.chunk}); "
            "the environment would run out of predicted actions mid-chunk"
        )
    if action.execute < 1 or action.chunk < 1:
        problems.append("action.chunk and action.execute must both be at least 1")

    if action.normalize.mode != "none":
        if action.normalize.stats is None:
            problems.append(
                f"action.normalize.mode is {action.normalize.mode!r} but no stats file is "
                "declared; without it the raw values would be used as if physical"
            )
        elif not action.normalize.stats.is_file():
            problems.append(f"action.normalize.stats not found: {action.normalize.stats}")
    if action.joint_map is not None and action.default == "hold":
        # ``hold`` needs the live joint positions, which only a joint-position
        # environment has; a cartesian model fills nothing.
        uncovered = set(range(native)) - set(action.joint_map)
        if uncovered and action.space != "joint_position":
            problems.append(
                f"action.default is 'hold' but {action.space} is not a joint space; "
                "holding a measured joint value is only defined for joint actions"
            )
    if action.clip is not None and action.clip[0] >= action.clip[1]:
        problems.append(f"action.clip lower bound {action.clip[0]} is not below {action.clip[1]}")

    if action.space != "cartesian_delta" and not action.affine.is_identity:
        # A unit conversion only means something together with the space it
        # converts into; on a joint space it would look like a calibration and
        # silently rescale joint angles.
        problems.append(
            "action.affine is set for a joint_position model. Unit conversion belongs "
            "to a cartesian space, where the scale is a property of the model's "
            "controller rather than of the robot; for joints, use action.normalize."
        )
    if action.clip is not None and action.space == "cartesian_delta" and action.clip != (-1.0, 1.0):
        problems.append(
            f"action.clip is {action.clip} but {action.space} values must lie in "
            "[-1, 1]; clipping to anything else still produces out-of-contract actions"
        )

    for name, spec in manifest.observation.images.items():
        if spec.source not in A3_OBSERVATION_KEYS:
            problems.append(
                f"observation.images.{name}.source is {spec.source!r}, which an A3 "
                f"observation cannot supply. Available: {sorted(A3_OBSERVATION_KEYS)}"
            )
        elif spec.resize is not None and (spec.resize[0] <= 0 or spec.resize[1] <= 0):
            problems.append(f"observation.images.{name}.resize must be positive")
    for name, spec in manifest.observation.vectors.items():
        if spec.source not in A3_OBSERVATION_KEYS:
            problems.append(
                f"observation.vectors.{name}.source is {spec.source!r}, which an A3 "
                f"observation cannot supply. Available: {sorted(A3_OBSERVATION_KEYS)}"
            )
        elif spec.slice is not None and spec.slice[0] >= spec.slice[1]:
            problems.append(
                f"observation.vectors.{name}.slice {spec.slice} is empty or reversed"
            )
        if spec.indices is not None:
            if sorted(spec.indices) != list(spec.indices):
                problems.append(
                    f"observation.vectors.{name}.indices must be ascending, got "
                    f"{list(spec.indices)}; the model's joint order is fixed by its "
                    f"checkpoint, and a permutation here would be silent"
                )
            elif len(set(spec.indices)) != len(spec.indices):
                problems.append(f"observation.vectors.{name}.indices repeats an index")
            elif min(spec.indices) < 0:
                problems.append(f"observation.vectors.{name}.indices has a negative entry")

    transport = manifest.transport
    if transport.kind == "subprocess":
        if not transport.command:
            problems.append("transport.kind is subprocess but transport.command is empty")
        if transport.cwd is not None and not transport.cwd.is_dir():
            problems.append(f"transport.cwd is not a directory: {transport.cwd}")
        if transport.runtime.kind == "conda" and not transport.runtime.env:
            problems.append("transport.runtime.kind is conda but no environment name is set")
        if transport.runtime.kind == "docker" and not transport.runtime.image:
            problems.append("transport.runtime.kind is docker but no image is set")
    elif transport.kind == "socket":
        if not transport.url:
            problems.append("transport.kind is socket but no url is set")
        elif not str(transport.url).startswith(("tcp://", "unix://")):
            problems.append(
                f"transport.url must start with tcp:// or unix://, got {transport.url!r}"
            )
    elif transport.kind == "replay":
        if transport.replay_path is None:
            problems.append("transport.kind is replay but no replay_path is set")
        elif not transport.replay_path.is_file():
            problems.append(f"transport.replay_path not found: {transport.replay_path}")

    evaluation = manifest.evaluation
    if evaluation.prompt_kind == "fixed_label" and not evaluation.prompt_labels:
        problems.append(
            "evaluation.prompt_kind is 'fixed_label' but no prompt_labels are listed.  "
            "A table-driven model can only be driven by a label it was trained on, so "
            "the set has to be recorded or the rollout cannot be reproduced -- the "
            "choice of label measurably changes the output."
        )
    if evaluation.prompt_kind == "none" and evaluation.prompt_labels:
        problems.append(
            "evaluation.prompt_kind is 'none' but prompt_labels are listed; language is "
            "either an input to this model or it is not"
        )
    if evaluation.regime == "zero_shot" and evaluation.primary_metric == "task":
        problems.append(
            "evaluation.primary_metric is 'task' for a zero_shot model.  Released weights "
            "were trained on other robots and other tasks, so a task metric is expected "
            "to be near zero and cannot separate 'not taught' from 'cannot learn'.  "
            "Lead with 'behavior' instead, or declare regime: finetuned."
        )

    return problems


def dropped_cameras(manifest: PolicyManifest) -> tuple[str, ...]:
    """A3 cameras this manifest does not consume.

    Recorded rather than discarded because dropping a view changes what a model
    can possibly do, and a comparison that ignored that would read a wiring
    difference as a capability difference.
    """

    used = {spec.source for spec in manifest.observation.images.values()}
    return tuple(key for key in A3_CAMERA_KEYS if key not in used)


def declared_native_width(manifest: PolicyManifest) -> int:
    return NATIVE_WIDTH[manifest.action.space]


def describe_manifest(manifest: PolicyManifest) -> dict[str, Any]:
    """Manifest summary plus the derived facts a rollout report should carry."""

    payload = manifest.to_dict()
    payload["derived"] = {
        "dropped_cameras": list(dropped_cameras(manifest)),
        "native_action_width": declared_native_width(manifest),
        "problems": validate_manifest(manifest),
    }
    return payload


def as_structured(array: Any, dtype: str = "float32") -> np.ndarray:
    """Coerce a manifest-declared vector to the wire dtype."""

    return np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
