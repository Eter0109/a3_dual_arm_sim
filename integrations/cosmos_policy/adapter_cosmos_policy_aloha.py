"""Cosmos Policy (ALOHA) adapter for the A3 bridge.

Lives in the Cosmos Policy checkout, not in the A3 repository, and imports
nothing from A3.  That is the whole arrangement: the model keeps its own
interpreter -- torch 2.7.0+cu128 here, which the A3 environment does not share --
and the two sides meet over the wire protocol.

This is the only file a Cosmos Policy integration needs.

It reuses Cosmos Policy's own ``PolicyServer`` rather than reimplementing model
loading and observation handling.  The HTTP layer is unused: the server object is
built and its request handler called directly, so the code path is theirs and only
the transport is ours.

Serve it with::

    python /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \\
        --adapter adapters/cosmos_policy_aloha.py \\
        --config ckpt=nvidia/Cosmos-Policy-ALOHA-Predict2-2B

The A3 side declares the joint mapping in a manifest; see
``configs/policies/cosmos_policy_aloha.yaml``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

#: The ALOHA fine-tune used in the paper: 185 demos over four real-world tasks.
#: There is no other ALOHA inference-only config in the released tree, so this is a
#: constant rather than a knob -- a wrong value fails while importing the model,
#: not silently.
ALOHA_CONFIG = (
    "cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_"
    "foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only"
)

#: ALOHA: six joints plus one gripper per arm.
ALOHA_DIM = 14

#: Labels the released embedding cache was built from.  Cosmos Policy looks the
#: task string up in a precomputed dictionary; a string that is not in it makes the
#: model try to build a T5-11B encoder on the spot, which is a 45 GB download and
#: not something a rollout should discover.  Read from the cache at load time
#: rather than hardcoded -- an earlier version of this adapter guessed
#: "fold the shirt" against a cache that says "fold shirt", and the mismatch
#: surfaced as that download.
_FALLBACK_TASKS: tuple[str, ...] = ()


def _available_tasks() -> tuple[str, ...]:
    """Task labels present in the precomputed T5 embedding cache.

    Read through the module rather than by ``from ... import``: the cache is a
    module-level dict that ``init_t5_text_embeddings_cache`` fills in place, so a
    bound name imported earlier would still be the empty dict it started as.
    """

    from cosmos_policy.experiments.robot import cosmos_utils

    cache = getattr(cosmos_utils, "t5_text_embeddings_cache", None)
    if isinstance(cache, dict):
        return tuple(str(key) for key in cache)
    return ()


def _resolve_task(
    task: str, configured: str = "", allow_encode: bool = False
) -> tuple[str, str | None]:
    """Pick a label the cache actually contains.  Returns ``(label, note)``.

    A configured label wins outright: the manifest knows which task the model is
    being pointed at, and guessing at inference time would make the rollout depend
    on something invisible.  Otherwise the A3 task string is matched, exactly first
    and then case- and whitespace-insensitively, because a caller writing
    "Fold  Shirt" means the same task and should not pay for a 45 GB encoder to
    find that out.

    When a configured label overrides a *different* A3 string, that is reported as a
    note.  The override is the one place where a caller's instruction is discarded
    rather than mapped, and it used to happen silently: ``cosmos_task`` did report
    the label actually used, but nothing said the caller's string had been ignored,
    so a rollout that asked for one thing and conditioned on another looked exactly
    like a model that refused to accept a prompt.  Saying it out loud costs nothing
    and removes the ambiguity.

    ``allow_encode`` turns off the substitution described above and lets an
    uncached instruction reach the model's own encoder.  It exists because the
    substitution is a *cost* guard, not a correctness one: the pkl is a cache that
    computes and writes back on a miss, so an uncached string is a supported input
    -- it just needs T5-XXL (45 GB) present, and building that encoder inside a
    rollout is not something anyone should discover by accident.  With the encoder
    already on disk, refusing the instruction becomes a limitation we impose on
    ourselves rather than one the model has, so the choice is exposed instead of
    being made silently either way.

    It defaults to off, and it overrides a configured pin rather than bending to
    it.  A pin plus an un-cached request is contradictory -- the pin says "use this
    label", the flag says "use the string I sent" -- and the failure mode of
    silently preferring the pin is precisely the confusion this module exists to
    stop repeating.  The note says which of the two was used.
    """

    available = _available_tasks()
    if not available:
        return configured or task, "no T5 cache is loaded, so the label is passed through"

    if allow_encode and task and task not in available:
        return (
            task,
            f"task {task!r} is not in the T5 cache, and allow_t5_encode is set, so "
            "it is passed through for the model's T5-XXL encoder to compute. The "
            "first request that does this is slow and needs the encoder on disk; "
            "the result is written back to the pkl, so later rollouts are cached.",
        )

    if configured:
        if configured in available:
            if task and task != configured:
                # The manifest pins the label, so this request's instruction never
                # reaches the model.  Report it instead of leaving the caller to
                # diff two fields to find out.
                return (
                    configured,
                    f"the manifest pins task={configured!r}, so this request's task "
                    f"string {task!r} was not sent. That is a configuration choice, "
                    "not a limit of the model: the interface accepts any string (and "
                    "a raw embedding array), but a label outside the cached six "
                    "needs T5-11B, a 45 GB download.",
                )
            # No note: cosmos_task already reports which label was used, and the
            # note field is reserved for things that need explaining.
            return configured, None
        known = sorted(available)
        return (
            available[0],
            f"the configured label {configured!r} is not in the T5 cache {known}; "
            f"substituted {available[0]!r}",
        )

    if task in available:
        return task, None

    normalised = " ".join(task.split()).lower()
    for candidate in available:
        if " ".join(candidate.split()).lower() == normalised:
            return candidate, f"matched {task!r} to cache label {candidate!r}"

    # Nothing matched.  Sending an unknown label is what triggers the T5 download,
    # so substitute a real one and say so, along with how to choose deliberately.
    # No semantic guess is made here: picking "the closest" of six unrelated
    # demonstrations would look like a decision while resting on nothing.
    chosen = _FALLBACK_TASKS[0] if _FALLBACK_TASKS else available[0]
    return (
        chosen,
        f"task {task!r} is not in the T5 cache {sorted(available)}; substituted "
        f"{chosen!r} rather than triggering a T5-11B download. Pass "
        "--config allow_t5_encode=true to let the model encode it instead, or "
        "--config task=<label> to choose deliberately.",
    )


def _require_aloha_platform() -> None:
    """Refuse to run unless Cosmos Policy resolved its ALOHA constants.

    ``cosmos_policy.constants`` chooses its action and proprio widths by searching
    ``sys.argv`` for "libero", "robocasa" or "aloha" -- and **defaults to LIBERO**
    when none appears.  LIBERO is 7-dimensional, so a bridge launched without the
    word "aloha" anywhere in its command line would load this checkpoint against
    LIBERO constants and produce wrong-shaped actions instead of an error.

    Checking here turns that silent mismatch into a startup failure.  The manifest
    guarantees the argv condition by passing ``platform=aloha``.
    """

    from cosmos_policy import constants

    if constants.ROBOT_PLATFORM != "ALOHA":
        raise RuntimeError(
            f"Cosmos Policy resolved ROBOT_PLATFORM={constants.ROBOT_PLATFORM!r} "
            f"(ACTION_DIM={constants.ACTION_DIM}), but the ALOHA checkpoint needs "
            f"{ALOHA_DIM}. The platform is detected from sys.argv and LIBERO is the "
            "fallback when no platform name appears, so the launch command must "
            "contain one -- the manifest does this via --config platform=aloha."
        )
    if constants.ACTION_DIM != ALOHA_DIM or constants.PROPRIO_DIM != ALOHA_DIM:
        raise RuntimeError(
            f"ALOHA constants are ACTION_DIM={constants.ACTION_DIM} "
            f"PROPRIO_DIM={constants.PROPRIO_DIM}; both must be {ALOHA_DIM}"
        )


def _alongside(checkpoint: str, filename: str) -> str:
    """Locate a companion file next to a checkpoint, however the checkpoint was named.

    ``ckpt_path`` may be a Hub repo id (``nvidia/Cosmos-Policy-ALOHA-Predict2-2B``),
    a single-file checkpoint (``.../Cosmos-Policy-ALOHA-Predict2-2B.pt``), or a
    directory.  Cosmos Policy's own defaults assume the first, so a local ``.pt``
    path would resolve to ``<file>.pt/aloha_dataset_statistics.json`` and fail the
    existence assertion.  Deriving the directory here makes both forms work, which
    matters because the local form is what a downloaded checkpoint looks like.
    """

    base = Path(checkpoint)
    if base.suffix == ".pt":
        base = base.parent
    if base.is_dir():
        candidate = base / filename
        if candidate.is_file():
            return str(candidate)
    # Not a local directory: treat it as a Hub repo id and let the caller's
    # downloader resolve it, as the upstream default does.
    return f"{checkpoint}/{filename}"


def load(config: dict[str, Any]) -> dict[str, Any]:
    """Build the policy server.  Called once, before the first request."""

    _require_aloha_platform()

    from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig, PolicyServer

    checkpoint = config.get("ckpt", "nvidia/Cosmos-Policy-ALOHA-Predict2-2B")
    chunk_size = int(config.get("chunk_size", 50))

    cfg = DeployConfig(
        suite="aloha",
        model_family="cosmos",
        config=config.get("config", ALOHA_CONFIG),
        ckpt_path=checkpoint,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=config.get(
            "dataset_stats", _alongside(checkpoint, "aloha_dataset_statistics.json")
        ),
        t5_text_embeddings_path=config.get(
            "t5_embeddings", _alongside(checkpoint, "aloha_t5_embeddings.pkl")
        ),
        # ALOHA: one third-person view plus two wrist views, which is exactly the
        # three cameras A3 supplies.  This is why this checkpoint is the right one
        # to try first -- the LIBERO and RoboCasa variants share fewer inputs.
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=2,
        use_proprio=True,
        normalize_proprio=True,
        # Physical units out, so the manifest declares units: raw and applies no
        # inverse.  With this False the numbers on the wire would be normalised and
        # the manifest would have to carry matching statistics -- the silent
        # mismatch class the manifest exists to prevent.
        unnormalize_actions=True,
        trained_with_image_aug=True,
        chunk_size=chunk_size,
        num_open_loop_steps=chunk_size,
        # ALOHA frames are stored upright; only the LIBERO renderer flips them.
        flip_images=False,
        use_jpeg_compression=False,
        num_denoising_steps_action=int(config.get("denoising_steps", 10)),
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        deterministic=True,
        seed=int(config.get("seed", 195)),
        # Planning (best-of-N over the value function) is off: it needs a second
        # fine-tuned checkpoint and multiplies inference cost by the query count.
        # Turning it on is a manifest change once the base path is known to work.
        num_queries_best_of_n=1,
        use_parallel_inference=False,
    )

    server = PolicyServer(cfg)
    return {
        "server": server,
        "cfg": cfg,
        "calls": 0,
        "checkpoint": checkpoint,
        "resize_size": getattr(cfg, "resize_size", None),
        # Which of the checkpoint's own task labels to use.  Cosmos Policy's
        # language input is a precomputed embedding per known label, so an A3 task
        # string that is not one of those labels has no embedding -- and asking for
        # one triggers a 45 GB T5-11B download instead of an error.
        "task_label": str(config.get("task", "")),
        # Off by default.  With it on, an instruction the cache has never seen is
        # handed to the model's own T5-XXL encoder instead of being replaced by a
        # cached label.  That is the model's documented behaviour on a cache miss
        # (``get_t5_embedding_from_cache`` computes and writes back), but the
        # encoder is 45 GB, so the caller asks for it explicitly rather than
        # getting it by surprise.
        "allow_t5_encode": str(config.get("allow_t5_encode", "")).strip().lower()
        in ("1", "true", "yes", "on"),
    }


def infer(model: dict[str, Any], request: Any) -> dict[str, Any]:
    """One inference: A3 observation in, one action chunk out."""

    proprio = request.vector("proprio")
    if proprio.shape != (ALOHA_DIM,):
        raise ValueError(
            f"Cosmos Policy ALOHA expects {ALOHA_DIM}-dimensional proprio (six joints "
            f"plus a gripper per arm); the request carried {proprio.shape}.  Check "
            "observation.vectors.proprio.indices in the A3 manifest: A3 has seven "
            "joints per arm, so two of its sixteen values have no ALOHA counterpart "
            "and must be dropped rather than appended."
        )

    task, note = _resolve_task(
        request.task,
        str(model.get("task_label", "")),
        allow_encode=bool(model.get("allow_t5_encode", False)),
    )
    # The payload matches Cosmos Policy's own /act endpoint field for field, so the
    # handler below is exercised exactly as a real ALOHA client would exercise it,
    # including its dtype coercions.
    payload = {
        "primary_image": request.image("primary_image"),
        "left_wrist_image": request.image("left_wrist_image"),
        "right_wrist_image": request.image("right_wrist_image"),
        "proprio": proprio,
        "task_description": task,
    }

    started = time.monotonic()
    response = model["server"].get_server_action(payload)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    model["calls"] += 1

    # ``get_server_action`` is a FastAPI handler and returns a JSONResponse.  Its
    # body is already serialised through json_numpy, which is what lets numpy
    # arrays survive a plain json.dumps; parsing it back gives plain lists.
    body = getattr(response, "body", response)
    decoded = json.loads(body if isinstance(body, (bytes, str)) else json.dumps(body))
    if "actions" not in decoded:
        raise ValueError(
            f"Cosmos Policy returned no actions; the response had {sorted(decoded)}.  "
            "Its server logs the traceback before returning an error body."
        )

    actions = np.asarray(decoded["actions"], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[1] != ALOHA_DIM:
        raise ValueError(
            f"the policy returned {actions.shape[1]}-dimensional actions; {ALOHA_DIM} "
            "was expected.  This usually means LIBERO constants were resolved instead "
            "of ALOHA."
        )

    info: dict[str, Any] = {
        "cosmos_calls": model["calls"],
        "cosmos_task": task,
        "cosmos_checkpoint": model["checkpoint"],
        "infer_ms": round(elapsed_ms, 2),
        "resize_size": model["resize_size"],
    }
    if note:
        # Carried into the episode result, because a substituted prompt changes
        # what the policy was asked to do and must not be invisible.
        info["cosmos_task_note"] = note
    # Cosmos Policy's own estimate of expected return.  Recorded because it is free
    # and because it is perfectly reproducible (identical across repeats when the
    # sampling seed is pinned), so it costs nothing to keep.
    #
    # Do not read it as a quality signal, though.  Measured over the six cached
    # labels on one fixed observation: the value spans 0.189 to 0.861 across labels
    # while the scene does not change at all, so it mostly reports *which words were
    # sent* rather than what the robot is facing.  And its ordering does not match
    # behaviour: "put purple eggplant on plate" carries the highest value (0.861)
    # and only middling behaviour, while "put candies in bowl" (0.229) sits within
    # 0.002 of the label this manifest uses (0.231) and behaves 1.5x worse on the
    # cookie scene.  The manifest's ``success.probe: env`` remains the verdict.
    value = decoded.get("value_prediction")
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if flat.size:
            info["cosmos_value"] = float(flat[0])

    return {
        "actions": actions,
        "space": "joint_position",
        # unnormalize_actions=True above, so these are physical joint angles.
        "units": "raw",
        "info": info,
    }
