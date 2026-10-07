"""Turn a model's action into the A3 contract.

Three things happen here, in this order, and the order matters:

1. **denormalise** -- model units to physical units, using the declared stats;
2. **rearrange** -- the model's own dimension order onto the native vector;
3. **translate** -- a non-native action space onto the native one.

Doing (3) before (2) would apply an inverse-kinematics solve to a vector whose
dimensions are still in the model's order, which is the kind of mistake that
produces plausible motion and wrong results.  Each step is a separate method so
a caller can test it on its own.

The environment's own ``CartesianDeltaAdapter`` does the space translation, so the
14-to-16 dimensional mapping and its IK are shared with teleoperation rather than
reimplemented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    CARTESIAN_ACTION_DIM,
    JOINT_ACTION_DIM,
    ActionMode,
    ContractError,
    validate_action,
)
from .manifest import NATIVE_WIDTH, ActionSpec
from .normalize import Normalizer
from .protocol import BridgeError


class ActionBridge:
    """Convert model-native action rows into the environment's action space."""

    def __init__(
        self,
        spec: ActionSpec,
        *,
        target_mode: ActionMode,
        env: Any = None,
    ) -> None:
        self._spec = spec
        self._target_mode = target_mode
        self._env = env
        self._normalizer = Normalizer(spec.normalize)
        self._joint_map = (
            None if spec.joint_map is None else np.asarray(spec.joint_map, dtype=np.int64)
        )
        self._clip = None if spec.clip is None else np.asarray(spec.clip, dtype=np.float64)
        #: Filled by ``latch_hold_values`` at episode start; see ``_held_native``.
        self._latched: np.ndarray | None = None

        if spec.space == "cartesian_delta" and target_mode == "joint_position":
            if env is None:
                raise BridgeError(
                    "converting cartesian_delta actions into joint positions needs the live "
                    "environment: the IK is solved against its model. Declare it as a "
                    "privileged policy (the CLI already passes the environment to factories "
                    "that accept one)."
                )
            # Imported here rather than at module scope because it pulls in MuJoCo,
            # and the rest of this package must stay importable on a host whose CPU
            # cannot run it (see the note in the package ``__init__``).
            from ..actions import CartesianDeltaAdapter
            from ..osc import OperationalSpaceAdapter

            # robosuite's own controller for a scene that uses robosuite's robot: a
            # joint-position servo does not survive that swap, measured both ways round --
            # see `osc.OperationalSpaceAdapter`.
            if getattr(env.config, "left_arm", "a3") == "panda":
                # **Share the environment's controller, do not build a second one.**  The
                # adapter latches a goal, and the environment's substep loop re-computes the
                # arm's torques for *its* adapter's goal every 2 ms -- which is the whole
                # point of the 500 Hz loop.  Two adapters would mean two goals, and the
                # environment's (latched at reset) would silently overwrite every command
                # this bridge converts: measured, the arm did not move at all, 0.00 mm for a
                # command worth 25.
                shared = getattr(env, "ik", None)
                if isinstance(shared, OperationalSpaceAdapter):
                    self._ik = shared
                else:
                    self._ik = OperationalSpaceAdapter(
                        env.model, env.config, site="gripper0_grip_site"
                    )
                    env.ik = self._ik
            else:
                self._ik = CartesianDeltaAdapter(env.model, env.config)
            if spec.dim != CARTESIAN_ACTION_DIM:
                raise BridgeError(
                    f"cartesian_delta actions are {CARTESIAN_ACTION_DIM}-dimensional, but the "
                    f"manifest declares {spec.dim}"
                )
        elif spec.space == "joint_position" and target_mode == "cartesian_delta":
            raise BridgeError(
                "a joint-position model cannot drive a cartesian_delta environment: the "
                "forward mapping needs the current pose, and the environment applies the IK "
                "to its own action space. Run this model against a joint_position environment."
            )
        else:
            self._ik = None

        if (
            spec.space == "joint_position"
            and spec.dim != JOINT_ACTION_DIM
            and self._joint_map is None
        ):
            raise BridgeError(
                f"a joint_position model emitting {spec.dim} values cannot fill the "
                f"{JOINT_ACTION_DIM}-dimensional native vector without joint_map"
            )

    @property
    def needs_env(self) -> bool:
        return self._ik is not None

    @property
    def normalized(self) -> bool:
        return self._normalizer.mode != "none"

    @property
    def holds_uncontrolled_joints(self) -> bool:
        """Whether unclaimed native dimensions keep a measured value.

        A3's home pose happens to have zero wrist roll, so a constant would work
        for the ALOHA mapping by coincidence.  Relying on that coincidence would
        break the moment either robot's zero changed, so ``hold`` reads the joint
        instead of assuming.
        """

        return self._spec.joint_map is not None and self._holds

    def latch_hold_values(self) -> np.ndarray | None:
        """Freeze the held joints at their present positions.

        Called from the gateway's ``reset``.  Latching at the start of an episode,
        rather than re-reading every step, is what makes the held joints actually
        held: commanding each joint to wherever it currently is means following it
        as dynamic coupling pushes it around, and a joint meant to be inert ends up
        drifting tens of degrees over an episode.  Measured on the ALOHA mapping,
        the two unclaimed wrist joints wandered to -0.10 and +0.09 rad within 32
        steps when re-read per step; freezing them keeps the only joints the model
        cannot command from confounding a comparison.
        """

        if not self._holds:
            return None
        self._latched = self._measure_native(NATIVE_WIDTH[self._spec.space])
        return self._latched.copy()

    def _measure_native(self, width: int) -> np.ndarray:
        """Read the environment's current action-space vector.

        ``getattr`` with a sentinel rather than ``hasattr``: the environment
        exposes this as a property, and ``hasattr`` runs the getter, so probing and
        then reading takes two measurements.  Two reads in a row look harmless and
        are not -- the latch would store the later one, which is the same mistake
        this latch exists to avoid, just smaller.
        """

        if self._env is None:
            raise BridgeError(
                "action.default is 'hold', which leaves unclaimed joints at a measured "
                "position; that needs the live environment, so construct the gateway "
                "with one"
            )
        current = getattr(self._env, "current_joint_action", None)
        if current is None:
            raise BridgeError(
                "action.default is 'hold', which needs the environment's current joint "
                f"positions, but {type(self._env).__name__} does not report any"
            )
        value = np.asarray(current, dtype=np.float64).reshape(-1)
        if value.shape != (width,):
            raise BridgeError(
                f"the environment reports {value.shape[0]} joints but {self._spec.space} "
                f"is {width}-dimensional"
            )
        return value

    def _held_native(self, width: int) -> np.ndarray:
        """The values held for unclaimed dimensions, latching on first use.

        Latching here as well as in ``reset`` keeps a caller that drives the bridge
        directly -- a test, or an offline scorer -- from getting the drifting
        behaviour the latch exists to remove.
        """

        if self._latched is None:
            self._latched = self._measure_native(width)
        return self._latched.copy()

    def native_row(self, row: np.ndarray) -> np.ndarray:
        """Model row -> native-width row in the model's own action space.

        No MjData is taken here: the only state-dependent step is the cross-space
        IK, which happens later in :meth:`convert_row`.
        """

        values = np.asarray(row, dtype=np.float64).reshape(-1)
        expected = self._spec.dim
        if values.shape != (expected,):
            raise ContractError(
                f"model returned {values.shape[0]} action values, but the manifest declares "
                f"dim={expected}. Either the manifest is wrong or the model changed."
            )
        if not np.all(np.isfinite(values)):
            raise ContractError("action contains NaN or infinity")
        values = self._normalizer.denormalize(values)
        # Unit conversion sits between the two: it undoes the model's *controller*
        # convention, while the normalizer undoes its *statistics*.  Applying them
        # the other way round would rescale already-physical numbers.
        if not self._spec.affine.is_identity:
            values = self._spec.affine.apply(values)
        if self._clip is not None:
            values = np.clip(values, self._clip[0], self._clip[1])
        if self._joint_map is not None:
            width = NATIVE_WIDTH[self._spec.space]
            native = (
                self._held_native(width)
                if self._holds
                else np.full(width, float(self._spec.default), dtype=np.float64)
            )
            native[self._joint_map] = values
            values = native
        return values

    @property
    def _holds(self) -> bool:
        return self._spec.default == "hold"

    def convert_row(self, row: np.ndarray, data: Any = None) -> np.ndarray:
        """Model row -> one environment action.

        ``data`` is the live ``mujoco.MjData``, needed when the translation
        crosses action spaces or when unclaimed joints are held.  It is read per
        step rather than per chunk so both see the state the action is actually
        applied from.
        """

        native = self.native_row(row)
        if self._ik is None:
            return validate_action(native, self._target_mode)
        if data is None:
            raise BridgeError("cartesian_delta translation needs the environment's MjData")
        return self._ik.convert(data, native)

    def convert_chunk(self, chunk: np.ndarray, data: Any = None) -> np.ndarray:
        """Convert every row of a chunk, for tests and offline scoring."""

        values = np.asarray(chunk, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return np.stack([self.convert_row(row, data) for row in values])
