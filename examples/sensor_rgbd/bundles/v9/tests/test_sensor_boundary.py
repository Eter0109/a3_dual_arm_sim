import unittest
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
from a3_continuability.sensor_boundary import (
    CAMERAS, IMAGE_KEYS, ExecutionPacket, SensorInputError, act_from_packet,
    policy_observation,
    needs_sensor_refresh,
)


def raw_observation():
    return {**{key: np.zeros((8, 8, 3), np.uint8) for key in IMAGE_KEYS},
            "observation.state": np.arange(16, dtype=float)/20,
            "time": 2., "object_poses": "MUST NOT ENTER", "contacts": [123],
            "cookies_in_target": 20, "observation.force": np.ones(18)}


def packet(raw=None, **kwargs):
    frames = SimpleNamespace(frames=tuple(SimpleNamespace(timestamp=2.) for _ in CAMERAS))
    return ExecutionPacket.create(raw or raw_observation(), frames,
        {name: (np.zeros(3), np.eye(3), 60.) for name in CAMERAS},
        timestamp=2., gripper_position_measured=True, **kwargs)


class RecordingController:
    def __init__(self):
        self.received = []

    def act(self, obs, calibrations, **kwargs):
        self.received.append((obs, calibrations, kwargs))
        return obs["observation.state"]+.001


class SensorBoundaryTests(unittest.TestCase):
    def test_phase_transition_cannot_skip_stationary_calibration_on_odd_frame(self):
        self.assertTrue(needs_sensor_refresh(101,0,'LOCATE','LOCATE'))
        self.assertTrue(needs_sensor_refresh(101,0,'VERIFY','VERIFY'))
        self.assertTrue(needs_sensor_refresh(101,2,'HELD','LIFT'))
        self.assertFalse(needs_sensor_refresh(101,3,'TRANSPORT','TRANSPORT'))

    def test_hidden_truth_poison_cannot_change_action_or_enter_controller(self):
        raw = raw_observation()
        first = RecordingController()
        a = act_from_packet(first, packet(raw), now=2.)
        raw.update(object_poses=np.full((500, 7), np.nan), contacts=None,
                   cookies_in_target=-123, native_success=False)
        raw["observation.force"] = np.full(18, np.nan)
        second = RecordingController()
        b = act_from_packet(second, packet(raw), now=2.)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(first.received[0][0]), {*IMAGE_KEYS, "observation.state"})
        self.assertEqual(set(second.received[0][0]), set(first.received[0][0]))
        self.assertEqual(a.shape, (8,))

    def test_arrays_are_copied_and_original_is_not_modified(self):
        raw = raw_observation()
        result = policy_observation(raw, gripper_position_measured=True)
        raw["observation.state"][:] = 99
        raw[IMAGE_KEYS[0]][:] = 99
        self.assertLess(result["observation.state"].max(), 1)
        self.assertEqual(result[IMAGE_KEYS[0]].max(), 0)
        with self.assertRaises(ValueError):
            result["observation.state"][0] = 50
        with self.assertRaises(TypeError):
            result["contacts"] = []

    def test_unknown_gripper_cannot_be_replaced_by_command_or_zero(self):
        with self.assertRaisesRegex(SensorInputError, "Measured gripper"):
            policy_observation(raw_observation(), gripper_position_measured=False)

    def test_missing_camera_or_bad_state_cannot_silently_fallback(self):
        raw = raw_observation()
        raw["observation.state"][3] = np.nan
        with self.assertRaises(SensorInputError):
            policy_observation(raw, gripper_position_measured=True)
        raw = raw_observation()
        del raw[IMAGE_KEYS[2]]
        with self.assertRaises(KeyError):
            policy_observation(raw, gripper_position_measured=True)

    def test_stale_and_future_samples_do_not_call_controller(self):
        controller = RecordingController()
        for now in (1.9, 2.101, float("nan")):
            with self.assertRaises(SensorInputError):
                act_from_packet(controller, packet(), now=now)
        self.assertEqual(controller.received, [])

    def test_unsynchronized_robot_and_depth_rejected(self):
        frames = SimpleNamespace(frames=tuple(SimpleNamespace(timestamp=t) for t in (2., 2., 2.02)))
        with self.assertRaisesRegex(SensorInputError, "Unsynchronized"):
            ExecutionPacket.create(raw_observation(), frames,
                {n: (np.zeros(3), np.eye(3), 60.) for n in CAMERAS},
                timestamp=2., gripper_position_measured=True)

    def test_right_arm_and_nonfinite_actions_rejected(self):
        class Bad:
            def act(self, *args, **kwargs):
                return np.zeros(16)
        with self.assertRaises(SensorInputError):
            act_from_packet(Bad(), packet(), now=2.)
        class Nonfinite:
            def act(self, *args, **kwargs):
                return np.full(8, np.nan)
        with self.assertRaises(SensorInputError):
            act_from_packet(Nonfinite(), packet(), now=2.)

    def test_missing_release_feedback_is_not_invented(self):
        controller = RecordingController()
        act_from_packet(controller, packet(), now=2.)
        self.assertIsNone(controller.received[0][2]["jaw_gap_m"])
        self.assertIsNone(controller.received[0][2]["pad_midpoint_world"])

    def test_calibration_is_copied_and_invalid_rotation_rejected(self):
        p = packet()
        with self.assertRaises(ValueError):
            p.calibrations["front"][0][0] = 20
        with self.assertRaises(SensorInputError):
            ExecutionPacket.create(raw_observation(), p.rgbd,
                {n: (np.zeros(3), np.zeros((3, 3)), 60.) for n in CAMERAS},
                timestamp=2., gripper_position_measured=True)

    def test_legacy_model_queue_kept_but_right_arm_and_truth_excluded(self):
        from a3_continuability.left8_executor import Left8Executor
        calls, resets = [], []
        class Tensor:
            def __init__(self, value):
                self.value = value
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        def batch(obs, task):
            self.assertEqual(obs["observation.state"].shape, (8,))
            self.assertEqual(set(obs), {*IMAGE_KEYS, "observation.state"})
            calls.append(task)
            return obs
        queue = [np.arange(8), np.arange(8)+1]
        adapter = SimpleNamespace(config=object(), pre=lambda x: x, post=lambda x: x,
            policy=SimpleNamespace(select_action=lambda x: Tensor(queue.pop(0))),
            reset=lambda context: resets.append(context))
        executor = Left8Executor(adapter)
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "pi05_adapter": SimpleNamespace(observation_batch=batch),
        }):
            np.testing.assert_equal(executor.act(raw_observation(), "batch"), np.arange(8))
            np.testing.assert_equal(executor.act(raw_observation(), "batch"), np.arange(8)+1)
        self.assertEqual(calls, ["batch", "batch"])
        self.assertEqual(resets, [])  # no accidental policy reset every frame
        executor.reset("new_batch")
        self.assertEqual(resets, ["new_batch"])


if __name__ == "__main__":
    unittest.main()
