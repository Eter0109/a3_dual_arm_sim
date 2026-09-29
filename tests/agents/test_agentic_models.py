from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from collections import deque
from types import SimpleNamespace
from urllib import error

import numpy as np
import pytest
from PIL import Image

from a3_dual_arm_sim.agents.backends import (
    AgenticModelError,
    ChatBackendConfig,
    LocalQwenVLBackend,
    ModelOutputError,
    OpenAICompatibleChatBackend,
    make_chat_backend,
)
from a3_dual_arm_sim.agents.planning import (
    MultimodalSkillPlanner,
    validate_single_box_plan,
)
from a3_dual_arm_sim.agents.verification import TemporalSkillVerifier, VerificationDecision
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest


class StubBackend:
    def __init__(self, *outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def complete(self, prompt, images):
        self.calls.append((prompt, images))
        return self.outputs.popleft()

    def close(self):
        pass


def plan_json(column=3):
    return json.dumps([
        {
            "skill": skill.value,
            "batch_index": batch,
            "source_column_number": column,
            "target_slot_number": batch + 1,
        }
        for batch in range(2)
        for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
    ])


def camera_frame(value=0, *, full_keys=False):
    prefix = "observation.images." if full_keys else ""
    return {
        prefix + "front": np.full((4, 5, 3), value, dtype=np.uint8),
        prefix + "left_wrist": np.full((4, 5, 3), value + 1, dtype=np.uint8),
    }


def test_real_image_planning_uses_restricted_skill_library_and_shared_prompts():
    backend = StubBackend(plan_json())
    frame = camera_frame(full_keys=True)
    planner = MultimodalSkillPlanner(backend)
    plan = planner.plan("Pack ten cookies into one box.", frame)
    assert [item.skill for item in plan] == [CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE] * 2
    assert [item.batch_index for item in plan] == [0, 0, 1, 1]
    assert [item.target_slot_number for item in plan] == [1, 1, 2, 2]
    assert all(item.source_column_number == 3 for item in plan)
    assert plan[0].instruction == "Pick up five cookies from source column 3, batch 1."
    prompt, images = backend.calls[0]
    assert "Atomic skill library" in prompt
    assert "User task: Pack ten cookies" in prompt
    assert '"skill": "PICK_FIVE"' in prompt
    assert '"skill": "PLACE_FIVE"' in prompt
    assert "not a prescribed choice" in prompt
    assert images[0] is frame["observation.images.front"]
    assert len(images) == 1
    assert planner.last_response == plan_json()


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_planner_allows_only_whole_response_json_fence_with_valid_semantics(fence):
    raw = fence + "\n" + plan_json(2) + "\n```"
    planner = MultimodalSkillPlanner(StubBackend(raw))
    result = planner.plan("Pack ten", camera_frame())
    assert all(item.source_column_number == 2 for item in result)
    assert planner.last_response == raw


@pytest.mark.parametrize("raw", [
    "Here is the plan:\n" + plan_json(),
    "```json\n" + plan_json(0) + "\n```",
    plan_json().replace("PICK_FIVE", "pick_five"),
])
def test_planner_diagnostics_do_not_repair_prose_or_invalid_plan(raw):
    planner = MultimodalSkillPlanner(StubBackend(raw))
    with pytest.raises(ModelOutputError):
        planner.plan("Pack ten", camera_frame())
    assert planner.last_response == raw


def test_planner_diagnostics_are_bounded_and_cleared_before_each_attempt():
    raw = "x" * 70000
    planner = MultimodalSkillPlanner(StubBackend(raw))
    with pytest.raises(ModelOutputError, match="size limit"):
        planner.plan("Pack ten", camera_frame())
    assert planner.last_response == raw[:16384]
    with pytest.raises(ValueError):
        planner.plan("", camera_frame())
    assert planner.last_response is None


@pytest.mark.parametrize("invalid", [
    "not JSON", "```json\n[]\n```", "{}", "[]",
    plan_json(0), plan_json(5), plan_json(True),
])
def test_malformed_or_out_of_range_plans_fail_without_fallback(invalid):
    with pytest.raises(ModelOutputError):
        MultimodalSkillPlanner(StubBackend(invalid)).plan("Pack ten", camera_frame())


@pytest.mark.parametrize("field,value", [
    ("skill", "PUSH_BOX"), ("batch_index", 1),
    ("target_slot_number", 2), ("source_column_number", 2.0),
    ("extra", "ignored?"),
])
def test_invalid_plan_objects_fail_strict_schema_or_sequence(field, value):
    items = json.loads(plan_json())
    items[0][field] = value
    with pytest.raises(ModelOutputError):
        MultimodalSkillPlanner(StubBackend(json.dumps(items))).plan("Pack ten", camera_frame())


def test_source_column_cannot_change_mid_plan_or_exceed_policy_coverage():
    items = json.loads(plan_json())
    items[2]["source_column_number"] = 2
    items[3]["source_column_number"] = 2
    with pytest.raises(ModelOutputError):
        MultimodalSkillPlanner(StubBackend(json.dumps(items))).plan("Pack ten", camera_frame())
    with pytest.raises(ModelOutputError):
        MultimodalSkillPlanner(StubBackend(plan_json()), allowed_source_columns=(1,)).plan(
            "Pack ten", camera_frame()
        )


def test_visual_inputs_are_required_and_bad_images_never_reach_backend():
    backend = StubBackend(plan_json())
    planner = MultimodalSkillPlanner(backend)
    with pytest.raises(ValueError, match="missing front"):
        planner.plan("Pack ten", {})
    with pytest.raises(ValueError, match="uint8 RGB"):
        planner.plan("Pack ten", {"front": np.zeros((4, 4), dtype=np.float32)})
    assert backend.calls == []


def test_completion_verifier_uses_both_views_in_time_order_and_stops_after_yes():
    backend = StubBackend("Yes")
    frames = [camera_frame(10), camera_frame(20, full_keys=True)]
    req = SkillRequest(CookieSkill.PICK_FIVE, 0, 2, 1)
    decision = TemporalSkillVerifier(backend).check(req, frames)
    assert decision.status == "completed"
    assert len(backend.calls) == 1
    prompt, images = backend.calls[0]
    assert "five" in prompt and "lifted" in prompt
    assert "oldest first" in prompt
    assert [int(image[0, 0, 0]) for image in images] == [10, 11, 20, 21]


@pytest.mark.parametrize("diagnosis,status", [("Stuck", "stuck"), ("StillTrying", "continuing")])
def test_unfinished_verification_runs_second_stage(diagnosis, status):
    backend = StubBackend("No", diagnosis)
    req = SkillRequest(CookieSkill.PLACE_FIVE, 1, 2, 2)
    decision = TemporalSkillVerifier(backend).check(req, [camera_frame(), camera_frame(2)])
    assert decision.status == status
    assert len(backend.calls) == 2
    assert "gripper empty" in backend.calls[0][0]
    assert "target box column 2" in backend.calls[0][0]
    assert "Stuck or StillTrying" in backend.calls[1][0]


@pytest.mark.parametrize("outputs", [("Probably yes",), ("No", "Continue"), ("Yes.",)])
def test_verifier_rejects_ambiguous_or_malformed_decisions(outputs):
    req = SkillRequest(CookieSkill.PICK_FIVE, 0, 1, 1)
    with pytest.raises(ModelOutputError):
        TemporalSkillVerifier(StubBackend(*outputs)).check(
            req, [camera_frame(), camera_frame(2)]
        )


def test_verifier_requires_temporal_pairs_and_left_wrist_view():
    verifier = TemporalSkillVerifier(StubBackend())
    req = SkillRequest(CookieSkill.PICK_FIVE, 0, 1, 1)
    with pytest.raises(ValueError, match="at least two"):
        verifier.check(req, [camera_frame()])
    with pytest.raises(ValueError, match="left_wrist"):
        verifier.check(req, [{"front": camera_frame()["front"]}] * 2)


def test_http_request_has_bounded_tokens_timeout_compressed_images_and_env_key(monkeypatch):
    seen = {}
    monkeypatch.setenv("A3_TEST_KEY", "secret-not-in-errors")

    def urlopen(outbound, timeout):
        seen["request"], seen["timeout"] = outbound, timeout
        return io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "Yes"}, "finish_reason": "stop"}]
        }).encode())

    monkeypatch.setattr("a3_dual_arm_sim.agents.backends.request.urlopen", urlopen)
    config = ChatBackendConfig(
        "http", "vision-model", base_url="https://provider.invalid/prefix/v1/",
        api_key_env="A3_TEST_KEY", timeout_seconds=7, max_tokens=33, image_max_edge=64,
    )
    backend = make_chat_backend(config)
    assert isinstance(backend, OpenAICompatibleChatBackend)
    assert backend.complete("inspect", [np.zeros((200, 100, 3), dtype=np.uint8)]) == "Yes"
    assert seen["timeout"] == 7
    outbound = seen["request"]
    assert outbound.full_url == "https://provider.invalid/prefix/v1/chat/completions"
    assert outbound.get_header("Authorization") == "Bearer secret-not-in-errors"
    payload = json.loads(outbound.data)
    assert payload["max_tokens"] == 33 and payload["stream"] is False
    encoded = payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.format == "JPEG" and max(decoded.size) == 64
    backend.close()


def test_http_errors_hide_credentials_and_do_not_retry(monkeypatch):
    calls = []
    monkeypatch.setenv("A3_TEST_KEY", "secret-api-key")

    def fail(outbound, timeout):
        calls.append(outbound)
        raise error.HTTPError(outbound.full_url, 401, "secret-api-key", {}, None)

    monkeypatch.setattr("a3_dual_arm_sim.agents.backends.request.urlopen", fail)
    backend = OpenAICompatibleChatBackend(ChatBackendConfig(
        "http", "qwen", base_url="https://provider.invalid/v1", api_key_env="A3_TEST_KEY"
    ))
    with pytest.raises(AgenticModelError, match="HTTP 401") as caught:
        backend.complete("inspect", [])
    assert "secret-api-key" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("body", [
    b"invalid", b"{}", b'{"choices": []}',
    b'{"choices": [{"message": {"content": null}}]}',
    b'{"choices": [{"message": {"content": "Yes"}, "finish_reason": "length"}]}',
])
def test_http_invalid_response_or_truncation_is_not_a_success(monkeypatch, body):
    monkeypatch.setattr(
        "a3_dual_arm_sim.agents.backends.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(body)
    )
    backend = OpenAICompatibleChatBackend(ChatBackendConfig(
        "http", "qwen", base_url="https://provider.invalid/v1", api_key_env=""
    ))
    with pytest.raises(ModelOutputError):
        backend.complete("inspect", [])


def test_missing_key_fails_before_network_request(monkeypatch):
    monkeypatch.delenv("A3_MISSING_TEST_KEY", raising=False)
    backend = OpenAICompatibleChatBackend(ChatBackendConfig(
        "http", "qwen", base_url="https://provider.invalid/v1", api_key_env="A3_MISSING_TEST_KEY"
    ))
    with pytest.raises(AgenticModelError, match="environment variable"):
        backend.complete("inspect", [])


@pytest.mark.parametrize("model_family", [None, "qwen3_5"])
@pytest.mark.parametrize("with_adapter", [False, True])
def test_local_qwen_is_lazy_uses_local_checkpoint_and_bounds_generation(
    monkeypatch, tmp_path, model_family, with_adapter,
):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_options"] = kwargs
            return "formatted"

        def __call__(self, **kwargs):
            calls["processor_inputs"] = kwargs
            return Inputs(input_ids=np.array([[1, 2, 3]]))

        def batch_decode(self, values, **kwargs):
            assert values.tolist() == [[4, 5]]
            return [" Yes "]

    class Model:
        device = "cpu"

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return np.array([[1, 2, 3, 4, 5]])

    def load_model(path, **kwargs):
        calls["load_model"] = (path, kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=contextlib.nullcontext, float16="mock-float16",
    ))
    model_class_name = (
        "Qwen3_5ForConditionalGeneration" if model_family == "qwen3_5"
        else "Qwen2_5_VLForConditionalGeneration"
    )
    # Only the selected family is available, proving lazy imports do not require
    # another model family to be installed or implicitly substitute its loader.
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(**{
        model_class_name: SimpleNamespace(from_pretrained=load_model),
        "AutoProcessor": SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Processor()),
    }))
    config_options = {} if model_family is None else {"model_family": model_family}
    if with_adapter:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "verifier_training_manifest.json").write_text(json.dumps({
            "base_model": "local/qwen-checkpoint",
            "model_family": model_family or "qwen2_5_vl",
        }), encoding="utf-8")

        def load_adapter(model, path, **kwargs):
            calls["load_adapter"] = (path, kwargs)
            return model

        monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ))
        config_options["adapter_path"] = str(adapter)
    config = ChatBackendConfig(
        "local_qwen", "local/qwen-checkpoint", dtype="float16", device="cpu", max_tokens=12,
        **config_options,
    )
    backend = make_chat_backend(config)
    assert isinstance(backend, LocalQwenVLBackend) and calls == {}
    assert config.fine_tuned is False
    assert config.model_family == (model_family or "qwen2_5_vl")
    assert backend.complete("inspect", [camera_frame()["front"]]) == "Yes"
    assert calls["load_model"][1]["local_files_only"] is True
    assert calls["load_model"][1]["torch_dtype"] == "mock-float16"
    assert calls["generation"]["max_new_tokens"] == 12
    assert calls["generation"]["do_sample"] is False
    assert backend.adapter_loaded is with_adapter
    if with_adapter:
        assert calls["load_adapter"][1] == {"is_trainable": False, "local_files_only": True}
    assert calls["template_options"]["tokenize"] is False
    assert calls["template_options"]["add_generation_prompt"] is True
    if model_family == "qwen3_5":
        assert calls["template_options"]["enable_thinking"] is False
    else:
        assert "enable_thinking" not in calls["template_options"]
    assert isinstance(calls["messages"][0]["content"][0]["image"], Image.Image)
    backend.close()
    assert backend._model is None and backend._processor is None
    assert backend.adapter_loaded is False
    if with_adapter:
        calls.pop("load_adapter")
        (adapter / "verifier_training_manifest.json").write_text(json.dumps({
            "base_model": "local/a-different-base",
            "model_family": model_family or "qwen2_5_vl",
        }), encoding="utf-8")
        with pytest.raises(AgenticModelError, match="could not load"):
            backend.complete("inspect", [camera_frame()["front"]])
        assert "load_adapter" not in calls and backend.adapter_loaded is False


def test_missing_qwen35_loader_fails_instead_of_falling_back_to_qwen25(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(),
        Qwen2_5_VLForConditionalGeneration=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: calls.append("wrong family")
        ),
    ))
    backend = make_chat_backend(ChatBackendConfig(
        "local_qwen", "local/qwen35", model_family="qwen3_5"
    ))
    with pytest.raises(AgenticModelError, match="qwen3_5"):
        backend.complete("inspect", [camera_frame()["front"]])
    assert calls == []


@pytest.mark.parametrize("kwargs", [
    {"kind": "invalid"}, {"model": ""}, {"timeout_seconds": 0},
    {"max_tokens": 0}, {"image_max_edge": 0}, {"jpeg_quality": 100},
    {"base_url": "https://secret@provider.invalid/v1"},
    {"model_family": "unsupported"},
    {"adapter_path": "local/adapter"},
])
def test_config_rejects_invalid_transport_limits(kwargs):
    values = {"kind": "http", "model": "qwen", "base_url": "https://provider.invalid/v1"}
    values.update(kwargs)
    with pytest.raises(ValueError):
        ChatBackendConfig(**values)


def test_invalid_status_and_incomplete_controller_plan_are_rejected():
    with pytest.raises(ValueError):
        VerificationDecision("unknown", "bad")
    with pytest.raises(ModelOutputError):
        validate_single_box_plan([])
