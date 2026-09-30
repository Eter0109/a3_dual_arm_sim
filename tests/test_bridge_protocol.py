"""Wire-protocol tests: framing, round trips, and the refusals.

The refusals matter more than the round trips.  A bridge that silently accepts a
truncated frame, a disallowed dtype or a mismatched protocol version would corrupt
actions rather than fail, and a corrupted action is indistinguishable from a model
that simply performs badly.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from a3_dual_arm_sim.bridge.protocol import (
    PROTOCOL,
    ActionChunk,
    BridgeError,
    Request,
    decode_message,
    decode_request,
    decode_response,
    encode_error,
    encode_message,
    encode_ready,
    encode_request,
    encode_response,
    read_exactly,
    read_frame,
    write_frame,
)


def test_arrays_round_trip_with_dtype_and_shape() -> None:
    arrays = {
        "front": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        "state": np.linspace(-1.0, 1.0, 16, dtype=np.float32),
    }
    kind, fields, decoded = decode_message(encode_message("infer", {"task": "x"}, arrays))

    assert kind == "infer"
    assert fields == {"task": "x"}
    assert set(decoded) == set(arrays)
    for name, original in arrays.items():
        assert decoded[name].dtype == original.dtype
        assert decoded[name].shape == original.shape
        assert np.array_equal(decoded[name], original)


def test_frame_carries_its_own_protocol_version() -> None:
    payload = encode_message("infer", {}, {})
    header_length = struct.unpack(">I", payload[4:8])[0]
    assert PROTOCOL.encode() in payload[8 : 8 + header_length]


def test_unknown_protocol_version_is_refused() -> None:
    payload = encode_message("infer", {}, {})
    header_length = struct.unpack(">I", payload[4:8])[0]
    header = payload[8 : 8 + header_length].replace(PROTOCOL.encode(), b"a3-bridge/9")
    tampered = payload[:8] + header + payload[8 + header_length :]

    with pytest.raises(BridgeError, match="protocol mismatch"):
        decode_message(tampered)


def test_a_truncated_frame_is_refused() -> None:
    payload = encode_message("infer", {}, {"state": np.zeros(16, dtype=np.float32)})
    with pytest.raises(BridgeError, match="truncated"):
        decode_message(payload[:-8])


def test_a_wrong_magic_word_is_refused() -> None:
    payload = encode_message("infer", {}, {})
    with pytest.raises(BridgeError, match="magic word"):
        decode_message(b"XXXX" + payload[4:])


def test_a_disallowed_dtype_is_refused() -> None:
    # float16 would be silently widened on arrival; refusing it keeps the two
    # ends from disagreeing about what the bytes mean.
    payload = encode_message("infer", {}, {"tiny": np.zeros(4, dtype=np.float16)})
    with pytest.raises(BridgeError, match="dtype"):
        decode_message(payload)


def test_a_byte_length_mismatch_is_refused() -> None:
    # The header says one shape and the frame carries another, which is the shape
    # of corruption a truncated write produces.  Rebuilding the array from it would
    # invent values, so it is refused instead.
    payload = encode_message("infer", {}, {"state": np.zeros(4, dtype=np.float32)})
    header_length = struct.unpack(">I", payload[4:8])[0]
    header = json.loads(payload[8 : 8 + header_length])
    header["arrays"]["state"]["shape"] = [5]
    new_header = json.dumps(header).encode("utf-8")
    tampered = (
        b"A3BR" + struct.pack(">I", len(new_header)) + new_header + payload[8 + header_length :]
    )

    with pytest.raises(BridgeError, match="carries"):
        decode_message(tampered)


def test_a_truncated_blob_is_refused() -> None:
    payload = encode_message("infer", {}, {"state": np.zeros(4, dtype=np.float32)})
    with pytest.raises(BridgeError, match="truncated"):
        decode_message(payload[:-4])


def test_request_round_trip_including_the_requested_chunk_size() -> None:
    payload = encode_request(
        episode_id="abc",
        step=7,
        task="pick the block",
        arrays={"state": np.zeros(16, dtype=np.float32)},
        dropped_images=("observation.images.left_wrist",),
        timestamp=1.5,
        chunk=50,
    )
    request = decode_request(payload)

    assert isinstance(request, Request)
    assert (request.episode_id, request.step, request.task, request.chunk) == (
        "abc",
        7,
        "pick the block",
        50,
    )
    assert request.timestamp == 1.5
    assert request.dropped_images == ("observation.images.left_wrist",)
    assert request.vector("state").shape == (16,)


def test_request_accessors_name_the_missing_key() -> None:
    payload = encode_request(
        episode_id="abc", step=0, task="t", arrays={"state": np.zeros(16, dtype=np.float32)}
    )
    request = decode_request(payload)
    with pytest.raises(KeyError, match="front"):
        request.image("front")


def test_action_chunk_round_trip() -> None:
    chunk = ActionChunk(
        actions=np.zeros((4, 16), dtype=np.float32),
        space="joint_position",
        units="raw",
        latency_ms=12.5,
        info={"latency_model": "fast"},
    )
    decoded = decode_response(encode_response(chunk))

    assert decoded.space == "joint_position"
    assert decoded.units == "raw"
    assert decoded.latency_ms == 12.5
    assert decoded.info == {"latency_model": "fast"}
    assert decoded.actions.shape == (4, 16)


def test_a_one_dimensional_action_becomes_a_single_step_chunk() -> None:
    chunk = ActionChunk(actions=np.zeros(16))
    assert chunk.actions.shape == (1, 16)
    assert chunk.horizon == 1
    assert chunk.dim == 16


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((0, 16)),  # empty
        np.full((4, 16), np.nan),  # not finite
        np.zeros((2, 3, 4)),  # wrong rank
    ],
)
def test_a_malformed_action_chunk_is_refused_at_construction(actions: np.ndarray) -> None:
    with pytest.raises(BridgeError):
        ActionChunk(actions=actions)


def test_a_wrong_frame_kind_is_refused() -> None:
    with pytest.raises(BridgeError, match="'infer'"):
        decode_request(encode_ready())
    with pytest.raises(BridgeError, match="'actions'"):
        decode_response(encode_error("boom"))


def test_stream_framing_round_trips_over_a_buffer() -> None:
    stream = io.BytesIO()
    write_frame(stream, b"first")
    write_frame(stream, b"second")
    stream.seek(0)

    assert read_frame(stream) == b"first"
    assert read_frame(stream) == b"second"
    # A clean end of stream is an empty read, not an error, so a caller can tell
    # "the bridge stopped" from "the bridge sent something malformed".
    assert read_frame(stream) == b""


def test_a_short_stream_raises_rather_than_returning_partial_data() -> None:
    with pytest.raises(EOFError):
        read_exactly(io.BytesIO(b"abc"), 5)
    with pytest.raises(BridgeError, match="length prefix"):
        read_frame(io.BytesIO(b"\x00\x00"))
