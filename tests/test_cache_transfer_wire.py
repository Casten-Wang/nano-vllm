import socket
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nanovllm.engine.cache_transfer import RankCacheTransfer
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.cache_transfer_wire import (
    PendingRankCacheReceive,
    PendingRankCacheSend,
    RankCacheReceiver,
    WIRE_HEADER,
    WIRE_MAGIC,
    WIRE_VERSION,
    receive_rank_cache_transfer,
    send_rank_cache_to_endpoint,
    send_rank_cache_transfer,
)


class BufferSocket:
    def __init__(self, data=b"", max_recv=17):
        self.data = bytearray(data)
        self.offset = 0
        self.max_recv = max_recv

    def sendall(self, data):
        self.data.extend(data)

    def recv_into(self, target):
        size = min(len(target), self.max_recv, len(self.data) - self.offset)
        if size <= 0:
            return 0
        target[:size] = self.data[self.offset : self.offset + size]
        self.offset += size
        return size


def make_payload(*, int8: bool = False) -> RankCacheTransfer:
    kv_dtype = torch.int8 if int8 else torch.float32
    kv_blocks = torch.arange(2 * 2 * 2 * 3 * 1 * 2).reshape(2, 2, 2, 3, 1, 2)
    kv_blocks = kv_blocks.to(kv_dtype)
    return RankCacheTransfer(
        format_version=1,
        transfer_id="request-1/attempt-2",
        tensor_parallel_rank=1,
        tensor_parallel_size=4,
        block_size=3,
        cached_tokens=5,
        kv_blocks=kv_blocks,
        kv_scales=(
            torch.arange(2 * 2 * 2 * 3 * 1, dtype=torch.float16).reshape(
                2, 2, 2, 3, 1
            )
            if int8
            else None
        ),
        recurrent_states=(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),),
        convolution_states=(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),),
    )


@pytest.mark.parametrize("int8", [False, True])
def test_socket_wire_round_trip_preserves_rank_payload(int8):
    sender, receiver = socket.socketpair()
    payload = make_payload(int8=int8)
    failure = []

    def send():
        try:
            send_rank_cache_transfer(sender, payload)
        except BaseException as exc:
            failure.append(exc)
        finally:
            sender.close()

    thread = Thread(target=send)
    thread.start()
    received = receive_rank_cache_transfer(receiver)
    receiver.close()
    thread.join()

    assert not failure
    assert received.transfer_id == payload.transfer_id
    assert received.tensor_parallel_rank == payload.tensor_parallel_rank
    assert received.cached_tokens == payload.cached_tokens
    torch.testing.assert_close(received.kv_blocks, payload.kv_blocks.cpu())
    if int8:
        torch.testing.assert_close(received.kv_scales, payload.kv_scales.cpu())
    else:
        assert received.kv_scales is None
    torch.testing.assert_close(received.recurrent_states[0], payload.recurrent_states[0])
    torch.testing.assert_close(
        received.convolution_states[0], payload.convolution_states[0]
    )


def test_socket_wire_rejects_payload_above_receiver_limit():
    sender, receiver = socket.socketpair()
    sender.sendall(WIRE_HEADER.pack(WIRE_MAGIC, WIRE_VERSION, 1, 2))
    sender.close()
    with pytest.raises(ValueError, match="exceeds configured byte limit"):
        receive_rank_cache_transfer(receiver, max_payload_bytes=1)
    receiver.close()


def test_socket_wire_rejects_truncated_payload():
    sender, receiver = socket.socketpair()
    sender.sendall(b"NVCT")
    sender.close()
    with pytest.raises(EOFError, match="ended unexpectedly"):
        receive_rank_cache_transfer(receiver)
    receiver.close()


def test_socket_wire_rejects_corrupted_tensor_bytes():
    sink = BufferSocket()
    send_rank_cache_transfer(sink, make_payload())
    sink.data[-33] ^= 1

    with pytest.raises(ValueError, match="checksum mismatch"):
        receive_rank_cache_transfer(BufferSocket(sink.data))


def test_tcp_endpoint_acknowledges_only_after_payload_validation():
    payload = make_payload(int8=True)
    received = []
    failure = []
    with RankCacheReceiver("127.0.0.1", 0) as receiver:
        def receive():
            try:
                received.append(receiver.receive())
            except BaseException as exc:
                failure.append(exc)

        thread = Thread(target=receive)
        thread.start()
        sent_bytes = send_rank_cache_to_endpoint(*receiver.address, payload)
        thread.join()

    assert sent_bytes > payload.kv_blocks.numel()
    assert not failure
    assert len(received) == 1
    assert received[0].transfer_id == payload.transfer_id
    torch.testing.assert_close(received[0].kv_blocks, payload.kv_blocks)


def test_tcp_endpoint_rejects_when_install_callback_fails():
    payload = make_payload()
    sender_failure = []
    with RankCacheReceiver("127.0.0.1", 0) as receiver:
        def send():
            try:
                send_rank_cache_to_endpoint(*receiver.address, payload)
            except BaseException as exc:
                sender_failure.append(exc)

        thread = Thread(target=send)
        thread.start()
        with pytest.raises(ValueError, match="destination rejected"):
            receiver.receive(
                on_verified=lambda _payload: (_ for _ in ()).throw(
                    ValueError("destination rejected")
                )
            )
        thread.join()

    assert len(sender_failure) == 1
    assert "rejected" in str(sender_failure[0])


def test_pending_receive_defers_sender_ack_until_install_commit():
    payload = make_payload()
    receiver = PendingRankCacheReceive("127.0.0.1", 0, timeout_s=2.0)
    sender_result = []
    sender_failure = []
    receiver.start()

    def send():
        try:
            sender_result.append(
                send_rank_cache_to_endpoint(
                    *receiver.address,
                    payload,
                    timeout_s=2.0,
                )
            )
        except BaseException as exc:
            sender_failure.append(exc)

    thread = Thread(target=send)
    thread.start()
    deadline = monotonic() + 2.0
    while receiver.poll()[0] == "receiving" and monotonic() < deadline:
        sleep(0.001)

    assert receiver.poll() == ("ready", None)
    assert thread.is_alive()
    torch.testing.assert_close(receiver.payload().kv_blocks, payload.kv_blocks)

    receiver.finish(accepted=True)
    thread.join()
    assert not sender_failure
    assert sender_result


def test_pending_receive_nacks_sender_when_install_is_rejected():
    payload = make_payload()
    receiver = PendingRankCacheReceive("127.0.0.1", 0, timeout_s=2.0)
    sender_failure = []
    receiver.start()

    def send():
        try:
            send_rank_cache_to_endpoint(
                *receiver.address,
                payload,
                timeout_s=2.0,
            )
        except BaseException as exc:
            sender_failure.append(exc)

    thread = Thread(target=send)
    thread.start()
    deadline = monotonic() + 2.0
    while receiver.poll()[0] == "receiving" and monotonic() < deadline:
        sleep(0.001)
    receiver.finish(accepted=False)
    thread.join()

    assert len(sender_failure) == 1
    assert "rejected" in str(sender_failure[0])


def test_pending_receive_enforces_end_to_end_deadline(monkeypatch):
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(
        "nanovllm.engine.cache_transfer_wire.monotonic",
        lambda: next(clock),
    )
    receiver = PendingRankCacheReceive("127.0.0.1", 0, timeout_s=2.0)

    state, error = receiver.poll()

    assert state == "failed"
    assert "deadline expired" in error
    receiver.finish(accepted=False)


def test_pending_send_enforces_end_to_end_deadline(monkeypatch):
    clock = iter((20.0, 22.0))
    monkeypatch.setattr(
        "nanovllm.engine.cache_transfer_wire.monotonic",
        lambda: next(clock),
    )
    sender = PendingRankCacheSend(
        "127.0.0.1",
        12345,
        make_payload(),
        timeout_s=2.0,
    )

    state, error = sender.poll()

    assert state == "failed"
    assert "deadline expired" in error
    with pytest.raises(RuntimeError, match="deadline expired"):
        sender.result()
    sender.finish()


def test_model_runner_rank_endpoint_exports_receives_and_installs():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    endpoints = [("127.0.0.1", port)]
    payload = make_payload()

    source = object.__new__(ModelRunner)
    source.rank = 0
    source.world_size = 1
    exported_to_host = []
    source.export_sequence_cache = (
        lambda seq, transfer_id, to_host=False: (
            exported_to_host.append(to_host) or payload
        )
    )
    destination = object.__new__(ModelRunner)
    destination.rank = 0
    destination.world_size = 1
    installed = []
    destination.import_sequence_cache = (
        lambda seq, received, transfer_id: installed.append(
            (seq, received, transfer_id)
        )
    )
    receive_result = []
    receiver_thread = Thread(
        target=lambda: receive_result.append(
            destination.receive_sequence_cache_from_endpoint(
                "destination-seq",
                payload.transfer_id,
                endpoints,
                timeout_s=2.0,
            )
        )
    )
    receiver_thread.start()

    send_result = source.send_sequence_cache_to_endpoint(
        "source-seq",
        payload.transfer_id,
        endpoints,
        timeout_s=2.0,
    )
    receiver_thread.join()

    assert send_result["rank"] == 0
    assert exported_to_host == [True]
    assert receive_result == [
        {"rank": 0, "cached_tokens": 5, "received_bytes": payload.nbytes}
    ]
    assert len(installed) == 1
    assert installed[0][0] == "destination-seq"
    assert installed[0][1].transfer_id == payload.transfer_id
    assert installed[0][2] == payload.transfer_id
    torch.testing.assert_close(installed[0][1].kv_blocks, payload.kv_blocks)


def test_model_runner_async_receive_polls_then_installs_before_ack():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    endpoints = [("127.0.0.1", port)]
    payload = make_payload()

    source = object.__new__(ModelRunner)
    source.rank = 0
    source.world_size = 1
    source.export_sequence_cache = lambda *_args, **_kwargs: payload
    destination = object.__new__(ModelRunner)
    destination.rank = 0
    destination.world_size = 1
    destination._pending_cache_receives = {}
    installed = []
    destination.import_sequence_cache = (
        lambda seq, received, transfer_id: installed.append(
            (seq, received, transfer_id)
        )
    )

    assert destination.start_sequence_cache_receive(
        payload.transfer_id,
        endpoints,
        2.0,
    ) == {"rank": 0, "started": 1}
    sender_result = []
    sender_thread = Thread(
        target=lambda: sender_result.append(
            source.send_sequence_cache_to_endpoint(
                "source-seq",
                payload.transfer_id,
                endpoints,
                timeout_s=2.0,
            )
        )
    )
    sender_thread.start()
    deadline = monotonic() + 2.0
    poll = destination.poll_sequence_cache_receive(payload.transfer_id)
    while poll["state"] == "receiving" and monotonic() < deadline:
        sleep(0.001)
        poll = destination.poll_sequence_cache_receive(payload.transfer_id)

    assert poll == {"rank": 0, "state": "ready"}
    assert sender_thread.is_alive()
    assert destination.install_sequence_cache_receive(
        "destination-seq",
        payload.transfer_id,
    ) == {"rank": 0, "cached_tokens": 5, "received_bytes": payload.nbytes}
    sender_thread.join()

    assert sender_result[0]["sent_bytes"] > 0
    assert len(installed) == 1
    assert payload.transfer_id not in destination._pending_cache_receives


def test_model_runner_batches_receive_states_in_one_result():
    runner = object.__new__(ModelRunner)
    runner.rank = 2
    runner._pending_cache_receives = {
        "ready": SimpleNamespace(poll=lambda: ("ready", None)),
        "failed": SimpleNamespace(poll=lambda: ("failed", "checksum mismatch")),
    }

    assert runner.poll_sequence_cache_receives(["ready", "failed"]) == {
        "rank": 2,
        "receives": {
            "ready": {"state": "ready"},
            "failed": {"state": "failed", "error": "checksum mismatch"},
        },
    }

    with pytest.raises(ValueError, match="unique non-empty"):
        runner.poll_sequence_cache_receives(["ready", "ready"])


def test_pending_receive_finish_joins_listener_thread_without_waiting_for_deadline():
    receive = PendingRankCacheReceive(
        "127.0.0.1",
        0,
        timeout_s=30.0,
    )
    receive.start()

    receive.finish(accepted=False)

    assert not receive._thread.is_alive()


def test_model_runner_async_receive_nacks_payload_smaller_than_preflight():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    endpoints = [("127.0.0.1", port)]
    payload = make_payload()

    source = object.__new__(ModelRunner)
    source.rank = 0
    source.world_size = 1
    source.export_sequence_cache = lambda *_args, **_kwargs: payload
    destination = object.__new__(ModelRunner)
    destination.rank = 0
    destination.world_size = 1
    destination._pending_cache_receives = {}
    destination.import_sequence_cache = Mock()
    destination.start_sequence_cache_receive(
        payload.transfer_id,
        endpoints,
        2.0,
        expected_payload_bytes=[payload.nbytes + 1],
    )
    sender_errors = []

    def send_payload():
        try:
            source.send_sequence_cache_to_endpoint(
                "source-seq",
                payload.transfer_id,
                endpoints,
                timeout_s=2.0,
            )
        except BaseException as exc:
            sender_errors.append(exc)

    sender_thread = Thread(target=send_payload)
    sender_thread.start()
    deadline = monotonic() + 2.0
    poll = destination.poll_sequence_cache_receive(payload.transfer_id)
    while poll["state"] == "receiving" and monotonic() < deadline:
        sleep(0.001)
        poll = destination.poll_sequence_cache_receive(payload.transfer_id)

    assert poll == {"rank": 0, "state": "ready"}
    with pytest.raises(ValueError, match="differ from the preflight"):
        destination.install_sequence_cache_receive(
            "destination-seq",
            payload.transfer_id,
            [payload.nbytes + 1],
        )
    sender_thread.join(timeout=2.0)

    assert not sender_thread.is_alive()
    assert len(sender_errors) == 1
    assert "receiver rejected" in str(sender_errors[0])
    destination.import_sequence_cache.assert_not_called()


def test_model_runner_async_send_waits_for_receiver_ack():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    endpoints = [("127.0.0.1", port)]
    payload = make_payload()
    source = object.__new__(ModelRunner)
    source.rank = 0
    source.world_size = 1
    source._pending_cache_sends = {}
    source.export_sequence_cache = lambda *_args, **_kwargs: payload
    receiver = PendingRankCacheReceive("127.0.0.1", port, timeout_s=2.0)
    receiver.start()

    assert source.start_sequence_cache_send(
        "source-seq",
        payload.transfer_id,
        endpoints,
        timeout_s=2.0,
    ) == {"rank": 0, "started": 1, "staged_bytes": payload.nbytes}
    deadline = monotonic() + 2.0
    receive_state, _ = receiver.poll()
    while receive_state == "receiving" and monotonic() < deadline:
        sleep(0.001)
        receive_state, _ = receiver.poll()
    assert receive_state == "ready"
    assert source.poll_sequence_cache_send(payload.transfer_id) == {
        "rank": 0,
        "state": "sending",
        "staged_bytes": 0,
    }
    assert source._pending_cache_sends[payload.transfer_id].staged_bytes == 0

    receiver.finish(accepted=True)
    send_poll = source.poll_sequence_cache_send(payload.transfer_id)
    while send_poll["state"] == "sending" and monotonic() < deadline:
        sleep(0.001)
        send_poll = source.poll_sequence_cache_send(payload.transfer_id)
    assert send_poll == {"rank": 0, "state": "ready", "staged_bytes": 0}
    result = source.finish_sequence_cache_send(payload.transfer_id)
    assert result["rank"] == 0
    assert result["sent_bytes"] > 0
    assert payload.transfer_id not in source._pending_cache_sends


def test_model_runner_batches_send_states_in_one_result():
    runner = object.__new__(ModelRunner)
    runner.rank = 2
    runner._pending_cache_sends = {
        "ready": SimpleNamespace(poll=lambda: ("ready", None), staged_bytes=0),
        "failed": SimpleNamespace(
            poll=lambda: ("failed", "receiver rejected"), staged_bytes=0
        ),
    }

    assert runner.poll_sequence_cache_sends(["ready", "failed"]) == {
        "rank": 2,
        "sends": {
            "ready": {"state": "ready", "staged_bytes": 0},
            "failed": {
                "state": "failed",
                "staged_bytes": 0,
                "error": "receiver rejected",
            },
        },
    }

    with pytest.raises(ValueError, match="unique non-empty"):
        runner.poll_sequence_cache_sends(["ready", "ready"])


def test_pending_send_finish_joins_thread_waiting_for_ack():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    send = PendingRankCacheSend(
        host,
        port,
        make_payload(),
        timeout_s=30.0,
    )
    try:
        send.start()
        deadline = monotonic() + 2.0
        while send.staged_bytes and monotonic() < deadline:
            sleep(0.001)

        send.finish()

        assert not send._thread.is_alive()
    finally:
        listener.close()
