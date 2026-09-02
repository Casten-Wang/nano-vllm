import socket
from threading import Thread

import pytest
import torch

from nanovllm.engine.cache_transfer import RankCacheTransfer
from nanovllm.engine.cache_transfer_wire import (
    WIRE_HEADER,
    WIRE_MAGIC,
    WIRE_VERSION,
    receive_rank_cache_transfer,
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
