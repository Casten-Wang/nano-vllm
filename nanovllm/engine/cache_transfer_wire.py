"""Streaming wire format for one tensor-parallel rank's cache payload."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
from threading import Lock, Thread, current_thread
from time import monotonic, sleep
from collections.abc import Callable, Iterable

import torch

from nanovllm.engine.cache_transfer import (
    HostStagingBufferPool,
    RankCacheTransfer,
)


WIRE_MAGIC = b"NVCT"
WIRE_VERSION = 1
WIRE_HEADER = struct.Struct("!4sBQQ")
WIRE_DIGEST_BYTES = hashlib.sha256().digest_size
MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 16 * 1024**3
_CHUNK_BYTES = 1024 * 1024
_TRANSFER_ACK = b"\x01"
_TRANSFER_NACK = b"\x00"

_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int8: "int8",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _payload_tensors(
    payload: RankCacheTransfer,
) -> Iterable[tuple[str, torch.Tensor]]:
    yield "kv_blocks", payload.kv_blocks
    if payload.kv_scales is not None:
        yield "kv_scales", payload.kv_scales
    for index, tensor in enumerate(payload.recurrent_states):
        yield f"recurrent/{index}", tensor
    for index, tensor in enumerate(payload.convolution_states):
        yield f"convolution/{index}", tensor


def _host_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in _DTYPE_TO_NAME:
        raise ValueError(f"unsupported cache transfer dtype: {tensor.dtype}")
    if tensor.ndim == 0:
        raise ValueError("cache transfer tensors must not be scalars")
    return tensor.detach().to(device="cpu").contiguous()


def _byte_view(tensor: torch.Tensor) -> memoryview:
    return memoryview(tensor.view(torch.uint8).reshape(-1).numpy())


def _send_bytes(sock, data: bytes | memoryview, digest=None) -> None:
    view = memoryview(data)
    for start in range(0, len(view), _CHUNK_BYTES):
        chunk = view[start : start + _CHUNK_BYTES]
        sock.sendall(chunk)
        if digest is not None:
            digest.update(chunk)


def send_rank_cache_transfer(sock, payload: RankCacheTransfer) -> int:
    """Send one payload without materializing a second full byte buffer."""

    tensors = [(name, _host_tensor(tensor)) for name, tensor in _payload_tensors(payload)]
    descriptors = []
    body_bytes = 0
    for name, tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        body_bytes += nbytes
        descriptors.append(
            {
                "name": name,
                "dtype": _DTYPE_TO_NAME[tensor.dtype],
                "shape": list(tensor.shape),
                "nbytes": nbytes,
            }
        )
    header = json.dumps(
        {
            "format_version": payload.format_version,
            "transfer_id": payload.transfer_id,
            "tensor_parallel_rank": payload.tensor_parallel_rank,
            "tensor_parallel_size": payload.tensor_parallel_size,
            "block_size": payload.block_size,
            "cached_tokens": payload.cached_tokens,
            "tensors": descriptors,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("cache transfer wire header is too large")

    sock.sendall(WIRE_HEADER.pack(WIRE_MAGIC, WIRE_VERSION, len(header), body_bytes))
    digest = hashlib.sha256()
    _send_bytes(sock, header, digest)
    for _, tensor in tensors:
        _send_bytes(sock, _byte_view(tensor), digest)
    sock.sendall(digest.digest())
    return WIRE_HEADER.size + len(header) + body_bytes + WIRE_DIGEST_BYTES


def _recv_exact(sock, target: memoryview) -> None:
    offset = 0
    while offset < len(target):
        received = sock.recv_into(target[offset:])
        if received == 0:
            raise EOFError("cache transfer stream ended unexpectedly")
        offset += received


def _recv_bytes(sock, size: int, digest=None) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    _recv_exact(sock, view)
    if digest is not None:
        digest.update(view)
    return data


def _aligned_storage_offsets(
    tensors: list[tuple[str, torch.dtype, list[int], int]],
) -> tuple[list[int], int]:
    """Lay out heterogeneous tensor bytes in one aligned host storage."""

    offsets = []
    end = 0
    for _name, dtype, _shape, nbytes in tensors:
        alignment = dtype.itemsize
        end = (end + alignment - 1) // alignment * alignment
        offsets.append(end)
        end += nbytes
    return offsets, end


def receive_rank_cache_transfer(
    sock,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    host_staging_pool: HostStagingBufferPool | None = None,
) -> RankCacheTransfer:
    """Receive and verify one payload into newly owned CPU tensors."""

    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    prefix = _recv_bytes(sock, WIRE_HEADER.size)
    magic, wire_version, header_bytes, body_bytes = WIRE_HEADER.unpack(prefix)
    if magic != WIRE_MAGIC or wire_version != WIRE_VERSION:
        raise ValueError("unsupported cache transfer wire format")
    if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
        raise ValueError("cache transfer wire header size is invalid")
    if body_bytes > max_payload_bytes:
        raise ValueError("cache transfer payload exceeds configured byte limit")

    digest = hashlib.sha256()
    raw_header = _recv_bytes(sock, header_bytes, digest)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cache transfer wire header is invalid") from exc
    descriptors = header.get("tensors")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("cache transfer wire tensor descriptors are invalid")

    tensor_descriptors: list[tuple[str, torch.dtype, list[int], int]] = []
    tensor_names: set[str] = set()
    described_body_bytes = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("cache transfer wire tensor descriptor is invalid")
        name = descriptor.get("name")
        dtype = _NAME_TO_DTYPE.get(descriptor.get("dtype"))
        shape = descriptor.get("shape")
        nbytes = descriptor.get("nbytes")
        if (
            not isinstance(name, str)
            or not name
            or name in tensor_names
            or dtype is None
            or not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            or not isinstance(nbytes, int)
            or nbytes <= 0
        ):
            raise ValueError("cache transfer wire tensor descriptor is invalid")
        expected_nbytes = dtype.itemsize
        for dimension in shape:
            expected_nbytes *= dimension
        if nbytes != expected_nbytes:
            raise ValueError("cache transfer wire tensor byte size is invalid")
        described_body_bytes += nbytes
        if described_body_bytes > body_bytes:
            raise ValueError("cache transfer wire body size is inconsistent")
        tensor_names.add(name)
        tensor_descriptors.append((name, dtype, shape, nbytes))
    if described_body_bytes != body_bytes:
        raise ValueError("cache transfer wire body size is inconsistent")

    offsets, storage_bytes = _aligned_storage_offsets(tensor_descriptors)
    lease = (
        host_staging_pool.acquire(storage_bytes, pin_memory=False)
        if host_staging_pool is not None
        else None
    )
    storage = (
        lease.storage
        if lease is not None
        else torch.empty(storage_bytes, dtype=torch.uint8)
    )
    if storage is None:
        raise RuntimeError("host staging lease was released before receive")
    try:
        tensors: dict[str, torch.Tensor] = {}
        for (name, dtype, shape, nbytes), offset in zip(
            tensor_descriptors,
            offsets,
        ):
            tensor_storage = storage[offset : offset + nbytes]
            storage_view = _byte_view(tensor_storage)
            _recv_exact(sock, storage_view)
            digest.update(storage_view)
            tensors[name] = tensor_storage.view(dtype).reshape(shape)

        received_digest = _recv_bytes(sock, WIRE_DIGEST_BYTES)
        if not hmac.compare_digest(digest.digest(), received_digest):
            raise ValueError("cache transfer payload checksum mismatch")

        kv_blocks = tensors.pop("kv_blocks")
        kv_scales = tensors.pop("kv_scales", None)
        recurrent = tuple(
            tensors.pop(f"recurrent/{index}")
            for index in range(sum(name.startswith("recurrent/") for name in tensors))
        )
        convolution = tuple(
            tensors.pop(f"convolution/{index}")
            for index in range(sum(name.startswith("convolution/") for name in tensors))
        )
        if tensors:
            raise ValueError("cache transfer wire tensor names are invalid")
        return RankCacheTransfer(
            format_version=int(header["format_version"]),
            transfer_id=header["transfer_id"],
            tensor_parallel_rank=int(header["tensor_parallel_rank"]),
            tensor_parallel_size=int(header["tensor_parallel_size"]),
            block_size=int(header["block_size"]),
            cached_tokens=int(header["cached_tokens"]),
            kv_blocks=kv_blocks,
            kv_scales=kv_scales,
            recurrent_states=recurrent,
            convolution_states=convolution,
            host_staging_lease=lease,
        )
    except KeyError as exc:
        if lease is not None:
            lease.release()
        raise ValueError("cache transfer wire data is incomplete") from exc
    except (TypeError, ValueError):
        if lease is not None:
            lease.release()
        raise
    except BaseException:
        if lease is not None:
            lease.release()
        raise


class RankCacheReceiver:
    """One-shot TCP listener that ACKs only a fully verified rank payload."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 30.0,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        host_staging_pool: HostStagingBufferPool | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("cache transfer receiver host must not be empty")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("cache transfer receiver port must be in [0, 65535]")
        if timeout_s <= 0:
            raise ValueError("cache transfer receiver timeout must be positive")
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.max_payload_bytes = max_payload_bytes
        self.host_staging_pool = host_staging_pool
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.settimeout(timeout_s)
        try:
            self._listener.bind((host, port))
            self._listener.listen(1)
        except BaseException:
            self._listener.close()
            raise

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._listener.getsockname()[:2]
        return str(host), int(port)

    def receive(
        self,
        *,
        timeout_s: float = 30.0,
        on_verified: Callable[[RankCacheTransfer], None] | None = None,
    ) -> RankCacheTransfer:
        if timeout_s <= 0:
            raise ValueError("cache transfer connection timeout must be positive")
        connection, _peer = self._listener.accept()
        with connection:
            connection.settimeout(timeout_s)
            try:
                payload = receive_rank_cache_transfer(
                    connection,
                    max_payload_bytes=self.max_payload_bytes,
                    host_staging_pool=self.host_staging_pool,
                )
                if on_verified is not None:
                    on_verified(payload)
            except BaseException:
                if "payload" in locals():
                    payload.release_host_staging()
                try:
                    connection.sendall(_TRANSFER_NACK)
                except OSError:
                    pass
                raise
            connection.sendall(_TRANSFER_ACK)
            return payload

    def close(self) -> None:
        self._listener.close()

    def __enter__(self) -> "RankCacheReceiver":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class PendingRankCacheReceive:
    """Receive one rank payload on CPU and defer its ACK until GPU install."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 30.0,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        host_staging_pool: HostStagingBufferPool | None = None,
    ) -> None:
        self._receiver = RankCacheReceiver(
            host,
            port,
            timeout_s=timeout_s,
            max_payload_bytes=max_payload_bytes,
            host_staging_pool=host_staging_pool,
        )
        self._timeout_s = timeout_s
        self._deadline = monotonic() + timeout_s
        self._lock = Lock()
        self._connection: socket.socket | None = None
        self._payload: RankCacheTransfer | None = None
        self._error: BaseException | None = None
        self._terminal = False
        self._thread = Thread(target=self._run, daemon=True)
        self._thread_started = False
        self._receiver._listener.settimeout(min(timeout_s, 0.05))

    @property
    def address(self) -> tuple[str, int]:
        return self._receiver.address

    def start(self) -> None:
        self._thread.start()
        self._thread_started = True

    def _run(self) -> None:
        connection = None
        try:
            while True:
                with self._lock:
                    if self._terminal:
                        return
                try:
                    connection, _peer = self._receiver._listener.accept()
                    break
                except socket.timeout:
                    if monotonic() >= self._deadline:
                        raise TimeoutError(
                            "cache transfer receive deadline expired"
                        )
            connection.settimeout(self._timeout_s)
            with self._lock:
                if self._terminal:
                    connection.close()
                    return
                self._connection = connection
            payload = receive_rank_cache_transfer(
                connection,
                max_payload_bytes=self._receiver.max_payload_bytes,
                host_staging_pool=self._receiver.host_staging_pool,
            )
            release_payload = False
            with self._lock:
                if not self._terminal:
                    self._payload = payload
                else:
                    release_payload = True
            if release_payload:
                payload.release_host_staging()
        except BaseException as exc:
            with self._lock:
                if not self._terminal:
                    self._error = exc
            if connection is not None:
                try:
                    connection.sendall(_TRANSFER_NACK)
                except OSError:
                    pass
                connection.close()
        finally:
            self._receiver.close()

    def poll(self) -> tuple[str, str | None]:
        with self._lock:
            if self._terminal:
                return "closed", None
            if self._error is not None:
                return "failed", str(self._error)
            if self._payload is not None:
                return "ready", None
            if monotonic() >= self._deadline:
                self._error = TimeoutError(
                    "cache transfer receive deadline expired"
                )
                return "failed", str(self._error)
            return "receiving", None

    def payload(self) -> RankCacheTransfer:
        with self._lock:
            if self._terminal:
                raise RuntimeError("cache receive is already closed")
            if self._error is not None:
                raise RuntimeError(f"cache receive failed: {self._error}")
            if self._payload is None:
                raise RuntimeError("cache receive payload is not ready")
            return self._payload

    def finish(self, *, accepted: bool) -> None:
        payload = None
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
            connection = self._connection
            self._connection = None
            payload = self._payload
            self._payload = None
        if payload is not None:
            payload.release_host_staging()
        if connection is not None:
            try:
                connection.sendall(_TRANSFER_ACK if accepted else _TRANSFER_NACK)
            finally:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        self._receiver.close()
        if self._thread_started and current_thread() is not self._thread:
            self._thread.join()


class PendingRankCacheSend:
    """Send one host-staged rank payload without blocking model scheduling."""

    def __init__(
        self,
        host: str,
        port: int,
        payload: RankCacheTransfer,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("cache transfer endpoint host must not be empty")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("cache transfer endpoint port must be in [1, 65535]")
        if timeout_s <= 0:
            raise ValueError("cache transfer endpoint timeout must be positive")
        self._host = host
        self._port = port
        self._payload = payload
        self._staged_bytes = payload.nbytes
        self._deadline = monotonic() + timeout_s
        self._lock = Lock()
        self._connection: socket.socket | None = None
        self._sent_bytes: int | None = None
        self._error: BaseException | None = None
        self._terminal = False
        self._thread = Thread(target=self._run, daemon=True)
        self._thread_started = False

    def start(self) -> None:
        self._thread.start()
        self._thread_started = True

    def _run(self) -> None:
        connection = None
        payload = None
        failure = None
        try:
            while True:
                with self._lock:
                    if self._terminal:
                        return
                try:
                    connection = socket.create_connection(
                        (self._host, self._port),
                        timeout=min(
                            max(self._deadline - monotonic(), 0.001),
                            0.05,
                        ),
                    )
                    break
                except (ConnectionRefusedError, TimeoutError, socket.timeout):
                    remaining = self._deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "cache transfer endpoint connection timed out"
                        )
                    sleep(min(0.01, remaining))
            with self._lock:
                if self._terminal:
                    connection.close()
                    return
                self._connection = connection
            connection.settimeout(
                max(self._deadline - monotonic(), 0.001)
            )
            with self._lock:
                payload = self._payload
            if payload is None:
                return
            sent_bytes = send_rank_cache_transfer(connection, payload)
            with self._lock:
                self._payload = None
                self._staged_bytes = 0
            payload.release_host_staging()
            payload = None
            acknowledgement = _recv_bytes(connection, 1)
            if acknowledgement != _TRANSFER_ACK:
                raise RuntimeError("cache transfer receiver rejected the payload")
            with self._lock:
                if not self._terminal and self._error is None:
                    self._sent_bytes = sent_bytes
                    self._payload = None
        except BaseException as exc:
            failure = exc
        finally:
            with self._lock:
                retained_payload = self._payload
                self._payload = None
                self._staged_bytes = 0
            if payload is None:
                payload = retained_payload
            if payload is not None:
                payload.release_host_staging()
            if connection is not None:
                connection.close()
            with self._lock:
                self._connection = None
                if (
                    failure is not None
                    and not self._terminal
                    and self._error is None
                ):
                    self._error = failure

    def poll(self) -> tuple[str, str | None]:
        timed_out = False
        connection = None
        with self._lock:
            if self._terminal:
                return "closed", None
            if self._error is not None:
                return "failed", str(self._error)
            if self._sent_bytes is not None:
                return "ready", None
            if monotonic() >= self._deadline:
                self._error = TimeoutError(
                    "cache transfer send deadline expired"
                )
                connection = self._connection
                timed_out = True
            else:
                return "sending", None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if timed_out and self._thread_started and current_thread() is not self._thread:
            self._thread.join()
        with self._lock:
            return "failed", str(self._error)

    @property
    def staged_bytes(self) -> int:
        """Return host payload bytes still retained by the sender."""

        with self._lock:
            return self._staged_bytes

    def result(self) -> int:
        with self._lock:
            if self._terminal:
                raise RuntimeError("cache send is already closed")
            if self._error is not None:
                raise RuntimeError(f"cache send failed: {self._error}")
            if self._sent_bytes is None:
                raise RuntimeError("cache send is not ready")
            return self._sent_bytes

    def finish(self) -> None:
        payload = None
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
            connection = self._connection
            self._connection = None
            if not self._thread_started:
                payload = self._payload
                self._payload = None
                self._staged_bytes = 0
        if payload is not None:
            payload.release_host_staging()
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if self._thread_started and current_thread() is not self._thread:
            self._thread.join()


def send_rank_cache_to_endpoint(
    host: str,
    port: int,
    payload: RankCacheTransfer,
    *,
    timeout_s: float = 30.0,
) -> int:
    """Connect, stream a payload, and wait for receiver validation."""

    if not isinstance(host, str) or not host:
        raise ValueError("cache transfer endpoint host must not be empty")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("cache transfer endpoint port must be in [1, 65535]")
    if timeout_s <= 0:
        raise ValueError("cache transfer endpoint timeout must be positive")
    deadline = monotonic() + timeout_s
    try:
        while True:
            try:
                connection = socket.create_connection(
                    (host, port),
                    timeout=max(deadline - monotonic(), 0.001),
                )
                break
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("cache transfer endpoint connection timed out")
                sleep(min(0.01, remaining))
        with connection:
            connection.settimeout(timeout_s)
            sent_bytes = send_rank_cache_transfer(connection, payload)
            acknowledgement = _recv_bytes(connection, 1)
        if acknowledgement != _TRANSFER_ACK:
            raise RuntimeError("cache transfer receiver rejected the payload")
        return sent_bytes
    finally:
        payload.release_host_staging()
