from datetime import timedelta
import os
from queue import Empty
import sys
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanovllm.engine.llm_engine import _create_distributed_store


def _run_gloo_worker(port: int, rank: int, result_queue) -> None:
    try:
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        store = dist.TCPStore(
            "127.0.0.1",
            port,
            None,
            False,
            timedelta(seconds=10),
        )
        dist.init_process_group(
            "gloo",
            store=store,
            world_size=2,
            rank=rank,
            timeout=timedelta(seconds=10),
        )
        value = torch.tensor([rank + 1.0])
        dist.all_reduce(value)
        result_queue.put(("ok", port, rank, value.item()))
        dist.destroy_process_group()
    except BaseException:
        result_queue.put(("error", port, rank, traceback.format_exc()))


@pytest.mark.skipif(
    not dist.is_gloo_available(),
    reason="requires the PyTorch Gloo backend",
)
def test_parent_owned_stores_initialize_isolated_spawned_process_groups():
    stores = [_create_distributed_store(None) for _ in range(2)]
    assert stores[0].port != stores[1].port

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_run_gloo_worker,
            args=(store.port, rank, result_queue),
        )
        for store in stores
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        try:
            results = [result_queue.get(timeout=20) for _ in processes]
        except Empty as error:
            raise AssertionError("spawned process group did not complete") from error
        assert all(result[0] == "ok" for result in results), results
        assert sorted(result[3] for result in results) == [3.0] * 4
        assert {result[1] for result in results} == {store.port for store in stores}
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
