import torch.distributed as dist

from nanovllm.engine.llm_engine import _create_distributed_store


def test_parent_owned_stores_reserve_distinct_ports():
    first = _create_distributed_store(None)
    second = _create_distributed_store(None)

    assert first.port != second.port
    first_client = dist.TCPStore("127.0.0.1", first.port, None, False)
    second_client = dist.TCPStore("127.0.0.1", second.port, None, False)
    first.set("engine", "first")
    second.set("engine", "second")
    assert first_client.get("engine") == b"first"
    assert second_client.get("engine") == b"second"
