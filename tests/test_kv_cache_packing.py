import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nanovllm" / "engine" / "kv_cache_packing.py"
SPEC = importlib.util.spec_from_file_location("kv_cache_packing", MODULE_PATH)
kv_cache_packing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kv_cache_packing)
build_packed_block_metadata = kv_cache_packing.build_packed_block_metadata


class DummySequence:
    def __init__(self, length: int, block_table: list[int]):
        self.length = length
        self.block_table = block_table

    def __len__(self):
        return self.length


def test_build_packed_block_metadata_deduplicates_blocks_in_stable_order():
    seqs = [
        DummySequence(3 * 256, [7, 10, 25]),
        DummySequence(2 * 256, [3, 8]),
        DummySequence(2 * 256, [10, 30]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=256)

    assert metadata.selected_block_ids == [7, 10, 25, 3, 8, 30]
    assert metadata.packed_block_tables == [
        [0, 1, 2],
        [3, 4, -1],
        [1, 5, -1],
    ]


def test_build_packed_block_metadata_reuses_shared_physical_block():
    seqs = [
        DummySequence(256, [42]),
        DummySequence(256, [42]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=256)

    assert metadata.selected_block_ids == [42]
    assert metadata.packed_block_tables == [
        [0],
        [0],
    ]


def test_build_packed_block_metadata_includes_partial_last_block():
    seqs = [
        DummySequence(5, [11, 12]),
        DummySequence(8, [12, 13]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=4)

    assert metadata.selected_block_ids == [11, 12, 13]
    assert metadata.packed_block_tables == [
        [0, 1],
        [1, 2],
    ]


def test_build_packed_block_metadata_rejects_missing_block_table_entries():
    seqs = [DummySequence(9, [5, 6])]

    try:
        build_packed_block_metadata(seqs, block_size=4)
    except ValueError as exc:
        assert "block_table has 2 blocks" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_packed_block_metadata_handles_empty_batch():
    metadata = build_packed_block_metadata([], block_size=256)

    assert metadata.selected_block_ids == []
    assert metadata.packed_block_tables == []


def test_build_packed_block_metadata_selects_only_sliding_window_blocks():
    seqs = [
        DummySequence(14, [20, 7, 35, 41]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=4, sliding_window_size=6)

    assert metadata.selected_block_ids == [35, 41]
    assert metadata.packed_block_tables == [
        [-1, -1, 0, 1],
    ]


def test_build_packed_block_metadata_window_deduplicates_across_requests():
    seqs = [
        DummySequence(14, [20, 7, 35, 41]),
        DummySequence(9, [5, 6, 35]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=4, sliding_window_size=6)

    assert metadata.selected_block_ids == [35, 41, 5, 6]
    assert metadata.packed_block_tables == [
        [-1, -1, 0, 1],
        [2, 3, 0, -1],
    ]


def test_build_packed_block_metadata_window_keeps_short_contexts_complete():
    seqs = [
        DummySequence(5, [11, 12]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=4, sliding_window_size=32)

    assert metadata.selected_block_ids == [11, 12]
    assert metadata.packed_block_tables == [
        [0, 1],
    ]


def test_build_packed_block_metadata_can_pack_prefix_length_only():
    seqs = [
        DummySequence(20, [1, 2, 3, 4, 5]),
    ]

    metadata = build_packed_block_metadata(seqs, block_size=4, seq_lens=[9])

    assert metadata.selected_block_ids == [1, 2, 3]
    assert metadata.packed_block_tables == [
        [0, 1, 2],
    ]


def test_build_packed_block_metadata_window_covers_whole_prefill_chunk():
    seqs = [
        DummySequence(20, [1, 2, 3, 4, 5]),
    ]

    metadata = build_packed_block_metadata(
        seqs,
        block_size=4,
        sliding_window_size=3,
        seq_lens=[16],
        query_start_lens=[8],
    )

    assert metadata.selected_block_ids == [2, 3, 4]
    assert metadata.packed_block_tables == [
        [-1, 0, 1, 2],
    ]


def test_build_packed_block_metadata_rejects_query_start_beyond_sequence_length():
    seqs = [DummySequence(8, [1, 2])]

    try:
        build_packed_block_metadata(seqs, block_size=4, query_start_lens=[8])
    except ValueError as exc:
        assert "query_start_lens[0] is 8" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_packed_block_metadata_rejects_seq_len_beyond_sequence_length():
    seqs = [DummySequence(8, [1, 2])]

    try:
        build_packed_block_metadata(seqs, block_size=4, seq_lens=[9])
    except ValueError as exc:
        assert "sequence length is only 8" in str(exc)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_build_packed_block_metadata_deduplicates_blocks_in_stable_order()
    test_build_packed_block_metadata_reuses_shared_physical_block()
    test_build_packed_block_metadata_includes_partial_last_block()
    test_build_packed_block_metadata_rejects_missing_block_table_entries()
    test_build_packed_block_metadata_handles_empty_batch()
    test_build_packed_block_metadata_selects_only_sliding_window_blocks()
    test_build_packed_block_metadata_window_deduplicates_across_requests()
    test_build_packed_block_metadata_window_keeps_short_contexts_complete()
    test_build_packed_block_metadata_can_pack_prefix_length_only()
    test_build_packed_block_metadata_window_covers_whole_prefill_chunk()
    test_build_packed_block_metadata_rejects_query_start_beyond_sequence_length()
    test_build_packed_block_metadata_rejects_seq_len_beyond_sequence_length()
