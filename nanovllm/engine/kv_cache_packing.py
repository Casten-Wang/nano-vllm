from dataclasses import dataclass
from typing import Protocol


class BlockTableSequence(Protocol):
    block_table: list[int]

    def __len__(self) -> int:
        ...


@dataclass(slots=True)
class PackedBlockMetadata:
    selected_block_ids: list[int]
    packed_block_tables: list[list[int]]


def build_packed_block_metadata(
    seqs: list[BlockTableSequence],
    block_size: int,
    sliding_window_size: int | None = None,
    seq_lens: list[int] | None = None,
    query_start_lens: list[int] | None = None,
) -> PackedBlockMetadata:
    """Build deterministic metadata for INT8 selective KV dequantization.

    selected_block_ids maps packed block ids back to the original physical KV
    block ids. packed_block_tables keeps each sequence's logical block order
    but rewrites physical block ids to packed ids, padding shorter rows with -1.

    When sliding_window_size is set, only logical blocks intersecting the query
    window are selected. For decode, the query starts at seq_len - 1. For
    chunked prefill, query_start_lens should point to the first query token in
    the chunk, so the selected blocks cover the union of all local-attention
    windows inside that chunk.

    Rows still keep full logical block positions, because FlashAttention
    interprets block_table entries together with cache_seqlens and window_size.
    """
    assert block_size > 0
    assert sliding_window_size is None or sliding_window_size > 0
    if seq_lens is not None and len(seq_lens) != len(seqs):
        raise ValueError("seq_lens must have the same length as seqs")
    if query_start_lens is not None and len(query_start_lens) != len(seqs):
        raise ValueError("query_start_lens must have the same length as seqs")

    physical_to_packed: dict[int, int] = {}
    selected_block_ids: list[int] = []
    packed_block_tables: list[list[int]] = []
    max_num_blocks = 0

    for seq_idx, seq in enumerate(seqs):
        seq_len = len(seq) if seq_lens is None else seq_lens[seq_idx]
        if seq_len < 0:
            raise ValueError(f"seq_lens[{seq_idx}] must be non-negative")
        if seq_len > len(seq):
            raise ValueError(
                f"seq_lens[{seq_idx}] is {seq_len}, "
                f"but sequence length is only {len(seq)}"
            )
        num_blocks = (seq_len + block_size - 1) // block_size
        if len(seq.block_table) < num_blocks:
            raise ValueError(
                f"block_table has {len(seq.block_table)} blocks, "
                f"but sequence length requires {num_blocks}"
            )
        if query_start_lens is None:
            query_start = max(0, seq_len - 1)
        else:
            query_start = query_start_lens[seq_idx]
            if query_start < 0:
                raise ValueError(f"query_start_lens[{seq_idx}] must be non-negative")
            if seq_len == 0:
                if query_start != 0:
                    raise ValueError(
                        f"query_start_lens[{seq_idx}] is {query_start}, "
                        "but sequence length is zero"
                    )
            elif query_start >= seq_len:
                raise ValueError(
                    f"query_start_lens[{seq_idx}] is {query_start}, "
                    f"but sequence length is only {seq_len}"
                )
        max_num_blocks = max(max_num_blocks, num_blocks)

        if sliding_window_size is None:
            start_block = 0
        else:
            # The earliest query token controls how far left this chunk can
            # attend. Decode is the special case query_start == seq_len - 1.
            # Later query tokens only move the right edge forward, which is
            # already covered by num_blocks.
            window_start = max(0, query_start - sliding_window_size + 1)
            start_block = window_start // block_size

        row: list[int] = [-1] * num_blocks
        for logical_block_id in range(start_block, num_blocks):
            physical_block_id = seq.block_table[logical_block_id]
            if physical_block_id not in physical_to_packed:
                physical_to_packed[physical_block_id] = len(selected_block_ids)
                selected_block_ids.append(physical_block_id)
            row[logical_block_id] = physical_to_packed[physical_block_id]
        packed_block_tables.append(row)

    for row in packed_block_tables:
        row.extend([-1] * (max_num_blocks - len(row)))

    return PackedBlockMetadata(
        selected_block_ids=selected_block_ids,
        packed_block_tables=packed_block_tables,
    )
