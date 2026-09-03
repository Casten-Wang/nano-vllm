from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.prefix_cache_queries = 0
        self.prefix_cache_checked_blocks = 0
        self.prefix_cache_hit_blocks = 0

    @property
    def num_total_blocks(self):
        return len(self.blocks)

    @property
    def num_used_blocks(self):
        return len(self.used_block_ids)

    @property
    def num_free_blocks(self):
        return len(self.free_block_ids)

    @property
    def usage(self):
        if not self.blocks:
            return 0.0
        return len(self.used_block_ids) / len(self.blocks)

    @property
    def prefix_cache_hit_rate(self):
        if self.prefix_cache_checked_blocks == 0:
            return 0.0
        return self.prefix_cache_hit_blocks / self.prefix_cache_checked_blocks

    def cache_stats(self):
        return {
            "prefix_cache_queries": self.prefix_cache_queries,
            "prefix_cache_checked_blocks": self.prefix_cache_checked_blocks,
            "prefix_cache_hit_blocks": self.prefix_cache_hit_blocks,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
        }

    def reset_cache_stats(self):
        self.prefix_cache_queries = 0
        self.prefix_cache_checked_blocks = 0
        self.prefix_cache_hit_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("no free KV blocks available")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_num_cached_blocks(self, seq: Sequence) -> int:
        """Return the reusable complete prefix blocks for ``seq``.

        The final logical block is intentionally excluded, matching the
        original nano-vLLM prefix-cache policy: a request's current tail block
        is owned by that request until a later request can reuse it safely.
        """

        return self._count_cached_blocks(seq, record_stats=True)

    def peek_num_cached_blocks(self, seq: Sequence) -> int:
        """Forecast reusable prefix blocks without changing hit metrics."""

        return self._count_cached_blocks(seq, record_stats=False)

    def _count_cached_blocks(
        self,
        seq: Sequence,
        *,
        record_stats: bool,
    ) -> int:
        if record_stats:
            self.prefix_cache_queries += 1
        h = -1
        num_cached_blocks = 0
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            if record_stats:
                self.prefix_cache_checked_blocks += 1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            if record_stats:
                self.prefix_cache_hit_blocks += 1
            num_cached_blocks += 1
        return num_cached_blocks

    def _get_cached_block_ids(
        self,
        seq: Sequence,
        num_cached_blocks: int,
    ) -> list[int]:
        """Resolve and validate the cached block IDs used by ``seq``.

        ``get_num_cached_blocks`` validates token IDs while walking the
        prefix-cache index.  Allocation and admission must repeat that
        validation because a free block may have been reused between the two
        calls, invalidating the old hash mapping.
        """

        if not 0 <= num_cached_blocks <= seq.num_blocks:
            raise ValueError(
                f"num_cached_blocks must be in [0, {seq.num_blocks}], "
                f"got {num_cached_blocks}"
            )
        h = -1
        cached_block_ids = []
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1:
                raise RuntimeError(
                    "prefix-cache entry disappeared before allocation"
                )
            block = self.blocks[block_id]
            if block.token_ids != token_ids:
                raise RuntimeError(
                    "prefix-cache entry changed before allocation"
                )
            cached_block_ids.append(block_id)
        return cached_block_ids

    def can_allocate(
        self,
        seq: Sequence,
        num_blocks: int | None = None,
        num_cached_blocks: int | None = None,
        reserve_free_blocks: int = 0,
    ) -> int:
        """Return reusable prefix blocks if enough free blocks are available.

        ``num_blocks`` limits allocation to the part of a prompt that will be
        processed in the current prefill chunk. This lets dynamic chunked
        prefill grow a request's block table incrementally instead of reserving
        every prompt block up front.
        """

        target_num_blocks = seq.num_blocks if num_blocks is None else num_blocks
        if not 0 <= target_num_blocks <= seq.num_blocks:
            raise ValueError(
                f"num_blocks must be in [0, {seq.num_blocks}], got {target_num_blocks}"
            )
        if num_cached_blocks is None:
            num_cached_blocks = self.get_num_cached_blocks(seq)
        if not 0 <= num_cached_blocks <= target_num_blocks:
            raise ValueError(
                "num_cached_blocks must be between zero and num_blocks"
            )
        if reserve_free_blocks < 0:
            raise ValueError("reserve_free_blocks must be non-negative")
        cached_block_ids = self._get_cached_block_ids(seq, num_cached_blocks)
        cached_blocks_requiring_free_slots = sum(
            block_id not in self.used_block_ids for block_id in cached_block_ids
        )
        num_new_blocks = (
            target_num_blocks
            - num_cached_blocks
            + cached_blocks_requiring_free_slots
        )
        if num_new_blocks and (
            len(self.free_block_ids) < num_new_blocks + reserve_free_blocks
        ):
            return -1
        return num_cached_blocks

    def allocate(
        self,
        seq: Sequence,
        num_cached_blocks: int,
        num_blocks: int | None = None,
    ):
        target_num_blocks = seq.num_blocks if num_blocks is None else num_blocks
        assert not seq.block_table
        if not 0 <= num_cached_blocks <= target_num_blocks <= seq.num_blocks:
            raise ValueError("invalid cached/target block counts")
        cached_block_ids = self._get_cached_block_ids(seq, num_cached_blocks)
        for i, block_id in enumerate(cached_block_ids):
            token_ids = seq.block(i)
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, target_num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def can_grow(
        self,
        seq: Sequence,
        num_blocks: int,
        reserve_free_blocks: int = 0,
    ) -> bool:
        if not len(seq.block_table) <= num_blocks <= seq.num_blocks:
            raise ValueError(
                f"num_blocks must be in [{len(seq.block_table)}, {seq.num_blocks}]"
            )
        if reserve_free_blocks < 0:
            raise ValueError("reserve_free_blocks must be non-negative")
        growth = num_blocks - len(seq.block_table)
        return growth == 0 or len(self.free_block_ids) >= growth + reserve_free_blocks

    def grow(self, seq: Sequence, num_blocks: int):
        if not self.can_grow(seq, num_blocks):
            raise RuntimeError("insufficient free KV blocks to grow sequence")
        while len(seq.block_table) < num_blocks:
            seq.block_table.append(self._allocate_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        self._hash_block_range(seq, start, end)

    def hash_imported_prompt(self, seq: Sequence):
        """Index complete prompt blocks whose KV was populated remotely."""

        self._hash_block_range(seq, 0, seq.num_prompt_tokens // self.block_size)

    def _hash_block_range(self, seq: Sequence, start: int, end: int):
        if start == end:
            return
        if not 0 <= start < end <= len(seq.block_table):
            raise ValueError("KV hash range exceeds the allocated block table")
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
