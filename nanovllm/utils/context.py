from dataclasses import dataclass
import torch


StatePrefillGroup = tuple[
    int,
    tuple[tuple[int, int], ...],
    torch.Tensor,
]


def build_state_prefill_groups(
    state_token_ranges: tuple[tuple[int, int], ...],
    state_slots: torch.Tensor | None,
    decode_token_count: int,
) -> tuple[StatePrefillGroup, ...]:
    """Group equal-length prefills once for reuse by every recurrent layer."""

    if not state_token_ranges or state_slots is None:
        return ()
    if decode_token_count < 0:
        raise ValueError("decode token count must be non-negative")
    if decode_token_count + len(state_token_ranges) > state_slots.numel():
        raise ValueError("recurrent prefill ranges exceed available state slots")
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for range_index, (start, end) in enumerate(state_token_ranges):
        if end <= start:
            raise ValueError("recurrent prefill ranges must be non-empty")
        grouped.setdefault(end - start, []).append(
            (start, end, decode_token_count + range_index)
        )
    result = []
    for sequence_length, ranges in grouped.items():
        slot_indices = state_slots.new_tensor(
            [slot_index for _, _, slot_index in ranges],
            dtype=torch.long,
        )
        result.append(
            (
                sequence_length,
                tuple(ranges),
                state_slots.index_select(0, slot_indices).to(torch.long),
            )
        )
    return tuple(result)


def build_state_reset_slots(
    state_slots: torch.Tensor | None,
    state_reset_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Select recurrent slots to reset once for reuse by every layer."""

    if state_slots is None or state_reset_mask is None:
        return None
    return state_slots[state_reset_mask]


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int = 0
    block_tables: torch.Tensor | None = None
    state_slots: torch.Tensor | None = None
    dequant_block_ids: torch.Tensor | None = None
    dequant_block_tables: torch.Tensor | None = None
    sliding_window_size: int | None = None
    is_mixed: bool = False
    decode_token_count: int = 0
    prefill_token_count: int = 0
    decode_context_lens: torch.Tensor | None = None
    decode_max_context_len: int = 0
    decode_block_tables: torch.Tensor | None = None
    decode_dequant_block_ids: torch.Tensor | None = None
    decode_dequant_block_tables: torch.Tensor | None = None
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None
    prefill_dequant_block_ids: torch.Tensor | None = None
    prefill_dequant_block_tables: torch.Tensor | None = None
    state_reset_mask: torch.Tensor | None = None
    state_reset_slots: torch.Tensor | None = None
    state_token_ranges: tuple[tuple[int, int], ...] = ()
    state_prefill_groups: tuple[StatePrefillGroup, ...] = ()
    decode_state_span: tuple[int, int] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    dequant_block_ids=None,
    dequant_block_tables=None,
    sliding_window_size=None,
    is_mixed=False,
    decode_token_count=0,
    prefill_token_count=0,
    decode_context_lens=None,
    decode_block_tables=None,
    decode_dequant_block_ids=None,
    decode_dequant_block_tables=None,
    prefill_cu_seqlens_q=None,
    prefill_cu_seqlens_k=None,
    prefill_max_seqlen_q=0,
    prefill_max_seqlen_k=0,
    prefill_block_tables=None,
    prefill_dequant_block_ids=None,
    prefill_dequant_block_tables=None,
    max_context_len=0,
    decode_max_context_len=0,
    state_slots=None,
    state_reset_mask=None,
    state_reset_slots=None,
    state_token_ranges=(),
    decode_state_span=None,
):
    global _CONTEXT
    state_prefill_groups = build_state_prefill_groups(
        state_token_ranges,
        state_slots,
        decode_token_count,
    )
    if state_reset_slots is None:
        state_reset_slots = build_state_reset_slots(
            state_slots,
            state_reset_mask,
        )
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        max_context_len=max_context_len,
        block_tables=block_tables,
        state_slots=state_slots,
        dequant_block_ids=dequant_block_ids,
        dequant_block_tables=dequant_block_tables,
        sliding_window_size=sliding_window_size,
        is_mixed=is_mixed,
        decode_token_count=decode_token_count,
        prefill_token_count=prefill_token_count,
        decode_context_lens=decode_context_lens,
        decode_max_context_len=decode_max_context_len,
        decode_block_tables=decode_block_tables,
        decode_dequant_block_ids=decode_dequant_block_ids,
        decode_dequant_block_tables=decode_dequant_block_tables,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=prefill_cu_seqlens_k,
        prefill_max_seqlen_q=prefill_max_seqlen_q,
        prefill_max_seqlen_k=prefill_max_seqlen_k,
        prefill_block_tables=prefill_block_tables,
        prefill_dequant_block_ids=prefill_dequant_block_ids,
        prefill_dequant_block_tables=prefill_dequant_block_tables,
        state_reset_mask=state_reset_mask,
        state_reset_slots=state_reset_slots,
        state_token_ranges=state_token_ranges,
        state_prefill_groups=state_prefill_groups,
        decode_state_span=decode_state_span,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
