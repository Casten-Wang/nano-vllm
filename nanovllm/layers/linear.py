import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)
        self.weight.safetensors_loader = self.safetensors_loader
        if self.bias is not None:
            self.bias.safetensors_loader = self.safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def safetensors_loader(self, param: nn.Parameter, loaded_weight):
        shape = tuple(loaded_weight.get_shape())
        expected_shape = list(param.shape)
        expected_shape[self.tp_dim] *= self.tp_size
        if shape != tuple(expected_shape):
            raise ValueError(
                f"invalid column-parallel weight shape: {shape}; "
                f"expected {tuple(expected_shape)}"
            )
        shard_size = param.shape[self.tp_dim]
        start = self.tp_rank * shard_size
        index = [slice(None)] * len(shape)
        index[self.tp_dim] = slice(start, start + shard_size)
        param.data.copy_(loaded_weight[tuple(index)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
        self.weight.packed_safetensors_loader = self.packed_safetensors_loader
        if self.bias is not None:
            self.bias.packed_safetensors_loader = self.packed_safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def packed_safetensors_loader(
        self,
        param: nn.Parameter,
        loaded_weight,
        loaded_shard_id: int,
    ):
        if not 0 <= loaded_shard_id < len(self.output_sizes):
            raise ValueError("invalid merged column shard id")
        shape = tuple(loaded_weight.get_shape())
        expected_shape = (self.output_sizes[loaded_shard_id], *param.shape[1:])
        if shape != expected_shape:
            raise ValueError(
                f"invalid merged column weight shape: {shape}; "
                f"expected {expected_shape}"
            )
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        source_start = self.tp_rank * shard_size
        target_start = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        target = param.data.narrow(self.tp_dim, target_start, shard_size)
        index = (slice(source_start, source_start + shard_size),) + (
            slice(None),
        ) * (len(shape) - 1)
        target.copy_(loaded_weight[index])


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        output_size = (
            self.num_heads + 2 * self.num_kv_heads
        ) * self.head_size * tp_size
        super().__init__(hidden_size, output_size, bias)
        self.weight.packed_safetensors_loader = self.packed_safetensors_loader
        if self.bias is not None:
            self.bias.packed_safetensors_loader = self.packed_safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        shard_rank = (
            self.tp_rank
            if loaded_shard_id == "q"
            else self.tp_rank // self.num_kv_head_replicas
        )
        loaded_weight = loaded_weight.narrow(
            self.tp_dim,
            shard_rank * shard_size,
            shard_size,
        )
        param_data.copy_(loaded_weight)

    def packed_safetensors_loader(
        self,
        param: nn.Parameter,
        loaded_weight,
        loaded_shard_id: str,
    ):
        if loaded_shard_id not in ("q", "k", "v"):
            raise ValueError("invalid QKV shard id")
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            target_start = 0
            source_rank = self.tp_rank
            source_rows = shard_size * self.tp_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            target_start = self.num_heads * self.head_size
            if loaded_shard_id == "v":
                target_start += shard_size
            source_rank = self.tp_rank // self.num_kv_head_replicas
            source_rows = shard_size * (
                self.tp_size // self.num_kv_head_replicas
            )
        shape = tuple(loaded_weight.get_shape())
        expected_shape = (source_rows, *param.shape[1:])
        if shape != expected_shape:
            raise ValueError(
                f"invalid QKV {loaded_shard_id} weight shape: {shape}; "
                f"expected {expected_shape}"
            )
        source_start = source_rank * shard_size
        target = param.data.narrow(self.tp_dim, target_start, shard_size)
        index = (slice(source_start, source_start + shard_size),) + (
            slice(None),
        ) * (len(shape) - 1)
        target.copy_(loaded_weight[index])


class KVParallelLinear(ColumnParallelLinear):
    """Shard KV heads, replicating source heads when TP is larger."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_kv_heads: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        self.head_size = head_size
        self.total_num_kv_heads = total_num_kv_heads
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        super().__init__(
            hidden_size,
            self.num_kv_heads * head_size * tp_size,
            bias,
        )
        self.weight.safetensors_loader = self.safetensors_loader
        if self.bias is not None:
            self.bias.safetensors_loader = self.safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = self.num_kv_heads * self.head_size
        source_rank = self.tp_rank // self.num_kv_head_replicas
        param.data.copy_(
            loaded_weight.narrow(self.tp_dim, source_rank * shard_size, shard_size)
        )

    def safetensors_loader(self, param: nn.Parameter, loaded_weight):
        shape = tuple(loaded_weight.get_shape())
        expected_shape = (
            self.total_num_kv_heads * self.head_size,
            *param.shape[1:],
        )
        if shape != expected_shape:
            raise ValueError(
                f"invalid KV-parallel weight shape: {shape}; "
                f"expected {expected_shape}"
            )
        shard_size = self.num_kv_heads * self.head_size
        source_rank = self.tp_rank // self.num_kv_head_replicas
        start = source_rank * shard_size
        index = (slice(start, start + shard_size),) + (slice(None),) * (
            len(shape) - 1
        )
        param.data.copy_(loaded_weight[index])


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)
        self.weight.safetensors_loader = self.safetensors_loader
        if self.bias is not None:
            self.bias.safetensors_loader = self.safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def safetensors_loader(self, param: nn.Parameter, loaded_weight):
        shape = tuple(loaded_weight.get_shape())
        if param.ndim == 1:
            if shape != tuple(param.shape):
                raise ValueError("invalid row-parallel bias shape")
            param.data.copy_(loaded_weight[:])
            return
        expected_shape = (param.shape[0], param.shape[1] * self.tp_size)
        if shape != expected_shape:
            raise ValueError(
                f"invalid row-parallel weight shape: {shape}; "
                f"expected {expected_shape}"
            )
        shard_size = param.shape[self.tp_dim]
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight[:, start : start + shard_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
