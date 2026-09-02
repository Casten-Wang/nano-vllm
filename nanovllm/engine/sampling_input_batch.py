from __future__ import annotations

import torch


class SamplingInputBatch:
    """Persistent host/device buffers for per-request sampling parameters."""

    def __init__(
        self,
        capacity: int,
        *,
        device: torch.device | str = "cuda",
        pin_memory: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("sampling input capacity must be positive")
        self.capacity = capacity
        self.host_temperatures = torch.empty(
            capacity,
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_top_ks = torch.empty(
            capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_top_ps = torch.empty(
            capacity,
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.device_temperatures = torch.empty(
            capacity,
            dtype=torch.float32,
            device=device,
        )
        self.device_top_ks = torch.empty(
            capacity,
            dtype=torch.int32,
            device=device,
        )
        self.device_top_ps = torch.empty(
            capacity,
            dtype=torch.float32,
            device=device,
        )
        self._temperature_array = self.host_temperatures.numpy()
        self._top_k_array = self.host_top_ks.numpy()
        self._top_p_array = self.host_top_ps.numpy()

    def update(
        self,
        temperatures: list[float],
        top_ks: list[int],
        top_ps: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = len(temperatures)
        if len(top_ks) != size or len(top_ps) != size:
            raise ValueError("sampling parameter batch sizes must match")
        if not 0 < size <= self.capacity:
            raise ValueError(
                f"sampling batch size must be in [1, {self.capacity}]"
            )
        self._temperature_array[:size] = temperatures
        self._top_k_array[:size] = top_ks
        self._top_p_array[:size] = top_ps
        self.device_temperatures[:size].copy_(
            self.host_temperatures[:size],
            non_blocking=True,
        )
        self.device_top_ks[:size].copy_(
            self.host_top_ks[:size],
            non_blocking=True,
        )
        self.device_top_ps[:size].copy_(
            self.host_top_ps[:size],
            non_blocking=True,
        )
        return (
            self.device_temperatures[:size],
            self.device_top_ks[:size],
            self.device_top_ps[:size],
        )
