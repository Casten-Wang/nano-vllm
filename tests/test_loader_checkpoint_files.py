import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nanovllm.utils.loader import load_model


def _write_sharded_checkpoint(root: Path, *, complete: bool) -> None:
    names = [f"model-{index:05d}-of-00002.safetensors" for index in (1, 2)]
    tensors = (
        {"0.weight": torch.ones(2, 2)},
        {"1.weight": torch.full((2, 2), 2.0)},
    )
    save_file(tensors[0], root / names[0])
    if complete:
        save_file(tensors[1], root / names[1])
    weight_map = {
        weight: names[index]
        for index, shard in enumerate(tensors)
        for weight in shard
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def test_empty_model_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors files found"):
        load_model(nn.Linear(2, 2, bias=False), str(tmp_path))


def test_missing_indexed_shard_is_rejected_before_partial_loading(tmp_path):
    _write_sharded_checkpoint(tmp_path, complete=False)
    model = nn.Sequential(*(nn.Linear(2, 2, bias=False) for _ in range(2)))
    untouched = model[0].weight.detach().clone()

    with pytest.raises(
        FileNotFoundError,
        match="model-00002-of-00002.safetensors",
    ):
        load_model(model, str(tmp_path))

    torch.testing.assert_close(model[0].weight, untouched)


def test_invalid_checkpoint_index_is_rejected_before_loading(tmp_path):
    save_file({"weight": torch.ones(2, 2)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}}),
        encoding="utf-8",
    )
    model = nn.Linear(2, 2, bias=False)
    untouched = model.weight.detach().clone()

    with pytest.raises(ValueError, match="invalid safetensors weight map"):
        load_model(model, str(tmp_path))

    torch.testing.assert_close(model.weight, untouched)


def test_complete_indexed_checkpoint_loads_all_shards(tmp_path):
    _write_sharded_checkpoint(tmp_path, complete=True)
    model = nn.Sequential(*(nn.Linear(2, 2, bias=False) for _ in range(2)))

    load_model(model, str(tmp_path))

    torch.testing.assert_close(model[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(model[1].weight, torch.full((2, 2), 2.0))
