"""Audit Hugging Face safetensors shapes using HTTP headers only."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_checkpoint_mapping import (
    instantiate_meta_model,
    parameter_storage_bytes,
    validate_weight_shape,
)
from nanovllm.utils.loader import resolve_packed_parameter
from nanovllm.models.quantization_spec import (
    QuantizationSpec,
    resolve_quantization_spec,
)


def fetch_bytes(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        payload = response.read()
        if byte_range is not None:
            expected = byte_range[1] - byte_range[0] + 1
            if response.status != 206 or len(payload) != expected:
                raise RuntimeError(
                    f"server did not honor byte range for {url}: "
                    f"status={response.status}, bytes={len(payload)}"
                )
        return payload


def fetch_safetensors_header(url: str, max_header_bytes: int) -> tuple[dict, int]:
    prefix = fetch_bytes(url, (0, 7))
    header_length = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_length <= max_header_bytes:
        raise ValueError(
            f"invalid safetensors header length {header_length} for {url}"
        )
    raw_header = fetch_bytes(url, (8, 7 + header_length))
    document = json.loads(raw_header)
    tensors = {
        name: metadata
        for name, metadata in document.items()
        if name != "__metadata__"
    }
    return tensors, 8 + header_length


class HeaderSlice:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata

    def get_shape(self):
        return self.metadata["shape"]

    def get_dtype(self):
        return self.metadata["dtype"]


class HeaderCheckpoint:
    def __init__(self, headers: dict[str, dict]) -> None:
        self.headers = headers

    def get_slice(self, name: str) -> HeaderSlice:
        return HeaderSlice(self.headers[name])


ALLOWED_TEXT_ONLY_SKIP_PREFIXES = ("model.visual.", "mtp.")


def _tensor_shape(metadata: dict, name: str) -> tuple[int, ...]:
    shape = metadata.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise ValueError(f"invalid shape metadata for {name}")
    return tuple(shape)


def _logical_header(metadata: dict, shape: tuple[int, ...]) -> dict:
    return {"dtype": "BF16", "shape": list(shape)}


def audit_quantized_tensor_groups(
    headers: dict[str, dict],
    quantization: QuantizationSpec,
) -> tuple[dict[str, dict], dict]:
    """Validate serialized quantization groups and reconstruct logical weights."""

    logical_headers = {}
    errors = []
    groups = []
    auxiliary_names = set()

    if quantization.format == "fp8_block":
        block_rows, block_columns = quantization.weight_block_size or (0, 0)
        for name, metadata in sorted(headers.items()):
            if not name.endswith(".weight") or metadata.get("dtype") != "F8_E4M3":
                continue
            module_name = name.removesuffix(".weight")
            scale_name = f"{name}_scale_inv"
            scale = headers.get(scale_name)
            auxiliary_names.add(name)
            auxiliary_names.add(scale_name)
            shape = _tensor_shape(metadata, name)
            if len(shape) != 2:
                errors.append(f"{name}: FP8 weight must be rank 2, got {shape}")
                continue
            if quantization.ignores_module(module_name):
                errors.append(f"{name}: ignored module is stored as FP8")
            if scale is None:
                errors.append(f"{name}: missing {scale_name}")
            else:
                scale_shape = _tensor_shape(scale, scale_name)
                expected_scale_shape = (
                    ceil(shape[0] / block_rows),
                    ceil(shape[1] / block_columns),
                )
                if scale.get("dtype") not in ("F16", "BF16", "F32"):
                    errors.append(
                        f"{scale_name}: scale dtype must be floating point"
                    )
                if scale_shape != expected_scale_shape:
                    errors.append(
                        f"{scale_name}: shape {scale_shape} != "
                        f"{expected_scale_shape}"
                    )
            groups.append(name)
            logical_headers[name] = _logical_header(metadata, shape)

        recognized_suffixes = (".weight_scale_inv",)
    elif quantization.format == "gptq_int4":
        pack_factor = 32 // quantization.weight_bits
        group_size = quantization.group_size or 0
        for name, qweight in sorted(headers.items()):
            if not name.endswith(".qweight"):
                continue
            module_name = name.removesuffix(".qweight")
            names = {
                suffix: f"{module_name}.{suffix}"
                for suffix in ("qweight", "qzeros", "scales", "g_idx")
            }
            auxiliary_names.update(names.values())
            tensors = {suffix: headers.get(item) for suffix, item in names.items()}
            missing = [names[suffix] for suffix, value in tensors.items() if value is None]
            if missing:
                errors.append(f"{module_name}: missing quantized tensors {missing}")
                continue
            shapes = {
                suffix: _tensor_shape(value, names[suffix])
                for suffix, value in tensors.items()
            }
            if quantization.ignores_module(module_name):
                errors.append(f"{module_name}: ignored module is stored as GPTQ")
            if tensors["qweight"].get("dtype") != "I32":
                errors.append(f"{names['qweight']}: dtype must be I32")
            if tensors["qzeros"].get("dtype") != "I32":
                errors.append(f"{names['qzeros']}: dtype must be I32")
            if tensors["g_idx"].get("dtype") != "I32":
                errors.append(f"{names['g_idx']}: dtype must be I32")
            if tensors["scales"].get("dtype") not in ("F16", "BF16", "F32"):
                errors.append(f"{names['scales']}: dtype must be floating point")
            if len(shapes["g_idx"]) != 1 or len(shapes["scales"]) != 2:
                errors.append(f"{module_name}: invalid g_idx or scales rank")
                continue
            input_size = shapes["g_idx"][0]
            group_count = ceil(input_size / group_size)
            output_size = shapes["scales"][1]
            expected = {
                "qweight": (ceil(input_size / pack_factor), output_size),
                "qzeros": (group_count, ceil(output_size / pack_factor)),
                "scales": (group_count, output_size),
                "g_idx": (input_size,),
            }
            for suffix, expected_shape in expected.items():
                if shapes[suffix] != expected_shape:
                    errors.append(
                        f"{names[suffix]}: shape {shapes[suffix]} != "
                        f"{expected_shape}"
                    )
            groups.append(module_name)
            logical_name = f"{module_name}.weight"
            logical_headers[logical_name] = _logical_header(
                qweight,
                (output_size, input_size),
            )

        recognized_suffixes = (".qweight", ".qzeros", ".scales", ".g_idx")
    else:
        raise ValueError(
            f"quantized tensor audit does not support {quantization.format!r}"
        )

    for name, metadata in headers.items():
        if name in auxiliary_names:
            continue
        if name.endswith(recognized_suffixes):
            errors.append(f"{name}: orphan quantization tensor")
            continue
        logical_headers[name] = metadata

    return logical_headers, {
        "format": quantization.format,
        "quantized_group_count": len(groups),
        "logical_tensor_count": len(logical_headers),
        "errors": errors,
        "valid": bool(groups) and not errors,
    }


def audit_quantized_tp_layout(
    logical_headers: dict[str, dict],
    quantization: QuantizationSpec,
    tp_size: int,
) -> dict:
    """Report tensor-parallel splits that cross serialized quantization units."""

    column_parallel = {"gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"}
    row_parallel = {"down_proj", "o_proj"}
    errors = []
    partial_units = []
    partial_unit_count = 0
    checked = 0
    for name, metadata in sorted(logical_headers.items()):
        if not name.endswith(".weight"):
            continue
        if name.startswith(ALLOWED_TEXT_ONLY_SKIP_PREFIXES):
            continue
        module = name.removesuffix(".weight")
        leaf = module.rsplit(".", 1)[-1]
        if leaf in column_parallel:
            axis = 0
        elif leaf in row_parallel:
            axis = 1
        else:
            continue
        if quantization.ignores_module(module):
            continue
        shape = _tensor_shape(metadata, name)
        if len(shape) != 2:
            continue
        checked += 1
        dimension = shape[axis]
        if dimension % tp_size:
            errors.append(
                f"{name}: sharded dimension {dimension} is not divisible by TP{tp_size}"
            )
            continue
        local_dimension = dimension // tp_size
        unit = (
            quantization.weight_block_size[axis]
            if quantization.weight_block_size is not None
            else quantization.group_size if axis == 1 else None
        )
        if unit is not None and local_dimension % unit:
            partial_unit_count += 1
            if len(partial_units) < 16:
                partial_units.append(
                    {
                        "weight": name,
                        "axis": axis,
                        "local_dimension": local_dimension,
                        "quantization_unit": unit,
                    }
                )
    return {
        "checked_sharded_weight_count": checked,
        "errors": errors,
        "partial_quantization_unit_count": partial_unit_count,
        "partial_quantization_units": partial_units,
        "requires_partial_unit_loader": partial_unit_count > 0,
        "valid": not errors,
    }


EXPERT_WEIGHT = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def coalesce_expert_logical_headers(
    logical_headers: dict[str, dict],
) -> tuple[dict[str, dict], dict]:
    """Convert per-expert serialized weights to nano-vllm's stacked view."""

    result = dict(logical_headers)
    grouped = {}
    for name, metadata in logical_headers.items():
        match = EXPERT_WEIGHT.fullmatch(name)
        if match is None:
            continue
        key = (match.group("prefix"), match.group("projection"))
        grouped.setdefault(key, []).append(
            (int(match.group("expert")), name, metadata)
        )

    errors = []
    normalized = {}
    for (prefix, projection), entries in sorted(grouped.items()):
        entries.sort()
        expert_ids = [expert_id for expert_id, _, _ in entries]
        if expert_ids != list(range(len(entries))):
            errors.append(
                f"{prefix}.{projection}: expert ids are not contiguous from zero"
            )
            continue
        shapes = {
            _tensor_shape(metadata, name) for _, name, metadata in entries
        }
        if len(shapes) != 1:
            errors.append(f"{prefix}.{projection}: expert shapes differ")
            continue
        shape = shapes.pop()
        for _, name, _ in entries:
            del result[name]
        normalized[(prefix, projection)] = (len(entries), shape, entries[0][2])

    prefixes = {prefix for prefix, _ in normalized}
    stacked_parameter_count = 0
    for prefix in sorted(prefixes):
        down = normalized.get((prefix, "down_proj"))
        gate = normalized.get((prefix, "gate_proj"))
        up = normalized.get((prefix, "up_proj"))
        if down is None or gate is None or up is None:
            errors.append(f"{prefix}: missing down, gate, or up expert projection")
            continue
        if gate[:2] != up[:2]:
            errors.append(f"{prefix}: gate and up expert layouts differ")
            continue
        if down[0] != gate[0] or down[1] != (gate[1][1], gate[1][0]):
            errors.append(f"{prefix}: down and gate/up expert layouts disagree")
            continue
        result[f"{prefix}.gate_up_proj"] = _logical_header(
            gate[2],
            (gate[0], gate[1][0] * 2, gate[1][1]),
        )
        result[f"{prefix}.down_proj"] = _logical_header(
            down[2],
            (down[0], *down[1]),
        )
        stacked_parameter_count += 2
    return result, {
        "stacked_parameter_count": stacked_parameter_count,
        "serialized_expert_weight_count": sum(map(len, grouped.values())),
        "errors": errors,
        "valid": not errors,
    }


def audit_model_headers(model, headers: dict[str, dict]) -> dict:
    expected = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    packed_mapping = getattr(model, "packed_modules_mapping", {})
    map_name = getattr(model, "map_weight_name", lambda name: name)
    checkpoint = HeaderCheckpoint(headers)
    loaded = set()
    skipped = []
    skipped_by_prefix = {
        prefix: 0 for prefix in ALLOWED_TEXT_ONLY_SKIP_PREFIXES
    }
    unclassified_skipped = []
    unexpected = []
    shape_errors = []
    mapped_checkpoint_bytes = 0
    estimated_local_payload_bytes = 0
    lazy_tensor_count = 0
    full_tensor_count = 0
    for source_name in sorted(headers):
        mapped = map_name(source_name)
        if mapped is None:
            skipped.append(source_name)
            for prefix in ALLOWED_TEXT_ONLY_SKIP_PREFIXES:
                if source_name.startswith(prefix):
                    skipped_by_prefix[prefix] += 1
                    break
            else:
                unclassified_skipped.append(source_name)
            continue
        packed = resolve_packed_parameter(mapped, packed_mapping)
        target = packed[0] if packed is not None else mapped
        if target not in expected:
            unexpected.append({"source": source_name, "mapped": target})
            continue
        try:
            source_bytes, local_bytes, lazy = validate_weight_shape(
                model,
                checkpoint,
                source_name,
                target,
                packed,
            )
        except (AssertionError, IndexError, RuntimeError, ValueError) as error:
            shape_errors.append(
                {"source": source_name, "mapped": target, "error": str(error)}
            )
            continue
        mapped_checkpoint_bytes += source_bytes
        estimated_local_payload_bytes += local_bytes
        if lazy:
            lazy_tensor_count += 1
        else:
            full_tensor_count += 1
        loaded.add(target)
    missing = sorted(expected - loaded)
    return {
        "expected_parameter_count": len(expected),
        "mapped_parameter_count": len(loaded),
        "skipped_tensor_count": len(skipped),
        "skipped_by_prefix": skipped_by_prefix,
        "unclassified_skipped_weights": unclassified_skipped,
        "missing_parameters": missing,
        "unexpected_weights": unexpected,
        "shape_errors": shape_errors,
        "local_parameter_bytes": parameter_storage_bytes(model),
        "checkpoint_loading": {
            "mapped_checkpoint_bytes": mapped_checkpoint_bytes,
            "estimated_local_payload_bytes": estimated_local_payload_bytes,
            "estimated_avoided_payload_bytes": (
                mapped_checkpoint_bytes - estimated_local_payload_bytes
            ),
            "estimated_local_payload_fraction": (
                estimated_local_payload_bytes / mapped_checkpoint_bytes
                if mapped_checkpoint_bytes
                else 0.0
            ),
            "lazy_tensor_count": lazy_tensor_count,
            "full_tensor_count": full_tensor_count,
        },
        "valid": (
            not missing
            and not unexpected
            and not shape_errors
            and not unclassified_skipped
        ),
    }


def parse_tp_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("expected comma-separated positive TP sizes")
    return sizes


def official_shard_metadata(model_metadata: dict, shards: list[str]) -> list[dict]:
    siblings = {
        item.get("rfilename"): item
        for item in model_metadata.get("siblings", ())
        if isinstance(item, dict)
    }
    results = []
    for shard in shards:
        item = siblings.get(shard, {})
        lfs = item.get("lfs") or {}
        sha256 = lfs.get("sha256")
        size = lfs.get("size")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError(f"missing valid official LFS identity for {shard}")
        results.append(
            {"name": shard, "size_bytes": size, "sha256": sha256}
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face model repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, default=(4, 8))
    parser.add_argument("--max-header-mib", type=float, default=16.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/remote_checkpoint_header_audit.json"),
    )
    args = parser.parse_args()
    if args.max_header_mib <= 0:
        parser.error("--max-header-mib must be positive")
    base = (
        f"https://huggingface.co/{args.repo}/resolve/"
        f"{quote(args.revision, safe='')}"
    )
    config_bytes = fetch_bytes(f"{base}/config.json")
    config_document = json.loads(config_bytes)
    quantization = resolve_quantization_spec(
        SimpleNamespace(
            quantization_config=config_document.get("quantization_config")
        )
    )
    index_bytes = fetch_bytes(f"{base}/model.safetensors.index.json")
    model_metadata = json.loads(
        fetch_bytes(
            f"https://huggingface.co/api/models/{args.repo}/revision/"
            f"{quote(args.revision, safe='')}?blobs=true"
        )
    )
    index = json.loads(index_bytes)
    shards = sorted(set(index["weight_map"].values()))
    checkpoint_shards = official_shard_metadata(model_metadata, shards)
    headers = {}
    downloaded_header_bytes = 0
    for shard in shards:
        tensors, downloaded = fetch_safetensors_header(
            f"{base}/{quote(shard, safe='/')}",
            int(args.max_header_mib * 1024**2),
        )
        duplicate = set(headers).intersection(tensors)
        if duplicate:
            raise ValueError(f"duplicate tensors across shards: {sorted(duplicate)[:3]}")
        headers.update(tensors)
        downloaded_header_bytes += downloaded
    indexed_names = set(index["weight_map"])
    if set(headers) != indexed_names:
        raise ValueError(
            "safetensors index and remote shard headers contain different tensors"
        )

    quantization_audit = None
    logical_headers = headers
    unstacked_logical_headers = headers
    expert_layout_audit = None
    if quantization.is_quantized:
        unstacked_logical_headers, quantization_audit = audit_quantized_tensor_groups(
            headers,
            quantization,
        )
        logical_headers, expert_layout_audit = coalesce_expert_logical_headers(
            unstacked_logical_headers
        )

    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "config.json").write_bytes(config_bytes)
        results = {}
        for tp_size in args.tp_sizes:
            model = instantiate_meta_model(str(model_dir), tp_size)
            result = audit_model_headers(model, logical_headers)
            if quantization.is_quantized:
                result["scope"] = (
                    "logical parameter names and shapes reconstructed from "
                    "quantized headers; payload loading is not validated"
                )
                result["checkpoint_loading"] = None
                result["quantized_tp_layout"] = audit_quantized_tp_layout(
                    unstacked_logical_headers,
                    quantization,
                    tp_size,
                )
                result["valid"] = (
                    result["valid"]
                    and result["quantized_tp_layout"]["valid"]
                    and quantization_audit["valid"]
                    and expert_layout_audit["valid"]
                )
            results[f"tp{tp_size}"] = result

    report = {
        "scope": "remote_headers_only; tensor payloads and GPU execution are not validated",
        "repo": args.repo,
        "revision": args.revision,
        "resolved_revision": model_metadata["sha"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "headers_sha256": hashlib.sha256(
            json.dumps(
                headers,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "tensor_parallel_sizes": list(args.tp_sizes),
        "shard_count": len(shards),
        "checkpoint_shards": checkpoint_shards,
        "source_tensor_count": len(headers),
        "logical_tensor_count": len(logical_headers),
        "downloaded_header_bytes": downloaded_header_bytes,
        "quantization": quantization_audit,
        "expert_layout": expert_layout_audit,
        "results": results,
        "valid": all(result["valid"] for result in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
