"""Audit Hugging Face safetensors shapes using HTTP headers only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
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

    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "config.json").write_bytes(config_bytes)
        results = {}
        for tp_size in args.tp_sizes:
            model = instantiate_meta_model(str(model_dir), tp_size)
            results[f"tp{tp_size}"] = audit_model_headers(model, headers)

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
        "downloaded_header_bytes": downloaded_header_bytes,
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
