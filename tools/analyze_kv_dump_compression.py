# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze compression ratios for dumped KV tensors.

This script measures the compressibility of dumped KV payloads using the raw
bytes of the `key` and `value` tensors instead of the serialized `.pt` file
size. It recursively scans an input directory for `event_*.pt` files created by
`vllm.v1.debug.kv_dump` and reports aggregate compression ratios for:

- all key tensors concatenated as one byte stream
- all value tensors concatenated as one byte stream
- key+value concatenated together as one byte stream
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Compressor(Protocol):
    def update(self, data: bytes) -> None: ...
    def finish(self) -> int: ...


class _StreamingZstdCompressor:
    def __init__(self, level: int) -> None:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Algorithm 'zstd' requires the 'zstandard' package."
            ) from exc
        self._compressor = zstd.ZstdCompressor(level=level).compressobj()
        self._size = 0

    def update(self, data: bytes) -> None:
        self._size += len(self._compressor.compress(data))

    def finish(self) -> int:
        self._size += len(self._compressor.flush())
        return self._size


class _AnsCompressor:
    def __init__(self, level: int) -> None:
        del level
        self._chunks: list[bytes] = []

    def update(self, data: bytes) -> None:
        self._chunks.append(data)

    def finish(self) -> int:
        try:
            import constriction
        except ImportError as exc:
            raise RuntimeError(
                "Algorithm 'ans' requires the 'constriction' package."
            ) from exc

        raw = b"".join(self._chunks)
        if not raw:
            return 0

        symbols = np.frombuffer(raw, dtype=np.uint8)
        counts = np.bincount(symbols, minlength=256).astype(np.float64)
        probabilities = counts / counts.sum()
        probabilities[probabilities == 0] = 1e-12
        probabilities /= probabilities.sum()

        try:
            model = constriction.stream.model.Categorical(
                probabilities, perfect=False
            )
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols, model)
            compressed = coder.get_compressed()
        except AttributeError as exc:
            raise RuntimeError(
                "Installed 'constriction' package does not expose the expected ANS API."
            ) from exc

        return len(compressed.tobytes())


SUPPORTED_ALGORITHMS = ("zstd", "ans")


def build_compressor(name: str, level: int) -> Compressor:
    if name == "zstd":
        return _StreamingZstdCompressor(level)
    if name == "ans":
        return _AnsCompressor(level)
    raise ValueError(f"Unsupported algorithm: {name}")


@dataclass
class StreamStats:
    raw_bytes: int
    compressed_bytes: dict[str, int]


@dataclass
class AnalysisResult:
    num_events: int
    key: StreamStats
    value: StreamStats
    combined: StreamStats


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def find_event_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("event_*.pt"))


def analyze(input_dir: Path, algorithms: list[str], level: int) -> AnalysisResult:
    event_files = find_event_files(input_dir)
    key_compressors = {name: build_compressor(name, level) for name in algorithms}
    value_compressors = {name: build_compressor(name, level) for name in algorithms}
    combined_compressors = {name: build_compressor(name, level) for name in algorithms}

    key_raw = 0
    value_raw = 0

    for path in event_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "key" not in payload or "value" not in payload:
            raise ValueError(f"Missing key/value tensors in dump file: {path}")

        key_bytes = tensor_to_bytes(payload["key"])
        value_bytes = tensor_to_bytes(payload["value"])
        key_raw += len(key_bytes)
        value_raw += len(value_bytes)

        for compressor in key_compressors.values():
            compressor.update(key_bytes)
        for compressor in value_compressors.values():
            compressor.update(value_bytes)
        for compressor in combined_compressors.values():
            compressor.update(key_bytes)
            compressor.update(value_bytes)

    return AnalysisResult(
        num_events=len(event_files),
        key=StreamStats(
            raw_bytes=key_raw,
            compressed_bytes={name: c.finish() for name, c in key_compressors.items()},
        ),
        value=StreamStats(
            raw_bytes=value_raw,
            compressed_bytes={name: c.finish() for name, c in value_compressors.items()},
        ),
        combined=StreamStats(
            raw_bytes=key_raw + value_raw,
            compressed_bytes={
                name: c.finish() for name, c in combined_compressors.items()
            },
        ),
    )


def summarize_stream(stream: StreamStats) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, compressed_bytes in stream.compressed_bytes.items():
        ratio = (
            compressed_bytes / stream.raw_bytes if stream.raw_bytes else 0.0
        )
        reduction = 1.0 - ratio if stream.raw_bytes else 0.0
        summary[name] = {
            "raw_bytes": stream.raw_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": ratio,
            "reduction": reduction,
        }
    return summary


def print_text_report(result: AnalysisResult) -> None:
    print(f"num_events: {result.num_events}")
    for stream_name, stream in (
        ("key", result.key),
        ("value", result.value),
        ("combined", result.combined),
    ):
        print(f"\n[{stream_name}]")
        for algo, stats in summarize_stream(stream).items():
            print(
                f"{algo:>5} raw={stats['raw_bytes']} "
                f"compressed={stats['compressed_bytes']} "
                f"ratio={stats['ratio']:.6f} reduction={stats['reduction']:.6f}"
            )


def to_jsonable(result: AnalysisResult) -> dict[str, object]:
    return {
        "num_events": result.num_events,
        "key": summarize_stream(result.key),
        "value": summarize_stream(result.value),
        "combined": summarize_stream(result.combined),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze compression ratios for dumped KV tensors."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing KV dump event_*.pt files.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(SUPPORTED_ALGORITHMS),
        choices=SUPPORTED_ALGORITHMS,
        help="Compression algorithms to evaluate.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=6,
        help="Compression level/preset used for all algorithms.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the summary as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    result = analyze(args.input_dir, args.algorithms, args.level)
    print_text_report(result)

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(to_jsonable(result), indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
