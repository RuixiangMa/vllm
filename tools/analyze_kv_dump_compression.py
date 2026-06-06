# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze compression ratios for dumped KV tensors or post-layout cache blocks."""

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

        symbols = np.frombuffer(raw, dtype=np.uint8).astype(np.int64, copy=False)
        counts = np.bincount(symbols, minlength=256).astype(np.float64)
        probabilities = counts / counts.sum()
        probabilities[probabilities == 0] = 1e-12
        probabilities /= probabilities.sum()

        try:
            model = constriction.stream.model.Categorical(
                probabilities.astype(np.float32), perfect=False
            )
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols.tolist(), model)
            compressed = coder.get_compressed()
        except AttributeError as exc:
            raise RuntimeError(
                "Installed 'constriction' package does not expose the expected ANS API."
            ) from exc

        return len(np.asarray(compressed).tobytes())


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
    streams: dict[str, StreamStats]


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def find_event_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("event_*.pt"))


def _payload_streams(payload: dict[str, object], payload_mode: str) -> dict[str, bytes]:
    if payload_mode == "raw_kv":
        if "key" not in payload or "value" not in payload:
            raise ValueError("Missing key/value tensors in dump file")
        key_bytes = tensor_to_bytes(payload["key"])
        value_bytes = tensor_to_bytes(payload["value"])
        return {
            "key": key_bytes,
            "value": value_bytes,
            "combined": key_bytes + value_bytes,
        }
    if payload_mode == "cache_blocks":
        if "cache_blocks" not in payload:
            raise ValueError("Missing cache_blocks tensor in dump file")
        return {"cache_blocks": tensor_to_bytes(payload["cache_blocks"])}
    raise ValueError(f"Unsupported payload mode: {payload_mode}")


def analyze(
    input_dir: Path, algorithms: list[str], level: int, payload_mode: str
) -> AnalysisResult:
    event_files = find_event_files(input_dir)
    stream_compressors: dict[str, dict[str, Compressor]] = {}
    stream_raw_bytes: dict[str, int] = {}

    for path in event_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        streams = _payload_streams(payload, payload_mode)
        for stream_name, stream_bytes in streams.items():
            if stream_name not in stream_compressors:
                stream_compressors[stream_name] = {
                    name: build_compressor(name, level) for name in algorithms
                }
                stream_raw_bytes[stream_name] = 0
            stream_raw_bytes[stream_name] += len(stream_bytes)
            for compressor in stream_compressors[stream_name].values():
                compressor.update(stream_bytes)

    return AnalysisResult(
        num_events=len(event_files),
        streams={
            stream_name: StreamStats(
                raw_bytes=stream_raw_bytes[stream_name],
                compressed_bytes={
                    name: c.finish()
                    for name, c in stream_compressors[stream_name].items()
                },
            )
            for stream_name in stream_compressors
        },
    )


def summarize_stream(stream: StreamStats) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, compressed_bytes in stream.compressed_bytes.items():
        ratio = compressed_bytes / stream.raw_bytes if stream.raw_bytes else 0.0
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
    for stream_name, stream in result.streams.items():
        print(f"\n[{stream_name}]")
        for algo, stats in summarize_stream(stream).items():
            print(
                f"{algo:>5} raw={stats['raw_bytes']} "
                f"compressed={stats['compressed_bytes']} "
                f"ratio={stats['ratio']:.6f} reduction={stats['reduction']:.6f}"
            )


def to_jsonable(result: AnalysisResult) -> dict[str, object]:
    data: dict[str, object] = {"num_events": result.num_events}
    for stream_name, stream in result.streams.items():
        data[stream_name] = summarize_stream(stream)
    return data


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
    parser.add_argument(
        "--payload-mode",
        choices=["raw_kv", "cache_blocks"],
        default="raw_kv",
        help="Which payload layout to analyze: raw key/value tensors or post-layout cache blocks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    result = analyze(args.input_dir, args.algorithms, args.level, args.payload_mode)
    print_text_report(result)

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(to_jsonable(result), indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
