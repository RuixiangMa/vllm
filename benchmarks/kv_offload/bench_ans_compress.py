# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark for nvCOMP ANS compression on KV cache offload path.

Usage:
    python benchmarks/kv_offload/bench_ans_compress.py \
        --num-blocks 1024 --page-size-bytes 16384 \
        --num-layers 2 --blocks-per-chunk 1 \
        --cross-layer --iterations 10
"""
from __future__ import annotations

import argparse
import time

import torch

from vllm.v1.kv_offload.cpu.ans_compress import (
    ANSCompressContext,
    CompressionConfig,
    CompressionGranularity,
    HAS_NVCOMP,
)


def benchmark_compression(
    num_blocks: int,
    page_size_bytes: int,
    num_layers: int,
    blocks_per_chunk: int,
    cross_layer: bool,
    iterations: int,
):
    if not HAS_NVCOMP:
        print("nvCOMP not available, cannot benchmark.")
        return

    gpu_tensors = [
        torch.randint(
            -128, 127, (num_blocks, page_size_bytes), dtype=torch.int8, device="cuda"
        )
        for _ in range(num_layers)
    ]
    cpu_tensors = [
        torch.zeros(
            (num_blocks, page_size_bytes), dtype=torch.int8, device="cpu", pin_memory=True
        )
        for _ in range(num_layers)
    ]

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=blocks_per_chunk, cross_layer=cross_layer
        ),
    )
    ctx = ANSCompressContext(
        gpu_tensors=gpu_tensors,
        cpu_tensors=cpu_tensors,
        num_cpu_blocks=num_blocks,
        config=config,
    )

    layer_indices = list(range(num_layers))
    block_ids = list(range(num_blocks))

    compress_times = []
    all_compress_data: list[list[tuple[int, bool]]] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk_results: list[tuple[int, bool]] = []
        ci = 0
        while ci < len(block_ids):
            comp_size, count, fits = ctx.compress_one_chunk(
                block_ids, ci, layer_indices
            )
            chunk_results.append((comp_size, fits))
            ci += count
        torch.cuda.synchronize()
        compress_times.append(time.perf_counter() - t0)
        all_compress_data.append(chunk_results)

    total_uncompressed = num_blocks * page_size_bytes * num_layers
    total_compressed = sum(s for chunk_list in all_compress_data for s, _ in chunk_list)
    ratio = total_compressed / total_uncompressed if total_uncompressed > 0 else 1.0

    decompress_times = []
    for chunk_list in all_compress_data:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ci = 0
        for comp_size, fits in chunk_list:
            if fits and comp_size > 0:
                ctx.decompress_one_chunk(
                    block_ids, ci, layer_indices, comp_size
                )
            ci += blocks_per_chunk
        torch.cuda.synchronize()
        decompress_times.append(time.perf_counter() - t0)

    d2h_uncompressed_us = total_uncompressed / 32e9 * 1e6
    d2h_compressed_us = total_compressed / 32e9 * 1e6

    print(f"=== ANS Compression Benchmark ===")
    print(f"Config: {num_blocks} blocks x {page_size_bytes} B/page x {num_layers} layers")
    print(f"Granularity: blocks_per_chunk={blocks_per_chunk}, cross_layer={cross_layer}")
    print(f"Compression ratio: {ratio:.3f} ({total_compressed}/{total_uncompressed} bytes)")
    print(f"Compress latency (avg): {sum(compress_times)/len(compress_times)*1e3:.2f} ms")
    print(f"Decompress latency (avg): {sum(decompress_times)/len(decompress_times)*1e3:.2f} ms")
    print(f"D2H uncompressed @32GB/s: {d2h_uncompressed_us:.1f} us")
    print(f"D2H compressed @32GB/s: {d2h_compressed_us:.1f} us")
    print(f"Bandwidth saved: {(1-ratio)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--page-size-bytes", type=int, default=16384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--blocks-per-chunk", type=int, default=1)
    parser.add_argument("--cross-layer", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    benchmark_compression(
        args.num_blocks, args.page_size_bytes, args.num_layers,
        args.blocks_per_chunk, args.cross_layer, args.iterations,
    )


if __name__ == "__main__":
    main()
