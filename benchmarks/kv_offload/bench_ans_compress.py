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

import numpy as np
import torch

from vllm.v1.kv_offload.cpu.ans_c_ext import (
    ANSCompressContextCExt,
    CompressionConfig,
    CompressionGranularity,
    HAS_ANS_C_EXT,
)


def benchmark_compression(
    num_blocks: int,
    page_size_bytes: int,
    num_layers: int,
    blocks_per_chunk: int,
    cross_layer: bool,
    iterations: int,
):
    if not HAS_ANS_C_EXT:
        print("ANS C++ extension not available, cannot benchmark.")
        return

    num_kv = 2
    total_tensors = num_layers * num_kv
    cpu_slot_bytes = page_size_bytes * num_layers if cross_layer else page_size_bytes

    gpu_tensors = [
        torch.randn(
            num_blocks, page_size_bytes // 2, dtype=torch.float16, device="cuda"
        ).view(torch.int8)
        for _ in range(total_tensors)
    ]
    cpu_tensors = [
        torch.zeros(
            (num_blocks, cpu_slot_bytes), dtype=torch.int8, device="cpu", pin_memory=True
        )
        for _ in range(total_tensors)
    ]

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=blocks_per_chunk, cross_layer=cross_layer
        ),
    )
    ctx = ANSCompressContextCExt(
        gpu_tensors=gpu_tensors,
        cpu_tensors=cpu_tensors,
        num_cpu_blocks=num_blocks,
        config=config,
    )

    gpu_block_ids = np.arange(num_blocks, dtype=np.int64)
    cpu_block_ids = np.arange(num_blocks, dtype=np.int64)

    compress_times = []
    compress_bytes = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_comp = ctx.transfer_comp(
            gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
        )
        torch.cuda.synchronize()
        compress_times.append(time.perf_counter() - t0)
        compress_bytes.append(total_comp)

    decompress_times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx.transfer_decomp(
            gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
        )
        torch.cuda.synchronize()
        decompress_times.append(time.perf_counter() - t0)

    total_uncompressed = num_blocks * page_size_bytes * total_tensors
    total_compressed = compress_bytes[-1]
    ratio = total_compressed / total_uncompressed if total_uncompressed > 0 else 1.0

    d2h_uncompressed_us = total_uncompressed / 32e9 * 1e6
    d2h_compressed_us = total_compressed / 32e9 * 1e6

    print(f"=== ANS Compression Benchmark (CExt) ===")
    print(f"Config: {num_blocks} blocks x {page_size_bytes} B/page x {num_layers} layers x {num_kv} kv")
    print(f"Granularity: blocks_per_chunk={blocks_per_chunk}, cross_layer={cross_layer}")
    print(f"Compression ratio: {ratio:.3f} ({total_compressed}/{total_uncompressed} bytes)")
    print(f"Compress latency (avg): {sum(compress_times)/len(compress_times)*1e3:.2f} ms")
    print(f"Decompress latency (avg): {sum(decompress_times)/len(decompress_times)*1e3:.2f} ms")
    print(f"D2H uncompressed @32GB/s: {d2h_uncompressed_us:.1f} us")
    print(f"D2H compressed @32GB/s: {d2h_compressed_us:.1f} us")
    print(f"Bandwidth saved: {(1-ratio)*100:.1f}%")

    ctx.destroy()


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
