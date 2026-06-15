# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.cpu.ans_c_ext import (
    ANSCompressContextCExt,
    CompressionConfig,
    CompressionGranularity,
    HAS_ANS_C_EXT,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


@pytest.mark.skipif(not HAS_ANS_C_EXT, reason="Requires ANS C++ extension")
def test_ans_roundtrip_per_layer():
    num_blocks = 4
    page_size_bytes = 32768
    num_layers = 2
    num_kv = 2
    total_tensors = num_layers * num_kv

    gpu_tensors = [
        torch.randn(
            num_blocks, page_size_bytes // 2, dtype=torch.float16, device="cuda"
        ).view(torch.int8)
        for _ in range(total_tensors)
    ]
    cpu_tensors = [
        torch.zeros(
            (num_blocks, page_size_bytes), dtype=torch.int8, device="cpu", pin_memory=True
        )
        for _ in range(total_tensors)
    ]

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=1, cross_layer=False
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

    original = [t.clone() for t in gpu_tensors]

    ctx.transfer_comp(
        gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
    )

    for t in gpu_tensors:
        t.zero_()

    ctx.transfer_decomp(
        gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
    )
    torch.cuda.synchronize()

    for i in range(total_tensors):
        torch.testing.assert_close(
            gpu_tensors[i], original[i], atol=0, rtol=0
        )

    ctx.destroy()


@pytest.mark.skipif(not HAS_ANS_C_EXT, reason="Requires ANS C++ extension")
def test_ans_roundtrip_cross_layer():
    num_blocks = 4
    page_size_bytes = 32768
    num_layers = 2
    num_kv = 2
    total_tensors = num_layers * num_kv

    gpu_tensors = [
        torch.randn(
            num_blocks, page_size_bytes // 2, dtype=torch.float16, device="cuda"
        ).view(torch.int8)
        for _ in range(total_tensors)
    ]
    cpu_tensors = [
        torch.zeros(
            (num_blocks, page_size_bytes * num_layers),
            dtype=torch.int8, device="cpu", pin_memory=True
        )
        for _ in range(total_tensors)
    ]

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=1, cross_layer=True
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

    original = [t.clone() for t in gpu_tensors]

    ctx.transfer_comp(
        gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
    )

    for t in gpu_tensors:
        t.zero_()

    ctx.transfer_decomp(
        gpu_block_ids, cpu_block_ids, 0, num_layers, torch.cuda.current_stream()
    )
    torch.cuda.synchronize()

    for i in range(total_tensors):
        torch.testing.assert_close(
            gpu_tensors[i], original[i], atol=0, rtol=0
        )

    ctx.destroy()


def test_compression_config_defaults():
    config = CompressionConfig()
    assert config.enable_compression is False
    assert config.granularity.blocks_per_chunk == 1
    assert config.granularity.cross_layer is True
    assert config.compress_layout == "post_layout"


def test_compression_config_from_extra_config():
    config = CompressionConfig.from_extra_config(
        {"compression": {"enable_compression": True, "granularity": {"blocks_per_chunk": 4}}}
    )
    assert config.enable_compression is True or not HAS_ANS_C_EXT
    assert config.granularity.blocks_per_chunk == 4


def test_compression_config_missing_key():
    config = CompressionConfig.from_extra_config({})
    assert config.enable_compression is False


def test_compression_config_invalid_compress_layout():
    with pytest.raises(ValueError):
        CompressionConfig(enable_compression=True, compress_layout="invalid")
