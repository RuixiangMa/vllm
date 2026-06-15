# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.kv_offload.cpu.ans_compress import (
    ANSCompressContext,
    CompressionConfig,
    CompressionGranularity,
    HAS_NVCOMP,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


@pytest.mark.skipif(not HAS_NVCOMP, reason="Requires nvidia-nvcomp")
def test_ans_roundtrip_per_layer():
    num_blocks = 4
    page_size_bytes = 4096
    gpu_tensor = torch.randint(
        0, 127, (num_blocks, page_size_bytes), dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        (num_blocks, page_size_bytes), dtype=torch.int8, device="cpu", pin_memory=True
    )

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=1, cross_layer=False
        ),
    )
    ctx = ANSCompressContext(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        num_cpu_blocks=num_blocks,
        config=config,
    )

    block_ids = [0, 1, 2, 3]
    original_data = gpu_tensor[block_ids].clone()

    for ci in range(len(block_ids)):
        comp_size, count, fits = ctx.compress_one_chunk(
            block_ids, ci, [0]
        )
        assert count == 1
        assert fits
        cpu_tensor[block_ids[ci]][:comp_size].copy_(
            ctx.compressed_buffer_gpu[:comp_size]
        )

    gpu_tensor[block_ids].zero_()

    for ci in range(len(block_ids)):
        comp_size = cpu_tensor[block_ids[ci]].nonzero().shape[0]
        if comp_size == 0:
            comp_size = page_size_bytes
        ctx.compressed_buffer_gpu[:comp_size].copy_(
            cpu_tensor[block_ids[ci]][:comp_size]
        )
        ctx.decompress_one_chunk(block_ids, ci, [0], comp_size)

    for i, block_id in enumerate(block_ids):
        torch.testing.assert_close(
            gpu_tensor[block_id], original_data[i], atol=0, rtol=0
        )


@pytest.mark.skipif(not HAS_NVCOMP, reason="Requires nvidia-nvcomp")
def test_ans_roundtrip_cross_layer():
    num_blocks = 4
    page_size_bytes = 4096
    num_layers = 2

    cpu_slot_bytes = page_size_bytes * num_layers
    gpu_tensors = [
        torch.randint(
            0, 127, (num_blocks, page_size_bytes), dtype=torch.int8, device="cuda"
        )
        for _ in range(num_layers)
    ]
    cpu_tensors = [
        torch.zeros(
            (num_blocks, cpu_slot_bytes), dtype=torch.int8, device="cpu", pin_memory=True
        )
        for _ in range(num_layers)
    ]

    config = CompressionConfig(
        enable_compression=True,
        granularity=CompressionGranularity(
            blocks_per_chunk=1, cross_layer=True
        ),
    )
    ctx = ANSCompressContext(
        gpu_tensors=gpu_tensors,
        cpu_tensors=cpu_tensors,
        num_cpu_blocks=num_blocks,
        config=config,
    )

    block_ids = [0, 1, 2, 3]
    layer_indices = list(range(num_layers))
    original_data = [t[block_ids].clone() for t in gpu_tensors]

    for ci in range(len(block_ids)):
        comp_size, count, fits = ctx.compress_one_chunk(
            block_ids, ci, layer_indices
        )
        assert count == 1
        assert fits
        cpu_tensors[0][block_ids[ci]][:comp_size].copy_(
            ctx.compressed_buffer_gpu[:comp_size]
        )

    for t in gpu_tensors:
        t[block_ids].zero_()

    for ci in range(len(block_ids)):
        comp_size = int(cpu_tensors[0][block_ids[ci]].nonzero().shape[0])
        ctx.compressed_buffer_gpu[:comp_size].copy_(
            cpu_tensors[0][block_ids[ci]][:comp_size]
        )
        ctx.decompress_one_chunk(block_ids, ci, layer_indices, comp_size)

    for layer_idx in range(num_layers):
        for i, block_id in enumerate(block_ids):
            torch.testing.assert_close(
                gpu_tensors[layer_idx][block_id],
                original_data[layer_idx][i],
                atol=0,
                rtol=0,
            )


def test_compression_config_defaults():
    config = CompressionConfig()
    assert config.enable_compression is False
    assert config.granularity.blocks_per_chunk == 1
    assert config.granularity.cross_layer is True


def test_compression_config_from_extra_config():
    config = CompressionConfig.from_extra_config(
        {"compression": {"enable_compression": True, "granularity": {"blocks_per_chunk": 4}}}
    )
    assert config.enable_compression is True or not HAS_NVCOMP
    assert config.granularity.blocks_per_chunk == 4


def test_compression_config_missing_key():
    config = CompressionConfig.from_extra_config({})
    assert config.enable_compression is False
