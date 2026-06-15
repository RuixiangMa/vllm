# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.ans_c_ext import (
    ANSCompressContextCExt,
    CompressionConfig,
    HAS_ANS_C_EXT,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    num_compressed_bytes: int
    num_uncompressed_bytes: int


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        output[:] = base_ptr + block_ids[:num_sub_blocks] * row_stride
        return

    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
    all_ptrs = (
        base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    rank = region.rank

    base_ptr = region._base.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


class SingleDirectionOffloadingHandler(OffloadingHandler):

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
        compress_ctx: ANSCompressContextCExt | None = None,
    ):
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            assert cpu_page_size == gpu_page_size * block_size_factor

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs

        self.src_block_size_factor = 1 if self.gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if self.gpu_to_cpu else 1

        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        self._mmap_region = mmap_region
        self._compress_ctx = compress_ctx
        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[Transfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def _compress_and_offload(
        self,
        gpu_block_ids: np.ndarray,
        cpu_block_ids: np.ndarray,
        start_layer_id: int,
        num_layers: int,
        stream: torch.cuda.Stream,
    ) -> int:
        ctx = self._compress_ctx
        assert ctx is not None
        return ctx.transfer_comp(
            gpu_block_ids, cpu_block_ids, start_layer_id, num_layers, stream
        )

    def _load_and_decompress(
        self,
        gpu_block_ids: np.ndarray,
        cpu_block_ids: np.ndarray,
        start_layer_id: int,
        num_layers: int,
        stream: torch.cuda.Stream,
    ) -> int:
        ctx = self._compress_ctx
        assert ctx is not None
        return ctx.transfer_decomp(
            gpu_block_ids, cpu_block_ids, start_layer_id, num_layers, stream
        )

    def _build_batch_copy_ops(
        self,
        src_spec: BlockIDsLoadStoreSpec,
        dst_spec: BlockIDsLoadStoreSpec,
        num_src_blocks: int,
        num_dst_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)

        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        all_src = np.empty(num_copy_ops, dtype=np.int64)
        all_dst = np.empty(num_copy_ops, dtype=np.int64)
        all_sizes = np.empty(num_copy_ops, dtype=np.int64)

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        num_transfer_bytes = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_block_size_factor
            dst_logical_blocks_to_skip = block_idx % self.dst_block_size_factor
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(
                dst_logical_blocks_count, self.dst_block_size_factor
            )
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(
                src_logical_blocks_count, self.src_block_size_factor
            )
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            group_src = src_spec.block_ids[src_offset:src_end_offset]
            group_dst = dst_spec.block_ids[dst_offset:dst_end_offset]

            for data_ref in group_data_refs:
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                compute_sub_block_ptrs(
                    group_src,
                    self.src_block_size_factor,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_block_size_factor,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        assert op_idx == num_copy_ops

        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        return batch_src, batch_dst, batch_sizes, num_transfer_bytes

    def _compute_uncompressed_bytes(
        self,
        src_spec: BlockIDsLoadStoreSpec,
        dst_spec: BlockIDsLoadStoreSpec,
    ) -> int:
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        num_bytes = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            for data_ref in group_data_refs:
                num_bytes += group_size * data_ref.page_size_bytes
        return num_bytes

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        use_compress = isinstance(self._compress_ctx, ANSCompressContextCExt)

        if use_compress:
            num_transfer_bytes = self._compute_uncompressed_bytes(
                src_spec, dst_spec
            )
            batch_src = batch_dst = batch_sizes = None
            num_copy_ops = 0
        else:
            batch_src, batch_dst, batch_sizes, num_transfer_bytes = (
                self._build_batch_copy_ops(
                    src_spec, dst_spec, num_src_blocks, num_dst_blocks
                )
            )
            num_copy_ops = len(batch_src)

        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            stream.wait_event(last_event)
        is_src_access_order_any = not self.gpu_to_cpu

        num_compressed_bytes = 0
        num_uncompressed_bytes = num_transfer_bytes

        with torch.cuda.stream(stream):
            start_event.record(stream)
            if use_compress:
                gpu_block_ids_np = src_blocks if self.gpu_to_cpu else dst_blocks
                cpu_block_ids_np = dst_blocks if self.gpu_to_cpu else src_blocks
                num_kv = 1 if self._compress_ctx.is_mla else 2
                num_layers = len(self.src_tensors) // num_kv
                if self.gpu_to_cpu:
                    num_compressed_bytes = self._compress_and_offload(
                        gpu_block_ids_np,
                        cpu_block_ids_np,
                        0,
                        num_layers,
                        stream,
                    )
                else:
                    num_compressed_bytes = self._load_and_decompress(
                        gpu_block_ids_np,
                        cpu_block_ids_np,
                        0,
                        num_layers,
                        stream,
                    )
            elif num_copy_ops > 0:
                ops.swap_blocks_batch(
                    batch_src,
                    batch_dst,
                    batch_sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                num_compressed_bytes=num_compressed_bytes,
                num_uncompressed_bytes=num_uncompressed_bytes,
            )
        )

        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
                num_compressed_bytes=transfer.num_compressed_bytes,
                num_uncompressed_bytes=transfer.num_uncompressed_bytes,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
        compression_config: CompressionConfig | None = None,
    ):
        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self._mmap_region = mmap_region
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        compress_ctx = None
        if compression_config is not None and compression_config.enable_compression:
            if HAS_ANS_C_EXT:
                compress_ctx = ANSCompressContextCExt(
                    gpu_tensors=gpu_tensors,
                    cpu_tensors=cpu_tensors,
                    num_cpu_blocks=num_cpu_blocks,
                    config=compression_config,
                )
                logger.info(
                    "KV cache ANS compression enabled (C++ ext): granularity=%s",
                    compression_config.granularity,
                )
            else:
                logger.warning(
                    "ANS compression requested but C++ extension not available, "
                    "falling back to uncompressed transfer"
                )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
            compress_ctx=compress_ctx,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            compress_ctx=compress_ctx,
        )
