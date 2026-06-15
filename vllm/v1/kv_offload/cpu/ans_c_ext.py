from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from vllm import _ans_transfer as _C
    HAS_ANS_C_EXT = True
except ImportError:
    HAS_ANS_C_EXT = False


@dataclass
class CompressionGranularity:
    blocks_per_chunk: int = 1
    cross_layer: bool = True


@dataclass
class CompressionConfig:
    enable_compression: bool = False
    algorithm: str = "ans"
    granularity: CompressionGranularity = None
    compress_layout: str = "post_layout"  # "post_layout" | "pre_layout"

    def __post_init__(self):
        if self.granularity is None:
            self.granularity = CompressionGranularity()
        if self.enable_compression and not HAS_ANS_C_EXT:
            logger.warning(
                "nvCOMP C++ extension is not available, disabling KV cache compression."
            )
            self.enable_compression = False

    @staticmethod
    def from_extra_config(extra_config: dict) -> CompressionConfig:
        if "compression" not in extra_config:
            return CompressionConfig()
        comp = extra_config["compression"]
        granularity = CompressionGranularity(
            blocks_per_chunk=comp.get("granularity", {}).get(
                "blocks_per_chunk", 1
            ),
            cross_layer=comp.get("granularity", {}).get("cross_layer", True),
        )
        return CompressionConfig(
            enable_compression=comp.get("enable_compression", False),
            algorithm=comp.get("algorithm", "ans"),
            granularity=granularity,
            compress_layout=comp.get("compress_layout", "post_layout"),
        )


class ANSCompressContextCExt:
    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        num_cpu_blocks: int,
        config: CompressionConfig,
    ):
        self.gpu_tensors = gpu_tensors
        self.cpu_tensors = cpu_tensors
        self.config = config
        num_kv = 2
        self.num_layers = len(gpu_tensors) // num_kv
        self.page_size_bytes = gpu_tensors[0].shape[1]
        self.cpu_slot_bytes = cpu_tensors[0].shape[1]

        self.chunk_size_bytes = self.page_size_bytes

        num_kv = 2
        max_chunks_per_batch = min(
            num_cpu_blocks * self.num_layers * num_kv,
            4096,
        )

        self.gpu_tensor_ptrs = torch.zeros(
            len(gpu_tensors), dtype=torch.int64, device="cpu"
        )
        for i, t in enumerate(gpu_tensors):
            self.gpu_tensor_ptrs[i] = t.data_ptr()
        self.gpu_tensor_ptrs = self.gpu_tensor_ptrs.cuda()

        self.cpu_tensor_ptrs = torch.zeros(
            len(cpu_tensors), dtype=torch.int64, device="cpu"
        )
        for i, t in enumerate(cpu_tensors):
            self.cpu_tensor_ptrs[i] = t.data_ptr()

        data_type = 0

        self.ctx = _C.ANSTransferContext(
            max_num_chunks=max_chunks_per_batch,
            max_chunk_size=self.chunk_size_bytes,
            cpu_slot_capacity=self.cpu_slot_bytes,
            data_type=data_type,
            transfer_sms=4,
        )

        size_table_numel = num_cpu_blocks * self.num_layers * num_kv
        self.size_table = torch.zeros(
            size_table_numel, dtype=torch.int32, device="cpu", pin_memory=True
        )

        self.cpu_size_table_block_stride = self.num_layers * num_kv
        self.cpu_size_table_layer_stride = num_kv
        self.is_mla = False

        logger.info(
            "ANS C++ extension context created: max_chunks=%d, "
            "chunk_size=%d, cpu_slot=%d",
            max_chunks_per_batch,
            self.chunk_size_bytes,
            self.cpu_slot_bytes,
        )

    def transfer_comp(
        self,
        gpu_block_ids: np.ndarray,
        cpu_block_ids: np.ndarray,
        start_layer_id: int,
        num_layers: int,
        stream: torch.cuda.Stream,
    ) -> int:
        gpu_ids_t = torch.from_numpy(gpu_block_ids).cuda()
        cpu_ids_t = torch.from_numpy(cpu_block_ids)

        total_comp = _C.transfer_kv_blocks_ans_comp(
            self.ctx,
            gpu_ids_t,
            self.gpu_tensor_ptrs,
            self.page_size_bytes,
            cpu_ids_t,
            self.cpu_tensor_ptrs,
            self.cpu_slot_bytes,
            self.chunk_size_bytes,
            start_layer_id,
            num_layers,
            self.is_mla,
            self.size_table.data_ptr(),
            self.cpu_size_table_block_stride,
            self.cpu_size_table_layer_stride,
            stream.cuda_stream,
        )
        return total_comp

    def transfer_decomp(
        self,
        gpu_block_ids: np.ndarray,
        cpu_block_ids: np.ndarray,
        start_layer_id: int,
        num_layers: int,
        stream: torch.cuda.Stream,
    ) -> int:
        gpu_ids_t = torch.from_numpy(gpu_block_ids).cuda()
        cpu_ids_t = torch.from_numpy(cpu_block_ids)

        total_comp = _C.transfer_kv_blocks_ans_decomp(
            self.ctx,
            gpu_ids_t,
            self.gpu_tensor_ptrs,
            self.page_size_bytes,
            cpu_ids_t,
            self.cpu_tensor_ptrs,
            self.cpu_slot_bytes,
            self.chunk_size_bytes,
            start_layer_id,
            num_layers,
            self.is_mla,
            self.size_table.data_ptr(),
            self.cpu_size_table_block_stride,
            self.cpu_size_table_layer_stride,
            stream.cuda_stream,
        )
        return total_comp

    def destroy(self):
        self.ctx.destroy()
