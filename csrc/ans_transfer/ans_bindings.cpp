#include "ans_transfer.cuh"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using vllm::ANSTransferContext;

static size_t ans_comp_binding(
    ANSTransferContext& ctx,
    const torch::Tensor& gpu_block_ids_tensor,
    const torch::Tensor& gpu_tensor_ptrs_tensor,
    int64_t gpu_block_stride,
    const torch::Tensor& cpu_block_ids_tensor,
    const torch::Tensor& cpu_tensor_ptrs_tensor,
    int64_t cpu_block_stride,
    int64_t chunk_size_in_bytes,
    int start_layer_id, int num_layers, bool is_mla,
    int64_t cpu_size_table_ptr,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    int64_t stream_ptr) {

    TORCH_CHECK(gpu_block_ids_tensor.dtype() == torch::kInt64,
                "gpu_block_ids must be int64");
    TORCH_CHECK(cpu_block_ids_tensor.dtype() == torch::kInt64,
                "cpu_block_ids must be int64");
    TORCH_CHECK(cpu_tensor_ptrs_tensor.dtype() == torch::kInt64,
                "cpu_tensor_ptrs must be int64");

    int num_blocks = gpu_block_ids_tensor.numel();
    const int64_t* gpu_block_ids = gpu_block_ids_tensor.data_ptr<int64_t>();
    const int64_t* cpu_block_ids_host = cpu_block_ids_tensor.data_ptr<int64_t>();
    void** gpu_tensor_ptrs = reinterpret_cast<void**>(
        gpu_tensor_ptrs_tensor.data_ptr<int64_t>());
    void** cpu_tensor_ptrs_host = reinterpret_cast<void**>(
        cpu_tensor_ptrs_tensor.data_ptr<int64_t>());

    return vllm::transfer_kv_blocks_ans_comp(
        &ctx,
        num_blocks, start_layer_id, num_layers,
        gpu_block_ids, gpu_tensor_ptrs, gpu_block_stride,
        cpu_block_ids_host, cpu_tensor_ptrs_host, cpu_block_stride,
        chunk_size_in_bytes, is_mla,
        reinterpret_cast<uint32_t*>(cpu_size_table_ptr),
        cpu_size_table_block_stride, cpu_size_table_layer_stride,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

static size_t ans_decomp_binding(
    ANSTransferContext& ctx,
    const torch::Tensor& gpu_block_ids_tensor,
    const torch::Tensor& gpu_tensor_ptrs_tensor,
    int64_t gpu_block_stride,
    const torch::Tensor& cpu_block_ids_tensor,
    const torch::Tensor& cpu_tensor_ptrs_tensor,
    int64_t cpu_block_stride,
    int64_t chunk_size_in_bytes,
    int start_layer_id, int num_layers, bool is_mla,
    int64_t cpu_size_table_ptr,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    int64_t stream_ptr) {

    TORCH_CHECK(gpu_block_ids_tensor.dtype() == torch::kInt64,
                "gpu_block_ids must be int64");
    TORCH_CHECK(cpu_block_ids_tensor.dtype() == torch::kInt64,
                "cpu_block_ids must be int64");
    TORCH_CHECK(cpu_tensor_ptrs_tensor.dtype() == torch::kInt64,
                "cpu_tensor_ptrs must be int64");

    int num_blocks = gpu_block_ids_tensor.numel();
    const int64_t* gpu_block_ids = gpu_block_ids_tensor.data_ptr<int64_t>();
    const int64_t* cpu_block_ids_host = cpu_block_ids_tensor.data_ptr<int64_t>();
    void** gpu_tensor_ptrs = reinterpret_cast<void**>(
        gpu_tensor_ptrs_tensor.data_ptr<int64_t>());
    void** cpu_tensor_ptrs_host = reinterpret_cast<void**>(
        cpu_tensor_ptrs_tensor.data_ptr<int64_t>());

    return vllm::transfer_kv_blocks_ans_decomp(
        &ctx,
        num_blocks, start_layer_id, num_layers,
        gpu_block_ids, gpu_tensor_ptrs, gpu_block_stride,
        cpu_block_ids_host, cpu_tensor_ptrs_host, cpu_block_stride,
        chunk_size_in_bytes, is_mla,
        reinterpret_cast<uint32_t*>(cpu_size_table_ptr),
        cpu_size_table_block_stride, cpu_size_table_layer_stride,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ANSTransferContext>(m, "ANSTransferContext")
        .def(py::init([](size_t max_num_chunks, size_t max_chunk_size,
                         size_t cpu_slot_capacity, int data_type,
                         int transfer_sms) {
            auto ctx = std::make_unique<ANSTransferContext>();
            vllm::ans_ctx_create(ctx.get(), max_num_chunks, max_chunk_size,
                                 cpu_slot_capacity, data_type, transfer_sms);
            return ctx;
        }),
        py::arg("max_num_chunks"), py::arg("max_chunk_size"),
        py::arg("cpu_slot_capacity"), py::arg("data_type") = 0,
        py::arg("transfer_sms") = -1)
        .def("destroy", [](ANSTransferContext& ctx) {
            vllm::ans_ctx_destroy(&ctx);
        })
        .def_readonly("max_num_chunks", &ANSTransferContext::max_num_chunks)
        .def_readonly("max_chunk_size", &ANSTransferContext::max_chunk_size)
        .def_readonly("max_comp_chunk_bytes",
                       &ANSTransferContext::max_comp_chunk_bytes)
        .def_readonly("cpu_slot_capacity",
                       &ANSTransferContext::cpu_slot_capacity);

    m.def("transfer_kv_blocks_ans_comp", &ans_comp_binding,
          "ANS compress on GPU then D2H transfer",
          py::arg("ctx"), py::arg("gpu_block_ids"),
          py::arg("gpu_tensor_ptrs"), py::arg("gpu_block_stride"),
          py::arg("cpu_block_ids"), py::arg("cpu_tensor_ptrs"),
          py::arg("cpu_block_stride"),
          py::arg("chunk_size_bytes"),
          py::arg("start_layer_id"), py::arg("num_layers"),
          py::arg("is_mla"), py::arg("cpu_size_table_ptr"),
          py::arg("cpu_size_table_block_stride"),
          py::arg("cpu_size_table_layer_stride"),
          py::arg("stream_ptr"));

    m.def("transfer_kv_blocks_ans_decomp", &ans_decomp_binding,
          "H2D transfer then ANS decompress on GPU",
          py::arg("ctx"), py::arg("gpu_block_ids"),
          py::arg("gpu_tensor_ptrs"), py::arg("gpu_block_stride"),
          py::arg("cpu_block_ids"), py::arg("cpu_tensor_ptrs"),
          py::arg("cpu_block_stride"),
          py::arg("chunk_size_bytes"),
          py::arg("start_layer_id"), py::arg("num_layers"),
          py::arg("is_mla"), py::arg("cpu_size_table_ptr"),
          py::arg("cpu_size_table_block_stride"),
          py::arg("cpu_size_table_layer_stride"),
          py::arg("stream_ptr"));
}
