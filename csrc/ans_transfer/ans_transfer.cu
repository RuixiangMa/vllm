#include "ans_transfer.cuh"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vllm {

#define ANS_NVCOMP_CHECK(call)                                          \
  do {                                                                  \
    nvcompStatus_t _s = (call);                                         \
    if (_s != nvcompSuccess) {                                          \
      fprintf(stderr, "[nvcomp] error %d at %s:%d\n",                   \
              (int)_s, __FILE__, __LINE__);                             \
      throw std::runtime_error("nvcomp ANS error");                     \
    }                                                                   \
  } while (0)

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
      fprintf(stderr, "[nvcomp] CUDA error: %s at %s:%d\n",            \
              cudaGetErrorString(_e), __FILE__, __LINE__);              \
      throw std::runtime_error(cudaGetErrorString(_e));                 \
    }                                                                   \
  } while (0)

__global__ void ans_build_gpu_chunk_ptrs_kernel(
    void** __restrict__ d_ptrs,
    void* const* __restrict__ gpu_tensor_ptrs,
    int64_t gpu_block_stride,
    const int64_t* __restrict__ gpu_block_ids,
    int start_layer_id, int kv_dim, int num_blocks,
    int batch_start, int bsz)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz;
         i += gridDim.x * blockDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;
        int t_idx = (start_layer_id + layer) * 2 + kv;
        d_ptrs[i] = static_cast<uint8_t*>(gpu_tensor_ptrs[t_idx])
                    + gpu_block_ids[b] * gpu_block_stride;
    }
}

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, size_t cpu_slot_capacity,
                    int data_type, int transfer_sms) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ans_ctx_create: ctx must be non-null");
    }
    ans_ctx_destroy(ctx);

    if (transfer_sms == -1) {
        transfer_sms = 4;
    }
    if (transfer_sms <= 0) {
        throw std::invalid_argument(
            "ans_ctx_create: transfer_sms must be positive or -1");
    }
    ctx->transfer_sms = transfer_sms;
    if (max_num_chunks == 0 || max_chunk_size == 0) {
        throw std::invalid_argument(
            "ans_ctx_create: max_num_chunks and max_chunk_size must be > 0");
    }

    CUDA_CHECK(cudaGetDevice(&ctx->device_id));
    ctx->max_num_chunks = max_num_chunks;
    ctx->max_chunk_size = max_chunk_size;
    ctx->cpu_slot_capacity = cpu_slot_capacity;

    try {
        ctx->comp_opts = nvcompBatchedANSCompressDefaultOpts;
        ctx->decomp_opts = nvcompBatchedANSDecompressDefaultOpts;
        if (data_type == 0) {
            ctx->comp_opts.data_type = NVCOMP_TYPE_FLOAT16;
        } else {
            ctx->comp_opts.data_type = NVCOMP_TYPE_CHAR;
        }

        const size_t max_total = max_num_chunks * max_chunk_size;

        ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetMaxOutputChunkSize(
            max_chunk_size, ctx->comp_opts, &ctx->max_comp_chunk_bytes));
        ctx->max_comp_chunk_bytes = (ctx->max_comp_chunk_bytes + 15) & ~size_t(15);

        ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetTempSizeAsync(
            max_num_chunks, max_chunk_size, ctx->comp_opts,
            &ctx->comp_temp_bytes, max_total));
        ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeAsync(
            max_num_chunks, max_chunk_size, ctx->decomp_opts,
            &ctx->decomp_temp_bytes, max_total));

        const size_t comp_staging_total = max_num_chunks * ctx->max_comp_chunk_bytes;
        const size_t ptr_bytes  = max_num_chunks * sizeof(void*);
        const size_t size_bytes = max_num_chunks * sizeof(size_t);
        const size_t status_bytes = max_num_chunks * sizeof(nvcompStatus_t);

        CUDA_CHECK(cudaMalloc(&ctx->d_comp_temp,       ctx->comp_temp_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_staging_base, 2 * comp_staging_total));
        ctx->d_comp_staging[0] = ctx->d_comp_staging_base;
        ctx->d_comp_staging[1] = ctx->d_comp_staging_base + comp_staging_total;
        CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_ptrs,     ptr_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_sizes,    size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[0],    ptr_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[1],    ptr_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[0],   size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[1],   size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_comp_statuses,   status_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_overflow,        sizeof(int)));

        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_temp,         ctx->decomp_temp_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[0],      ptr_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[1],      ptr_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[0], size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[1], size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_act_sizes,    size_bytes));
        CUDA_CHECK(cudaMalloc(&ctx->d_decomp_statuses,     status_bytes));

        ctx->h_ptr_scratch.resize(max_num_chunks);
        ctx->h_size_scratch.resize(max_num_chunks);

        for (int slot = 0; slot < 2; slot++) {
            for (size_t i = 0; i < max_num_chunks; i++)
                ctx->h_ptr_scratch[i] = ctx->d_comp_staging[slot]
                                        + i * ctx->max_comp_chunk_bytes;
            CUDA_CHECK(cudaMemcpy(ctx->d_comp_ptrs[slot],
                                  ctx->h_ptr_scratch.data(),
                                  ptr_bytes, cudaMemcpyHostToDevice));
        }

        for (size_t i = 0; i < max_num_chunks; i++)
            ctx->h_size_scratch[i] = max_chunk_size;
        CUDA_CHECK(cudaMemcpy(ctx->d_uncomp_sizes,
                              ctx->h_size_scratch.data(),
                              size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[0],
                              ctx->h_size_scratch.data(),
                              size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[1],
                              ctx->h_size_scratch.data(),
                              size_bytes, cudaMemcpyHostToDevice));

        {
            int least_priority, greatest_priority;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
                &least_priority, &greatest_priority));
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &ctx->cpu_transfer_stream, cudaStreamNonBlocking,
                greatest_priority));
        }
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &ctx->compress_done[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(
                &ctx->slot_done[i], cudaEventDisableTiming));
        }

        ctx->initialized = true;
    } catch (...) {
        ans_ctx_destroy(ctx);
        throw;
    }
}

void ans_ctx_destroy(ANSTransferContext* ctx) {
    if (ctx == nullptr) return;

    int saved_device = -1;
    cudaGetDevice(&saved_device);
    if (ctx->device_id >= 0) cudaSetDevice(ctx->device_id);

    if (ctx->d_comp_temp)          cudaFree(ctx->d_comp_temp);
    if (ctx->d_comp_staging_base)  cudaFree(ctx->d_comp_staging_base);
    for (int i = 0; i < 2; i++) {
        if (ctx->d_comp_ptrs[i])   cudaFree(ctx->d_comp_ptrs[i]);
        if (ctx->d_comp_sizes[i])  cudaFree(ctx->d_comp_sizes[i]);
        if (ctx->compress_done[i]) cudaEventDestroy(ctx->compress_done[i]);
        if (ctx->slot_done[i])     cudaEventDestroy(ctx->slot_done[i]);
    }
    if (ctx->d_comp_statuses)      cudaFree(ctx->d_comp_statuses);
    if (ctx->d_overflow)           cudaFree(ctx->d_overflow);
    if (ctx->cpu_transfer_stream)  cudaStreamDestroy(ctx->cpu_transfer_stream);
    if (ctx->d_uncomp_ptrs)        cudaFree(ctx->d_uncomp_ptrs);
    if (ctx->d_uncomp_sizes)       cudaFree(ctx->d_uncomp_sizes);
    if (ctx->d_decomp_temp)        cudaFree(ctx->d_decomp_temp);
    for (int i = 0; i < 2; i++) {
        if (ctx->d_decomp_ptrs[i])      cudaFree(ctx->d_decomp_ptrs[i]);
        if (ctx->d_decomp_buf_sizes[i]) cudaFree(ctx->d_decomp_buf_sizes[i]);
    }
    if (ctx->d_decomp_act_sizes)   cudaFree(ctx->d_decomp_act_sizes);
    if (ctx->d_decomp_statuses)    cudaFree(ctx->d_decomp_statuses);

    ctx->initialized = false;
    ctx->device_id = -1;
    ctx->max_num_chunks = 0;
    ctx->max_chunk_size = 0;
    ctx->max_comp_chunk_bytes = 0;
    ctx->comp_temp_bytes = 0;
    ctx->decomp_temp_bytes = 0;
    ctx->cpu_slot_capacity = 0;
    ctx->comp_opts = {};
    ctx->decomp_opts = {};
    ctx->d_comp_temp = nullptr;
    ctx->d_comp_staging_base = nullptr;
    ctx->d_uncomp_ptrs = nullptr;
    ctx->d_uncomp_sizes = nullptr;
    ctx->d_comp_statuses = nullptr;
    ctx->d_overflow = nullptr;
    ctx->d_decomp_temp = nullptr;
    ctx->d_decomp_act_sizes = nullptr;
    ctx->d_decomp_statuses = nullptr;
    ctx->cpu_transfer_stream = nullptr;
    ctx->transfer_sms = 0;
    for (int i = 0; i < 2; i++) {
        ctx->d_comp_staging[i] = nullptr;
        ctx->d_comp_ptrs[i] = nullptr;
        ctx->d_comp_sizes[i] = nullptr;
        ctx->d_decomp_ptrs[i] = nullptr;
        ctx->d_decomp_buf_sizes[i] = nullptr;
        ctx->compress_done[i] = nullptr;
        ctx->slot_done[i] = nullptr;
    }
    ctx->h_ptr_scratch.clear();
    ctx->h_size_scratch.clear();

    if (saved_device >= 0) cudaSetDevice(saved_device);
}

ANSTransferContext::~ANSTransferContext() {
    ans_ctx_destroy(this);
}

static void sync_streams(ANSTransferContext* ctx, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->cpu_transfer_stream));
}

static int decode_chunk_index(
    int g, int kv_dim, int num_blocks,
    int* out_layer, int* out_kv, int* out_b) {
    *out_layer = g / (kv_dim * num_blocks);
    *out_kv = (g % (kv_dim * num_blocks)) / num_blocks;
    *out_b = g % num_blocks;
    return 0;
}

static uint32_t* compute_size_table_entry(
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    int64_t cpu_block_id, int layer, int kv) {
    return cpu_size_table_base
           + cpu_block_id              * cpu_size_table_block_stride
           + (int64_t)(layer)          * cpu_size_table_layer_stride
           + (int64_t)(kv);
}

size_t transfer_kv_blocks_ans_comp(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    const int64_t* gpu_block_ids,
    void* const* gpu_tensor_ptrs,
    int64_t gpu_block_stride,
    const int64_t* cpu_block_ids_host,
    void* const* cpu_tensor_ptrs_host,
    int64_t cpu_block_stride,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    cudaStream_t stream) {

    if (!ctx || !ctx->initialized) {
        throw std::invalid_argument(
            "transfer_kv_blocks_ans_comp: ANSTransferContext not initialized");
    }

    const int kv_dim = is_mla ? 1 : 2;
    const int total_chunks = num_layers * kv_dim * num_blocks;
    const int batch_cap = static_cast<int>(ctx->max_num_chunks);
    const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

    if (chunk_size_in_bytes <= 0 ||
        static_cast<size_t>(chunk_size_in_bytes) != ctx->max_chunk_size) {
        throw std::invalid_argument(
            "transfer_kv_blocks_ans_comp: chunk_size_in_bytes must equal "
            "ctx->max_chunk_size");
    }

    CUDA_CHECK(cudaMemset(ctx->d_overflow, 0, sizeof(int)));

    std::vector<size_t> h_comp_sizes(batch_cap);

    for (int bi = 0; bi < num_batches; bi++) {
        const int bs  = bi * batch_cap;
        const int bsz = std::min(batch_cap, total_chunks - bs);
        const int cur = bi % 2;

        if (bi >= 2)
            CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->slot_done[cur], 0));

        {
            int threads = 256;
            int blocks = std::min((bsz + threads - 1) / threads,
                                  ctx->transfer_sms);
            ans_build_gpu_chunk_ptrs_kernel<<<blocks, threads, 0, stream>>>(
                ctx->d_uncomp_ptrs, gpu_tensor_ptrs, gpu_block_stride,
                gpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz);

            ANS_NVCOMP_CHECK(nvcompBatchedANSCompressAsync(
                (const void* const*)ctx->d_uncomp_ptrs,
                ctx->d_uncomp_sizes,
                chunk_size_in_bytes,
                bsz,
                ctx->d_comp_temp,
                ctx->comp_temp_bytes,
                ctx->d_comp_ptrs[cur],
                ctx->d_comp_sizes[cur],
                ctx->comp_opts,
                ctx->d_comp_statuses,
                stream));
            CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], stream));
        }

        {
            CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream,
                                           ctx->compress_done[cur], 0));
            CUDA_CHECK(cudaMemcpyAsync(h_comp_sizes.data(),
                                       ctx->d_comp_sizes[cur],
                                       bsz * sizeof(size_t),
                                       cudaMemcpyDeviceToHost,
                                       ctx->cpu_transfer_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx->cpu_transfer_stream));

            bool overflow = false;
            for (int i = 0; i < bsz; i++) {
                int layer, kv, b;
                decode_chunk_index(bs + i, kv_dim, num_blocks, &layer, &kv, &b);
                size_t sz = h_comp_sizes[i];
                int64_t cpu_bid = cpu_block_ids_host[b];
                int t_idx = (start_layer_id + layer) * 2 + kv;
                uint8_t* cpu_dest = static_cast<uint8_t*>(
                    cpu_tensor_ptrs_host[t_idx])
                    + cpu_bid * cpu_block_stride;

                if (sz > static_cast<size_t>(chunk_size_in_bytes)) {
                    overflow = true;
                    *compute_size_table_entry(cpu_size_table_base,
                        cpu_size_table_block_stride, cpu_size_table_layer_stride,
                        cpu_bid, start_layer_id + layer, kv) = 0;
                    continue;
                }

                CUDA_CHECK(cudaMemcpyAsync(
                    cpu_dest,
                    ctx->d_comp_staging[cur] + (size_t)i * ctx->max_comp_chunk_bytes,
                    sz,
                    cudaMemcpyDeviceToHost,
                    ctx->cpu_transfer_stream));

                *compute_size_table_entry(cpu_size_table_base,
                    cpu_size_table_block_stride, cpu_size_table_layer_stride,
                    cpu_bid, start_layer_id + layer, kv) = static_cast<uint32_t>(sz);
            }

            CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur],
                                       ctx->cpu_transfer_stream));

            if (overflow) {
                sync_streams(ctx, stream);
                throw std::runtime_error(
                    "nvcomp compressed payload exceeded the CPU chunk slot");
            }
        }
    }

    sync_streams(ctx, stream);

    size_t total_comp = 0;
    const int total_entries = num_layers * kv_dim * num_blocks;
    for (int g = 0; g < total_entries; g++) {
        int layer, kv, b;
        decode_chunk_index(g, kv_dim, num_blocks, &layer, &kv, &b);
        uint32_t* entry = compute_size_table_entry(cpu_size_table_base,
            cpu_size_table_block_stride, cpu_size_table_layer_stride,
            cpu_block_ids_host[b], start_layer_id + layer, kv);
        total_comp += static_cast<size_t>(*entry);
    }
    return total_comp;
}

size_t transfer_kv_blocks_ans_decomp(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    const int64_t* gpu_block_ids,
    void* const* gpu_tensor_ptrs,
    int64_t gpu_block_stride,
    const int64_t* cpu_block_ids_host,
    void* const* cpu_tensor_ptrs_host,
    int64_t cpu_block_stride,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    cudaStream_t stream) {

    if (!ctx || !ctx->initialized) {
        throw std::invalid_argument(
            "transfer_kv_blocks_ans_decomp: ANSTransferContext not initialized");
    }

    const int kv_dim = is_mla ? 1 : 2;
    const int total_chunks = num_layers * kv_dim * num_blocks;
    const int batch_cap = static_cast<int>(ctx->max_num_chunks);
    const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

    if (chunk_size_in_bytes <= 0 ||
        static_cast<size_t>(chunk_size_in_bytes) != ctx->max_chunk_size) {
        throw std::invalid_argument(
            "transfer_kv_blocks_ans_decomp: chunk_size_in_bytes must equal "
            "ctx->max_chunk_size");
    }

    for (int bi = 0; bi < num_batches; bi++) {
        const int bs  = bi * batch_cap;
        const int bsz = std::min(batch_cap, total_chunks - bs);
        const int cur = bi % 2;

        if (bi >= 2)
            CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream,
                                           ctx->slot_done[cur], 0));

        {
            int threads = 256;
            int blocks = std::min((bsz + threads - 1) / threads,
                                  ctx->transfer_sms);
            ans_build_gpu_chunk_ptrs_kernel<<<blocks, threads, 0, stream>>>(
                ctx->d_decomp_ptrs[cur], gpu_tensor_ptrs, gpu_block_stride,
                gpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz);
        }

        {
            std::vector<size_t> h_comp_sizes_batch(bsz);

            for (int i = 0; i < bsz; i++) {
                int layer, kv, b;
                decode_chunk_index(bs + i, kv_dim, num_blocks, &layer, &kv, &b);
                int64_t cpu_bid = cpu_block_ids_host[b];
                int t_idx = (start_layer_id + layer) * 2 + kv;
                const uint8_t* cpu_src = static_cast<const uint8_t*>(
                    cpu_tensor_ptrs_host[t_idx])
                    + cpu_bid * cpu_block_stride;

                uint32_t* entry = compute_size_table_entry(cpu_size_table_base,
                    cpu_size_table_block_stride, cpu_size_table_layer_stride,
                    cpu_bid, start_layer_id + layer, kv);
                size_t sz = static_cast<size_t>(*entry);
                h_comp_sizes_batch[i] = sz;

                CUDA_CHECK(cudaMemcpyAsync(
                    ctx->d_comp_staging[cur] + (size_t)i * ctx->max_comp_chunk_bytes,
                    cpu_src,
                    sz,
                    cudaMemcpyHostToDevice,
                    ctx->cpu_transfer_stream));
            }

            CUDA_CHECK(cudaMemcpyAsync(ctx->d_comp_sizes[cur],
                                       h_comp_sizes_batch.data(),
                                       bsz * sizeof(size_t),
                                       cudaMemcpyHostToDevice,
                                       ctx->cpu_transfer_stream));
        }

        CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur],
                                   ctx->cpu_transfer_stream));

        {
            CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->compress_done[cur], 0));

            ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
                (const void* const*)ctx->d_comp_ptrs[cur],
                ctx->d_comp_sizes[cur],
                ctx->d_decomp_buf_sizes[cur],
                ctx->d_decomp_act_sizes,
                bsz,
                ctx->d_decomp_temp,
                ctx->decomp_temp_bytes,
                ctx->d_decomp_ptrs[cur],
                ctx->decomp_opts,
                ctx->d_decomp_statuses,
                stream));
            CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur], stream));
        }
    }

    sync_streams(ctx, stream);

    size_t total_comp = 0;
    const int total_entries = num_layers * kv_dim * num_blocks;
    for (int g = 0; g < total_entries; g++) {
        int layer, kv, b;
        decode_chunk_index(g, kv_dim, num_blocks, &layer, &kv, &b);
        uint32_t* entry = compute_size_table_entry(cpu_size_table_base,
            cpu_size_table_block_stride, cpu_size_table_layer_stride,
            cpu_block_ids_host[b], start_layer_id + layer, kv);
        total_comp += static_cast<size_t>(*entry);
    }
    return total_comp;
}

#undef CUDA_CHECK
#undef ANS_NVCOMP_CHECK

} // namespace vllm
