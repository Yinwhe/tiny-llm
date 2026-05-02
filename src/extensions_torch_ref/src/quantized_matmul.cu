#include "tiny_llm_ext.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <stdexcept>

namespace tiny_llm_ext_torch_ref {

namespace {

__device__ constexpr int AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__device__ inline uint8_t unpack_int4(const uint32_t word, const int slot, const uint32_t mask) {
    return (word >> (slot * 4)) & mask;
}

__device__ inline uint8_t unpack_awq_int4(const uint32_t word, const int logical_slot, const uint32_t mask) {
    return unpack_int4(word, AWQ_REVERSE_ORDER[logical_slot], mask);
}

template <typename T>
__global__ void quantized_matmul_awq_w4a16(const T *scales,
                                           const int32_t *zeros,
                                           const T *a,
                                           const int32_t *b,
                                           T *out,
                                           const int M,
                                           const int N,
                                           const int K,
                                           const int group_size) {
    // Match the CPU outer loops directly: one thread computes one out[i, j].
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= K) {
        return;
    }

    constexpr int bits = 4;
    constexpr int packs_per_word = 32 / bits;
    const int groups_per_row = N / group_size;
    const int packed_k = K / packs_per_word;
    const int packed_col = j / packs_per_word;
    const int logical_slot = j % packs_per_word;
    const uint32_t pack_mask = (1u << bits) - 1u;

    const uint32_t *zeros_ptr = reinterpret_cast<const uint32_t *>(zeros);
    const uint32_t *b_ptr = reinterpret_cast<const uint32_t *>(b);

    float sum = 0.0f;

    for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
        const float scale = static_cast<float>(scales[group_idx * K + j]);
        const uint32_t packed_zero = zeros_ptr[group_idx * packed_k + packed_col];
        const uint8_t q_zero = unpack_awq_int4(packed_zero, logical_slot, pack_mask);

        for (int group_offset = 0; group_offset < group_size; ++group_offset) {
            const int in_col = group_idx * group_size + group_offset;

            // Logically this is q_weight[in_col, j]. Physically, 8 output
            // columns share one packed int32 word.
            const uint32_t packed_weight = b_ptr[in_col * packed_k + packed_col];
            const uint8_t q_weight = unpack_awq_int4(packed_weight, logical_slot, pack_mask);

            const float a_val = static_cast<float>(a[i * N + in_col]);
            const float weight_val =
                (static_cast<float>(q_weight) - static_cast<float>(q_zero)) * scale;
            sum += a_val * weight_val;
        }
    }

    out[i * K + j] = static_cast<T>(sum);
}

template <typename T>
void launch_quantized_matmul_cuda(const torch::Tensor &scales,
                                  const torch::Tensor &zeros,
                                  const torch::Tensor &a,
                                  const torch::Tensor &b,
                                  torch::Tensor &out,
                                  const int group_size) {
    const at::cuda::CUDAGuard device_guard(out.device());
    const auto stream = at::cuda::getCurrentCUDAStream(out.device().index());

    const int M = static_cast<int>(a.size(0));
    const int N = static_cast<int>(a.size(1));
    const int K = static_cast<int>(out.size(1));

    // One block covers a 32 x 8 tile of the output matrix.
    const dim3 threads_per_block(32, 8);
    const dim3 blocks((M + threads_per_block.x - 1) / threads_per_block.x,
                      (K + threads_per_block.y - 1) / threads_per_block.y);

    quantized_matmul_awq_w4a16<T><<<blocks, threads_per_block, 0, stream.stream()>>>(
        scales.const_data_ptr<T>(),
        zeros.const_data_ptr<int32_t>(),
        a.const_data_ptr<T>(),
        b.const_data_ptr<int32_t>(),
        out.data_ptr<T>(),
        M,
        N,
        K,
        group_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void quantized_matmul_cuda(const torch::Tensor &scales,
                           const torch::Tensor &zeros,
                           const torch::Tensor &a,
                           const torch::Tensor &b,
                           torch::Tensor &out,
                           const int group_size,
                           const int bits) {
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: CUDA path only supports bits=4");
    }
    if (group_size <= 0) {
        throw std::runtime_error("quantized_matmul: CUDA path requires group_size > 0");
    }

    if (out.scalar_type() == torch::kFloat16) {
        launch_quantized_matmul_cuda<c10::Half>(scales, zeros, a, b, out, group_size);
    } else if (out.scalar_type() == torch::kBFloat16) {
        launch_quantized_matmul_cuda<c10::BFloat16>(scales, zeros, a, b, out, group_size);
    } else {
        throw std::runtime_error("quantized_matmul: CUDA path only supports float16 or bfloat16");
    }
}

}  // namespace tiny_llm_ext_torch_ref
