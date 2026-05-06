#include "tiny_llm_ext.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <stdexcept>

namespace tiny_llm_ext_torch_ref {

namespace {

constexpr int BR = 32;
constexpr int BC = 32;
constexpr int MAX_E = 128;
constexpr float NEG_INF = -1.0e20f;

__device__ inline float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__global__ void flash_attention_f32_e128(const float *q,
                                         const float *k,
                                         const float *v,
                                         const float *mask,
                                         float *out,
                                         const int is_causal,
                                         const int N,
                                         const int L,
                                         const int S,
                                         const int E,
                                         const int num_kv_heads,
                                         const int num_heads,
                                         const float scale,
                                         const int Tc) {
    const int n = blockIdx.x;
    const int i = blockIdx.y;
    const int a = threadIdx.y;  // Query row inside the tile.
    const int b = threadIdx.x;  // Key column inside the tile.

    if (n >= N || a >= BR || b >= BC) {
        return;
    }

    const bool is_i_in_range = (i * BR + a) < L;
    const int q_kv_ratio = num_heads / num_kv_heads;
    const int kv_head = n / q_kv_ratio;

    const float *q_ptr = q + (n * L + i * BR) * E;
    const float *k_ptr_base = k + kv_head * S * E;
    const float *v_ptr_base = v + kv_head * S * E;
    float *out_ptr = out + n * L * E;
    const float *mask_ptr = mask + n * L * S;

    __shared__ float q_local[BR][MAX_E];
    __shared__ float o_i[BR][MAX_E];

    if (b == 0) {
        for (int c = 0; c < E; ++c) {
            o_i[a][c] = 0.0f;
            q_local[a][c] = is_i_in_range ? q_ptr[a * E + c] : 0.0f;
        }
    }
    __syncthreads();

    float m_i = NEG_INF;
    float l_i = 0.0f;

    for (int j = 0; j < Tc; ++j) {
        if (is_causal) {
            const int row_max = min((i + 1) * BR - 1, L - 1);
            const int col_min = j * BC;
            if (col_min > row_max + (S - L)) {
                continue;
            }
        }

        const bool is_j_in_range = (j * BC + b) < S;
        const float *k_ptr = k_ptr_base + j * BC * E;
        const float *v_ptr = v_ptr_base + j * BC * E;

        float s_a_b = 0.0f;
        if (is_i_in_range && is_j_in_range) {
            for (int c = 0; c < E; ++c) {
                s_a_b += q_local[a][c] * k_ptr[b * E + c];
            }
            s_a_b *= scale;

            const int row_min = i * BR;
            const int col_max = min((j + 1) * BC - 1, S - 1);
            const bool block_all_valid = is_causal && (col_max <= row_min + (S - L));
            if (!block_all_valid) {
                s_a_b += mask_ptr[(i * BR + a) * S + (j * BC + b)];
            }
        } else {
            s_a_b = NEG_INF;
        }

        const float rowmax = warp_reduce_max(s_a_b);
        const float new_max = fmaxf(m_i, rowmax);
        const float m_i_diff = m_i - new_max;
        const float m_i_diff_exp = expf(m_i_diff);
        m_i = new_max;

        const float p_a_b = (is_i_in_range && is_j_in_range) ? expf(s_a_b - m_i) : 0.0f;
        const float rowsum = warp_reduce_sum(p_a_b);
        l_i = m_i_diff_exp * l_i + rowsum;

        for (int c = 0; c < E; ++c) {
            const float pv = (is_i_in_range && is_j_in_range) ? p_a_b * v_ptr[b * E + c] : 0.0f;
            const float res = warp_reduce_sum(pv);
            if (b == 0 && is_i_in_range) {
                o_i[a][c] = m_i_diff_exp * o_i[a][c] + res;
            }
        }
    }

    if (b == 0 && is_i_in_range) {
        for (int c = 0; c < E; ++c) {
            out_ptr[(i * BR + a) * E + c] = o_i[a][c] / l_i;
        }
    }
}

}  // namespace

void flash_attention_cuda(const torch::Tensor &q,
                          const torch::Tensor &k,
                          const torch::Tensor &v,
                          const torch::Tensor &mask,
                          torch::Tensor &out,
                          const float scale,
                          const bool is_causal,
                          const int num_kv_heads,
                          const int num_heads) {
    const int64_t E = q.size(2);
    if (E > MAX_E) {
        throw std::runtime_error("flash_attention: CUDA path currently supports E <= 128");
    }

    const at::cuda::CUDAGuard device_guard(out.device());
    const auto stream = at::cuda::getCurrentCUDAStream(out.device().index());

    const int N = static_cast<int>(q.size(0));
    const int L = static_cast<int>(q.size(1));
    const int S = static_cast<int>(k.size(1));
    const int Tc = static_cast<int>((S + BC - 1) / BC);

    // One block computes one (head-batch, query-tile) pair.
    const dim3 threads_per_block(BC, BR);
    const dim3 blocks(N, (L + BR - 1) / BR);

    flash_attention_f32_e128<<<blocks, threads_per_block, 0, stream.stream()>>>(
        q.const_data_ptr<float>(),
        k.const_data_ptr<float>(),
        v.const_data_ptr<float>(),
        mask.const_data_ptr<float>(),
        out.data_ptr<float>(),
        is_causal ? 1 : 0,
        N,
        L,
        S,
        static_cast<int>(E),
        num_kv_heads,
        num_heads,
        scale,
        Tc);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tiny_llm_ext_torch_ref
