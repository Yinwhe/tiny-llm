#include <torch/extension.h>

#include "tiny_llm_ext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm_ext_torch_ref {

namespace {

void check_contiguous(const torch::Tensor &tensor, const char *name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string("flash_attention: ") + name + " must be contiguous");
    }
}

}  // namespace

torch::Tensor flash_attention(const torch::Tensor &q,
                              const torch::Tensor &k,
                              const torch::Tensor &v,
                              const torch::Tensor &mask,
                              const float scale,
                              const bool is_causal,
                              const int num_kv_heads,
                              const int num_heads) {
    if (q.dtype() != torch::kFloat32 || k.dtype() != torch::kFloat32 || v.dtype() != torch::kFloat32 ||
        mask.dtype() != torch::kFloat32) {
        throw std::runtime_error("flash_attention: all input tensors must be float32");
    }
    if (q.dim() != 3 || k.dim() != 3 || v.dim() != 3) {
        throw std::runtime_error("flash_attention: q, k and v must be 3D");
    }
    if (mask.dim() != 3) {
        throw std::runtime_error("flash_attention: mask must be 3D");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }

    // Q: [N, L, E]
    // K: [N_KV, S, E]
    // V: [N_KV, S, E]
    // O: [N, L, E]
    // M: [N, L, S]
    if (q.size(0) % num_heads != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (k.size(0) % num_kv_heads != 0 || v.size(0) % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: k.shape[0] and v.shape[0] must be divisible by num_kv_heads");
    }
    if (q.size(2) != k.size(2) || q.size(2) != v.size(2)) {
        throw std::runtime_error("flash_attention: q.shape[2] must equal k.shape[2] and v.shape[2]");
    }
    if (q.size(0) / num_heads != k.size(0) / num_kv_heads) {
        throw std::runtime_error("flash_attention: number of heads mismatch");
    }
    if (k.size(1) != v.size(1)) {
        throw std::runtime_error("flash_attention: k.shape[1] must equal v.shape[1]");
    }
    if (mask.size(0) != q.size(0) || mask.size(1) != q.size(1) || mask.size(2) != k.size(1)) {
        throw std::runtime_error("flash_attention: mask must have shape [N, L, S]");
    }
    if (q.device() != k.device() || q.device() != v.device() || q.device() != mask.device()) {
        throw std::runtime_error("flash_attention: all input tensors must be on the same device");
    }

    auto out = torch::empty_like(q);

    FlashAttention op(scale, is_causal, num_kv_heads, num_heads);
    std::vector<torch::Tensor> inputs{q, k, v, mask};
    std::vector<torch::Tensor> outputs{out};

    if (q.is_cuda()) {
        op.eval_gpu(inputs, outputs);
    } else {
        op.eval_cpu(inputs, outputs);
    }

    return outputs[0];
}

void FlashAttention::eval_cpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    check_contiguous(q, "q");
    check_contiguous(k, "k");
    check_contiguous(v, "v");
    check_contiguous(mask, "mask");
    check_contiguous(out, "out");

    const int64_t N = q.size(0);
    const int64_t L = q.size(1);
    const int64_t S = k.size(1);
    const int64_t E = q.size(2);

    const int64_t Br = 32;
    const int64_t Bc = 32;
    const int64_t Tr = (L + Br - 1) / Br;
    const int64_t Tc = (S + Bc - 1) / Bc;

    const int64_t q_head_stride = L * E;
    const int64_t kv_head_stride = S * E;
    const int64_t mask_stride = L * S;
    const int64_t q_kv_heads_ratio = num_heads_ / num_kv_heads_;
    const int64_t causal_offset = S - L;

    const float *q_ptr = q.const_data_ptr<float>();
    const float *k_ptr = k.const_data_ptr<float>();
    const float *v_ptr = v.const_data_ptr<float>();
    const float *mask_ptr = mask.const_data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        const float *q_batch = q_ptr + n * q_head_stride;
        const float *k_batch = k_ptr + (n / q_kv_heads_ratio) * kv_head_stride;
        const float *v_batch = v_ptr + (n / q_kv_heads_ratio) * kv_head_stride;
        const float *mask_batch = mask_ptr + n * mask_stride;
        float *out_batch = out_ptr + n * q_head_stride;

        for (int64_t i = 0; i < Tr; ++i) {
            const int64_t br_upper_bound = std::min(L - i * Br, Br);

            std::vector<float> q_i(Br * E, 0.0f);
            for (int64_t a = 0; a < br_upper_bound; ++a) {
                const int64_t q_row = (i * Br + a) * E;
                for (int64_t c = 0; c < E; ++c) {
                    q_i[a * E + c] = q_batch[q_row + c];
                }
            }

            std::vector<float> o_i(Br * E, 0.0f);
            std::vector<float> l_i(Br, 0.0f);
            std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());

            for (int64_t j = 0; j < Tc; ++j) {
                const int64_t row_max = i * Br + br_upper_bound - 1;
                const int64_t col_min = j * Bc;

                // If this K/V tile is entirely to the right of the causal
                // boundary, it contributes nothing and can be skipped.
                if (is_causal_ && col_min > row_max + causal_offset) {
                    continue;
                }

                const int64_t bc_upper_bound = std::min(S - j * Bc, Bc);

                std::vector<float> k_j(Bc * E, 0.0f);
                std::vector<float> v_j(Bc * E, 0.0f);
                for (int64_t b = 0; b < bc_upper_bound; ++b) {
                    const int64_t kv_row = (j * Bc + b) * E;
                    for (int64_t c = 0; c < E; ++c) {
                        k_j[b * E + c] = k_batch[kv_row + c];
                        v_j[b * E + c] = v_batch[kv_row + c];
                    }
                }

                std::vector<float> s_i(Br * Bc, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    for (int64_t b = 0; b < bc_upper_bound; ++b) {
                        for (int64_t c = 0; c < E; ++c) {
                            s_i[a * Bc + b] += q_i[a * E + c] * k_j[b * E + c];
                        }
                    }
                }

                const int64_t row_min = i * Br;
                const int64_t col_max = j * Bc + bc_upper_bound - 1;
                const bool block_all_valid = is_causal_ && (col_max <= row_min + causal_offset);

                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    for (int64_t b = 0; b < bc_upper_bound; ++b) {
                        s_i[a * Bc + b] *= scale_;
                        if (!block_all_valid) {
                            const int64_t mask_idx = (i * Br + a) * S + (j * Bc + b);
                            s_i[a * Bc + b] += mask_batch[mask_idx];
                        }
                    }
                }

                std::vector<float> m_i_diff(Br, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    float rowmax = -std::numeric_limits<float>::infinity();
                    for (int64_t b = 0; b < bc_upper_bound; ++b) {
                        rowmax = std::max(rowmax, s_i[a * Bc + b]);
                    }
                    const float new_max = std::max(m_i[a], rowmax);
                    m_i_diff[a] = m_i[a] - new_max;
                    m_i[a] = new_max;
                }

                std::vector<float> p(Br * Bc, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    for (int64_t b = 0; b < bc_upper_bound; ++b) {
                        p[a * Bc + b] = std::exp(s_i[a * Bc + b] - m_i[a]);
                    }
                }

                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    float rowsum = 0.0f;
                    for (int64_t b = 0; b < bc_upper_bound; ++b) {
                        rowsum += p[a * Bc + b];
                    }
                    l_i[a] = std::exp(m_i_diff[a]) * l_i[a] + rowsum;
                }

                for (int64_t a = 0; a < br_upper_bound; ++a) {
                    for (int64_t c = 0; c < E; ++c) {
                        float res = 0.0f;
                        for (int64_t b = 0; b < bc_upper_bound; ++b) {
                            res += p[a * Bc + b] * v_j[b * E + c];
                        }
                        o_i[a * E + c] = std::exp(m_i_diff[a]) * o_i[a * E + c] + res;
                    }
                }
            }

            for (int64_t a = 0; a < br_upper_bound; ++a) {
                for (int64_t c = 0; c < E; ++c) {
                    o_i[a * E + c] /= l_i[a];
                }
            }

            for (int64_t a = 0; a < br_upper_bound; ++a) {
                const int64_t out_row = (i * Br + a) * E;
                for (int64_t c = 0; c < E; ++c) {
                    out_batch[out_row + c] = o_i[a * E + c];
                }
            }
        }
    }
}

void FlashAttention::eval_gpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    check_contiguous(q, "q");
    check_contiguous(k, "k");
    check_contiguous(v, "v");
    check_contiguous(mask, "mask");
    check_contiguous(out, "out");

#ifdef _CUDA_
    flash_attention_cuda(q, k, v, mask, out, scale_, is_causal_, num_kv_heads_, num_heads_);
#else
    (void)q;
    (void)k;
    (void)v;
    (void)mask;
    (void)out;
    throw std::runtime_error("flash_attention: CUDA implementation not available");
#endif
}

void FlashAttention::print(std::ostream &os) const {
    os << name() << "(scale=" << scale_ << ", is_causal=" << is_causal_ << ", num_kv_heads=" << num_kv_heads_
       << ", num_heads=" << num_heads_ << ")";
}

std::pair<std::vector<torch::Tensor>, std::vector<int64_t>> FlashAttention::vmap(
    const std::vector<torch::Tensor> &inputs, const std::vector<int64_t> &axes) const {
    (void)inputs;
    (void)axes;
    throw std::runtime_error("FlashAttention has no vmap implementation.");
}

bool FlashAttention::is_equivalent(const FlashAttention &other) const {
    return scale_ == other.scale_ && is_causal_ == other.is_causal_ && num_kv_heads_ == other.num_kv_heads_ &&
           num_heads_ == other.num_heads_;
}

}  // namespace tiny_llm_ext_torch_ref
