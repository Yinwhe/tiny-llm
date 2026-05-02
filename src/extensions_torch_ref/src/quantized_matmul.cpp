#include <ATen/ops/tensor.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/types.h>

#include "tiny_llm_ext.h"

#include <cstdint>
#include <string>
#include <stdexcept>

namespace tiny_llm_ext_torch_ref {

torch::Tensor quantized_matmul(const torch::Tensor &scales,  // Per-group scaling factors
                               const torch::Tensor &zeros,   // Per-group zero points / packed offsets
                               const int group_size,         // Group size
                               const int bits,               // Number of bits
                               const torch::Tensor &a,       // Input activation tensor
                               const torch::Tensor &b,       // Packed quantized weight tensor
                               const bool transpose_b        // Whether to transpose b
) {
    // This reference implementation currently supports AWQ-style int4
    // weights, with `b` stored in transposed packed form.
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size <= 0) {
        throw std::runtime_error("quantized_matmul: group_size must be positive");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }

    if (scales.device() != a.device() || a.device() != b.device() || b.device() != zeros.device()) {
        throw std::runtime_error("quantized_matmul: scales, a, b and zeros must be in the same device");
    }

    if (scales.dtype() != torch::kFloat16 && scales.dtype() != torch::kBFloat16 && scales.dtype() != torch::kFloat32) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16 or float32");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must be the same dtype as scales");
    }

    if (zeros.dtype() != torch::kInt32) {
        throw std::runtime_error("quantized_matmul: zeros must be int32");
    }
    if (b.dtype() != torch::kInt32) {
        throw std::runtime_error("quantized_matmul: b must be int32");
    }

    if (scales.dim() != 2) {
        throw std::runtime_error("quantized_matmul: scales must be a 2D array");
    }
    if (zeros.dim() != 2) {
        throw std::runtime_error("quantized_matmul: zeros must be a 2D array");
    }
    if (a.dim() < 2) {
        throw std::runtime_error("quantized_matmul: a must have at least 2 dimensions");
    }
    if (b.dim() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }

    constexpr int64_t packs_per_word = 8;

    // a:      [..., N]
    // b:      [N, K / 8]
    // scales: [N / group_size, K]
    // zeros:  [N / group_size, K / 8]
    const int64_t n = a.size(-1);   // input features
    const int64_t k = scales.size(1);  // output features

    if (n % group_size != 0) {
        throw std::runtime_error("quantized_matmul: a.shape()[-1] must be divisible by group_size");
    }
    const int64_t num_groups = n / group_size;
    if (k % packs_per_word != 0) {
        throw std::runtime_error("quantized_matmul: scales.shape()[1] must be divisible by 8");
    }
    const int64_t packed_k = k / packs_per_word;

    if (scales.size(0) != num_groups) {
        throw std::runtime_error("quantized_matmul: scales.shape()[0] must equal a.shape()[-1] / group_size");
    }
    if (zeros.size(0) != num_groups) {
        throw std::runtime_error("quantized_matmul: zeros.shape()[0] must equal a.shape()[-1] / group_size");
    }
    if (zeros.size(1) != packed_k) {
        throw std::runtime_error("quantized_matmul: zeros.shape()[1] must equal scales.shape()[1] / 8");
    }
    if (b.size(0) != n) {
        throw std::runtime_error("quantized_matmul: b.shape()[0] must equal a.shape()[-1]");
    }
    if (b.size(1) != packed_k) {
        throw std::runtime_error("quantized_matmul: b.shape()[1] must equal scales.shape()[1] / 8");
    }

    std::vector<int64_t> out_shape(a.sizes().begin(), a.sizes().end());
    out_shape.back() = k;

    // The low-level kernels operate on 2D matrices, so we flatten the
    // batch prefix of `a` and reshape the result back afterwards.
    auto a_2d = a.reshape({-1, n}).contiguous();
    auto out_2d = torch::empty({a_2d.size(0), k}, torch::TensorOptions().device(a.device()).dtype(a.dtype()));

    QuantizedMatmul op(group_size, bits);
    std::vector<torch::Tensor> inputs{scales.contiguous(), zeros.contiguous(), a_2d, b.contiguous()};
    std::vector<torch::Tensor> outputs{out_2d};

    if (a.is_cuda()) {
        op.eval_gpu(inputs, outputs);
    } else {
        op.eval_cpu(inputs, outputs);
    }

    return outputs[0].reshape(out_shape);
}

namespace {

// AWQ packs 8 logical int4 values into one uint32 using the order
// [0, 2, 4, 6, 1, 3, 5, 7]. To recover logical slot `j % 8`, use the
// inverse mapping below before extracting the 4-bit nibble.
constexpr int64_t AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

inline uint8_t unpack_int4(const uint32_t word, const int64_t slot, const uint32_t mask) {
    return (word >> (slot * 4)) & mask;
}

inline uint8_t unpack_awq_int4(const uint32_t word, const int64_t logical_slot, const uint32_t mask) {
    return unpack_int4(word, AWQ_REVERSE_ORDER[logical_slot], mask);
}

void check_contiguous(const torch::Tensor &tensor, const char *name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string("quantized_matmul: ") + name + " must be contiguous");
    }
}

template <typename T>
void quantized_matmul_cpu_impl(const torch::Tensor &scales,
                               const torch::Tensor &zeros,
                               const torch::Tensor &a,
                               const torch::Tensor &b,
                               torch::Tensor &out,
                               const int group_size,
                               const int bits) {
    const int64_t m = a.size(0);
    const int64_t n = a.size(1);
    const int64_t k = out.size(1);

    const int64_t groups_per_row = n / group_size;
    const int64_t packs_per_word = 32 / bits;
    const int64_t packed_k = k / packs_per_word;

    const T *a_ptr = a.const_data_ptr<T>();
    const T *scales_ptr = scales.const_data_ptr<T>();
    const uint32_t *zeros_ptr = reinterpret_cast<const uint32_t *>(zeros.const_data_ptr<int32_t>());
    const uint32_t *b_ptr = reinterpret_cast<const uint32_t *>(b.const_data_ptr<int32_t>());
    T *out_ptr = out.data_ptr<T>();

    const uint32_t pack_mask = (1u << bits) - 1u;

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            const int64_t packed_col = j / packs_per_word;
            const int64_t logical_slot = j % packs_per_word;

            for (int64_t group_idx = 0; group_idx < groups_per_row; ++group_idx) {
                // This group contributes `group_size` consecutive input
                // features to the dot-product for out[i, j].
                const float scale = static_cast<float>(scales_ptr[group_idx * k + j]);
                const uint32_t packed_zero = zeros_ptr[group_idx * packed_k + packed_col];
                const uint8_t q_zero = unpack_awq_int4(packed_zero, logical_slot, pack_mask);

                for (int64_t group_offset = 0; group_offset < group_size; ++group_offset) {
                    const int64_t in_col = group_idx * group_size + group_offset;

                    // Logically this is q_weight[in_col, j]. Physically, 8
                    // output columns share one packed int32 word.
                    const uint32_t packed_weight = b_ptr[in_col * packed_k + packed_col];

                    const float a_val = static_cast<float>(a_ptr[i * n + in_col]);
                    const uint8_t q_weight = unpack_awq_int4(packed_weight, logical_slot, pack_mask);
                    const float weight_val =
                        (static_cast<float>(q_weight) - static_cast<float>(q_zero)) * scale;
                    sum += a_val * weight_val;
                }
            }

            out_ptr[i * k + j] = static_cast<T>(sum);
        }
    }
}

}  // namespace

void QuantizedMatmul::eval_cpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const {
    auto &scales = inputs[0];
    auto &zeros = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    check_contiguous(scales, "scales");
    check_contiguous(zeros, "zeros");
    check_contiguous(a, "a");
    check_contiguous(b, "b");

    if (a.scalar_type() == torch::kFloat16) {
        quantized_matmul_cpu_impl<c10::Half>(scales, zeros, a, b, out, group_size_, bits_);
    } else if (a.scalar_type() == torch::kFloat32) {
        quantized_matmul_cpu_impl<float>(scales, zeros, a, b, out, group_size_, bits_);
    } else if (a.scalar_type() == torch::kBFloat16) {
        quantized_matmul_cpu_impl<c10::BFloat16>(scales, zeros, a, b, out, group_size_, bits_);
    } else {
        throw std::runtime_error("quantized_matmul: unsupported dtype for CPU implementation");
    }
}

void QuantizedMatmul::eval_gpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const {
    auto &scales = inputs[0];
    auto &zeros = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    check_contiguous(scales, "scales");
    check_contiguous(zeros, "zeros");
    check_contiguous(a, "a");
    check_contiguous(b, "b");

#ifdef _CUDA_
    quantized_matmul_cuda(scales, zeros, a, b, out, group_size_, bits_);
#else
    (void)scales;
    (void)zeros;
    (void)a;
    (void)b;
    (void)out;
    throw std::runtime_error("QuantizedMatmul has no CUDA implementation.");
#endif
}

void QuantizedMatmul::print(std::ostream &os) const {
    os << name() << "(group_size=" << group_size_ << ", bits=" << bits_ << ")";
}

std::pair<std::vector<torch::Tensor>, std::vector<int64_t>> QuantizedMatmul::vmap(
    const std::vector<torch::Tensor> &inputs, const std::vector<int64_t> &axes) const {
    (void)inputs;
    (void)axes;

    throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
}

bool QuantizedMatmul::is_equivalent(const QuantizedMatmul &other) const {
    return group_size_ == other.group_size_ && bits_ == other.bits_;
}

}  // namespace tiny_llm_ext_torch_ref
