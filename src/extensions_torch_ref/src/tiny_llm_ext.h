#pragma once

#include <torch/extension.h>

#include <iosfwd>
#include <utility>
#include <vector>

namespace tiny_llm_ext_torch_ref {

void load_library(const char *device, const char *path);

torch::Tensor quantized_matmul(const torch::Tensor &scales,  // Per-group scaling factors
                               const torch::Tensor &zeros,   // Per-group zero points / packed offsets
                               const int group_size,         // Group size
                               const int bits,               // Number of bits
                               const torch::Tensor &a,       // Input activation tensor
                               const torch::Tensor &b,       // Packed quantized weight tensor
                               const bool transpose_b        // Whether to transpose b
);

#ifdef _CUDA_
void quantized_matmul_cuda(const torch::Tensor &scales,
                           const torch::Tensor &zeros,
                           const torch::Tensor &a,
                           const torch::Tensor &b,
                           torch::Tensor &out,
                           const int group_size,
                           const int bits);
#endif

class QuantizedMatmul {
public:
    explicit QuantizedMatmul(const int group_size, const int bits) : group_size_(group_size), bits_(bits) {};

    void eval_cpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const;
    void eval_gpu(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor> &outputs) const;

    std::pair<std::vector<torch::Tensor>, std::vector<int64_t>> vmap(const std::vector<torch::Tensor> &inputs,
                                                                     const std::vector<int64_t> &axes) const;

    void print(std::ostream &os) const;

    const char *name() const { return "QuantizedMatmul"; }

    bool is_equivalent(const QuantizedMatmul &other) const;

private:
    int group_size_;
    int bits_;
};

}  // namespace tiny_llm_ext_torch_ref
