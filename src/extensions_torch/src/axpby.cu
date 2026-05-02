#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

namespace tiny_llm_ext {

namespace {

__device__ inline int64_t elem_to_loc(int64_t elem, const int64_t *shape, const int64_t *strides, int ndim) {
    int64_t offset = 0;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        const int64_t index = elem % shape[dim];
        elem /= shape[dim];
        offset += index * strides[dim];
    }
    return offset;
}

template <typename T>
__global__ void axpby_general(const T *x,
                              const T *y,
                              T *out,
                              const float alpha_,
                              const float beta_,
                              const int64_t *shape,
                              const int64_t *x_strides,
                              const int64_t *y_strides,
                              const int ndim,
                              const int64_t size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    T alpha = static_cast<T>(alpha_);
    T beta = static_cast<T>(beta_);
    auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
    auto y_offset = elem_to_loc(index, shape, y_strides, ndim);
    out[index] = alpha * x[x_offset] + beta * y[y_offset];
}

template <typename T>
__global__ void axpby_contiguous(const T *x,
                                 const T *y,
                                 T *out,
                                 const float alpha_,
                                 const float beta_,
                                 const int64_t size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    T alpha = static_cast<T>(alpha_);
    T beta = static_cast<T>(beta_);
    out[index] = alpha * x[index] + beta * y[index];
}

template <typename T>
void launch_axpby_cuda(const torch::Tensor &x,
                       const torch::Tensor &y,
                       torch::Tensor &out,
                       const float alpha,
                       const float beta,
                       const bool contiguous_kernel) {
    const at::cuda::CUDAGuard device_guard(out.device());
    const auto stream = at::cuda::getCurrentCUDAStream(out.device().index());
    constexpr int threads = 256;
    const auto size = out.numel();
    const int blocks = static_cast<int>((size + threads - 1) / threads);

    if (contiguous_kernel) {
        axpby_contiguous<T><<<blocks, threads, 0, stream.stream()>>>(
            x.const_data_ptr<T>(),
            y.const_data_ptr<T>(),
            out.data_ptr<T>(),
            alpha,
            beta,
            size);
    } else {
        auto shape = torch::tensor(
            out.sizes().vec(),
            torch::TensorOptions().device(out.device()).dtype(torch::kInt64));
        auto x_strides = torch::tensor(
            x.strides().vec(),
            torch::TensorOptions().device(out.device()).dtype(torch::kInt64));
        auto y_strides = torch::tensor(
            y.strides().vec(),
            torch::TensorOptions().device(out.device()).dtype(torch::kInt64));
        axpby_general<T><<<blocks, threads, 0, stream.stream()>>>(
            x.const_data_ptr<T>(),
            y.const_data_ptr<T>(),
            out.data_ptr<T>(),
            alpha,
            beta,
            shape.const_data_ptr<int64_t>(),
            x_strides.const_data_ptr<int64_t>(),
            y_strides.const_data_ptr<int64_t>(),
            out.dim(),
            size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void axpby_cuda(const torch::Tensor &x,
                const torch::Tensor &y,
                torch::Tensor &out,
                const float alpha,
                const float beta) {
    const bool contiguous_kernel = x.is_contiguous() && y.is_contiguous() && out.is_contiguous();

    if (out.scalar_type() == torch::kFloat32) {
        return launch_axpby_cuda<float>(x, y, out, alpha, beta, contiguous_kernel);
    } else if (out.scalar_type() == torch::kFloat16) {
        return launch_axpby_cuda<c10::Half>(x, y, out, alpha, beta, contiguous_kernel);
    } else if (out.scalar_type() == torch::kBFloat16) {
        return launch_axpby_cuda<c10::BFloat16>(x, y, out, alpha, beta, contiguous_kernel);
    } else if (out.scalar_type() == torch::kComplexFloat) {
        return launch_axpby_cuda<c10::complex<float>>(x, y, out, alpha, beta, contiguous_kernel);
    } else {
        throw std::runtime_error("Axpby is only supported for floating point types.");
    }
}

}  // namespace tiny_llm_ext
