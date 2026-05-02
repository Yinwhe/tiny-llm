#include "axpby.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
torch::Tensor axpby(const torch::Tensor &x,  // Input tensor x
                    const torch::Tensor &y,  // Input tensor y
                    const float alpha,       // Scaling factor for x
                    const float beta         // Scaling factor for y
) {
    if (x.device() != y.device()) {
        std::ostringstream message;
        message << "x and y must be on the same device, got " << x.device() << " and " << y.device();
        throw std::runtime_error(message.str());
    }

    // Promote dtypes between x and y as needed
    auto promoted_dtype = c10::promoteTypes(x.scalar_type(), y.scalar_type());

    // Upcast to float32 for non-floating point inputs x and y
    auto out_dtype =
        (c10::isFloatingType(promoted_dtype) || c10::isComplexType(promoted_dtype)) ? promoted_dtype : torch::kFloat32;

    // Cast x and y up to the determined dtype
    auto x_casted = x.to(out_dtype);
    auto y_casted = y.to(out_dtype);

    // Broadcast the shapes of x and y
    auto broadcasted_inputs = torch::broadcast_tensors({x_casted, y_casted});
    auto out_shape = broadcasted_inputs[0].sizes();

    // This is the Torch/CUDA-native counterpart to the MLX teaching example:
    // same operation entry, but eager execution into a real tensor.
    auto out = torch::empty(
        out_shape,
        torch::TensorOptions()
            .device(broadcasted_inputs[0].device())
            .dtype(out_dtype));

    if (out.is_cuda()) {
#ifdef _CUDA_
        axpby_cuda(broadcasted_inputs[0], broadcasted_inputs[1], out, alpha, beta);
#else
        throw std::runtime_error("Axpby has no CUDA implementation.");
#endif
    } else {
        axpby_cpu(broadcasted_inputs[0], broadcasted_inputs[1], out, alpha, beta);
    }
    return out;
}

///////////////////////////////////////////////////////////////////////////////
// CPU Backend Implementation
///////////////////////////////////////////////////////////////////////////////

namespace {

inline int64_t elem_to_loc(int64_t elem, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides) {
    int64_t offset = 0;
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
        const int64_t index = elem % shape[dim];
        elem /= shape[dim];
        offset += index * strides[dim];
    }
    return offset;
}

template <typename T>
void axpby_impl(const torch::Tensor &x, const torch::Tensor &y, torch::Tensor &out, float alpha_, float beta_) {
    auto *out_ptr = out.data_ptr<T>();
    const auto *x_ptr = x.const_data_ptr<T>();
    const auto *y_ptr = y.const_data_ptr<T>();
    const auto size = out.numel();
    const auto shape = out.sizes().vec();
    const auto x_strides = x.strides().vec();
    const auto y_strides = y.strides().vec();

    T alpha = static_cast<T>(alpha_);
    T beta = static_cast<T>(beta_);

    for (int64_t out_idx = 0; out_idx < size; out_idx++) {
        // Map linear indices to offsets in x and y
        auto x_offset = elem_to_loc(out_idx, shape, x_strides);
        auto y_offset = elem_to_loc(out_idx, shape, y_strides);

        // We allocate the output to be contiguous and regularly strided
        // (defaults to row major) and hence it doesn't need additional mapping
        out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
    }
}

}  // namespace

void axpby_cpu(const torch::Tensor &x,
               const torch::Tensor &y,
               torch::Tensor &out,
               const float alpha,
               const float beta) {
    // Dispatch to the correct dtype
    if (out.scalar_type() == torch::kFloat32) {
        return axpby_impl<float>(x, y, out, alpha, beta);
    } else if (out.scalar_type() == torch::kFloat16) {
        return axpby_impl<c10::Half>(x, y, out, alpha, beta);
    } else if (out.scalar_type() == torch::kBFloat16) {
        return axpby_impl<c10::BFloat16>(x, y, out, alpha, beta);
    } else if (out.scalar_type() == torch::kComplexFloat) {
        return axpby_impl<c10::complex<float>>(x, y, out, alpha, beta);
    } else {
        throw std::runtime_error("Axpby is only supported for floating point types.");
    }
}

}  // namespace tiny_llm_ext
