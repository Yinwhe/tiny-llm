#pragma once

#include <torch/extension.h>

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
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
);

///////////////////////////////////////////////////////////////////////////////
// Backends
///////////////////////////////////////////////////////////////////////////////

void axpby_cpu(const torch::Tensor &x,
               const torch::Tensor &y,
               torch::Tensor &out,
               const float alpha,
               const float beta);

#ifdef _CUDA_
void axpby_cuda(const torch::Tensor &x,
                const torch::Tensor &y,
                torch::Tensor &out,
                const float alpha,
                const float beta);
#endif

}  // namespace tiny_llm_ext
