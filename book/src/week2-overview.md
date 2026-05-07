# Week 2: Tiny vLLM

In Week 2 of the course, we will focus on building serving infrastructure for the Qwen2 model. Essentially, this means creating a minimal version of the vLLM project from scratch. By the end of the week, you’ll be able to serve the Qwen2 model efficiently on a CUDA GPU using the infrastructure we’ve built together.

## What We’ll Cover

* Key-value cache implementation
* C++/CUDA kernels
    * Implementing a quantized matmul kernel
    * Implementing a flash attention kernel
    * Note: This week, we won’t focus on performance optimization. The kernels you build will likely be much slower than highly optimized production implementations. Optimizing them will be left as an exercise.
* Model serving infrastructure
    * Implementing chunked prefill
    * Implementing continuous batching

Additionally, the repo includes skeleton code for the Qwen3 model. If your GPU supports the bfloat16 data type, you’re encouraged to try implementing it and experiment with the Qwen3-series models as well.

{{#include copyright.md}}
