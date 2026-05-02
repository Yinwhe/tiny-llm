# Week 2 Day 2-3: Quantized Matmul

In this chapter, we will implement the quantized matrix multiplication. Quantization compresses model weights from 16-bit floating point to 4-bit integers, which is critical for efficient LLM serving on devices with limited memory bandwidth.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [Quantized Matmul on CPU (Video)](https://www.youtube.com/watch?v=es6s6T1bTtI)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Why Quantization?

As we learned in the KV Cache chapter, the decode phase of LLM inference is **memory-bandwidth bound**. Let's revisit the arithmetic intensity calculation for the Qwen2-0.5B model:

```plain
Per-token computation in decode phase:
- Input: 1 token × 896 dimensions = 896 float16 values = 1.792 KB
- MLP weights: 896 × 4864 × 3 matrices × 2 bytes = ~25 MB per layer
- Attention weights: 896 × 896 × 4 matrices × 2 bytes = ~6 MB per layer
- Total weights per layer: ~31 MB
- Total for 24 layers: ~750 MB

FLOPs (2 per multiply-accumulate):
- MLP per layer: 2 × 3 × 896 × 4864 ≈ 26M
- Attention per layer: 2 × 4 × 896 × 896 ≈ 6.4M
- 24 layers: ~780 million per token

Memory access: ~750 MB
Arithmetic intensity: 780M FLOPs / 750 MB ≈ 1.0 FLOPs/Byte
```

Using the same back-of-the-envelope example with a 400 GB/s, 10 TFLOPS device:

```plain
Memory-bound throughput: 400 GB/s × 1.0 FLOPs/Byte = 400 GFLOPS
Compute-bound throughput: 10 TFLOPS

We're using only ~4% of available compute!
```

### The Solution: Quantization

By compressing weights from 16 bits (float16/bfloat16) to 4 bits (int4), we:

- **Reduce memory bandwidth by 4×**: 750 MB → ~190 MB per token
- **Improve arithmetic intensity by 4×**: 1.0 → ~4.0 FLOPs/Byte
- **Increase throughput by ~4×**: 400 GFLOPS → ~1.6 TFLOPS

The tradeoff is minimal accuracy loss with proper quantization techniques.

### Group-wise Quantization

Instead of quantizing all weights uniformly, we divide them into **groups** and quantize each group independently. This preserves more information about the weight distribution.

For a weight matrix $W$ of shape $(N, K)$, we divide the input-feature dimension into groups of size $G$ (typically 64 or 128):

```plain
Original weight matrix W: N × K (float16/bfloat16)

Group size G = 64
Number of groups along the input dimension = N / G

For each group of consecutive values:
  1. Find min and max values
  2. Compute a scale and zero point to map the group into the int4 range [0, 15]
  3. Quantize each value using the group's scale and zero point
```

### Affine Quantization

We use **affine (asymmetric) quantization** which maps a floating-point range to the full integer range.

$$
\text{quantized} = \text{round}\left(\frac{\text{value}}{\text{scale}}\right) + \text{zero}
$$

$$
\text{dequantized} = (\text{quantized} - \text{zero}) \times \text{scale}
$$

The original MLX version described the equivalent `scale + bias` form. In the AWQ checkpoints used by our Torch version, the extension directly consumes per-group `scale` and packed `zero` values instead.

For 4-bit quantization, the quantized values are in the range $[0, 15]$.

**Example:**

```plain
Group values: [-0.5, -0.3, 0.1, 0.4, 0.8]

Choose scale ≈ 0.0867
Choose zero = 6

Quantization:
  -0.5 → round(-0.5 / 0.0867) + 6 ≈ 0
  -0.3 → round(-0.3 / 0.0867) + 6 ≈ 3
   0.1 → round( 0.1 / 0.0867) + 6 ≈ 7
   0.4 → round( 0.4 / 0.0867) + 6 ≈ 11
   0.8 → round( 0.8 / 0.0867) + 6 ≈ 15

Dequantization:
  value ≈ (quantized - 6) * 0.0867
```

### Storage Format

For efficient storage and computation, quantized weights are packed:

```plain
Original: N × K float16 (2 bytes each) = 2NK bytes
Quantized: N × K int4 (0.5 bytes each) = 0.5NK bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Logical weight matrix shape: N × K
Packed weight shape: N × (K / 8) int32
Scales shape: (N / G) × K float16/bfloat16
Zeros shape: (N / G) × (K / 8) int32
```

Example packing for 8 consecutive 4-bit values `[a, b, c, d, e, f, g, h]`:

```plain
uint32_value = (h << 28) | (g << 24) | (f << 20) | (e << 16) |
               (d << 12) | (c << 8)  | (b << 4)  | a

Unpacking:
  a = (uint32_value >> 0)  & 0xF
  b = (uint32_value >> 4)  & 0xF
  c = (uint32_value >> 8)  & 0xF
  ...
  h = (uint32_value >> 28) & 0xF
```

For the AWQ checkpoints used in this repo, the 8 logical int4 values are stored in a fixed permuted order inside each packed word, so your implementation will need a small slot remapping when unpacking.

## Quantized Matrix Multiplication

### Mathematical Formulation

For standard matrix multiplication $C = AB$ where:

- $A$: shape $(M, N)$, float16/bfloat16 (activations)
- $B$: shape $(N, K)$, **quantized** to int4 (weights)
- $C$: shape $(M, K)$, float16/bfloat16 (output)

Each element $C[i, k]$ is computed as:

$$
C[i, k] = \sum_{j=0}^{N-1} A[i, j] \times B[j, k]
$$

With quantization, $B[j, k]$ is represented as:

$$
B[j, k] = (B_{\text{quantized}}[j, k] - \text{zero}[g, k]) \times \text{scale}[g, k]
$$

where $g = \lfloor j / G \rfloor$ is the group index.

Substituting:

$$
C[i, k] = \sum_{g=0}^{N/G-1} \sum_{j'=0}^{G-1} A[i, g \times G + j'] \times ((B_{\text{quantized}}[g \times G + j', k] - \text{zero}[g, k]) \times \text{scale}[g, k])
$$

### Computation Flow

```plain
Input:
  A: M × N (float16, activations)
  B_quantized: N × (K/8) (int32, packed weights)
  scales: (N/G) × K (float16/bfloat16)
  zeros: (N/G) × (K/8) (int32, packed zero points)

Output:
  C: M × K (float16)

For each output element C[i, k]:
  sum = 0
  packed_col = k // 8
  slot = k % 8
  for each group g in 0..(N/G - 1):
    scale = scales[g, k]
    q_zero = unpack(zeros[g, packed_col], slot)

    for each offset t in 0..(G - 1):
      in_col = g*G + t
      q_weight = unpack(B_quantized[in_col, packed_col], slot)
      b_value = (q_weight - q_zero) * scale
      a_value = A[i, in_col]
      sum += a_value * b_value
  
  C[i, k] = sum
```

## Task 1: Implement QuantizedWeights

```
src/tiny_llm/quantize.py
```

First, familiarize yourself with the `QuantizedWeights` class, which stores quantized weight information:

| Field | Shape | Description |
|-------|-------|-------------|
| `weight` | $(N, K/8)$ int32 | Packed quantized weights. Each 32-bit word stores 8 logical int4 values for one output-column pack. |
| `scales` | $(N/G, K)$ float16/bfloat16 | Per-group scale factors for dequantization. |
| `zeros` | $(N/G, K/8)$ int32 | Packed per-group zero points. |
| `group_size` | int | Number of consecutive input features that share the same scale/zero (typically 64 or 128) |
| `bits` | int | Quantization bit width (typically 4, meaning values are in range $[0, 15]$) |

The `from_torch_layer` static method extracts these fields from the AWQ quantized linear layers when loading the model.

Next, implement the `quantized_linear` function, which is a wrapper around `quantized_matmul` that mimics the standard `linear` function interface. And we'll implement `quantized_matmul` in the next task.

## Task 2: Implement `quantized_matmul` (CPU version)

In this task, we will implement the quantized matmul as a Torch C++ extension. The pattern is still close to the existing `axpby` example in the codebase — read through `axpby.h`, `axpby.cpp`, and the corresponding binding in `bindings.cpp` first as your reference.

```
src/extensions_torch/src/tiny_llm_ext.h
src/extensions_torch/bindings.cpp
src/extensions_torch/src/quantized_matmul.cpp
src/extensions_torch/CMakeLists.txt
```

You need to touch four files, all within the `tiny_llm_ext` namespace:

- **`tiny_llm_ext.h`** — Declare the `quantized_matmul(...)` function signature and define a small `QuantizedMatmul` helper class to hold `group_size` and `bits`, and to organize the CPU / GPU entry points.
- **`bindings.cpp`** — Add an `m.def(...)` call to expose the function to Python.
- **`quantized_matmul.cpp`** — Implement the `quantized_matmul(...)` function (validate inputs, compute output shape, flatten `a` to 2D if needed, allocate the output tensor) and the `eval_cpu` method that runs the nested loop from the Computation Flow section above.

Inside the CPU loop, iterate over each output element `(i, k)`, accumulate in `float` (fp32) to avoid precision loss, and cast the result back to the output dtype when writing to the output tensor. Since the AWQ checkpoint packs both weights and zeros, you will need to unpack both before dequantizing each group.

Don't forget to add `src/quantized_matmul.cpp` to `target_sources` in `CMakeLists.txt`.

You can test your implementation by running:

```bash
pdm run build-ext-torch
pdm run test --week 2 --day 2 -- -k task_2
```

## Task 3: Implement `quantized_matmul` (GPU version)

```
src/extensions_torch/src/quantized_matmul.cu
src/extensions_torch/src/quantized_matmul.cpp
```

In this task, you will write the CUDA kernel for quantized matmul **and** wire up the `eval_gpu` method to dispatch it. Keep the math exactly the same as Task 2 (CPU); only the execution model changes.

### CUDA Kernel

You need to implement one kernel entry in `quantized_matmul.cu`:

- Use a **one-thread-per-output-element** mapping: each thread computes `out[i, k]`.
- The kernel should be templated on the data type (to support both `float16` and `bfloat16` activations).
- Apply the same group-wise dequantization loop as the CPU version:
  - Iterate over groups (`group_size` will typically be 64 or 128)
  - Unpack int4 values from packed `uint32`
  - Dequantize with `(q - zero) * scale`
  - Accumulate in `float` and cast to the output dtype at the end
- Add boundary checks (`i < M`, `k < K`) before writing output.

### GPU Dispatch

Complete the `eval_gpu` method in `quantized_matmul.cpp` to dispatch your CUDA kernel. Follow the same pattern as `axpby`'s GPU dispatch:

1. Get the current CUDA stream.
2. Select the correct kernel instantiation based on the activation dtype (`float16` or `bfloat16`).
3. Pass input/output pointers and dimension constants (`M`, `N`, `K`, `group_size`) in the same order as the kernel signature.
4. Launch a 2D grid so that each thread computes one output element.
5. Check for CUDA launch errors after dispatch.

You can test your implementation by running:

```bash
pdm run build-ext-torch
pdm run test --week 2 --day 2 -- -k task_3
```

## Task 4: Model Integration

```
src/tiny_llm/qwen2_week2.py
```

Integrate your quantized matmul into the Week 2 Qwen2 model so that inference runs on quantized AWQ weights end-to-end.

Change the weight type from `torch.Tensor` to `QuantizedWeights` for all linear layers in attention (`wq/wk/wv/wo`) and MLP (`w_gate/w_up/w_down`). Replace every `linear(x, w)` call with `quantized_linear(x, w)`. In the model loading code, use `QuantizedWeights.from_torch_layer(...)` to extract quantized weight information from each AWQ linear layer, instead of dequantizing to a full dense matrix first. Make sure the Week 1 loader still dequantizes (since Week 1 layers expect plain dense weights), while the Week 2 loader does **not** dequantize.

Note that the AWQ checkpoints used here store packed `qweight` / `qzeros`, and real models may use `group_size = 128` rather than `64`. If you see shape errors or garbage output, the most likely cause is a mismatch between the packed layout expected by the extension and the tensors extracted from the model.

You can test your implementation by running:

```bash
pdm run main --solution tiny_llm --loader week2 --model qwen2-0.5b
```

You can also benchmark throughput and compare your implementation with the reference solution:

```bash
pdm bench --solution tiny_llm --loader week2 --model qwen2-0.5b
pdm bench --solution tiny_llm_torch_ref --loader week2 --model qwen2-0.5b
```

{{#include copyright.md}}
