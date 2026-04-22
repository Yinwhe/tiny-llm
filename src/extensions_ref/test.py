from tiny_llm_ext_ref import quantized_matmul
import mlx.core as mx
import numpy as np

precision = np.float16
input = mx.array(np.random.randn(3, 64).astype(precision))
weight = mx.array(np.random.randn(5, 64).astype(precision))
w_q, scales, biases = mx.quantize(weight)
user_out = quantized_matmul(
    scales=scales,
    biases=biases,
    group_size=64,
    bits=4,
    a=input,
    b=w_q,
    transpose_b=True,
)
print(user_out)
