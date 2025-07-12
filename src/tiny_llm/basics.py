import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Linear layer.

    Args:
        x: Input tensor of shape (..., in_features)
        w: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features)

    Returns:
        Output tensor of shape (..., out_features)
    """
    if bias is None:
        return mx.matmul(x, w.swapaxes(-1, -2))
    else:
        return mx.matmul(x, w.swapaxes(-1, -2)) + bias


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
