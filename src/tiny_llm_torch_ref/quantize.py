from typing import Any

import torch
from extensions_torch_ref import tiny_llm_ext_torch_ref


AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


class QuantizedWeights:
    def __init__(
        self,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        bits: int,
        weight: torch.Tensor,
    ):
        """
        Shapes for Qwen2 AWQ int4 weights:
        - `scales`: [in_features / group_size, out_features]
        - `zeros`: [in_features / group_size, out_features / 8]
        - `weight`: [in_features, out_features / 8]
        """
        self.scales = scales
        self.zeros = zeros
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_torch_layer(torch_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=torch_layer.scales,
            zeros=torch_layer.qzeros,
            group_size=torch_layer.group_size,
            bits=torch_layer.w_bit,
            weight=torch_layer.qweight,
        )


def quantized_matmul(
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    """
    Shapes:
    - `scales`: [N / group_size, K]
    - `zeros`: [N / group_size, K / 8]
    - `a`: [..., N]
    - `b`: [N, K / 8] packed int4 weight tensor
    - returns: [..., K] if `transpose_b=True`
    """
    return tiny_llm_ext_torch_ref.quantized_matmul(
        scales.contiguous(),
        zeros.contiguous(),
        group_size,
        bits,
        a.contiguous(),
        b.contiguous(),
        transpose_b,
    )


def quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shapes:
    - `x`: [..., in_features]
    - `w.weight`: [in_features, out_features / 8]
    - `bias`: [out_features]
    - returns: [..., out_features]
    """
    if bias is None:
        return quantized_matmul(
            w.scales, w.zeros, w.group_size, w.bits, x, w.weight, True
        )
    else:
        return (
            quantized_matmul(w.scales, w.zeros, w.group_size, w.bits, x, w.weight, True)
            + bias
        )


def dequantize_linear(torch_layer: Any) -> torch.Tensor:
    """
    Input:
    - `torch_layer`: an AWQ quantized linear-like layer
      with `qweight`, `qzeros`, `scales`, `group_size`, `w_bit`

    Returns:
    - dequantized weight tensor with shape [out_features, in_features]
    """
    if not all(
        hasattr(torch_layer, attr)
        for attr in ("qweight", "qzeros", "scales", "group_size", "w_bit")
    ):
        raise RuntimeError(
            "dequantize_linear expects an AWQ layer with "
            "qweight/qzeros/scales/group_size/w_bit"
        )

    if int(torch_layer.w_bit) != 4:
        raise RuntimeError("dequantize_linear currently only supports int4 AWQ")

    qweight = torch_layer.qweight.detach().to(dtype=torch.int32)
    qzeros = torch_layer.qzeros.detach().to(dtype=torch.int32)
    scales = torch_layer.scales.detach().to(dtype=torch.float32)
    group_size = int(torch_layer.group_size)

    in_features = qweight.size(0)
    packed_out = qweight.size(1)
    out_features = packed_out * 8
    num_groups = in_features // group_size

    if in_features % group_size != 0:
        raise RuntimeError("dequantize_linear expects in_features divisible by group_size")
    if qzeros.shape != (num_groups, packed_out):
        raise RuntimeError("dequantize_linear: unexpected qzeros shape")
    if scales.shape != (num_groups, out_features):
        raise RuntimeError("dequantize_linear: unexpected scales shape")

    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    slot_order = torch.tensor(AWQ_REVERSE_ORDER, device=qweight.device, dtype=torch.long)

    unpacked_weight = torch.bitwise_right_shift(qweight[..., None], shifts).bitwise_and_(0xF)
    unpacked_zero = torch.bitwise_right_shift(qzeros[..., None], shifts).bitwise_and_(0xF)

    logical_weight = unpacked_weight.index_select(-1, slot_order).reshape(in_features, out_features)
    logical_zero = unpacked_zero.index_select(-1, slot_order).reshape(num_groups, out_features)

    repeated_zero = logical_zero.repeat_interleave(group_size, dim=0)
    repeated_scale = scales.repeat_interleave(group_size, dim=0)
    weight = (logical_weight.to(torch.float32) - repeated_zero.to(torch.float32)) * repeated_scale

    return weight.transpose(0, 1).contiguous().to(dtype=torch.float16)
