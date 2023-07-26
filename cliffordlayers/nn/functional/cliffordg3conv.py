import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


def get_clifford_g3_kernel(weights):
    assert len(weights) == 6

    scale = weights[4]

    square_1 = weights[0] * weights[0]
    square_2 = weights[1] * weights[1]
    square_3 = weights[2] * weights[2]
    square_4 = weights[3] * weights[3]

    norm = torch.sqrt(square_1 + square_2 + square_3 + square_4 + 0.0001)

    weight_1_n = weights[0] / norm
    weight_2_n = weights[1] / norm
    weight_3_n = weights[2] / norm
    weight_4_n = weights[3] / norm

    norm_factor = 2.0
    square_2 = norm_factor * (weight_2_n * weight_2_n)
    square_3 = norm_factor * (weight_3_n * weight_3_n)
    square_4 = norm_factor * (weight_4_n * weight_4_n)

    rot12 = norm_factor * weight_1_n * weight_2_n
    rot13 = norm_factor * weight_1_n * weight_3_n
    rot14 = norm_factor * weight_1_n * weight_4_n
    rot23 = norm_factor * weight_2_n * weight_3_n
    rot24 = norm_factor * weight_2_n * weight_4_n
    rot34 = norm_factor * weight_3_n * weight_4_n

    rot_kernel2 = torch.cat(
        [
            scale * (1.0 - (square_3 + square_4)),
            scale * (rot23 - rot14),
            scale * (rot24 + rot13),
        ],
        dim=1,
    )
    rot_kernel3 = torch.cat(
        [
            scale * (rot23 + rot14),
            scale * (1.0 - (square_2 + square_4)),
            scale * (rot34 - rot12),
        ],
        dim=1,
    )
    rot_kernel4 = torch.cat(
        [
            scale * (rot24 - rot13),
            scale * (rot34 + rot12),
            scale * (1.0 - (square_2 + square_3)),
        ],
        dim=1,
    )

    kernel = torch.cat([rot_kernel2, rot_kernel3, rot_kernel4], dim=0)
    return kernel


def clifford_g3convnd(
    input,
    weights,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    transposed=False,
):
    kernel = get_clifford_g3_kernel(weights)
    if bias is not None:
        bias = torch.cat([bias[0], bias[1], bias[2]], dim=0)

    if input.dim() == 3:
        convfunc = F.conv_transpose1d if transposed else F.conv1d
        padding = _single(padding)
        dilation = _single(dilation)
        stride = _single(stride)
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d if transposed else F.conv2d
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        stride = _pair(stride)
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d if transposed else F.conv3d
        padding = _triple(padding)
        dilation = _triple(dilation)
        stride = _triple(stride)
    else:
        raise NotImplementedError("input must be 3D, 4D or 5D")

    return convfunc(input, kernel, bias, stride, padding=padding, dilation=dilation, groups=groups)
