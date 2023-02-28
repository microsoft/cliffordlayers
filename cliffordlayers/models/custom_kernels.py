# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# CUSTOMIZED ENCODING/DECODING KERNELS AS USED In THE PAPER:                               #
# Clifford Neural Layers for PDE Modeling                                                 #
###########################################################################################
from typing import Tuple, Union

import torch
import torch.nn as nn

from cliffordlayers.nn.functional.utils import _w_assert


def get_2d_clifford_encoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebra encoding layers.
    This specific 2d embedding layer assumes scalar and vector input fields.
    I.e. each part of the input multivector field is present except the bivector part.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of dimension `(4, d~input, d~output, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        Tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d~output * 4, d~input * 3, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k


def get_2d_clifford_decoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebra decoding layers.
    This specific 2d decoding layer assumes scalar and vector output fields.
    I.e. each part of the output multivector field is present except the bivector part.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of dimension `(4, d~input, d~output, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        Tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d~output * 3, d~input * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3], g[1] * w[2]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0], -g[0] * w[1]], dim=1)
    k = torch.cat([k0, k1, k2], dim=0)
    return 3, k


def get_2d_clifford_rotation_encoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Rotational Clifford kernel for 2d Clifford algebra encoding layers.
    This specific 2d embedding layer assumes scalar and vector input fields.
    I.e. each part of the input multivector field is present except the bivector part.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of dimension `(6, d~input, d~output, ...)`.
        w[0], w[1], w[2], w[3] are the weight 2d Clifford weight tensors;
        w[4] is the scaling tensor;
        w[5] is the zero kernel tensor.

        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        Tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d_output * 4, d~input * 3, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    assert g[0] == -1 and g[1] == -1, "Wrong signature of Clifford algebra. Signature not suitable for rotation kernel."
    w = _w_assert(w)
    assert len(w) == 6

    # Adding scalar output kernel.
    k0 = torch.cat([w[0], -w[1], -w[2]], dim=1)

    # Rotational kernel from here onwards.
    s0 = w[0] * w[0]
    s1 = w[1] * w[1]
    s2 = w[2] * w[2]
    s3 = w[3] * w[3]
    norm = torch.sqrt(s0 + s1 + s2 + s3 + 0.0001)
    w0_n = w[0] / norm
    w1_n = w[1] / norm
    w2_n = w[2] / norm
    w3_n = w[3] / norm

    norm_factor = 2.0
    s1 = norm_factor * (w1_n * w1_n)
    s2 = norm_factor * (w2_n * w2_n)
    s3 = norm_factor * (w3_n * w3_n)
    rot01 = norm_factor * w0_n * w1_n
    rot02 = norm_factor * w0_n * w2_n
    rot03 = norm_factor * w0_n * w3_n
    rot12 = norm_factor * w1_n * w2_n
    rot13 = norm_factor * w1_n * w3_n
    rot23 = norm_factor * w2_n * w3_n

    scale = w[4]
    zero_kernel = w[5]

    k1 = torch.cat(
        [
            zero_kernel,
            scale * (1.0 - (s2 + s3)),
            scale * (rot12 - rot03),
        ],
        dim=1,
    )
    k2 = torch.cat(
        [
            zero_kernel,
            scale * (rot12 + rot03),
            scale * (1.0 - (s1 + s3)),
        ],
        dim=1,
    )
    k3 = torch.cat(
        [
            zero_kernel,
            scale * (rot13 - rot02),
            scale * (rot23 + rot01),
        ],
        dim=1,
    )
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k


def get_2d_clifford_rotation_decoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Rotational Clifford kernel for 2d Clifford algebra decoding layers.
    This specific 2d decoding layer assumes scalar and vector output fields.
    I.e. each part of the output multivector field is present except the bivector part.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of dimension `(6, d~input, d~output, ...)`.
        w[0], w[1], w[2], w[3] are the weight 2d Clifford weight tensors;
        w[4] is the scaling tensor;
        w[5] is the zero kernel tensor.

        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        Tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d~output * 3, d~input * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    assert g[0] == -1 and g[1] == -1, "Wrong signature of Clifford algebra. Signature not suitable for rotation kernel."
    w = _w_assert(w)
    assert len(w) == 6

    # Adding scalar output kernel.
    k0 = torch.cat([w[0], -w[1], -w[2], -w[3]], dim=1)

    # Rotational kernel from here onwards.
    s0 = w[0] * w[0]
    s1 = w[1] * w[1]
    s2 = w[2] * w[2]
    s3 = w[3] * w[3]
    norm = torch.sqrt(s0 + s1 + s2 + s3 + 0.0001)
    w0_n = w[0] / norm
    w1_n = w[1] / norm
    w2_n = w[2] / norm
    w3_n = w[3] / norm

    norm_factor = 2.0
    s1 = norm_factor * (w1_n * w1_n)
    s2 = norm_factor * (w2_n * w2_n)
    s3 = norm_factor * (w3_n * w3_n)
    rot01 = norm_factor * w0_n * w1_n
    rot02 = norm_factor * w0_n * w2_n
    rot03 = norm_factor * w0_n * w3_n
    rot12 = norm_factor * w1_n * w2_n
    rot13 = norm_factor * w1_n * w3_n
    rot23 = norm_factor * w2_n * w3_n

    scale = w[4]
    zero_kernel = w[5]

    k1 = torch.cat(
        [
            zero_kernel,
            scale * (1.0 - (s2 + s3)),
            scale * (rot12 - rot03),
            scale * (rot13 + rot02),
        ],
        dim=1,
    )
    k2 = torch.cat(
        [
            zero_kernel,
            scale * (rot12 + rot03),
            scale * (1.0 - (s1 + s3)),
            scale * (rot23 - rot01),
        ],
        dim=1,
    )
    k = torch.cat([k0, k1, k2], dim=0)
    return 3, k


def get_3d_clifford_encoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebra encoding layers.
    This specific 3d embedding layer assumes vector and bivector input fields.
    I.e. each part of the input multivector field is present except the scalar and the bivector parts.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor): Weight input of dimension `(8, d~input, d~output, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        Tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d~output * 8, d~input * 6, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)

    k0 = torch.cat(
        [
            w[1] * g[0],
            w[2] * g[1],
            w[3] * g[2],
            -w[4] * g[0] * g[1],
            -w[5] * g[0] * g[2],
            -w[6] * g[1] * g[2],
        ],
        dim=1,
    )
    k1 = torch.cat(
        [w[0], -w[4] * g[1], -w[5] * g[2], w[2] * g[1], w[3] * g[2], -w[7] * g[1] * g[2]],
        dim=1,
    )
    k2 = torch.cat(
        [w[4] * g[0], w[0], -w[6] * g[2], -w[1] * g[0], w[7] * g[0] * g[2], w[3] * g[2]],
        dim=1,
    )
    k3 = torch.cat(
        [w[5] * g[0], w[6] * g[1], w[0], -w[7] * g[0] * g[1], -w[1] * g[0], -w[2] * g[1]],
        dim=1,
    )
    k4 = torch.cat([w[2], -w[1], g[2] * w[7], w[0], -w[6] * g[2], w[5] * g[2]], dim=1)
    k5 = torch.cat([w[3], -w[7] * g[1], -w[1], w[6] * g[1], w[0], -w[4] * g[1]], dim=1)
    k6 = torch.cat([w[7] * g[0], w[3], -w[2], -w[5] * g[0], w[4] * g[0], w[0]], dim=1)
    k7 = torch.cat([w[6], -w[5], w[4], w[3], -w[2], w[1]], dim=1)
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return 8, k


def get_3d_clifford_decoding_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebra decoding layers.
    This specific 3d embedding layer assumes vector and bivector input fields.
    I.e. each part of the output multivector field is present except the scalar and the bivector parts.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor): Weight input of dimension `(8, d~input, d~output, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Returns:
        tuple[int, torch.Tensor]: Number of blades, weight output of dimension `(d~output * 6, d~input * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)

    k1 = torch.cat(
        [w[1], w[0], -w[4] * g[1], -w[5] * g[2], w[2] * g[1], w[3] * g[2], -w[7] * g[1] * g[2], -w[6] * g[2] * g[1]],
        dim=1,
    )
    k2 = torch.cat(
        [w[2], w[4] * g[0], w[0], -w[6] * g[2], -w[1] * g[0], w[7] * g[0] * g[2], w[3] * g[2], w[5] * g[2] * g[0]],
        dim=1,
    )
    k3 = torch.cat(
        [w[3], w[5] * g[0], w[6] * g[1], w[0], -w[7] * g[0] * g[1], -w[1] * g[0], -w[2] * g[1], -w[4] * g[0] * g[1]],
        dim=1,
    )
    k4 = torch.cat([w[4], w[2], -w[1], g[2] * w[7], w[0], -w[6] * g[2], w[5] * g[2], w[3] * g[2]], dim=1)
    k5 = torch.cat([w[5], w[3], -w[7] * g[1], -w[1], w[6] * g[1], w[0], -w[4] * g[1], -w[2] * g[1]], dim=1)
    k6 = torch.cat([w[6], w[7] * g[0], w[3], -w[2], -w[5] * g[0], w[4] * g[0], w[0], w[1] * g[0]], dim=1)
    k = torch.cat([k1, k2, k3, k4, k5, k6], dim=0)
    return 6, k
