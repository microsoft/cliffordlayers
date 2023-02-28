# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Union

import torch
import torch.nn as nn

from cliffordlayers.nn.functional.utils import _w_assert


def get_1d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 1d Clifford algebras, g = [-1] corresponds to a complex number kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(2, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]):  Number of output blades, weight output of shape `(d~output~ * 2, d~input~ * 2, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 1
    w = _w_assert(w)
    assert len(w) == 2

    k0 = torch.cat([w[0], g[0] * w[1]], dim=1)
    k1 = torch.cat([w[1], w[0]], dim=1)
    k = torch.cat([k0, k1], dim=0)
    return 2, k


def get_2d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebras, g = [-1, -1] corresponds to a quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3], g[1] * w[2]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0], -g[0] * w[1]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k


def get_2d_clifford_rotation_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Rotational Clifford kernel for 2d Clifford algebras, the vector part corresponds to quaternion rotation.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(6, d~input~, d~output~, ...)`.
                    `w[0]`, `w[1]`, `w[2]`, `w[3]` are the 2D Clifford weight tensors; 
                    `w[4]` is the scaling tensor; `w[5]` is the zero kernel tensor.

        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
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
    k3 = torch.cat(
        [
            zero_kernel,
            scale * (rot13 - rot02),
            scale * (rot23 + rot01),
            scale * (1.0 - (s1 + s2)),
        ],
        dim=1,
    )
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k


def get_3d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an octonion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)
    assert len(w) == 8

    k0 = torch.cat(
        [
            w[0],
            w[1] * g[0],
            w[2] * g[1],
            w[3] * g[2],
            -w[4] * g[0] * g[1],
            -w[5] * g[0] * g[2],
            -w[6] * g[1] * g[2],
            -w[7] * g[0] * g[1] * g[2],
        ],
        dim=1,
    )
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
    k7 = torch.cat([w[7], w[6], -w[5], w[4], w[3], -w[2], w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return 8, k


def get_complex_kernel(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Complex kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(2, d~input~, d~output~, ...)`.

    Returns:
        (torch.Tensor):  Weight output of shape `(d~output~ * 2, d~input~ * 2, ...)`.
    """
    w = _w_assert(w)
    assert len(w) == 2
    k0 = torch.cat([w[0], -w[1]], dim=1)
    k1 = torch.cat([w[1], w[0]], dim=1)
    k = torch.cat([k0, k1], dim=0)
    return k


def get_quaternion_kernel(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.


    Returns:
        (torch.Tensor):  Weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    w = _w_assert(w)
    assert len(w) == 4
    k0 = torch.cat([w[0], -w[1], -w[2], -w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], w[3], -w[2]], dim=1)
    k2 = torch.cat([w[2], -w[3], w[0], w[1]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return k


def get_quaternion_rotation_kernel(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Quaternion rotation, taken mostly from <https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks>

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(6, d~input~, d~output~, ...)`.
                    `w[0]`, `w[1]`, `w[2]`, `w[3]` are the quaternion w;
                    tensors; `w[4]` is the scaling tensor; `w[5]` is the zero kernel tensor.

    Returns:
        (torch.Tensor): Quaternion weight output of dimension `(d~output * 3, d~input * 4, ...)`.
    """
    w = _w_assert(w)
    assert len(w) == 6
    square_1 = w[0] * w[0]
    square_2 = w[1] * w[1]
    square_3 = w[2] * w[2]
    square_4 = w[3] * w[3]

    norm = torch.sqrt(square_1 + square_2 + square_3 + square_4 + 0.0001)

    w1_n = w[0] / norm
    w2_n = w[1] / norm
    w3_n = w[2] / norm
    w4_n = w[3] / norm

    norm_factor = 2.0
    square_2 = norm_factor * (w2_n * w2_n)
    square_3 = norm_factor * (w3_n * w3_n)
    square_4 = norm_factor * (w4_n * w4_n)

    rot12 = norm_factor * w1_n * w2_n
    rot13 = norm_factor * w1_n * w3_n
    rot14 = norm_factor * w1_n * w4_n
    rot23 = norm_factor * w2_n * w3_n
    rot24 = norm_factor * w2_n * w4_n
    rot34 = norm_factor * w3_n * w4_n

    scale = w[4]
    zero_kernel = w[5]

    rot_kernel2 = torch.cat(
        [
            zero_kernel,
            scale * (1.0 - (square_3 + square_4)),
            scale * (rot23 - rot14),
            scale * (rot24 + rot13),
        ],
        dim=1,
    )
    rot_kernel3 = torch.cat(
        [
            zero_kernel,
            scale * (rot23 + rot14),
            scale * (1.0 - (square_2 + square_4)),
            scale * (rot34 - rot12),
        ],
        dim=1,
    )
    rot_kernel4 = torch.cat(
        [
            zero_kernel,
            scale * (rot24 - rot13),
            scale * (rot34 + rot12),
            scale * (1.0 - (square_2 + square_3)),
        ],
        dim=1,
    )

    k= torch.cat([rot_kernel2, rot_kernel3, rot_kernel4], dim=0)
    return k


def get_octonion_kernel(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Octonion kernels.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.


    Returns:
        (torch.Tensor):  Weight output of shape `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    w = _w_assert(w)
    assert len(w) == 8
    k0 = torch.cat(
        [w[0], -w[1], -w[2], -w[3], -w[4], -w[5], -w[6], w[7]], dim=1
    )
    k1 = torch.cat(
        [w[1], w[0], w[4], w[5], -w[2], -w[3], -w[7], -w[6]], dim=1
    )
    k2 = torch.cat(
        [w[2], -w[4], w[0], w[6], w[1], w[7], -w[3], w[5]], dim=1
    )
    k3 = torch.cat(
        [w[3], -w[5], -w[6], w[0], -w[7], w[1], w[2], -w[4]], dim=1
    )
    k4 = torch.cat(
        [w[4], w[2], -w[1], -w[7], w[0], w[6], -w[5], -w[3]], dim=1
    )
    k5 = torch.cat(
        [w[5], w[3], w[7], -w[1], -w[6], w[0], w[4], w[2]], dim=1
    )
    k6 = torch.cat(
        [w[6], -w[7], w[3], -w[2], w[5], -w[4], w[0], -w[1]], dim=1
    )
    k7 = torch.cat(
        [w[7], w[6], -w[5], w[4], w[3], -w[2], w[1], w[0]], dim=1
    )
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return k