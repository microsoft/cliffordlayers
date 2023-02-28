# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional, Union

import torch
import torch.nn as nn


def clifford_convnd(
    conv_fn: Callable,
    x: torch.Tensor,
    output_blades: int,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Apply a Clifford convolution to a tensor.

    Args:
        conv_fn (Callable): The convolution function to use.

        x (torch.Tensor): Input tensor.

        output_blades (int): The output blades of the Clifford algebra.
        Different from the default n_blades when using encoding and decoding layers.

        weight (torch.Tensor): Weight tensor.

        bias (torch.Tensor, optional): Bias tensor. Defaults to None.

    Returns:
        torch.Tensor: Convolved output tensor.
    """
    # Reshape x such that the convolution function can be applied.
    B, *_ = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    x = x.permute(B_dim, -1, C_dim, *D_dims)
    x = x.reshape(B, -1, *x.shape[3:])
    # Apply convolution function
    output = conv_fn(x, weight, bias=bias, **kwargs)
    # Reshape back.
    output = output.view(B, output_blades, -1, *output.shape[2:])
    B_dim, I_dim, C_dim, *D_dims = range(len(output.shape))
    output = output.permute(B_dim, C_dim, *D_dims, I_dim)
    return output


def _w_assert(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Convert Clifford weights to tensor .

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Clifford weights.

    Raises:
        ValueError: Unknown weight type.

    Returns:
        torch.Tensor: Clifford weights as torch.Tensor.
    """
    if type(w) in (tuple, list):
        w = torch.stack(w)
        return w
    elif isinstance(w, torch.Tensor):
        return w
    elif isinstance(w, nn.Parameter):
        return w
    elif isinstance(w, nn.ParameterList):
        return w
    else:
        raise ValueError(f"Unknown weight type.")


def batchmul1d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """1d batch multiplication of the form
    (batch, in_channel, x), (out_channel, in_channel, x) -> (batch, out_channel, x).

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.einsum("bix,oix->box", x, weights)


def batchmul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """2d batch multiplication of the form
    (batch, in_channel, x, y), (out_channel, in_channel, x, y) -> (batch, out_channel, x, y).

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.einsum("bixy,oixy->boxy", x, weights)


def batchmul3d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """3d batch multiplication of the form
    (batch, in_channel, x, y, z), (out_channel, in_channel, x, y, z) -> (batch, out_channel, x, y, z).

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.einsum("bixyz,oixyz->boxyz", x, weights)
