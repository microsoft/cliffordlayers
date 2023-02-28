# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


def test_complex_layer():
    """Test Clifford1d linear module against complex linear module using g = [-1].
    """    
    in_channels = 8
    out_channels = 16
    x = torch.randn(4, in_channels, 2)
    clifford_linear = CliffordLinear(
        g = [-1], 
        in_channels = in_channels, 
        out_channels = out_channels
        )
    output_clifford_linear = clifford_linear(x)
    w_c = torch.view_as_complex(clifford_linear.weight.permute(1, 2, 0).contiguous())
    b_c = torch.view_as_complex(clifford_linear.bias.permute(1, 0).contiguous())
    input_c = torch.view_as_complex(x)
    output_c = F.linear(input_c, w_c, b_c)
    torch.testing.assert_close(output_clifford_linear, torch.view_as_real(output_c))


def test_clifford_linear_vs_clifford_convolution_1d():
    """Test Clifford1d linear module against Clifford convolution module with kernel_size = 1.
    """
    in_channels = 8
    out_channels = 16
    x = torch.randn(4, in_channels, 2)
    clifford_conv = CliffordConv1d(
        g = [-1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 1
        )
    output_clifford_conv = clifford_conv(x[..., None, :])
    clifford_linear = CliffordLinear(
        g = [-1],
        in_channels = in_channels, 
        out_channels = out_channels
        )
    clifford_linear.weight = nn.Parameter(
        torch.Tensor(torch.stack([weight.clone().detach() for weight in clifford_conv.weight])).squeeze()
    )
    clifford_linear.bias = nn.Parameter(clifford_conv.bias)
    output_clifford_linear = clifford_linear(x)
    torch.testing.assert_close(output_clifford_conv.squeeze(), output_clifford_linear)


def test_clifford_linear_vs_clifford_convolution_2d():
    """Test Clifford1d linear module against Clifford convolution module with kernel_size = 1.
    """
    in_channels = 8
    out_channels = 16
    x = torch.randn(4, in_channels, 4)
    clifford_conv = CliffordConv2d(
        g = [-1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 1
        )
    output_clifford_conv = clifford_conv(x[..., None, None, :])
    clifford_linear = CliffordLinear(
        g = [-1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels
        )
    clifford_linear.weight = nn.Parameter(
        torch.Tensor(torch.stack([weight.clone().detach() for weight in clifford_conv.weight])).squeeze()
    )
    clifford_linear.bias = nn.Parameter(clifford_conv.bias)
    output_clifford_linear = clifford_linear(x)
    torch.testing.assert_close(output_clifford_conv.squeeze(), output_clifford_linear)


def test_clifford_linear_vs_clifford_convolution_3d():
    """Test Clifford1d linear module against Clifford convolution module with kernel_size = 1.
    """
    in_channels = 8
    out_channels = 16
    x = torch.randn(4, in_channels, 8)
    clifford_conv = CliffordConv3d(
        g = [-1, -1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 1
        )
    output_clifford_conv = clifford_conv(x[..., None, None, None, :])
    clifford_linear = CliffordLinear(
        g = [-1, -1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels
        )
    clifford_linear.weight = nn.Parameter(
        torch.Tensor(torch.stack([weight.clone().detach() for weight in clifford_conv.weight]).squeeze())
    )
    clifford_linear.bias = nn.Parameter(clifford_conv.bias)
    output_clifford_linear = clifford_linear(x)
    torch.testing.assert_close(output_clifford_conv.squeeze(), output_clifford_linear)
