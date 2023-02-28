# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_complex_convolution():
    """Test Clifford1d convolution module against complex convolution module using g = [-1].
    """    
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 128, 2)
    clifford_conv = CliffordConv1d(
        g = [-1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 3
        )
    output_clifford_conv = clifford_conv(x)
    w_c = torch.view_as_complex(torch.stack((clifford_conv.weight[0], clifford_conv.weight[1]), -1))
    b_c = torch.view_as_complex(clifford_conv.bias.permute(1, 0).contiguous())
    input_c = torch.view_as_complex(x)
    output_c = F.conv1d(input_c, w_c, b_c)
    torch.testing.assert_close(output_clifford_conv, torch.view_as_real(output_c))


def test_Clifford1d_conv_shapes():
    """Test shapes of Clifford1d convolution module.
    """
    in_channels = 8 
    x = torch.randn(1, 8, 128, 2)
    clifford_conv = CliffordConv1d(
        g = [1], 
        in_channels = in_channels, 
        out_channels = in_channels, 
        kernel_size = 3, 
        padding=1
        )
    x_out = clifford_conv(x)
    assert x.shape == x_out.shape


def test_Clifford2d_conv_shapes():
    """Test shapes of Clifford2d convolution module.
    """
    in_channels = 8
    x = torch.randn(1, 8, 128, 128, 4)
    clifford_conv = CliffordConv2d(
        g = [1, 1], 
        in_channels = in_channels, 
        out_channels = in_channels, 
        kernel_size=3, 
        padding=1
        )
    x_out = clifford_conv(x)
    clifford_conv_rotation = CliffordConv2d(
        g=[-1, -1], in_channels=8, out_channels=8, kernel_size=3, padding=1, rotation=True
    )
    x_out_rot = clifford_conv_rotation(x)
    assert x.shape == x_out.shape
    assert x.shape == x_out_rot.shape


def test_Clifford2d_conv_params():
    """Test parameters of Clifford2d convolution using g = [-1, -1] vs Clifford2d rotational convolution.
    When bias is set to False the ration needs to be 4/5.
    """
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 128, 128, 4)
    clifford_conv = CliffordConv2d(
        g = [-1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 3, 
        padding = 1, 
        bias = False
        )
    clifford_conv_rotation = CliffordConv2d(
        g = [-1, -1], 
        in_channels = in_channels, 
        out_channels = out_channels, 
        kernel_size = 3, 
        padding = 1, 
        bias = False, 
        rotation = True
    )
    torch.testing.assert_close(float(count_params(clifford_conv) / count_params(clifford_conv_rotation)), 0.8)


def test_Clifford3d_conv_shapes():
    """Test shapes of Clifford2d convolution module.
    """
    in_channels = 8
    x = torch.randn(1, in_channels, 32, 32, 32, 8)
    clifford_conv = CliffordConv3d(
        g = [1, 1, 1], 
        in_channels = in_channels, 
        out_channels = in_channels, 
        kernel_size = 3, 
        padding = 1
        )
    x_out = clifford_conv(x)
    assert x.shape == x_out.shape
