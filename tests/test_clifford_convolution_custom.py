# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from cliffordlayers.models.basic.custom_layers import (
    CliffordConv2dScalarVectorEncoder,
    CliffordConv2dScalarVectorDecoder,
    CliffordConv3dMaxwellEncoder,
    CliffordConv3dMaxwellDecoder,
)


def test_CliffordFluidNet2d_conv_encoding():
    """Test shapes of custom CliffordFluidNet2d encoding modules."""
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 128, 128, 3)
    clifford_conv = CliffordConv2dScalarVectorEncoder(
        g=[1, 1], in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
    )
    x_out = clifford_conv(x)
    clifford_conv_rotation = CliffordConv2dScalarVectorEncoder(
        g=[-1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, rotation=True
    )
    x_out_rot = clifford_conv_rotation(x)
    assert x_out.shape == (1, out_channels, 128, 128, 4)
    assert x_out_rot.shape == (1, out_channels, 128, 128, 4)


def test_Clifford2dFluidNet_conv_decoding():
    """Test shapes of custom CliffordFluidNet2d decoding modules."""
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 128, 128, 4)
    clifford_conv = CliffordConv2dScalarVectorDecoder(
        g=[1, 1],
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )
    x_out = clifford_conv(x)
    clifford_conv_rotation = CliffordConv2dScalarVectorDecoder(
        g=[-1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, rotation=True
    )
    x_out_rot = clifford_conv_rotation(x)
    assert x_out.shape == (1, out_channels, 128, 128, 3)
    assert x_out_rot.shape == (1, out_channels, 128, 128, 3)


def test_Clifford3dMaxwell_conv_encoding():
    """Test shapes of custom CliffordMaxwell3d encoding modules."""
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 64, 64, 64, 6)
    clifford_conv = CliffordConv3dMaxwellEncoder(
        g=[1, 1, 1], in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0
    )
    x_out = clifford_conv(x)
    assert x_out.shape == (1, out_channels, 64, 64, 64, 8)


def test_Clifford3dMaxwell_conv_decoding():
    """Test shapes of custom CliffordMaxwell3d decoding modules."""
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 64, 64, 64, 8)
    clifford_conv = CliffordConv3dMaxwellDecoder(
        g=[1, 1, 1], in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0
    )
    x_out = clifford_conv(x)
    assert x_out.shape == (1, out_channels, 64, 64, 64, 6)
