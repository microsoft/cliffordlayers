# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordfourier_deprecated import (
    CliffordSpectralConv2d_deprecated, 
    CliffordSpectralConv3d_deprecated,
)
from cliffordlayers.nn.modules.cliffordfourier import (
    CliffordSpectralConv2d, 
    CliffordSpectralConv3d,
)


def test_clifford_fourier_layer_2d():
    """Test 2d CFNO implementation for g=[1, 1] vs deprecated implementation. 
    """    
    # The deprecated CFNO only works if number of input and output channels are the same.
    old_cfno2d = CliffordSpectralConv2d_deprecated(
        in_channels = 8,
        out_channels = 8,
        modes1 = 16,
        modes2 = 16,
        )
    input = torch.rand(1, 8, 128, 128, 4)
    input_spinor = torch.cat((input[..., 0].unsqueeze(-1), input[..., 3].unsqueeze(-1)), dim=-1)
    input_vector = torch.cat((input[..., 1].unsqueeze(-1), input[..., 2].unsqueeze(-1)), dim=-1)
    output_old = old_cfno2d(torch.view_as_complex(input_vector), torch.view_as_complex(input_spinor)) 
    
    new_cfn2d = CliffordSpectralConv2d(
        g = [1, 1],
        in_channels = 8,
        out_channels = 8,
        modes1 = 16,
        modes2 = 16,
    )
    new_cfn2d.weights = nn.Parameter(old_cfno2d.weights.permute(0, 2, 1, 3, 4))
    output_new = new_cfn2d(input)
    vector, spinor = output_old
    output_old_trans = torch.cat(
        (
            spinor.real.unsqueeze(-1),
            vector.real.unsqueeze(-1),
            vector.imag.unsqueeze(-1),
            spinor.imag.unsqueeze(-1),
        ),
        dim = -1
    )

    torch.testing.assert_close(output_old_trans, output_new)


def test_clifford_fourier_layer_3d():
    """Test 3d CFNO implementation for g=[1, 1, 1] vs deprecated implementation. 
    """ 
    # The old CFNO only works if number of input and output channels are the same.
    old_cfno3d = CliffordSpectralConv3d_deprecated(
        in_channels = 4,
        out_channels = 4,
        modes1 = 16,
        modes2 = 16,
        modes3 = 16,
        )
    input = torch.rand(1, 4, 32, 32, 32, 8)
    input_dual_1 = torch.cat((input[..., 0].unsqueeze(-1), input[..., 7].unsqueeze(-1)), dim=-1)
    input_dual_2 = torch.cat((input[..., 1].unsqueeze(-1), input[..., 6].unsqueeze(-1)), dim=-1)
    input_dual_3 = torch.cat((input[..., 2].unsqueeze(-1), input[..., 5].unsqueeze(-1)), dim=-1)
    input_dual_4 = torch.cat((input[..., 3].unsqueeze(-1), input[..., 4].unsqueeze(-1)), dim=-1)
    dual_1, dual_2, dual_3, dual_4 = old_cfno3d(
        torch.view_as_complex(input_dual_1), 
        torch.view_as_complex(input_dual_2), 
        torch.view_as_complex(input_dual_3), 
        torch.view_as_complex(input_dual_4),
        )
    new_cfno3d = CliffordSpectralConv3d(
        g = [1, 1, 1],
        in_channels = 4,
        out_channels = 4,
        modes1 = 16,
        modes2 = 16,
        modes3 = 16,
    )
    new_cfno3d.weights = nn.Parameter(old_cfno3d.weights.permute(0, 2, 1, 3, 4, 5))
    output_new = new_cfno3d(input)
    output_old_trans = torch.cat(
        (
            dual_1.real.unsqueeze(-1),
            dual_2.real.unsqueeze(-1),
            dual_3.real.unsqueeze(-1),
            dual_4.real.unsqueeze(-1),
            dual_4.imag.unsqueeze(-1),
            dual_3.imag.unsqueeze(-1),
            dual_2.imag.unsqueeze(-1),
            dual_1.imag.unsqueeze(-1),
        ),
        dim = -1
    )

    torch.testing.assert_close(output_old_trans, output_new)


def test_unit_weights_clifford_fourier_layer_2d(): 
    """Test 2d CFNO implementation vs CFFT and inverse CFFT without weight multiplication.
    Input and output channels need to be the same; Fourier modes have to correspond to spatial resolution. 
    """ 
    in_channels = 8
    nx = 128
    ny = 128
    input = torch.rand(1, 8, nx, ny, 4)
    cfn2d = CliffordSpectralConv2d(
        g = [1, 1],
        in_channels = in_channels,
        out_channels = in_channels,
        modes1 = nx,
        modes2 = ny,
        multiply = False,
    )
    output = cfn2d(input)
    torch.testing.assert_close(output, input)


def test_unit_weights_clifford_fourier_layer_3d():
    """Test 3d CFNO implementation vs CFFT and inverse CFFT without weight multiplication. 
    Input and output channels need to be the same; Fourier modes have to correspond to spatial resolution. 
    """
    in_channels = 8
    nx = 32
    ny = 32
    nz = 32
    input = torch.rand(1, 8, nx, ny, nz, 8)
    cfn3d = CliffordSpectralConv3d(
        g = [1, 1, 1],
        in_channels = in_channels,
        out_channels = in_channels,
        modes1 = nx,
        modes2 = ny,
        modes3 = nz,
        multiply = False,
    )
    output = cfn3d(input)
    torch.testing.assert_close(output, input)

