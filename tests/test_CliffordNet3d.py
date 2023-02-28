# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F 
from cliffordlayers.models.models_3d import (
    CliffordNet3d,
    CliffordFourierBasicBlock3d,
)


def test_clifford_fourier_resnet():
    """Test shape compatibility of Clifford3d Fourier model.
    """
    x = torch.randn(8, 4, 64, 64, 64, 6)
    model = CliffordNet3d(
        g = [1, 1, 1],
        block = CliffordFourierBasicBlock3d,
        num_blocks = [1, 1, 1, 1],
        in_channels = 4,
        out_channels = 1,
        hidden_channels = 16,
        activation = F.gelu,
        norm = False,
    )
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, 1, 64, 64, 64, 6)


def test_clifford_fourier_net_norm():
    """Test shape compatibility of Clifford2d Fourier model using normalization.
    """
    x = torch.randn(8, 4, 64, 64, 64, 6)
    model = CliffordNet3d(
        g = [1, 1, 1],
        block = CliffordFourierBasicBlock3d,
        num_blocks = [1, 1, 1, 1],
        in_channels = 4,
        out_channels = 6,
        hidden_channels = 16,
        activation = F.gelu,
        norm = True,
        num_groups = 2,
    )
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, 6, 64, 64, 64, 6)
