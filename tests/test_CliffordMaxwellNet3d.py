# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from cliffordlayers.models.basic.threed import (
    CliffordMaxwellNet3d,
    CliffordFourierBasicBlock3d,
)


def test_clifford_fourier_resnet():
    """Test shape compatibility of CliffordMaxwellNet3d Fourier model."""
    x = torch.randn(8, 4, 32, 32, 32, 6)
    model = CliffordMaxwellNet3d(
        g=[1, 1, 1],
        block=CliffordFourierBasicBlock3d,
        num_blocks=[1, 1, 1, 1],
        in_channels=4,
        out_channels=1,
        hidden_channels=16,
        activation=F.gelu,
        norm=False,
    )
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, 1, 32, 32, 32, 6)


def test_clifford_fourier_net_norm():
    """Test shape compatibility of CliffordMaxwellNet2d Fourier model using normalization."""
    x = torch.randn(8, 4, 32, 32, 32, 6)
    model = CliffordMaxwellNet3d(
        g=[1, 1, 1],
        block=CliffordFourierBasicBlock3d,
        num_blocks=[1, 1, 1, 1],
        in_channels=4,
        out_channels=6,
        hidden_channels=16,
        activation=F.gelu,
        norm=True,
        num_groups=2,
    )
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, 6, 32, 32, 32, 6)
