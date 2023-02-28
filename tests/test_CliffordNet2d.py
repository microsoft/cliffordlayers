# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F 
from cliffordlayers.models.utils import partialclass
from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordBasicBlock2d,
    CliffordFourierBasicBlock2d,
)


def test_clifford_resnet():
    """Test shape compatibility of Clifford2d ResNet model.
    """
    x = torch.randn(8, 4, 128, 128, 3)
    in_channels = 4
    out_channels = 1
    model = CliffordNet2d(
        g = [1, 1],
        block = CliffordBasicBlock2d,
        num_blocks = [2, 2, 2, 2],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = False,
        rotation = False,
    )
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_clifford_resnet_norm():
    """Test shape compatibility of Clifford2d ResNet model using normalization.
    """
    in_channels = 4
    out_channels = 1
    x = torch.randn(8, in_channels, 128, 128, 3)
    model = CliffordNet2d(
        g = [1, 1],
        block = CliffordBasicBlock2d,
        num_blocks = [2, 2, 2, 2],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = False,
    )
    out = model(x)
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_clifford_rotational_resnet_norm():
    """Test shape compatibility of Clifford2d rotational ResNet model using normalization.
    """
    in_channels = 4
    out_channels = 1
    x = torch.randn(8, in_channels, 128, 128, 3)
    model = CliffordNet2d(
        g = [-1, -1],
        block = CliffordBasicBlock2d,
        num_blocks = [2, 2, 2, 2],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = True,
    )
    out = model(x)
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_clifford_fourier_net():
    """Test shape compatibility of Clifford2d Fourier model.
    """
    in_channels = 4
    out_channels = 1
    x = torch.randn(8, in_channels, 128, 128, 3)
    model = CliffordNet2d(
        g = [1, 1],
        block = partialclass(
                "CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=32, modes2=32
            ),
        num_blocks = [1, 1, 1, 1],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = False,
        rotation = False,
    )
    out = model(x)
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_clifford_fourier_net_norm():
    """Test shape compatibility of Clifford2d Fourier model using normalization.
    """
    in_channels = 4
    out_channels = 1
    x = torch.randn(8, in_channels, 128, 128, 3)
    model = CliffordNet2d(
        g = [1, 1],
        block = partialclass(
                "CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=32, modes2=32
            ),
        num_blocks = [1, 1, 1, 1],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = False,
    )
    out = model(x)
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_clifford_fourier_rotational_net_norm():
    """Test shapes compatibility of Clifford2d Fourier model using normalization (and rotation).
    """
    in_channels = 4
    out_channels = 1
    x = torch.randn(8, in_channels, 128, 128, 3)
    model = CliffordNet2d(
        g = [-1, -1],
        block = partialclass(
                "CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=32, modes2=32
            ),
        num_blocks = [1, 1, 1, 1],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = True,
    )
    out = model(x)
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    assert out.shape == (8, out_channels, 128, 128, 3)
