# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# 2D MODELS AS USED In THE PAPER:                                                         #
# Clifford Neural Layers for PDE Modeling                                                 #
###########################################################################################
from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as F

from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d
from cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv2d
from cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm2d
from cliffordlayers.models.basic.custom_layers import CliffordConv2dScalarVectorEncoder, CliffordConv2dScalarVectorDecoder


class CliffordBasicBlock2d(nn.Module):
    """2D building block for Clifford ResNet architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 1.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: Callable = F.gelu,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        rotation: bool = False,
        norm: bool = False,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = CliffordConv2d(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.conv2 = CliffordConv2d(
            g,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.norm1 = CliffordGroupNorm2d(g, num_groups, in_channels) if norm else nn.Identity()
        self.norm2 = CliffordGroupNorm2d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordBasicBlock2d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(self.norm1(x)))
        out = self.conv2(self.activation(self.norm2(out)))
        return out + x


class CliffordFourierBasicBlock2d(nn.Module):
    """2D building block for Clifford FNO architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 1.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
        modes1 (int, optional): Number of Fourier modes in the first dimension. Defaults to 16.
        modes2 (int, optional): Number of Fourier modes in the second dimension. Defaults to 16.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: Callable = F.gelu,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        rotation: bool = False,
        norm: bool = False,
        num_groups: int = 1,
        modes1: int = 16,
        modes2: int = 16,
    ):
        super().__init__()
        self.fourier = CliffordSpectralConv2d(
            g,
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
        )
        self.conv = CliffordConv2d(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.norm = CliffordGroupNorm2d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordFourierBasicBlock2d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier(x)
        x2 = self.conv(x)
        return self.activation(self.norm(x1 + x2))


class CliffordFluidNet2d(nn.Module):
    """2D building block for Clifford architectures for fluid mechanics (vector field+scalar field)
    with ResNet backbone network. The backbone networks follows these three steps:
        1. Clifford scalar+vector field encoding.
        2. Basic blocks as provided.
        3. Clifford scalar+vector field decoding.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        block (nn.Module): Choice of basic blocks.
        num_blocks (list): List of basic blocks in each residual block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    # For periodic boundary conditions, set padding = 0.
    padding = 9

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        block: nn.Module,
        num_blocks: list,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: Callable,
        rotation: False,
        norm: bool = False,
        num_groups: int = 1,
    ):
        super().__init__()

        self.activation = activation
        # Encoding and decoding layers
        self.encoder = CliffordConv2dScalarVectorEncoder(
            g,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            rotation=rotation,
        )
        self.decoder = CliffordConv2dScalarVectorDecoder(
            g,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            rotation=rotation,
        )

        # Residual blocks
        self.layers = nn.ModuleList(
            [
                self._make_basic_block(
                    g,
                    block,
                    hidden_channels,
                    num_blocks[i],
                    activation=activation,
                    rotation=rotation,
                    norm=norm,
                    num_groups=num_groups,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_basic_block(
        self,
        g,
        block: nn.Module,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        rotation: bool,
        norm: bool,
        num_groups: int,
    ) -> nn.Sequential:
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                block(
                    g,
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    rotation=rotation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5

        # Encoding layer
        x = self.encoder(self.activation(x))

        # Embed for non-periodic boundaries
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = F.pad(x, [0, self.padding, 0, self.padding])
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Apply residual layers
        for layer in self.layers:
            x = layer(x)

        # Decoding layer
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = x[..., : -self.padding, : -self.padding]
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Output layer
        x = self.decoder(x)
        return x
