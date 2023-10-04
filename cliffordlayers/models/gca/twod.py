from typing import Callable, List, Tuple, Union

import torch
from torch import nn

from cliffordlayers.nn.modules.gcan import (
    CliffordG3Conv2d,
    CliffordG3ConvTranspose2d,
    CliffordG3GroupNorm,
    CliffordG3LinearVSiLU,
    CliffordG3SumVSiLU,
    CliffordG3MeanVSiLU,
)


def get_activation(activation: str, channels: int) -> Callable:
    if activation == "vsum":
        return CliffordG3SumVSiLU()
    elif activation == "vmean":
        return CliffordG3MeanVSiLU()
    elif activation == "vlin":
        return CliffordG3LinearVSiLU(channels)
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")


class CliffordG3BasicBlock2d(nn.Module):
    """
    Basic block for G3 convolutions on 2D grids, comprising two G3 Clifford convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution operation. Defaults to 1.
        padding (int, optional): Padding added to both sides of the input. Defaults to 1.
        activation (str, optional): Type of activation function. Defaults to "vlin".
        norm (bool, optional): If True, normalization is applied. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "vlin",
        norm: bool = True,
        num_groups: int = 1,
        prenorm: bool = True,
    ):
        super().__init__()
        self.conv1 = CliffordG3Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.conv2 = CliffordG3Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.norm1 = CliffordG3GroupNorm(num_groups, in_channels, 3) if norm else nn.Identity()
        self.norm2 = CliffordG3GroupNorm(num_groups, out_channels, 3) if norm else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = CliffordG3Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

        self.act1 = get_activation(activation, in_channels)
        self.act2 = get_activation(activation, out_channels)

        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            out = self.conv1(self.act1(self.norm1(x)))
            out = self.conv2(self.act2(self.norm2(out)))
        else:
            out = self.conv1(self.norm1(self.act1(x)))
            out = self.conv2(self.norm2(self.act2(out)))

        return out + self.shortcut(x)


class CliffordG3ResNet2d(nn.Module):
    """
    ResNet for G3 Clifford convolutions on 2D grids.

    Args:
        num_blocks (list): Number of blocks at each resolution.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels.
        activation (str, optional): Type of activation function. Defaults to "vlin".
        block (nn.Module, optional): Type of block. Defaults to CliffordG3BasicBlock2d.
        norm (bool, optional): If True, normalization is applied. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
    """

    padding = 9

    def __init__(
        self,
        num_blocks: list,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str = "vlin",
        block: nn.Module = CliffordG3BasicBlock2d,
        norm: bool = False,
        num_groups: int = 1,
        prenorm=True,
    ):
        super().__init__()

        # Embedding layers
        self.conv_in1 = CliffordG3Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )

        self.conv_in2 = CliffordG3Conv2d(
            hidden_channels,
            hidden_channels,
        )

        # Output layers
        self.conv_out1 = CliffordG3Conv2d(
            hidden_channels,
            hidden_channels,
        )
        self.conv_out2 = CliffordG3Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        # ResNet blocks
        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    hidden_channels,
                    num_blocks[i],
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    prenorm=prenorm,
                )
                for i in range(len(num_blocks))
            ]
        )

        self.act1 = get_activation(activation, hidden_channels)
        self.act2 = get_activation(activation, hidden_channels)

    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        num_blocks: int,
        activation: str,
        num_groups: int,
        norm: bool = True,
        prenorm: bool = True,
    ) -> nn.Sequential:
        layers = []
        for _ in range(num_blocks):
            layers.append(
                block(channels, channels, activation=activation, norm=norm, prenorm=prenorm, num_groups=num_groups)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5

        h = self.conv_in1(x)
        h = self.act1(h)

        # Second embedding layer
        h = self.conv_in2(h)

        for layer in self.layers:
            h = layer(h)

        # Output layers
        h = self.conv_out1(h)
        h = self.act2(h)
        h = self.conv_out2(h)

        # return output
        return h


class CliffordG3DownBlock(nn.Module):
    """
    UNet encoder block for G3 Clifford convolutions on 2D grids.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Type of activation function.
        norm (bool, optional): If True, normalization is applied. Defaults to False.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        norm: bool = False,
        prenorm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = CliffordG3BasicBlock2d(
            in_channels, out_channels, activation=activation, norm=norm, prenorm=prenorm, num_groups=num_groups
        )

    def forward(self, x):
        return self.block(x)


class CliffordG3Downsample(nn.Module):
    """
    Scale down the two-dimensional G3 Clifford feature map by a half.

    Args:
        n_channels (int): Number of channels.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.conv = CliffordG3Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class CliffordG3MiddleBlock(nn.Module):
    """
    UNet middle block for G3 Clifford convolutions on 2D grids.

    Args:
        n_channels (int): Number of channels.
        activation (str): Type of activation function.
        norm (bool, optional): If True, normalization is applied. Defaults to False.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
    """

    def __init__(
        self,
        n_channels: int,
        activation: str,
        norm: bool = False,
        prenorm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.res1 = CliffordG3BasicBlock2d(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            prenorm=prenorm,
            num_groups=num_groups,
        )
        self.res2 = CliffordG3BasicBlock2d(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            prenorm=prenorm,
            num_groups=num_groups,
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x


class CliffordG3UpBlock(nn.Module):
    """
    UNet decoder block for G3 Clifford convolutions on 2D grids.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Type of activation function.
        norm (bool, optional): If True, normalization is applied. Defaults to False.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        norm: bool = False,
        prenorm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = CliffordG3BasicBlock2d(
            in_channels + out_channels,
            out_channels,
            activation=activation,
            norm=norm,
            prenorm=prenorm,
            num_groups=num_groups,
        )

    def forward(self, x):
        return self.res(x)


class CliffordUpsample(nn.Module):
    """
    Scale up the two-dimensional G3 Clifford feature map by a factor of two.

    Args:
        n_channels (int): Number of channels.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = CliffordG3ConvTranspose2d(
            n_channels,
            n_channels,
            4,
            2,
            1,
        )

    def forward(self, x):
        return self.conv(x)


class CliffordG3UNet2d(nn.Module):
    """
    U-Net architecture with Clifford G3 convolutions for 2D grids.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of channels in the first hidden convolutional layer.
        activation (str, optional): Type of activation function. Defaults to "vlin".
        norm (bool, optional): If True, normalization is applied. Defaults to False.
        ch_mults (Union[Tuple[int, ...], List[int]], optional): Multipliers for the number of channels at each depth.
                                                            Defaults to (1, 2, 2, 2).
        n_blocks (int, optional): Number of convolutional blocks at each resolution. Defaults to 2.
        prenorm (bool, optional): If True, normalization is applied before activation, otherwise after. Defaults to True.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str = "vlin",
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
        n_blocks: int = 2,
        prenorm: bool = True,
        num_groups: int = 1,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        # Number of resolutions
        n_resolutions = len(ch_mults)

        self.conv1 = CliffordG3Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        # Decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    CliffordG3DownBlock(
                        in_channels,
                        out_channels,
                        activation=activation,
                        norm=norm,
                        prenorm=prenorm,
                        num_groups=num_groups,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(
                    CliffordG3Downsample(
                        in_channels,
                    )
                )

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = CliffordG3MiddleBlock(out_channels, activation=activation, norm=norm, prenorm=prenorm)

        # Increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    CliffordG3UpBlock(
                        in_channels,
                        out_channels,
                        activation=activation,
                        norm=norm,
                        prenorm=prenorm,
                        num_groups=num_groups,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                CliffordG3UpBlock(
                    in_channels,
                    out_channels,
                    activation=activation,
                    norm=norm,
                    prenorm=prenorm,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(
                    CliffordUpsample(
                        in_channels,
                    )
                )

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.activation = get_activation(activation, out_channels)

        if norm:
            self.norm = CliffordG3GroupNorm(num_groups, out_channels, 3)
        else:
            self.norm = nn.Identity()

        # Output layers
        self.conv2 = CliffordG3Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5

        x = self.conv1(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, CliffordUpsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.activation(self.norm(x))
        x = self.conv2(x)
        return x
