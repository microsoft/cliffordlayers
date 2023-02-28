# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# CUSTOMIZED ENCODING/DECODING LAYERS AS USED In THE PAPER:                               #
# Clifford Neural Layers for PDE Modeling                                                 #
###########################################################################################
import torch
import torch.nn.functional as F
from typing import Union
from cliffordlayers.nn.modules.cliffordconv import(
    CliffordConv2d,
    CliffordConv3d,
)
from cliffordlayers.models.custom_kernels import(
    get_2d_clifford_encoding_kernel,
    get_2d_clifford_decoding_kernel,
    get_2d_clifford_rotation_encoding_kernel,
    get_2d_clifford_rotation_decoding_kernel,
    get_3d_clifford_encoding_kernel,
    get_3d_clifford_decoding_kernel,
)


class CliffordConv2dEncoder(CliffordConv2d):
    """2d Clifford convolution encoder which inherits from CliffordConv2d.

    """
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rotation: bool = False,
    ):
        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            rotation,
        )

        if rotation:
            self._get_kernel = get_2d_clifford_rotation_encoding_kernel
        else:
            self._get_kernel = get_2d_clifford_encoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(CliffordConv2d, self).forward(x, F.conv2d)


class CliffordConv2dDecoder(CliffordConv2d):
    """2d Clifford convolution decoder which inherits from CliffordConv2d.

    """
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        rotation: bool = False,
    ):
        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            rotation,
        )

        if rotation:
            self._get_kernel = get_2d_clifford_rotation_decoding_kernel
        else:
            self._get_kernel = get_2d_clifford_decoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is True:
            raise ValueError(f"Bias needs to be set to False for 2d Clifford decoding layers.")
        return super(CliffordConv2d, self).forward(x, F.conv2d)


class CliffordConv3dEncoder(CliffordConv3d):
    """3d Clifford convolution encoder which inherits from CliffordConv3d.

    """
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self._get_kernel = get_3d_clifford_encoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(CliffordConv3d, self).forward(x, F.conv3d)


class CliffordConv3dDecoder(CliffordConv3d):
    """3d Clifford convolution decoder which inherits from CliffordConv3d.

    """
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self._get_kernel = get_3d_clifford_decoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is True:
            raise ValueError(f"Bias needs to be set to False for 3d Clifford decoding layers.")
        return super(CliffordConv3d, self).forward(x, F.conv3d)

