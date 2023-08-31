# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# CUSTOMIZED ENCODING/DECODING LAYERS AS USED In THE PAPER:                               #
# Clifford Neural Layers for PDE Modeling                                                 #
###########################################################################################
import torch
import torch.nn.functional as F
from typing import Union
from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv2d,
    CliffordConv3d,
)
from cliffordlayers.models.basic.custom_kernels import (
    get_2d_scalar_vector_encoding_kernel,
    get_2d_scalar_vector_decoding_kernel,
    get_2d_rotation_scalar_vector_encoding_kernel,
    get_2d_rotation_scalar_vector_decoding_kernel,
    get_3d_maxwell_encoding_kernel,
    get_3d_maxwell_decoding_kernel,
)


class CliffordConv2dScalarVectorEncoder(CliffordConv2d):
    """2d Clifford convolution encoder for scalar+vector input fields which inherits from CliffordConv2d."""

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
            self._get_kernel = get_2d_rotation_scalar_vector_encoding_kernel
        else:
            self._get_kernel = get_2d_scalar_vector_encoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(CliffordConv2d, self).forward(x, F.conv2d)


class CliffordConv2dScalarVectorDecoder(CliffordConv2d):
    """2d Clifford convolution decoder for scalar+vector output fields which inherits from CliffordConv2d."""

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
            self._get_kernel = get_2d_rotation_scalar_vector_decoding_kernel
        else:
            self._get_kernel = get_2d_scalar_vector_decoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is True:
            raise ValueError("Bias needs to be set to False for 2d Clifford decoding layers.")
        return super(CliffordConv2d, self).forward(x, F.conv2d)


class CliffordConv3dMaxwellEncoder(CliffordConv3d):
    """3d Clifford convolution encoder for vector+bivector inputs which inherits from CliffordConv3d."""

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

        self._get_kernel = get_3d_maxwell_encoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(CliffordConv3d, self).forward(x, F.conv3d)


class CliffordConv3dMaxwellDecoder(CliffordConv3d):
    """3d Clifford convolution decoder for vector+bivector inputs which inherits from CliffordConv3d."""

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

        self._get_kernel = get_3d_maxwell_decoding_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is True:
            raise ValueError("Bias needs to be set to False for 3d Clifford decoding layers.")
        return super(CliffordConv3d, self).forward(x, F.conv3d)
