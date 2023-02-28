# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)
from ...signature import CliffordSignature


class CliffordLinear(nn.Module):
    """Clifford linear layer.

    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.

    """

    def __init__(
        self,
        g,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        self.n_blades = sig.n_blades

        if self.dim == 1:
            self._get_kernel = get_1d_clifford_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        nn.init.kaiming_uniform_(
            self.weight.view(self.out_channels, self.in_channels * self.n_blades),
            a=math.sqrt(5),
        )
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight.view(self.out_channels, self.in_channels * self.n_blades)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that the Clifford kernel can be applied.
        B, _, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        B_dim, C_dim, I_dim = range(len(x.shape))
        x = x.permute(B_dim, -1, C_dim)
        x = x.reshape(B, -1)
        # Get Clifford kernel, apply it.
        _, weight = self._get_kernel(self.weight, self.g)
        output = F.linear(x, weight, self.bias.view(-1))
        # Reshape back.
        output = output.view(B, I, -1)
        B_dim, I_dim, C_dim = range(len(output.shape))
        output = output.permute(B_dim, C_dim, I_dim)
        return output
