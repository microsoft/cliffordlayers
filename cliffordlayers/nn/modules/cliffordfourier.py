# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from typing import Union

from ...cliffordkernels import get_2d_clifford_kernel, get_3d_clifford_kernel
from ...signature import CliffordSignature
from ..functional.utils import batchmul2d, batchmul3d


class CliffordSpectralConv2d(nn.Module):
    """2d Clifford Fourier layer.
    Performs following three steps:
        1. Clifford Fourier transform over the multivector of 2d Clifford algebras, based on complex Fourier transforms using [pytorch.fft.fft2](https://pytorch.org/docs/stable/generated/torch.fft.fft2.html#torch.fft.fft2).
        2. Weight multiplication in the Clifford Fourier space using the geometric product.
        3. Inverse Clifford Fourier transform, based on inverse complex Fourier transforms using [pytorch.fft.ifft2](https://pytorch.org/docs/stable/generated/torch.fft.ifft2.html#torch.fft.ifft2).

    Args:
        g ((Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of non-zero Fourier modes in the first dimension.
        modes2 (int): Number of non-zero Fourier modes in the second dimension.
        multiply (bool): Multipliation in the Fourier space. If set to False this class only crops high-frequency modes.

    """

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        multiply: bool = True,
    ) -> None:
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        if self.dim != 2:
            raise ValueError("g must be a 2D Clifford algebra")

        self.n_blades = sig.n_blades
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.multiply = multiply

        # Initialize weight parameters.
        if multiply:
            scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                scale * torch.rand(4, out_channels, in_channels, self.modes1 * 2, self.modes2 * 2, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that FFT can be applied to dual pairs.
        B, _, *D, I = x.shape
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")

        dual_1 = torch.view_as_complex(torch.stack((x[..., 0], x[..., 3]), dim=-1))
        dual_2 = torch.view_as_complex(torch.stack((x[..., 1], x[..., 2]), dim=-1))
        dual_1_ft = torch.fft.fft2(dual_1)
        dual_2_ft = torch.fft.fft2(dual_2)

        # Add dual pairs again to multivector in the Fourier space.
        multivector_ft = torch.cat(
            (
                dual_1_ft.real,
                dual_2_ft.real,
                dual_2_ft.imag,
                dual_1_ft.imag,
            ),
            dim=1,
        )

        # Reserve Cifford output Fourier modes.
        out_ft = torch.zeros(
            B,
            self.out_channels * self.n_blades,
            *D,
            dtype=torch.float,
            device=multivector_ft.device,
        )

        # Concatenate positive and negative modes, such that the geometric product can be applied in one go.
        input_mul = torch.cat(
            (
                torch.cat(
                    (
                        multivector_ft[:, :, : self.modes1, : self.modes2],
                        multivector_ft[:, :, : self.modes1, -self.modes2 :],
                    ),
                    -1,
                ),
                torch.cat(
                    (
                        multivector_ft[:, :, -self.modes1 :, : self.modes2],
                        multivector_ft[:, :, -self.modes1 :, -self.modes2 :],
                    ),
                    -1,
                ),
            ),
            -2,
        )

        # Get Clifford weight tensor and apply the geometric product in the Fourier space.
        if self.multiply:
            _, kernel = get_2d_clifford_kernel(self.weights, self.g)
            output_mul = batchmul2d(input_mul, kernel)
        else:
            output_mul = input_mul

        # Fill the output modes, i.e. cut away high-frequency modes.
        out_ft[:, :, : self.modes1, : self.modes2] = output_mul[:, :, : self.modes1, : self.modes2]
        out_ft[:, :, -self.modes1 :, : self.modes2] = output_mul[:, :, -self.modes1 :, : self.modes2]
        out_ft[:, :, : self.modes1, -self.modes2 :] = output_mul[:, :, : self.modes1, -self.modes2 :]
        out_ft[:, :, -self.modes1 :, -self.modes2 :] = output_mul[:, :, -self.modes1 :, -self.modes2 :]

        # Reshape output such that inverse FFTs can be applied to the dual pairs.
        out_ft = out_ft.reshape(B, I, -1, *out_ft.shape[-2:])
        B_dim, I_dim, C_dim, *D_dims = range(len(out_ft.shape))
        out_ft = out_ft.permute(B_dim, C_dim, *D_dims, I_dim)
        out_dual_1 = torch.view_as_complex(torch.stack((out_ft[..., 0], out_ft[..., 3]), dim=-1))
        out_dual_2 = torch.view_as_complex(torch.stack((out_ft[..., 1], out_ft[..., 2]), dim=-1))
        dual_1_ifft = torch.fft.ifft2(out_dual_1, s=(out_dual_1.size(-2), out_dual_1.size(-1)))
        dual_2_ifft = torch.fft.ifft2(out_dual_2, s=(out_dual_2.size(-2), out_dual_2.size(-1)))

        # Finally, return to the multivector in the spatial domain.
        output = torch.stack(
            (
                dual_1_ifft.real,
                dual_2_ifft.real,
                dual_2_ifft.imag,
                dual_1_ifft.imag,
            ),
            dim=-1,
        )

        return output


class CliffordSpectralConv3d(nn.Module):
    """3d Clifford Fourier layer.
    Performs following three steps:
        1. Clifford Fourier transform over the multivector of 3d Clifford algebras, based on complex Fourier transforms using [pytorch.fft.fftn](https://pytorch.org/docs/stable/generated/torch.fft.fftn.html#torch.fft.fftn).
        2. Weight multiplication in the Clifford Fourier space using the geometric product.
        3. Inverse Clifford Fourier transform, based on inverse complex Fourier transforms using [pytorch.fft.ifftn](https://pytorch.org/docs/stable/generated/torch.fft.fftn.html#torch.fft.ifftn).

    Args:
        g ((Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of non-zero Fourier modes in the first dimension.
        modes2 (int): Number of non-zero Fourier modes in the second dimension.
        modes3 (int): Number of non-zero Fourier modes in the second dimension.
        multiply (bool): Multipliation in the Fourier space. If set to False this class only crops high-frequency modes.

    """

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        multiply: bool = True,
    ) -> None:
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        if self.dim != 3:
            raise ValueError("g must be a 3D Clifford algebra")
        self.n_blades = sig.n_blades

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.multiply = multiply

        # Initialize weight parameters.
        if self.multiply:
            scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                scale
                * torch.rand(
                    8,
                    out_channels,
                    in_channels,
                    self.modes1 * 2,
                    self.modes2 * 2,
                    self.modes3 * 2,
                    dtype=torch.float32,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that FFT can be applied to dual pairs.
        B, _, *D, I = x.shape
        *_, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")

        dual_1 = torch.view_as_complex(torch.stack((x[..., 0], x[..., 7]), dim=-1))
        dual_2 = torch.view_as_complex(torch.stack((x[..., 1], x[..., 6]), dim=-1))
        dual_3 = torch.view_as_complex(torch.stack((x[..., 2], x[..., 5]), dim=-1))
        dual_4 = torch.view_as_complex(torch.stack((x[..., 3], x[..., 4]), dim=-1))
        dual_1_ft = torch.fft.fftn(dual_1, dim=[-3, -2, -1])
        dual_2_ft = torch.fft.fftn(dual_2, dim=[-3, -2, -1])
        dual_3_ft = torch.fft.fftn(dual_3, dim=[-3, -2, -1])
        dual_4_ft = torch.fft.fftn(dual_4, dim=[-3, -2, -1])

        # Add dual pairs again to multivector in the Fourier space.
        multivector_ft = torch.cat(
            (
                dual_1_ft.real,
                dual_2_ft.real,
                dual_3_ft.real,
                dual_4_ft.real,
                dual_4_ft.imag,
                dual_3_ft.imag,
                dual_2_ft.imag,
                dual_1_ft.imag,
            ),
            dim=1,
        )

        # Reserve Cifford output Fourier modes.
        out_ft = torch.zeros(
            B,
            self.out_channels * self.n_blades,
            *D,
            dtype=torch.float,
            device=multivector_ft.device,
        )

        # Concatenate positive and negative modes, such that the geometric product can be applied in one go.
        input_mul = torch.cat(
            (
                torch.cat(
                    (
                        torch.cat(
                            (
                                multivector_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
                                multivector_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :],
                            ),
                            -1,
                        ),
                        torch.cat(
                            (
                                multivector_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3],
                                multivector_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :],
                            ),
                            -1,
                        ),
                    ),
                    -2,
                ),
                torch.cat(
                    (
                        torch.cat(
                            (
                                multivector_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                                multivector_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :],
                            ),
                            -1,
                        ),
                        torch.cat(
                            (
                                multivector_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
                                multivector_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :],
                            ),
                            -1,
                        ),
                    ),
                    -2,
                ),
            ),
            -3,
        )

        # Get Clifford weight tensor and apply the geometric product in the Fourier space.
        if self.multiply:
            _, kernel = get_3d_clifford_kernel(self.weights, self.g)
            output_mul = batchmul3d(input_mul, kernel)
        else:
            output_mul = input_mul

        # Fill the output modes, i.e. cut away high-frequency modes.
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = output_mul[
            :, :, : self.modes1, : self.modes2, : self.modes3
        ]
        out_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :] = output_mul[
            :, :, : self.modes1, : self.modes2, -self.modes3 :
        ]
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = output_mul[
            :, :, : self.modes1, -self.modes2 :, : self.modes3
        ]
        out_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :] = output_mul[
            :, :, : self.modes1, -self.modes2 :, -self.modes3 :
        ]
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = output_mul[
            :, :, -self.modes1 :, : self.modes2, : self.modes3
        ]
        out_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :] = output_mul[
            :, :, : -self.modes1 :, : self.modes2, -self.modes3 :
        ]
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = output_mul[
            :, :, -self.modes1 :, -self.modes2 :, : self.modes3
        ]
        out_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :] = output_mul[
            :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :
        ]

        # Reshape output such that inverse FFTs can be applied to the dual pairs.
        out_ft = out_ft.reshape(B, I, -1, *out_ft.shape[-3:])
        B_dim, I_dim, C_dim, *D_dims = range(len(out_ft.shape))
        out_ft = out_ft.permute(B_dim, C_dim, *D_dims, I_dim)

        out_dual_1 = torch.view_as_complex(torch.stack((out_ft[..., 0], out_ft[..., 7]), dim=-1))
        out_dual_2 = torch.view_as_complex(torch.stack((out_ft[..., 1], out_ft[..., 6]), dim=-1))
        out_dual_3 = torch.view_as_complex(torch.stack((out_ft[..., 2], out_ft[..., 5]), dim=-1))
        out_dual_4 = torch.view_as_complex(torch.stack((out_ft[..., 3], out_ft[..., 4]), dim=-1))
        dual_1_ifft = torch.fft.ifftn(out_dual_1, s=(out_dual_1.size(-3), out_dual_1.size(-2), out_dual_1.size(-1)))
        dual_2_ifft = torch.fft.ifftn(out_dual_2, s=(out_dual_2.size(-3), out_dual_2.size(-2), out_dual_2.size(-1)))
        dual_3_ifft = torch.fft.ifftn(out_dual_3, s=(out_dual_3.size(-3), out_dual_3.size(-2), out_dual_3.size(-1)))
        dual_4_ifft = torch.fft.ifftn(out_dual_4, s=(out_dual_4.size(-3), out_dual_4.size(-2), out_dual_4.size(-1)))

        # Finally, return to the multivector in the spatial domain.
        output = torch.stack(
            (
                dual_1_ifft.real,
                dual_2_ifft.real,
                dual_3_ifft.real,
                dual_4_ifft.real,
                dual_4_ifft.imag,
                dual_3_ifft.imag,
                dual_2_ifft.imag,
                dual_1_ifft.imag,
            ),
            dim=-1,
        )

        return output
