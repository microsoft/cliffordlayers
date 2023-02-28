# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# THIS IS AN OLD IMPLEMENTATION OF THE CLIFFORD FOURIER TRANSFORM LAYERS.                 #
# WE KEEP IT FOR UNIT TESTING FOR THE TIME BEING.                                         #
###########################################################################################
import torch
from torch import nn


def batchmul2d(input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)


def batchmul3d(input, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


def get_clifford_linear_kernel_2d(weights):
    assert len(weights) == 4 or weights.size(0) == 4
    kernel1 = torch.cat([weights[0], weights[1], weights[2], -weights[3]], dim=0)
    kernel2 = torch.cat([weights[1], weights[0], -weights[3], weights[2]], dim=0)
    kernel3 = torch.cat([weights[2], weights[3], weights[0], -weights[1]], dim=0)
    kernel4 = torch.cat([weights[3], weights[2], -weights[1], weights[0]], dim=0)
    kernel = torch.cat([kernel1, kernel2, kernel3, kernel4], dim=1)
    return kernel


def get_clifford_linear_kernel_3d(weights):
    kernel1 = torch.cat(
        [
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            -weights[4],
            -weights[5],
            -weights[6],
            -weights[7],
        ],
        dim=0,
    )
    kernel2 = torch.cat(
        [
            weights[1],
            weights[0],
            -weights[4],
            -weights[5],
            weights[2],
            weights[3],
            -weights[7],
            -weights[6],
        ],
        dim=0,
    )
    kernel3 = torch.cat(
        [
            weights[2],
            weights[4],
            weights[0],
            -weights[6],
            -weights[1],
            weights[7],
            weights[3],
            weights[5],
        ],
        dim=0,
    )
    kernel4 = torch.cat(
        [
            weights[3],
            weights[5],
            weights[6],
            weights[0],
            -weights[7],
            -weights[1],
            -weights[2],
            -weights[4],
        ],
        dim=0,
    )
    kernel5 = torch.cat(
        [
            weights[4],
            weights[2],
            -weights[1],
            weights[7],
            weights[0],
            -weights[6],
            weights[5],
            weights[3],
        ],
        dim=0,
    )
    kernel6 = torch.cat(
        [
            weights[5],
            weights[3],
            -weights[7],
            -weights[1],
            weights[6],
            weights[0],
            -weights[4],
            -weights[2],
        ],
        dim=0,
    )
    kernel7 = torch.cat(
        [
            weights[6],
            weights[7],
            weights[3],
            -weights[2],
            -weights[5],
            weights[4],
            weights[0],
            weights[1],
        ],
        dim=0,
    )
    kernel8 = torch.cat(
        [
            weights[7],
            weights[6],
            -weights[5],
            weights[4],
            weights[3],
            -weights[2],
            weights[1],
            weights[0],
        ],
        dim=0,
    )
    kernel = torch.cat([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8], dim=1)
    return kernel


class CliffordSpectralConv2d_deprecated(nn.Module):
    """2d Clifford Fourier transform.
    Performs (i) Clifford Fourier transform over the multivector of 2d Clifford algebras,
    (ii) weight multiplication in the Clifford Fourier space using the geometric product,
    (iii) inverse Clifford Fourier transform.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes to use in the first dimension.
        modes2 (int): Number of Fourier modes to use in the second dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(4, in_channels, out_channels, self.modes1 * 2, self.modes2 * 2, dtype=torch.float32)
        )

    def forward(self, vector: torch.Tensor, spinor: torch.Tensor) -> torch.Tensor:
        # TODO: : should the inputs and outputs be Multivectors?
        B = vector.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        vector_ft = torch.fft.fft2(vector)
        spinor_ft = torch.fft.fft2(spinor)
        multivector_ft = torch.cat(
            (
                spinor_ft.real,
                vector_ft.real,
                vector_ft.imag,
                spinor_ft.imag,
            ),
            dim=1,
        )

        # Clifford Fourier modes
        out_ft = torch.zeros_like(
            multivector_ft,
            dtype=torch.float,
            device=multivector_ft.device,
        )

        input = torch.cat(
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
        # TODO: refactor
        # This is a bit ugly and likely doesn't need this function and should be using something from `cliffordkernels`
        kernel = get_clifford_linear_kernel_2d(self.weights)
        output = batchmul2d(input, kernel)
        out_ft[:, :, : self.modes1, : self.modes2] = output[:, :, : self.modes1, : self.modes2]
        out_ft[:, :, -self.modes1 :, : self.modes2] = output[:, :, -self.modes1 :, : self.modes2]
        out_ft[:, :, : self.modes1, -self.modes2 :] = output[:, :, : self.modes1, -self.modes2 :]
        out_ft[:, :, -self.modes1 :, -self.modes2 :] = output[:, :, -self.modes1 :, -self.modes2 :]

        out_ft = out_ft.reshape(out_ft.size(0), 4, -1, *out_ft.shape[-2:])
        out_vector_ft = torch.complex(out_ft[:, 1], out_ft[:, 2])
        out_spinor_ft = torch.complex(out_ft[:, 0], out_ft[:, 3])
        # Return to physical space
        vector = torch.fft.ifft2(out_vector_ft, s=(vector.size(-2), vector.size(-1)))
        spinor = torch.fft.ifft2(out_spinor_ft, s=(spinor.size(-2), spinor.size(-1)))
        return vector, spinor


class CliffordSpectralConv3d_deprecated(nn.Module):
    """3d Clifford Fourier transform.
    Performs (i) Clifford Fourier transform over the multivector of 3d Clifford algebras,
    (ii) weight multiplication in the Clifford Fourier space using the geometric product,
    (iii) inverse Clifford Fourier transform.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes to use in the first dimension.
        modes2 (int): Number of Fourier modes to use in the second dimension.
        modes3 (int): Number of Fourier modes to use in the third dimension.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        modes1: int, 
        modes2: int, 
        modes3: int
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.rand(
                8,
                in_channels,
                out_channels,
                self.modes1 * 2,
                self.modes2 * 2,
                self.modes3 * 2,
                dtype=torch.float32,
            )
        )

    def forward(
        self, dual_pair_1: torch.Tensor, dual_pair_2: torch.Tensor, dual_pair_3: torch.Tensor, dual_pair_4: torch.Tensor
    ) -> torch.Tensor:
        # TODO: should the inputs and outputs be Multivectors?
        x_dual_pair_1_ft = torch.fft.fftn(dual_pair_1, dim=[-3, -2, -1])
        x_dual_pair_2_ft = torch.fft.fftn(dual_pair_2, dim=[-3, -2, -1])
        x_dual_pair_3_ft = torch.fft.fftn(dual_pair_3, dim=[-3, -2, -1])
        x_dual_pair_4_ft = torch.fft.fftn(dual_pair_4, dim=[-3, -2, -1])
        multivector_ft = torch.stack(
            (
                x_dual_pair_1_ft.real,
                x_dual_pair_2_ft.real,
                x_dual_pair_3_ft.real,
                x_dual_pair_4_ft.real,
                x_dual_pair_4_ft.imag,
                x_dual_pair_3_ft.imag,
                x_dual_pair_2_ft.imag,
                x_dual_pair_1_ft.imag,
            ),
            dim=1,
        )

        # Clifford Fourier modes
        out_ft = torch.zeros_like(
            multivector_ft,
            dtype=torch.float,
            device=multivector_ft.device,
        )

        input = torch.cat(
            (
                torch.cat(
                    (
                        torch.cat(
                            (
                                multivector_ft[:, :, :, : self.modes1, : self.modes2, : self.modes3],
                                multivector_ft[:, :, :, : self.modes1, : self.modes2, -self.modes3 :],
                            ),
                            -1,
                        ),
                        torch.cat(
                            (
                                multivector_ft[:, :, :, : self.modes1, -self.modes2 :, : self.modes3],
                                multivector_ft[:, :, :, : self.modes1, -self.modes2 :, -self.modes3 :],
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
                                multivector_ft[:, :, :, -self.modes1 :, : self.modes2, : self.modes3],
                                multivector_ft[:, :, :, -self.modes1 :, : self.modes2, -self.modes3 :],
                            ),
                            -1,
                        ),
                        torch.cat(
                            (
                                multivector_ft[:, :, :, -self.modes1 :, -self.modes2 :, : self.modes3],
                                multivector_ft[:, :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :],
                            ),
                            -1,
                        ),
                    ),
                    -2,
                ),
            ),
            -3,
        )
        kernel = get_clifford_linear_kernel_3d(self.weights)
        bs = input.size(0)
        out = batchmul3d(input.reshape(bs, -1, *input.size()[3:]), kernel)
        output = out.reshape(bs, 8, -1, *out.shape[-3:])
        
        out_ft[:, :, :, : self.modes1, : self.modes2, : self.modes3] = output[
            :, :, :, : self.modes1, : self.modes2, : self.modes3
        ]
        out_ft[:, :, :, : self.modes1, : self.modes2, -self.modes3 :] = output[
            :, :, :, : self.modes1, : self.modes2, -self.modes3 :
        ]
        out_ft[:, :, :, : self.modes1, -self.modes2 :, : self.modes3] = output[
            :, :, :, : self.modes1, -self.modes2 :, : self.modes3
        ]
        out_ft[:, :, :, : self.modes1, -self.modes2 :, -self.modes3 :] = output[
            :, :, :, : self.modes1, -self.modes2 :, -self.modes3 :
        ]
        out_ft[:, :, :, -self.modes1 :, : self.modes2, : self.modes3] = output[
            :, :, :, -self.modes1 :, : self.modes2, : self.modes3
        ]
        out_ft[:, :, :, -self.modes1 :, : self.modes2, -self.modes3 :] = output[
            :, :, :, : -self.modes1 :, : self.modes2, -self.modes3 :
        ]
        out_ft[:, :, :, -self.modes1 :, -self.modes2 :, : self.modes3] = output[
            :, :, :, -self.modes1 :, -self.modes2 :, : self.modes3
        ]
        out_ft[:, :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :] = output[
            :, :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :
        ]

        out_x_dual_pair_1_ft = torch.complex(out_ft[:, 0], out_ft[:, 7])
        out_x_dual_pair_2_ft = torch.complex(out_ft[:, 1], out_ft[:, 6])
        out_x_dual_pair_3_ft = torch.complex(out_ft[:, 2], out_ft[:, 5])
        out_x_dual_pair_4_ft = torch.complex(out_ft[:, 3], out_ft[:, 4])
        # Return to physical space
        out_x_dual_pair_1 = torch.fft.ifftn(
            out_x_dual_pair_1_ft,
            s=(dual_pair_1.size(-3), dual_pair_1.size(-2), dual_pair_1.size(-1)),
        )
        out_x_dual_pair_2 = torch.fft.ifftn(
            out_x_dual_pair_2_ft,
            s=(dual_pair_2.size(-3), dual_pair_2.size(-2), dual_pair_2.size(-1)),
        )
        out_x_dual_pair_3 = torch.fft.ifftn(
            out_x_dual_pair_3_ft,
            s=(dual_pair_3.size(-3), dual_pair_3.size(-2), dual_pair_3.size(-1)),
        )
        out_x_dual_pair_4 = torch.fft.ifftn(
            out_x_dual_pair_4_ft,
            s=(dual_pair_4.size(-3), dual_pair_4.size(-2), dual_pair_4.size(-1)),
        )
        return out_x_dual_pair_1, out_x_dual_pair_2, out_x_dual_pair_3, out_x_dual_pair_4