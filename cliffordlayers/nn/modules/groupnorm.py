# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch

from ..functional.groupnorm import clifford_group_norm, complex_group_norm
from .batchnorm import _CliffordBatchNorm, _ComplexBatchNorm


class _ComplexGroupNorm(_ComplexBatchNorm):
    def __init__(
        self,
        num_groups: int,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
    ):
        self.num_groups = num_groups
        super().__init__(
            int(channels / num_groups),
            eps,
            momentum,
            affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return complex_group_norm(
            x,
            self.num_groups,
            self.running_mean,
            self.running_cov,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self):
        return (
            "{num_groups}, {channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )


class ComplexGroupNorm1d(_ComplexGroupNorm):
    """Complex-valued group normalization for 2D or 3D data.

    The input complex-valued data is expected to be at least 2d, with shape `(B, C, D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension (if present).
    """

    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input).")


class ComplexGroupNorm2d(_ComplexGroupNorm):
    """Complex-valued group normalization for 4 data.

    The input complex-valued data is expected to be 4d, with shape `(B, C, *D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 2 dimensions.
    """

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input).")


class ComplexGroupNorm3d(_ComplexGroupNorm):
    """Complex-valued group normalization for 5 data.

    The input complex-valued data is expected to be 5d, with shape `(B, C, *D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 3 dimensions.
    """

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input).")


class _CliffordGroupNorm(_CliffordBatchNorm):
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        num_groups: int,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
    ):
        self.num_groups = num_groups
        super().__init__(
            g,
            int(channels / num_groups),
            eps,
            momentum,
            affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return clifford_group_norm(
            x,
            self.n_blades,
            self.num_groups,
            self.running_mean,
            self.running_cov,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self):
        return (
            "{num_groups}, {channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )


class CliffordGroupNorm1d(_CliffordGroupNorm):
    """Clifford group normalization for 2D or 3D data.

    The input data is expected to be at least 3d, with shape `(B, C, D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension (if present).
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 3 and x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")


class CliffordGroupNorm2d(_CliffordGroupNorm):
    """Clifford group normalization for 4D data.

    The input data is expected to be 4D, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 2 dimensions.
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 5:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")


class CliffordGroupNorm3d(_CliffordGroupNorm):
    """Clifford group normalization for 4D data.

    The input data is expected to be 5D, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 3 dimensions.
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 6:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")
