# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch
import torch.nn as nn
from torch.nn import init

from ...signature import CliffordSignature
from ..functional.batchnorm import clifford_batch_norm, complex_batch_norm


class _ComplexBatchNorm(nn.Module):
    def __init__(
        self, channels: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats=True
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, channels))
            self.bias = torch.nn.Parameter(torch.empty(2, channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(2, channels))
            self.register_buffer("running_cov", torch.empty(2, 2, channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_cov", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_running_stats()
        self.reset_parameters()

    def reset_running_stats(self):
        if not self.track_running_stats:
            return

        self.num_batches_tracked.zero_()
        self.running_mean.zero_()
        self.running_cov.copy_(torch.eye(2, 2).unsqueeze(-1))

    def reset_parameters(self):
        if not self.affine:
            return

        self.weight.data.copy_(torch.eye(2, 2).unsqueeze(-1))
        init.zeros_(self.bias)

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    # use exponential moving average
                    exponential_average_factor = self.momentum

        return complex_batch_norm(
            x,
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
            "{channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )


class ComplexBatchNorm1d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 2D or 3D data.

    The input complex-valued data is expected to be at least 2d, with shape `(B, C, D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension (if present).
    See [torch.nn.BatchNorm1d][] for details.
    """

    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input).")


class ComplexBatchNorm2d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 4D data.

    The input complex-valued data is expected to be 4d, with shape `(B, C, *D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 2 dimensions.
    See [torch.nn.BatchNorm2d][] for details.
    """

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input).")


class ComplexBatchNorm3d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 5D data.

    The input complex-valued data is expected to be 5d, with shape `(B, C, *D)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining 3 dimensions.
    See [torch.nn.BatchNorm3d][] for details.
    """

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (got {x.dim()}D input).")


class _CliffordBatchNorm(nn.Module):
    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        sig = CliffordSignature(g)
        self.g = sig.g
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(self.n_blades, self.n_blades, channels))
            self.bias = torch.nn.Parameter(torch.empty(self.n_blades, channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(self.n_blades, channels))
            self.register_buffer("running_cov", torch.empty(self.n_blades, self.n_blades, channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_cov", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_running_stats()
        self.reset_parameters()

    def reset_running_stats(self):
        if not self.track_running_stats:
            return

        self.num_batches_tracked.zero_()
        self.running_mean.zero_()
        self.running_cov.copy_(torch.eye(self.n_blades, self.n_blades).unsqueeze(-1))

    def reset_parameters(self):
        if not self.affine:
            return

        self.weight.data.copy_(torch.eye(self.n_blades, self.n_blades).unsqueeze(-1))
        init.zeros_(self.bias)

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    # Use cumulative moving average.
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    # Use exponential moving average.
                    exponential_average_factor = self.momentum

        return clifford_batch_norm(
            x,
            self.n_blades,
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
            "{channels}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**vars(self))
        )


class CliffordBatchNorm1d(_CliffordBatchNorm):
    """Clifford batch normalization for 2D or 3D data.

    The input data is expected to be at least 3d, with shape `(B, C, D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension (if present).
    See [torch.nn.BatchNorm1d] for details.
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 3 and x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")


class CliffordBatchNorm2d(_CliffordBatchNorm):
    """Clifford batch normalization for 4D data.

    The input data is expected to be 4d, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension 2 dimensions.
    See [torch.nn.BatchNorm2d][] for details.
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 5:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")


class CliffordBatchNorm3d(_CliffordBatchNorm):
    """Clifford batch normalization for 5D data.
    The input data is expected to be 5d, with shape `(B, C, *D, I)`,
    where `B` is the batch dimension, `C` the channels/features, and D the remaining dimension 3 dimensions.
    See [torch.nn.BatchNorm2d][] for details.
    """

    def _check_input_dim(self, x):
        *_, I = x.shape
        if not I == self.n_blades:
            raise ValueError(f"Wrong number of Clifford blades. Expected {self.n_blades} blades, but {I} were given.")
        if x.dim() != 6:
            raise ValueError(f"Expected 3D or 4D input (got {x.dim()}D input).")
