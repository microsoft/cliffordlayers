# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import torch

from cliffordlayers.nn.functional.groupnorm import (
    clifford_group_norm,
    complex_group_norm,
)
from cliffordlayers.nn.modules.groupnorm import (
    CliffordGroupNorm1d,
    CliffordGroupNorm2d,
    CliffordGroupNorm3d,
    ComplexGroupNorm1d,
)
from cliffordlayers.signature import CliffordSignature


def test_clifford_instance_norm1d_vs_complex_instance_norm():
    """Test Clifford1d groupnorm function against complex groupnorm function using num_groups=1 and g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    x_norm_clifford = clifford_group_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        num_groups=1,
    )
    x_norm_complex = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=1,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_layer_norm1d_vs_complex_layer_norm():
    """Test Clifford1d groupnorm function against complex groupnorm function using num_groups=channels and g = [-1]."""
    channels = 16
    x = torch.randn(4, channels, 8, 2)
    x_norm_clifford = clifford_group_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        num_groups=channels,
    )
    x_norm_complex = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=channels,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_groupnorm1d_vs_complex_groupnorm_scaled():
    """Test Clifford1d groupnorm function against complex groupnorm function using num_groups=2 and g = [-1],
    where an affine transformation is applied.
    """
    channels = 16
    num_groups = 2
    x = torch.randn(4, channels, 8, 2)
    w = torch.randn(2, 2, int(channels / num_groups))
    b = torch.randn(2, int(channels / num_groups))

    x_norm_clifford = clifford_group_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        num_groups=num_groups,
        weight=w,
        bias=b,
    )
    x_norm_complex = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=num_groups,
        weight=w,
        bias=b,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_groupnorm1d_vs_complex_groupnorm_scaled_validation():
    """Test Clifford1d groupnorm function against complex groupnorm function in the validation setting using num_groups=2 and g = [-1],
    where an affine transformation is applied.
    """
    channels = 16
    num_groups = 2
    x = torch.randn(4, channels, 8, 2)
    w = torch.randn(2, 2, int(channels / num_groups))
    b = torch.randn(2, int(channels / num_groups))

    x_norm_clifford = clifford_group_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        num_groups=num_groups,
        weight=w,
        bias=b,
        training=False,
    )
    x_norm_complex = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=num_groups,
        weight=w,
        bias=b,
        training=False,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_complex_groupnorm_valid():
    """Test complex group norm implementation for num_groups=8:
    validation setting where running_mean and running_cov are provided
    is tested against training setting where exactly this running_mean and running_cov should be calculated.
    """
    channels = 16
    num_groups = 8
    x = torch.randn(1, channels, 64, 2)
    B, C, *D, I = x.shape

    # Now reshape x as done in the group norm routine and calculate mean and covariance accordingly.
    x_r = x.view(1, int(B * C / num_groups), num_groups, *D, I)
    B, C, *D, I = x_r.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x_r.shape))
    shape = 1, C, *([1] * (x_r.dim() - 3))
    mean = x_r.mean(dim=(B_dim, *D_dims))
    x_mean = x_r - mean.reshape(*shape, I)
    X = x_mean.permute(C_dim, I_dim, B_dim, *D_dims).flatten(2, -1)
    cov = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]

    assert mean.shape == (int(channels / num_groups), 2)
    assert cov.shape == (int(channels / num_groups), 2, 2)

    x_norm_valid = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=num_groups,
        running_mean=mean.permute(1, 0),
        running_cov=cov.permute(1, 2, 0),
        training=False,
    )

    x_norm = complex_group_norm(
        torch.view_as_complex(x),
        num_groups=num_groups,
    )

    torch.testing.assert_close(x_norm, x_norm_valid)


def get_mean_cov(x: torch.Tensor, num_groups: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Getting mean and covariance tensor for arbitrary Clifford algebras.

    Args:
        x (torch.Tensor): Input tensor of shape `(B, C, *D, I)` where I is the blade of the algebra.
        num_groups (int): Number of groups.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and covariance tensors of shapes `(I, C/num_groups)` and `(I, I, C/num_groups)`.
    """
    B, C, *D, I = x.shape

    # Now reshape x as done in the group norm routine and calculate mean and covariance accordingly.
    x = x.view(1, int(B * C / num_groups), num_groups, *D, I)
    B, C, *D, I = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    shape = 1, C, *([1] * (x.dim() - 3))
    mean = x.mean(dim=(B_dim, *D_dims))
    x_mean = x - mean.reshape(*shape, I)
    X = x_mean.permute(C_dim, I_dim, B_dim, *D_dims).flatten(2, -1)
    cov = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]

    return mean, cov


def test_clifford_groupnorm1d_valid():
    """Test Clifford1d group norm implementation for num_groups=8 and g=[1]:
    validation setting where running_mean and running_cov are provided
    is tested against training setting where exactly this running_mean and running_cov should be calculated.
    """
    channels = 16
    num_groups = 8
    x = torch.randn(1, channels, 64, 2)

    mean, cov = get_mean_cov(x, num_groups)
    assert mean.shape == (int(channels / num_groups), 2)
    assert cov.shape == (int(channels / num_groups), 2, 2)

    x_norm_valid = clifford_group_norm(
        x,
        CliffordSignature([1]).n_blades,
        num_groups=num_groups,
        running_mean=mean.permute(1, 0),
        running_cov=cov.permute(1, 2, 0),
        training=False,
    )

    x_norm = clifford_group_norm(
        x,
        CliffordSignature([1]).n_blades,
        num_groups=num_groups,
    )

    torch.testing.assert_close(x_norm, x_norm_valid)


def test_clifford_groupnorm2d_valid():
    """Test Clifford2d group norm implementation for num_groups=4 and g=[1, 1]:
    validation setting where running_mean and running_cov are provided
    is tested against training setting where exactly this running_mean and running_cov should be calculated.
    """
    channels = 32
    num_groups = 4
    x = torch.randn(1, channels, 64, 64, 4)

    mean, cov = get_mean_cov(x, num_groups)
    assert mean.shape == (int(channels / num_groups), 4)
    assert cov.shape == (int(channels / num_groups), 4, 4)

    x_norm_valid = clifford_group_norm(
        x,
        CliffordSignature([1, 1]).n_blades,
        num_groups=num_groups,
        running_mean=mean.permute(1, 0),
        running_cov=cov.permute(1, 2, 0),
        training=False,
    )

    x_norm = clifford_group_norm(
        x,
        CliffordSignature([1, 1]).n_blades,
        num_groups=num_groups,
    )

    torch.testing.assert_close(x_norm, x_norm_valid)


def test_clifford_groupnorm3d_valid():
    """Test Clifford3d group norm implementation for num_groups=4 and g=[1, 1, 1]:
    validation setting where running_mean and running_cov are provided
    is tested against training setting where exactly this running_mean and running_cov should be calculated.
    """
    channels = 32
    num_groups = 4
    x = torch.randn(1, channels, 32, 32, 32, 8)

    mean, cov = get_mean_cov(x, num_groups)
    assert mean.shape == (int(channels / num_groups), 8)
    assert cov.shape == (int(channels / num_groups), 8, 8)

    x_norm_valid = clifford_group_norm(
        x,
        CliffordSignature([1, 1, 1]).n_blades,
        num_groups=num_groups,
        running_mean=mean.permute(1, 0),
        running_cov=cov.permute(1, 2, 0),
        training=False,
    )

    x_norm = clifford_group_norm(
        x,
        CliffordSignature([1, 1, 1]).n_blades,
        num_groups=num_groups,
    )

    torch.testing.assert_close(x_norm, x_norm_valid)


def test_modules_clifford_groupnorm1d_vs_complex_instancenorm1d():
    """Test Clifford1d groupnorm module against complex groupnorm module using num_groups=1 and g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    complex_norm = ComplexGroupNorm1d(
        channels=16,
        num_groups=1,
    )
    x_norm_complex = complex_norm(torch.view_as_complex(x))
    clifford_norm = CliffordGroupNorm1d(
        [-1],
        channels=16,
        num_groups=1,
    )
    x_norm_clifford = clifford_norm(x)

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_modules_clifford_layernorm1d_vs_complex_layernorm1d():
    """Test Clifford1d groupnorm module against complex groupnorm module using num_groups=num_channels and g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    complex_norm = ComplexGroupNorm1d(
        channels=16,
        num_groups=16,
    )
    x_norm_complex = complex_norm(torch.view_as_complex(x))
    clifford_norm = CliffordGroupNorm1d(
        [-1],
        channels=16,
        num_groups=16,
    )
    x_norm_clifford = clifford_norm(x)

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_modules_clifford_groupnorm1d_vs_complex_groupnorm1d():
    """Test Clifford1d groupnorm module against complex groupnorm module using num_groups=2 and g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    complex_norm = ComplexGroupNorm1d(
        channels=16,
        num_groups=2,
    )
    x_norm_complex = complex_norm(torch.view_as_complex(x))
    clifford_norm = CliffordGroupNorm1d(
        [-1],
        channels=16,
        num_groups=2,
    )
    x_norm_clifford = clifford_norm(x)

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_module_clifford_groupnorm2d():
    """Test Clifford2d groupnorm module for correct shapes using num_groups=2 and g = [-1, -1]."""
    x = torch.randn(4, 16, 64, 64, 4)
    clifford_norm = CliffordGroupNorm2d(
        [-1, -1],
        num_groups=2,
        channels=16,
    )
    x_norm_clifford = clifford_norm(x)

    assert x.shape == x_norm_clifford.shape


def test_module_clifford_groupnorm3d():
    """Test Clifford3d groupnorm module for correct shapes using num_groups=2 and g = [-1, -1, -1]."""
    x = torch.randn(4, 16, 64, 64, 64, 8)
    clifford_norm = CliffordGroupNorm3d(
        [-1, -1, -1],
        num_groups=8,
        channels=16,
    )
    x_norm_clifford = clifford_norm(x)

    assert x.shape == x_norm_clifford.shape
