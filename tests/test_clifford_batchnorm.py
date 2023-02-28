# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from cliffordlayers.nn.functional.batchnorm import (
    clifford_batch_norm,
    complex_batch_norm,
)
from cliffordlayers.nn.modules.batchnorm import (
    CliffordBatchNorm1d,
    CliffordBatchNorm2d,
    CliffordBatchNorm3d,
    ComplexBatchNorm1d,
)
from cliffordlayers.signature import CliffordSignature


def test_clifford_batchnorm1d_vs_complex_batchnorm():
    """Test Clifford1d batchnorm function against complex batchnorm function using g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
    )
    x_norm_complex = complex_batch_norm(torch.view_as_complex(x))

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_batchnorm1d_vs_complex_batchnorm_scaled():
    """Test Clifford1d batchnorm function against complex batchnorm function using g = [-1],
    where an affine transformation is applied.
    """
    x = torch.randn(4, 16, 8, 2)
    w = torch.randn(2, 2, 16)
    b = torch.randn(2, 16)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        weight=w,
        bias=b,
    )
    x_norm_complex = complex_batch_norm(
        torch.view_as_complex(x),
        weight=w,
        bias=b,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_batchnorm1d_vs_complex_batchnorm_scaled_validation():
    """Test Clifford1d batchnorm function against complex batchnorm function in the validation setting using g = [-1],
    where an affine transformation is applied.
    """
    x = torch.randn(4, 16, 8, 2)
    w = torch.randn(2, 2, 16)
    b = torch.randn(2, 16)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        weight=w,
        bias=b,
        training=False,
    )
    x_norm_complex = complex_batch_norm(
        torch.view_as_complex(x),
        weight=w,
        bias=b,
        training=False,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_batchnorm1d_vs_complex_batchnorm_running_mean():
    """Test Clifford1d batchnorm function against complex batchnorm function using g = [-1],
    where running mean is provided.
    """
    x = torch.randn(4, 16, 8, 2)
    mean = torch.randn(2, 16)
    # For the running covariance matrix, we need a positive definite form.
    X = torch.randn(16, 2, 2)
    cov = X @ X.mT
    cov = cov.add_(torch.eye(2)).permute(1, 2, 0)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature(
            [
                -1,
            ]
        ).n_blades,
        running_mean=mean,
        running_cov=cov,
        training=True,
    )
    x_norm_complex = complex_batch_norm(
        torch.view_as_complex(x),
        running_mean=mean,
        running_cov=cov,
        training=True,
    )

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_modules_clifford_batchnorm1d_vs_complex_batchnorm1d():
    """Test Clifford1d batchnorm module against complex batchnorm module using g = [-1]."""
    x = torch.randn(4, 16, 8, 2)
    complex_norm = ComplexBatchNorm1d(
        channels=16,
    )
    x_norm_complex = complex_norm(torch.view_as_complex(x))
    clifford_norm = CliffordBatchNorm1d(
        [
            -1,
        ],
        channels=16,
    )
    x_norm_clifford = clifford_norm(x)

    torch.testing.assert_close(x_norm_clifford, torch.view_as_real(x_norm_complex))


def test_clifford_batchnorm2d():
    """Test Clifford2d batchnorm function for correct outputs using g = [1, 1]."""
    x = torch.randn(4, 16, 8, 4)
    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature([1, 1]).n_blades,
    )

    assert x_norm_clifford.shape == x.shape


def test_clifford_batchnorm2d_scaled():
    """Test Clifford2d batchnorm function for correct outputs using g = [1, 1],
    where an affine transformation is applied.
    """
    x = torch.randn(4, 16, 8, 4)
    w = torch.randn(4, 4, 16)
    b = torch.randn(4, 16)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature([1, 1]).n_blades,
        weight=w,
        bias=b,
    )

    assert x_norm_clifford.shape == x.shape


def test_clifford_batchnorm3d():
    """Test Clifford3d batchnorm function for correct outputs using g = [1, 1, 1]."""
    x = torch.randn(4, 16, 32, 32, 32, 8)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature([1, 1, 1]).n_blades,
    )

    assert x_norm_clifford.shape == x.shape


def test_clifford_batchnorm3d_scaled():
    """Test Clifford3d batchnorm function for correct outputs using g = [1, 1, 1],
    where an affine transformation is applied.
    """
    x = torch.randn(4, 16, 32, 32, 32, 8)
    w = torch.randn(8, 8, 16)
    b = torch.randn(8, 16)

    x_norm_clifford = clifford_batch_norm(
        x,
        CliffordSignature([1, 1, 1]).n_blades,
        weight=w,
        bias=b,
    )

    assert x_norm_clifford.shape == x.shape


def test_module_clifford_batchnorm2d():
    """Test Clifford2d batchnorm module for correct outputs using g = [1, 1]."""
    x = torch.randn(4, 16, 64, 64, 4)
    clifford_norm = CliffordBatchNorm2d(
        [-1, -1],
        channels=16,
    )
    x_norm_clifford = clifford_norm(x)

    assert x.shape == x_norm_clifford.shape


def test_module_clifford_batchnorm3d():
    """Test Clifford3d batchnorm module for correct outputs using g = [1, 1]."""
    x = torch.randn(4, 16, 64, 64, 64, 8)
    clifford_norm = CliffordBatchNorm3d(
        [-1, -1, -1],
        channels=16,
    )
    x_norm_clifford = clifford_norm(x)

    assert x.shape == x_norm_clifford.shape
