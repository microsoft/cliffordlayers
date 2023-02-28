# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from cliffordlayers.cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
    get_complex_kernel,
    get_quaternion_kernel,
    get_octonion_kernel,
)
from cliffordlayers.signature import CliffordSignature


def test_complex_kernel():
    """Test Clifford1d kernels against complex kernels using g = [-1]."""
    d_input = 16
    d_output = 24
    weights = torch.randn(2, d_input, d_output)
    complex_kernel = get_complex_kernel(weights)
    _, clifford_kernel = get_1d_clifford_kernel(weights, CliffordSignature([-1]).g)
    torch.testing.assert_close(complex_kernel, clifford_kernel)


def test_quaternion_kernel():
    """Test Clifford2d kernels against quaternion kernels using g = [-1, -1]."""
    d_input = 16
    d_output = 24
    weights = torch.rand(4, d_input, d_output)
    quaternion_kernel = get_quaternion_kernel(weights)
    _, clifford_kernel = get_2d_clifford_kernel(weights, CliffordSignature([-1, -1]).g)
    torch.testing.assert_close(quaternion_kernel, clifford_kernel)


def test_octonion_kernel():
    """Test Clifford3d kernels against octonion kernels using g = [-1, -1, -1]."""
    d_input = 16
    d_output = 24
    weights = torch.rand(8, d_input, d_output)
    octonion_kernel = get_octonion_kernel(weights)
    _, clifford_kernel = get_3d_clifford_kernel(weights, CliffordSignature([-1, -1, -1]).g)
    torch.testing.assert_close(octonion_kernel, clifford_kernel)
