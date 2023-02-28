# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from cliffordlayers.cliffordkernels import get_2d_clifford_rotation_kernel, get_quaternion_rotation_kernel
from cliffordlayers.signature import CliffordSignature


def test_2d_clifford_rotation_kernel():
    """Test Clifford2d rotational kernels against quaternion kernels using g = [-1, -1].
    The scalar parts of the Clifford2d rotational kernel are ignored.
    """
    in_channels = 8
    out_channels = 16
    weights = [
        torch.randn(3, 4),
        torch.randn(3, 4),
        torch.randn(3, 4),
        torch.randn(3, 4),
    ]
    # Append scaling weights.
    weights.append(torch.randn(3, 4))
    # Append zero weights.
    weights.append(torch.zeros(3, 4))
    quaternion_kernel = get_quaternion_rotation_kernel(weights)
    _, clifford_kernel = get_2d_clifford_rotation_kernel(weights, CliffordSignature([-1, -1]).g)
    torch.testing.assert_close(quaternion_kernel, clifford_kernel[3:])
