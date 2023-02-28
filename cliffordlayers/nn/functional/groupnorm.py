# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union

import torch
import torch.nn as nn

from .batchnorm import clifford_batch_norm, complex_batch_norm


def complex_group_norm(
    x: torch.Tensor,
    num_groups: int = 1,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
):
    """Group normalization for complex-valued tensors.

    Args:
        x (torch.Tensor): The input complex-valued data is expected to be at least 2d, with
                          shape `(B, C, *D)`, where `B` is the batch dimension, `C` the
                          channels/features, and *D the remaining dimensions (if present).

        num_groups (int): Number of groups for which normalization is calculated. Defaults to 1.
                          For `num_groups == 1`, it effectively applies complex-valued layer normalization;
                          for `num_groups == C`, it effectively applies complex-valued instance normalization.

        running_mean (torch.Tensor, optional): The tensor with running mean statistics having shape `(2, C / num_groups)`. Defaults to None.
        running_cov (torch.Tensor, optional): The tensor with running real-imaginary covariance statistics having shape `(2, 2, C / num_groups)`. Defaults to None.

        weight (Union[torch.Tensor, nn.Parameter], optional): Additional weight tensor which is applied post normalization, and has the shape `(2, 2, C/ num_groups)`. Defaults to None.

        bias (Union[torch.Tensor, nn.Parameter], optional): Additional bias tensor which is applied post normalization, and has the shape `(2, C / num_groups)`. Defaults to None.

        training (bool, optional): Whether to use the running mean and variance. Defaults to True.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to 0.1.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to 1e-05.

    Returns:
        (torch.Tensor): Normalized input as complex tensor of shape `(B, C, *D)`.
    """

    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    B, C, *D = x.shape
    assert C % num_groups == 0, "Number of channels should be evenly divisible by the number of groups."
    assert num_groups <= C
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (2, 2, int(C / num_groups))
        assert bias.shape == (2, int(C / num_groups))
        weight = weight.repeat(1, 1, B)
        bias = bias.repeat(1, B)

    def _instance_norm(
        x,
        num_groups,
        running_mean,
        running_cov,
        weight,
        bias,
        training,
        momentum,
        eps,
    ):
        if running_mean is not None and running_cov is not None:
            assert running_mean.shape == (2, int(C / num_groups))
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(1, B)
            assert running_cov.shape == (2, 2, int(C / num_groups))
            running_cov_orig = running_cov
            running_cov = running_cov_orig.repeat(1, 1, B)

        # Reshape such that batch normalization can be applied.
        # For num_groups == 1, it defaults to layer normalization,
        # for num_groups == C, it defaults to instance normalization.
        x_reshaped = x.view(1, int(B * C / num_groups), num_groups, *D)

        x_norm = complex_batch_norm(
            x_reshaped,
            running_mean,
            running_cov,
            weight=weight,
            bias=bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )

        # Reshape back running mean and running var.
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(2, B, int(C / num_groups)).mean(1, keepdim=False))
        if running_cov is not None:
            running_cov_orig.copy_(running_cov.view(2, 2, B, int(C / num_groups)).mean(2, keepdim=False))

        return x_norm.view(B, C, *D)

    return _instance_norm(
        x,
        num_groups,
        running_mean,
        running_cov,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def clifford_group_norm(
    x: torch.Tensor,
    n_blades: int,
    num_groups: int = 1,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> torch.Tensor:
    """Clifford group normalization

    Args:
        x (torch.Tensor): Input tensor of shape `(B, C, *D, I)` where I is the blade of the algebra.

        n_blades (int): Number of blades of the Clifford algebra.

        num_groups (int): Number of groups for which normalization is calculated. Defaults to 1.
                          For `num_groups == 1`, it effectively applies Clifford layer normalization, for `num_groups == C`, it effectively applies Clifford instance normalization.

        running_mean (torch.Tensor, optional): The tensor with running mean statistics having shape `(I, C / num_groups)`. Defaults to None.
        running_cov (torch.Tensor, optional): The tensor with running real-imaginary covariance statistics having shape `(I, I, C / num_groups)`. Defaults to None.

        weight (Union[torch.Tensor, nn.Parameter], optional): Additional weight tensor which is applied post normalization, and has the shape `(I, I, C / num_groups)`. Defaults to None.

        bias (Union[torch.Tensor, nn.Parameter], optional): Additional bias tensor which is applied post normalization, and has the shape `(I, C / num_groups)`. Defaults to None.

        training (bool, optional): Whether to use the running mean and variance. Defaults to True.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to 0.1.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to 1e-05.

    Returns:
        (torch.Tensor): Group normalized input of shape `(B, C, *D, I)`.
    """

    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    B, C, *D, I = x.shape
    assert num_groups <= C
    assert C % num_groups == 0, "Number of channels should be evenly divisible by the number of groups."
    assert I == n_blades
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (I, I, int(C / num_groups))
        assert bias.shape == (I, int(C / num_groups))
        weight = weight.repeat(1, 1, B)
        bias = bias.repeat(1, B)

    def _instance_norm(
        x,
        num_groups,
        running_mean,
        running_cov,
        weight,
        bias,
        training,
        momentum,
        eps,
    ):

        if running_mean is not None and running_cov is not None:
            assert running_mean.shape == (I, int(C / num_groups))
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(1, B)
            assert running_cov.shape == (I, I, int(C / num_groups))
            running_cov_orig = running_cov
            running_cov = running_cov_orig.repeat(1, 1, B)

        # Reshape such that batch normalization can be applied.
        # For num_groups == 1, it defaults to layer normalization,
        # for num_groups == C, it defaults to instance normalization.
        x_reshaped = x.reshape(1, int(B * C / num_groups), num_groups, *D, I)

        x_norm = clifford_batch_norm(
            x_reshaped,
            n_blades,
            running_mean,
            running_cov,
            weight,
            bias,
            training,
            momentum,
            eps,
        )

        # Reshape back running mean and running var.
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(I, B, int(C / num_groups)).mean(1, keepdim=False))
        if running_cov is not None:
            running_cov_orig.copy_(running_cov.view(I, I, B, int(C / num_groups)).mean(1, keepdim=False))

        return x_norm.view(B, C, *D, I)

    return _instance_norm(
        x,
        num_groups,
        running_mean,
        running_cov,
        weight,
        bias,
        training,
        momentum,
        eps,
    )
