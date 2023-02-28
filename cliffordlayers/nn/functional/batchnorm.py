#####################################################################################
# Major ideas of complex batch norm taken from https://github.com/ivannz/cplxmodule 
# MIT License
#####################################################################################
from typing import Optional, Union

import torch
import torch.nn as nn


def whiten_data(
    x: torch.Tensor,
    training: bool = True,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Jointly whiten features in tensors `(B, C, *D, I)`: take n_blades(I)-dim vectors
    and whiten individually for each channel dimension C over `(B, *D)`.
    I is the number of blades in the respective Clifford algebra, e.g. I = 2 for complex numbers.

    Args:
        x (torch.Tensor): The tensor to whiten.
        training (bool, optional): Wheter to update the running mean and covariance. Defaults to `True`.
        running_mean (torch.Tensor, optional): The running mean of shape `(I, C). Defaults to `None`.
        running_cov (torch.Tensor, optional): The running covariance of shape `(I, I, C)` Defaults to `None`.
        momentum (float, optional): The momentum to use for the running mean and covariance. Defaults to `0.1`.
        eps (float, optional): A small number to add to the covariance. Defaults to 1e-5.

    Returns:
        (torch.Tensor): Whitened data of shape `(B, C, *D, I)`.
    """

    assert x.dim() >= 3
    # Get whitening shape of [1, C, ...]
    _, C, *_, I = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    shape = 1, C, *([1] * (x.dim() - 3))

    # Get feature mean.
    if not (running_mean is None or running_mean.shape == (I, C)):
        raise ValueError(f"Running_mean expected to be none, or of shape ({I}, {C}).")
    if training or running_mean is None:
        mean = x.mean(dim=(B_dim, *D_dims))
        if running_mean is not None:
            running_mean += momentum * (mean.data.permute(1, 0) - running_mean)
    else:
        mean = running_mean.permute(1, 0)

    # Get feature covariance.
    x = x - mean.reshape(*shape, I)
    if not (running_cov is None or running_cov.shape == (I, I, C)):
        raise ValueError(f"Running_cov expected to be none, or of shape ({I}, {I}, {C}).")
    if training or running_cov is None:
        # B, C, *D, I -> C, I, B, *D
        X = x.permute(C_dim, I_dim, B_dim, *D_dims).flatten(2, -1)
        # Covariance XX^T matrix of shape C x I x I
        cov = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]
        if running_cov is not None:
            running_cov += momentum * (cov.data.permute(1, 2, 0) - running_cov)

    else:
        cov = running_cov.permute(2, 0, 1)

    # Upper triangle Cholesky decomposition of covariance matrix: U^T U = Cov
    eye = eps * torch.eye(I, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    U = torch.linalg.cholesky(cov + eye).mH
    # Invert Cholesky decomposition, returns tensor of shape [B, C, *D, I]
    x_whiten = torch.linalg.solve_triangular(
        U.reshape(*shape, I, I),
        x.unsqueeze(-1),
        upper=True,
    ).squeeze(-1)
    return x_whiten


def complex_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> torch.Tensor:
    """Applies complex-valued Batch Normalization as described in
    (Trabelsi et al., 2018) for each channel across a batch of data.

    Args:
        x (torch.Tensor): The input complex-valued data is expected to be at least 2d, with shape `(B, C, *D)`, where `B` is the batch dimension, `C` the channels/features, and *D the remaining dimensions (if present).

        running_mean (Union[torch.Tensor, nn.Parameter], optional): The tensor with running mean statistics having shape `(2, C)`.
        running_cov (Union[torch.Tensor, nn.Parameter], optional): The tensor with running real-imaginary covariance statistics having shape `(2, 2, C)`.
        weight (torch.Tensor, optional): Additional weight tensor which is applied post normalization, and has the shape `(2, 2, C)`.
        bias (torch.Tensor, optional): Additional bias tensor which is applied post normalization, and has the shape `(2, C)`.
        training (bool, optional): Whether to use the running mean and variance. Defaults to `True`.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to `0.1`.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to `1e-05`.

    Returns:
        (torch.Tensor): Normalized input as complex tensor of shape `(B, C, *D)`.
    """

    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)
    x = torch.view_as_real(x)
    _, C, *_, I = x.shape
    assert I == 2

    # Whiten and apply affine transformation.
    x_norm = whiten_data(
        x,
        training,
        running_mean,
        running_cov,
        momentum,
        eps,
    )
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (2, 2, C)
        assert bias.shape == (2, C)
        # Unsqueeze weight and bias for each dimension except the channel dimension.
        shape = 1, C, *([1] * (x.dim() - 3))
        weight = weight.reshape(2, 2, *shape)
        # Apply additional affine transformation post normalization.
        weight_idx = list(range(weight.dim()))
        # TODO weight multiplication should be changed to complex product.
        weight = weight.permute(*weight_idx[2:], *weight_idx[:2])
        x_norm = weight.matmul(x_norm[..., None]).squeeze(-1) + bias.reshape(*shape, 2)

    return torch.view_as_complex(x_norm)


def clifford_batch_norm(
    x: torch.Tensor,
    n_blades: int,
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> torch.Tensor:
    """Clifford batch normalization for each channel across a batch of data.

    Args:
        x (torch.Tensor): Input tensor of shape `(B, C, *D, I)` where I is the blade of the algebra.
        n_blades (int): Number of blades of the Clifford algebra.
        running_mean (torch.Tensor, optional): The tensor with running mean statistics having shape `(I, C)`.
        running_cov (torch.Tensor, optional): The tensor with running covariance statistics having shape `(I, I, C)`.
        weight (Union[torch.Tensor, nn.Parameter], optional): Additional weight tensor which is applied post normalization, and has the shape `(I, I, C)`.
        bias (Union[torch.Tensor, nn.Parameter], optional): Additional bias tensor which is applied post normalization, and has the shape `(I, C)`.
        training (bool, optional): Whether to use the running mean and variance. Defaults to True. Defaults to True.
        momentum (float, optional): Momentum for the running mean and variance. Defaults to 0.1.
        eps (float, optional): Epsilon for the running mean and variance. Defaults to 1e-05.

    Returns:
        (torch.Tensor): Normalized input of shape `(B, C, *D, I)`
    """

    # Check arguments.
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    # Whiten and apply affine transformation
    _, C, *_, I = x.shape
    assert I == n_blades
    x_norm = whiten_data(
        x,
        training=training,
        running_mean=running_mean,
        running_cov=running_cov,
        momentum=momentum,
        eps=eps,
    )
    if weight is not None and bias is not None:
        # Check if weight and bias tensors are of correct shape.
        assert weight.shape == (I, I, C)
        assert bias.shape == (I, C)
        # Unsqueeze weight and bias for each dimension except the channel dimension.
        shape = 1, C, *([1] * (x.dim() - 3))
        weight = weight.reshape(I, I, *shape)
        # Apply additional affine transformation post normalization.
        weight_idx = list(range(weight.dim()))
        # TODO: weight multiplication should be changed to geometric product.
        weight = weight.permute(*weight_idx[2:], *weight_idx[:2])
        x_norm = weight.matmul(x_norm[..., None]).squeeze(-1) + bias.reshape(*shape, I)

    return x_norm
