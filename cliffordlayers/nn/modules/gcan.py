import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from cliffordlayers.nn.functional.cliffordg3conv import clifford_g3convnd


def get_clifford_left_kernel(M, w, flatten=True):
    """
    Obtains the matrix that computes the geometric product from the left.
    When the output is flattened, it can be used to apply a fully connected
    layer on the multivectors.

    Args:
        M (Tensor): Cayley table that defines the geometric relation.
        w (Tensor): Input tensor with shape (o, i, c) where o is the number of output channels,
                    i is the number of input channels, and c is the number of blades.
        flatten (bool, optional): If True, the resulting matrix will be reshaped for subsequent
                                  fully connected operations. Defaults to True.

    """
    o, i, c = w.size()
    k = torch.einsum("ijk, pqi->jpkq", M, w)
    if flatten:
        k = k.reshape(o * c, i * c)
    return k


def get_clifford_right_kernel(M, w, flatten=True):
    """
    Obtains the matrix that computes the geometric product from the right.
    When the output is flattened, it can be used to apply a fully connected
    layer on the multivectors.

    Args:
        M (Tensor): Cayley table that defines the geometric relation.
        w (Tensor): Input tensor with shape (o, i, c) where o is the number of output channels,
                    i is the number of input channels, and c is the number of blades.
        flatten (bool, optional): If True, the resulting matrix will be reshaped for subsequent
                                    fully connected operations. Defaults to True.
    """
    o, i, c = w.size()
    k = torch.einsum("ijk, pqk->jpiq", M, w)
    if flatten:
        k = k.reshape(o * c, i * c)
    return k


class PGAConjugateLinear(nn.Module):
    """
    Linear layer that applies the PGA conjugation to the input.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        algebra (Algebra): Algebra object that defines the geometric product.
        input_blades (tuple): Nonnegative blades of the input multivectors.
        action_blades (tuple, optional): Blades of the action. Defaults to (0, 5, 6, 7, 8, 9, 10, 15),
                                         which encodes rotation and translation.
    """

    def __init__(
        self,
        in_features,
        out_features,
        algebra,
        input_blades,
        action_blades=(0, 5, 6, 7, 8, 9, 10, 15),
    ):
        super().__init__()
        assert torch.all(algebra.metric == torch.tensor([0, 1, 1, 1]))
        self.input_blades = input_blades
        self.in_features = in_features
        self.out_features = out_features
        self.algebra = algebra
        self.action_blades = action_blades
        self.n_action_blades = len(action_blades)
        self._action = nn.Parameter(torch.empty(out_features, in_features, self.n_action_blades))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.embed_e0 = nn.Parameter(torch.zeros(in_features, 1))

        self.inverse = algebra.reverse

        self.reset_parameters()

    def reset_parameters(self):
        # Init the rotation parts uniformly.
        torch.nn.init.uniform_(self._action[..., 0], -1, 1)
        torch.nn.init.uniform_(self._action[..., 4:7], -1, 1)

        # Init the translation parts with zeros.
        torch.nn.init.zeros_(self._action[..., 1:4])
        torch.nn.init.zeros_(self._action[..., 7])

        norm = self.algebra.norm(self.algebra.embed(self._action.data, self.action_blades))
        assert torch.allclose(norm[..., 1:], torch.tensor(0.0), atol=1e-3)
        norm = norm[..., :1]
        self._action.data = self._action.data / norm

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def action(self):
        return self.algebra.embed(self._action, self.action_blades)

    def forward(self, input):
        M = self.algebra.cayley
        k = self.action
        k_ = self.inverse(k)
        x = self.algebra.embed(input, self.input_blades)
        x[..., 14:15] = self.embed_e0
        # x[..., 14:15] = 1

        k_l = get_clifford_left_kernel(M, k, flatten=False)
        k_r = get_clifford_right_kernel(M, k_, flatten=False)

        x = torch.einsum("oi,poqi,qori,bir->bop", self.weight, k_r, k_l, x)

        x = self.algebra.get(x, self.input_blades)

        return x


class MultiVectorAct(nn.Module):
    """
    A module to apply multivector activations to the input.

    Args:
        channels (int): Number of channels in the input.
        algebra: The algebra object that defines the geometric product.
        input_blades (list, tuple): The nonnegative input blades.
        kernel_blades (list, tuple, optional): The blades that will be used to compute the activation. Defaults to all input blades.
        agg (str, optional): The aggregation method to be used. Options include "linear", "sum", and "mean". Defaults to "linear".
    """

    def __init__(self, channels, algebra, input_blades, kernel_blades=None, agg="linear"):
        super().__init__()
        self.algebra = algebra
        self.input_blades = tuple(input_blades)
        if kernel_blades is not None:
            self.kernel_blades = tuple(kernel_blades)
        else:
            self.kernel_blades = self.input_blades

        if agg == "linear":
            self.conv = nn.Conv1d(channels, channels, kernel_size=len(self.kernel_blades), groups=channels)
        self.agg = agg

    def forward(self, input):
        v = self.algebra.embed(input, self.input_blades)
        if self.agg == "linear":
            v = v * torch.sigmoid(self.conv(v[..., self.kernel_blades]))
        elif self.agg == "sum":
            v = v * torch.sigmoid(v[..., self.kernel_blades].sum(dim=-1, keepdim=True))
        elif self.agg == "mean":
            v = v * torch.sigmoid(v[..., self.kernel_blades].mean(dim=-1, keepdim=True))
        else:
            raise ValueError(f"Aggregation {self.agg} not implemented.")
        v = self.algebra.get(v, self.input_blades)
        return v


class _CliffordG3ConvNd(nn.Module):
    """
    A Clifford geometric algebra convolutional layer for N-dimensional fields where the features are vectors in G3.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution operation. Defaults to 1.
        padding (int, optional): Padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Dilation rate of the kernel. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        transposed (bool, optional): If True, performs a transposed convolution. Defaults to False.
        bias (bool, optional): If True, adds a bias term to the output. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        transposed: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if transposed:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size)) for _ in range(4)]
            )
        else:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size)) for _ in range(4)]
            )
        if bias:
            self.bias = nn.ParameterList([nn.Parameter(torch.empty(out_channels)) for _ in range(3)])
        else:
            self.register_parameter("bias", None)

        self.scale_param = nn.Parameter(torch.Tensor(self.weights[0].shape))
        self.zero_kernel = nn.Parameter(torch.zeros(self.weights[0].shape), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.scale_param, a=math.sqrt(5))
        self.weights.append(self.scale_param)
        self.weights.append(self.zero_kernel)

        if self.bias is not None:
            for i, bias in enumerate(self.bias):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[i])
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"

        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"

        return s.format(**self.__dict__)


class CliffordG3Conv2d(_CliffordG3ConvNd):
    """
    2D convolutional layer where the features are vectors in G3.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution operation. Defaults to 1.
        padding (int or str, optional): Padding added to both sides of the input or padding mode. Defaults to 0.
        dilation (int, optional): Dilation rate of the kernel. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a bias term to the output. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            transposed=False,
            bias=bias,
        )

    def forward(self, input):
        x = torch.cat([input[..., 0], input[..., 1], input[..., 2]], dim=1)

        x = clifford_g3convnd(
            x,
            self.weights,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        e_0 = x[:, : self.out_channels, :, :]
        e_1 = x[:, self.out_channels : self.out_channels * 2, :, :]
        e_2 = x[:, self.out_channels * 2 : self.out_channels * 3, :, :]

        return torch.stack([e_0, e_1, e_2], dim=-1)


class CliffordG3ConvTranspose2d(_CliffordG3ConvNd):
    """
    2D transposed convolutional layer where the features are vectors in G3.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution operation. Defaults to 1.
        padding (int or str, optional): Padding added to both sides of the input or padding mode. Defaults to 0.
        dilation (int, optional): Dilation rate of the kernel. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a bias term to the output. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            transposed=True,
            bias=bias,
        )

    def forward(self, input):
        x = torch.cat([input[..., 0], input[..., 1], input[..., 2]], dim=1)

        x = clifford_g3convnd(
            x,
            self.weights,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            transposed=True,
        )
        e_0 = x[:, : self.out_channels, :, :]
        e_1 = x[:, self.out_channels : self.out_channels * 2, :, :]
        e_2 = x[:, self.out_channels * 2 : self.out_channels * 3, :, :]

        return torch.stack([e_0, e_1, e_2], dim=-1)


class CliffordG3LinearVSiLU(nn.Module):
    """
    A module that applies the vector SiLU using a linear combination to vectors in G3.

    Args:
        channels (int): Number of channels in the input.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (1, 1, 3), groups=channels)

    def forward(self, input):
        return input * torch.sigmoid(self.conv(input))


class CliffordG3SumVSiLU(nn.Module):
    """
    A module that applies the vector SiLU using vector sum to vectors in G3.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(input.sum(-1, keepdim=True)) * input


class CliffordG3MeanVSiLU(nn.Module):
    """
    A module that applies the vector SiLU using vector mean to vectors in G3.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(input.mean(-1, keepdim=True)) * input


class CliffordG3GroupNorm(nn.Module):
    """
    A module that applies group normalization to vectors in G3.

    Args:
        num_groups (int): Number of groups to normalize over.
        num_features (int): Number of features in the input.
        num_blades (int): Number of blades in the input.
        scale_norm (bool, optional): If True, the output is scaled by the norm of the input. Defaults to False.
    """

    def __init__(self, num_groups, num_features, num_blades, scale_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features, num_blades))
        self.num_groups = num_groups
        self.scale_norm = scale_norm
        self.num_blades = num_blades
        self.num_features = num_features

    def forward(self, x):
        N, C, *D, I = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1, I)
        mean = x.mean(-2, keepdim=True)
        x = x - mean
        if self.scale_norm:
            norm = x.norm(dim=-1, keepdim=True).mean(dim=-2, keepdims=True)
            x = x / norm

        x = x.view(len(x), self.num_features, -1, self.num_blades)

        return (x * self.weight[None, :, None, None] + self.bias[None, :, None]).view(N, C, *D, I)
