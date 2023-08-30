import torch
from cliffordlayers.nn.modules.gcan import (
    CliffordG3ConvTranspose2d,
    CliffordG3GroupNorm,
    CliffordG3Conv2d,
    CliffordG3LinearVSiLU,
    CliffordG3MeanVSiLU,
    CliffordG3SumVSiLU,
    PGAConjugateLinear,
    MultiVectorAct,
)
from cliffordlayers.cliffordalgebra import CliffordAlgebra


def test_g3convtranspose2d():
    g3convtranspose2d = CliffordG3ConvTranspose2d(8, 8)
    x = torch.randn(4, 8, 32, 32, 3)
    y = g3convtranspose2d(x)
    assert y.shape == (4, 8, 32, 32, 3)


def test_g3groupnorm():
    g3groupnorm = CliffordG3GroupNorm(8, 8, 3)
    x = torch.randn(4, 8, 3)
    y = g3groupnorm(x)
    assert y.shape == (4, 8, 3)


def test_g3vector_act():
    g3vectorlinear = CliffordG3LinearVSiLU(8)
    g3vectorsum = CliffordG3SumVSiLU()
    g3vectormean = CliffordG3MeanVSiLU()
    x = torch.randn(4, 8, 32, 32, 3)
    y = g3vectorlinear(x)
    assert y.shape == (4, 8, 32, 32, 3)
    y = g3vectorsum(x)
    assert y.shape == (4, 8, 32, 32, 3)
    y = g3vectormean(x)
    assert y.shape == (4, 8, 32, 32, 3)


def test_g3conv2d():
    # We operate on the vectors of the G3 algebra.
    conv = CliffordG3Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    x = torch.randn(4, 3, 3, 3, 3)
    output = conv(x)

    assert output.shape == (4, 4, 3, 3, 3)


def test_pga_conjugate_linear():
    in_features = 8
    out_features = 16
    algebra = CliffordAlgebra([0, 1, 1, 1])

    linear = PGAConjugateLinear(
        algebra=algebra,
        in_features=in_features,
        out_features=out_features,
        input_blades=(1, 2, 3),  # (11, 12, 13) for points.
    )

    vector = torch.randn(4, in_features, 3)
    output = linear(vector)
    assert output.shape == (4, out_features, 3)


def test_multivectoract():
    algebra = CliffordAlgebra([0, 1, 1, 1])
    input = torch.randn(4, 3, 3)
    act = MultiVectorAct(channels=3, algebra=algebra, input_blades=(1, 2, 3))
    output = act(input)
    assert output.shape == (4, 3, 3)


if __name__ == "__main__":
    test_pga_conjugate_linear()
    test_multivectoract()
    test_g3conv2d()
    test_g3vector_act()
    test_g3groupnorm()
    test_g3convtranspose2d()
