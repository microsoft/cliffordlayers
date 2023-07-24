import torch
from cliffordlayers.cliffordalgebra import CliffordAlgebra


def test_geometric_product_vs_complex_multiplication():
    algebra = CliffordAlgebra([-1])

    x = torch.randn(4, 8, 2)
    y = torch.randn(4, 8, 2)

    x_complex = torch.complex(x[..., 0], x[..., 1])
    y_complex = torch.complex(y[..., 0], y[..., 1])

    x = algebra.embed(x, (0, 1))
    y = algebra.embed(y, (0, 1))

    xy = algebra.geometric_product(x, y)
    xy_complex = x_complex * y_complex
    torch.testing.assert_close(xy, torch.view_as_real(xy_complex))


def test_reverse():
    algebra = CliffordAlgebra([-1.0, -1.0])

    v1 = algebra.embed(torch.randn(4, 8, 2), (1, 2))
    v2 = algebra.embed(torch.randn(4, 8, 2), (1, 2))

    v1v2 = algebra.geometric_product(v1, v2)
    v2v1 = algebra.geometric_product(v2, v1)

    v1v2_reversed = algebra.reverse(v1v2)

    torch.testing.assert_close(v1v2_reversed, v2v1)


def test_mag2():
    algebra = CliffordAlgebra([1.0, 1.0])

    v = torch.randn(4, 8, 2)
    normsq = torch.sum(v**2, dim=-1, keepdim=True)
    v_algebra = algebra.embed(v, (1, 2))
    mag2 = algebra.mag2(v_algebra)

    torch.testing.assert_close(mag2[..., :1], normsq)


if __name__ == "__main__":
    test_geometric_product_vs_complex_multiplication()
    test_reverse()
    test_mag2()
