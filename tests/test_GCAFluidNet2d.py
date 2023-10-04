# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from cliffordlayers.models.gca.twod import CliffordG3ResNet2d, CliffordG3UNet2d


def test_gca_resnet():
    x = torch.randn(8, 4, 128, 128, 3)
    in_channels = 4
    out_channels = 1

    model = CliffordG3ResNet2d(
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        norm=True,
    )

    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, out_channels, 128, 128, 3)

    model = CliffordG3ResNet2d(
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        norm=False,
    )

    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, out_channels, 128, 128, 3)


def test_gca_unet():
    x = torch.randn(8, 4, 128, 128, 3)
    in_channels = 4
    out_channels = 1

    model = CliffordG3UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        norm=True,
    )

    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, out_channels, 128, 128, 3)

    model = CliffordG3UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        norm=False,
    )

    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    out = model(x)
    assert out.shape == (8, out_channels, 128, 128, 3)


if __name__ == "__main__":
    test_gca_resnet()
    test_gca_unet()
