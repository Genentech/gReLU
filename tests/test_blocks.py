import torch
from torch import Tensor

from grelu.model.blocks import (
    ChannelTransformBlock,
    ConvBlock,
    ConvTower,
    FeedForwardBlock,
    GRUBlock,
    LinearBlock,
    SeparableConv,
    TransformerTower,
)


def test_LinearBlock():
    block = LinearBlock(in_len=5, out_len=3, act_func="relu")
    x = torch.randn(1, 4, 5)
    output = block(x)
    assert output.shape == (1, 4, 3)


def test_ConvBlock():
    x = torch.randn(1, 4, 6)
    # Simple
    block = ConvBlock(in_channels=4, out_channels=3, kernel_size=2, dilation=1)
    output = block(x)
    assert output.shape == (1, 3, 6)

    # Pooling
    block = ConvBlock(
        in_channels=4,
        out_channels=3,
        kernel_size=2,
        dilation=1,
        pool_func="max",
        pool_size=3,
    )
    output = block(x)
    assert output.shape == (1, 3, 2)

    # return_pre_pool
    block = ConvBlock(
        in_channels=4,
        out_channels=3,
        kernel_size=2,
        dilation=1,
        pool_func="max",
        pool_size=3,
        return_pre_pool=True,
    )
    output, pre = block(x)
    assert output.shape == (1, 3, 2)
    assert pre.shape == (1, 3, 6)
    assert torch.allclose(output[0, 0, 0], pre[0, 0, :3].max(axis=-1).values)


def test_ConvBlock_order():
    # Conv output
    x = Tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 2.0, -1.0, 1]]).unsqueeze(1)

    # Default order
    block = ConvBlock(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        bias=False,
        norm=True,
        pool_func="avg",
        pool_size=2,
        act_func="relu",
        residual=True,
        dropout=0.0,
    )
    output = block(x)
    assert torch.allclose(output, block.pool(block.act(block.norm(block.conv(x)) + x)))

    # Non-default order
    block = ConvBlock(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        bias=False,
        norm=True,
        pool_func="avg",
        pool_size=2,
        act_func="relu",
        residual=True,
        dropout=0.0,
        order="NCADR",
    )
    output = block(x)
    assert torch.allclose(output, block.pool(block.act(block.conv(block.norm(x))) + x))


def test_ChannelTransform():
    # Simple
    x = torch.randn(1, 4, 5)
    block = ChannelTransformBlock(in_channels=4, out_channels=2)
    output = block(x)
    assert output.shape == (1, 2, 5)

    # if_equal = false
    block = ChannelTransformBlock(
        in_channels=4, out_channels=4, norm=False, act_func=None, dropout=0.0
    )
    output = block(x)
    assert torch.allclose(output, x)

    # if_equal = true
    block = ChannelTransformBlock(
        in_channels=4,
        out_channels=4,
        norm=False,
        act_func=None,
        dropout=0.0,
        if_equal=True,
    )
    output = block(x)
    assert output.shape == x.shape
    assert not torch.allclose(output, x)


def test_ChannelTransform_order():
    x = torch.randn(1, 4, 5)

    # Default order
    block = ChannelTransformBlock(
        in_channels=4, out_channels=2, dropout=0.0, norm=True, act_func="gelu"
    )
    output = block(x)
    assert torch.allclose(output, block.act(block.norm(block.conv(x))))

    # Non-default order
    block = ChannelTransformBlock(
        in_channels=4,
        out_channels=2,
        dropout=0.0,
        norm=True,
        act_func="gelu",
        order="NCDA",
    )
    output = block(x)
    assert torch.allclose(output, block.act(block.conv(block.norm(x))))


def test_separable_conv():
    x = torch.randn(1, 4, 5)
    block = SeparableConv(in_channels=4, kernel_size=3)
    output = block(x)
    assert output.shape == x.shape


def test_ConvTower():
    # Define input parameters
    params = {
        "stem_channels": 8,
        "stem_kernel_size": 11,
        "n_blocks": 3,
        "channel_init": 16,
        "channel_mult": 2,
        "kernel_size": 3,
        "dilation_init": 1,
        "dilation_mult": 2,
        "act_func": "relu",
        "norm": False,
        "pool_func": None,
        "residual": False,
        "order": "CDNRA",
        "dropout": 0.0,
    }

    # Create instance of ConvTower
    conv_tower = ConvTower(**params)
    # Generate random input data
    x = torch.randn(1, 4, 64)

    # Compute output
    output = conv_tower(x)
    # Check output shape
    expected_shape = (1, 32, 64)
    assert output.shape == expected_shape

    # check output tensor values
    assert torch.all(torch.isfinite(output))

    # Check dilation
    assert conv_tower.blocks[0].conv.dilation == (1,)
    assert conv_tower.blocks[1].conv.dilation == (1,)
    assert conv_tower.blocks[2].conv.dilation == (2,)

    # Check channels
    assert conv_tower.blocks[0].conv.out_channels == 8
    assert conv_tower.blocks[1].conv.out_channels == 16
    assert conv_tower.blocks[2].conv.out_channels == 32

    # Check receptive field
    assert conv_tower.receptive_field == 17
    assert conv_tower.pool_factor == 1


def test_feedforward():
    x = torch.randn(1, 5, 4)
    block = FeedForwardBlock(in_len=4, dropout=0.0, act_func="relu")
    y = block(x)
    assert y.shape == x.shape
    assert torch.all(torch.isfinite(y))


def test_GRUBlock():
    block = GRUBlock(
        in_channels=4,
        n_layers=2,
        dropout=0.1,
    )

    # create test input tensor
    x = torch.randn(2, 4, 100)

    # compute output tensor
    y = block(x)

    # check output tensor shape
    assert y.shape == x.shape

    # check output tensor values
    assert torch.all(torch.isfinite(y))


def test_transformer_tower():
    # Define input tensor
    x = torch.randn(2, 16, 100)  # N, in_channels, L

    # Instantiate encoder with default parameters
    transformer_tower = TransformerTower(in_channels=16)

    # Pass input through encoder
    y = transformer_tower(x)

    # Check output shape
    assert y.shape == x.shape
    # check output tensor values
    assert torch.all(torch.isfinite(y))
