import numpy as np
import pytest
import torch
import torch.nn as nn

from grelu.model.layers import (
    Activation,
    AdaptivePool,
    Attention,
    ChannelTransform,
    Crop,
    Dropout,
    Norm,
    Pool,
)


def test_activation():
    # Input data
    x = torch.randn(2, 4, 5)

    # Test ReLU activation
    activation = Activation("relu")
    output = activation(x)
    expected_output = torch.relu(x)
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output)

    # Test ELU activation
    activation = Activation("elu")
    output = activation(x)
    expected_output = torch.nn.functional.elu(x)
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output)

    # Test GELU activation
    activation = Activation("gelu")
    output = activation(x)
    expected_output = torch.nn.functional.gelu(x)
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output)

    # Test Identity activation
    activation = Activation(None)
    output = activation(x)
    expected_output = x
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output)

    # Test exponential activation
    activation = Activation("exp")
    output = activation(x)
    expected_output = torch.exp(x)
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output)

    # Test unsupported activation function
    with pytest.raises(NotImplementedError):
        activation = Activation("invalid_func")


def test_pool():
    # Create a 1D tensor with 10 channels and 100 positions
    x = torch.randn(2, 4, 100)

    # Test avg pooling with pool_size=2
    pool_avg = Pool(func="avg", pool_size=2)
    y_avg = pool_avg(x)
    assert y_avg.shape == (2, 4, 50)
    assert np.allclose(y_avg[0, 0, 0], torch.mean(x[0, 0, :2]))

    # Test max pooling with pool_size=3
    pool_max = Pool(func="max", pool_size=3)
    y_max = pool_max(x)
    assert y_max.shape == (2, 4, 33)
    assert np.allclose(y_max[0, 0, 0], torch.max(x[0, 0, :3]))

    # Test attention pooling with pool_size=4 and in_channels=10
    pool_attn = Pool(func="attn", pool_size=4, in_channels=4)
    y_attn = pool_attn(x)
    assert y_attn.shape == (2, 4, 25)

    # Test identity function
    pool_identity = Pool(func=None)
    y_identity = pool_identity(x)
    assert np.allclose(y_identity, x)


def test_adaptive_pool():
    # Create a 1D tensor with 10 channels and 100 positions
    x = torch.randn(2, 4, 100)

    # Test avg pooling
    pool_avg = AdaptivePool(func="avg")
    y_avg = pool_avg(x)
    assert y_avg.shape == (2, 4, 1)
    assert np.allclose(y_avg, x.mean(-1, keepdims=True))

    # Test identity function
    pool_identity = AdaptivePool(func=None)
    y_identity = pool_identity(x)
    assert np.allclose(y_identity, x)


def test_norm():
    # Create input tensor
    x = torch.rand(2, 3, 4)

    # Test BatchNorm
    norm_layer = Norm(func="batch", in_dim=3)
    y_batch = norm_layer(x)
    assert isinstance(norm_layer.layer, nn.BatchNorm1d)
    assert torch.allclose(y_batch.mean(), torch.tensor(0.0), atol=1e-2)
    assert torch.allclose(y_batch.std(), torch.tensor(1.0), atol=3e-2)

    # Test LayerNorm
    norm_layer = Norm(func="layer", in_dim=4)
    _ = norm_layer(x)
    assert isinstance(norm_layer.layer, nn.LayerNorm)

    # Test Identity
    norm_layer = Norm(func=None)
    y_identity = norm_layer(x)
    assert torch.allclose(x, y_identity)


def test_channel_transform():
    # Set up layer and input tensor
    in_channels = 3
    out_channels = 5
    layer = ChannelTransform(in_channels, out_channels)
    x = torch.randn((2, in_channels, 10))

    # Apply layer and check output shape and number of channels
    y = layer(x)
    assert y.shape == (2, out_channels, 10)

    # Check that the identity layer is working correctly
    layer = ChannelTransform(in_channels, in_channels, if_equal=False)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x)

    # Check if_equal
    layer = ChannelTransform(in_channels, in_channels, if_equal=True)
    y = layer(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_dropout():
    # create input tensor
    x = torch.rand(10, 5)

    # initialize dropout layer with p=0
    dropout = Dropout(p=0)
    y_identity = dropout(x)
    assert torch.allclose(y_identity, x)

    # initialize dropout layer with p=0.5
    dropout = Dropout(p=0.5)
    y_dropout = dropout(x)

    # assert that dropout was applied
    assert (y_identity == y_dropout).sum() < (x.numel())

    # assert that dropout is disabled during evaluation mode
    dropout.eval()
    y_eval = dropout(x)
    assert not torch.allclose(y_dropout, y_eval)


def test_crop():
    x = torch.randn(2, 4, 5)

    # Test with a target length
    crop = Crop(crop_len=1)
    y = crop(x)
    assert list(y.shape) == [2, 4, 3]
    assert np.allclose(x[:, :, 1:4], y)

    # Test with receptive field
    crop = Crop(crop_len="auto", receptive_field=4)
    y = crop(x)
    assert list(y.shape) == [2, 4, 1]
    assert np.allclose(x[:, :, [2]], y)


def test_attention():
    x = torch.randn(2, 5, 4)
    attn = Attention(in_len=4, key_len=4, value_len=8, n_heads=1, n_pos_features=2)
    # Test intermediates
    q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
    assert q.shape == (2, 5, 4)
    assert k.shape == (2, 5, 4)
    assert v.shape == (2, 5, 8)
    # Test attention scores
    a = attn.get_attn_scores(x)
    assert a.shape == (2, 1, 5, 5)
    a, v = attn.get_attn_scores(x, return_v=True)
    assert a.shape == (2, 1, 5, 5)
    assert v.shape == (2, 1, 5, 8)
    # Test layer output
    y = attn(x)
    assert y.shape == x.shape
