import torch

from grelu.model.heads import ConvHead, MLPHead


def test_conv_head():
    x = torch.rand(1, 4, 2)

    # Global average pooling
    head = ConvHead(n_tasks=5, in_channels=4, act_func=None, pool_func="avg")
    assert head(x).shape == (1, 5, 1)

    # No pooling
    head = ConvHead(n_tasks=5, in_channels=4, act_func=None, pool_func=None)
    assert head(x).shape == (1, 5, 2)


def test_mlp_head():
    x = torch.rand(1, 4, 32)

    # 0 hidden layers
    head = MLPHead(n_tasks=5, in_channels=4, in_len=32, act_func="relu", hidden_size=[])
    assert head(x).shape == (1, 5, 1)
    assert head.n_tasks == 5
    assert len(head.blocks) == 1

    # 1 hidden layer
    head = MLPHead(
        n_tasks=5, in_channels=4, in_len=32, act_func="relu", hidden_size=[8]
    )
    assert head(x).shape == (1, 5, 1)
    assert head.n_tasks == 5
    assert len(head.blocks) == 2

    # 2 hidden layers
    head = MLPHead(
        n_tasks=5, in_channels=4, in_len=32, act_func="relu", hidden_size=[8, 16]
    )
    assert head(x).shape == (1, 5, 1)
    assert head.n_tasks == 5
    assert len(head.blocks) == 3
