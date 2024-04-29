import torch
from torch import Tensor

from grelu.model.position import get_central_mask, get_exponential_embedding

x = torch.rand(1, 2, 1)


def test_central_mask():
    emb = get_central_mask(x, out_channels=4)
    assert torch.allclose(
        emb,
        Tensor([[0.0, 1.0, -0.0, -1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
    )


def test_exponential():
    emb = get_exponential_embedding(x, out_channels=4)
    assert torch.allclose(
        emb,
        Tensor(
            [
                [1.0905, 1.4142, -1.0905, -1.4142],
                [1.0000, 1.0000, 0.0000, 0.0000],
                [0.9170, 0.7071, 0.9170, 0.7071],
            ]
        ),
    )
