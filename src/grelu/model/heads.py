"""
`grelu.model.heads` contains 'head' layers for sequence-to-function deep
learning models. All heads inherit from the `torch.nn.Module` class, and
define a `forward` function that takes sequence embeddings produced
by earlier layers of the model (tensors of shape (N, embedding_dim, embedding_length))
and returns tensors of shape (N, tasks, output_length).
"""

from typing import List, Optional

import torch
from einops import rearrange
from torch import nn

from grelu.model.blocks import ChannelTransformBlock, LinearBlock
from grelu.model.layers import AdaptivePool


class ConvHead(nn.Module):
    """
    A 1x1 Conv layer that transforms the the number of channels in the input and then
    optionally pools along the length axis.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the convolutional layer
        pool_func: Pooling function.
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        act_func: Optional[str] = None,
        pool_func: Optional[str] = None,
        norm: bool = False,
        norm_kwargs: Optional[dict] = dict(),
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        # Save all params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.act_func = act_func
        self.pool_func = pool_func
        self.norm = norm

        # Create layers
        self.channel_transform = ChannelTransformBlock(
            self.in_channels,
            self.n_tasks,
            act_func=self.act_func,
            norm=self.norm,
            norm_kwargs=norm_kwargs,
            dtype=dtype,
            device=device,
        )
        self.pool = AdaptivePool(self.pool_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        x = self.channel_transform(x)
        x = self.pool(x)
        return x


class MLPHead(nn.Module):
    """
    This block implements the multi-layer perceptron (MLP) module.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        in_len: Length of the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the linear layers
        hidden_size: A list of dimensions for each hidden layer of the MLP.
        dropout: Dropout probability for the linear layers.
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        in_len: int,
        act_func: Optional[str] = None,
        hidden_size: List[int] = [],
        norm: bool = False,
        dropout: float = 0.0,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Save params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.in_len = in_len
        self.act_func = act_func
        self.hidden_size = hidden_size
        self.norm = norm
        self.dropout = dropout

        # Create layers
        self.blocks = nn.ModuleList()
        in_len = self.in_len * self.in_channels

        # hidden layers
        for h in self.hidden_size:
            self.blocks.append(
                LinearBlock(
                    in_len,
                    h,
                    norm=self.norm,
                    act_func=self.act_func,
                    dropout=self.dropout,
                    dtype=dtype,
                    device=device,
                )
            )
            in_len = h  # Output len of this block is the input len of next block

        # Final layer
        self.blocks.append(
            LinearBlock(
                in_len,
                self.n_tasks,
                norm=self.norm,
                act_func=None,
                dropout=self.dropout,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        # Concatenate channels into the length axis
        x = rearrange(x, "b t l -> b 1 (t l)")
        # Apply linear blocks on the length axis
        for block in self.blocks:
            x = block(x)
        # Swap output tasks back to the channels axis
        x = rearrange(x, "b 1 l -> b l 1")
        return x
