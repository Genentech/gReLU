"""
`grelu.model.layers` contains common types of layers to build deep learning models,
implemented as torch.nn.Module classes.

These layers are used by classes in `grelu.model.blocks` and `grelu.model.heads`.
"""

from typing import Optional

import torch
from einops import rearrange
from enformer_pytorch.modeling_enformer import GELU, AttentionPool, relative_shift
from torch import Tensor, einsum, nn

from grelu.model.position import get_central_mask


class Activation(nn.Module):
    """
    A nonlinear activation layer.

    Args:
        func: The type of activation function. Supported values are 'relu',
            'elu', 'softplus', 'gelu', 'gelu_borzoi', 'gelu_enformer' and 'exp'. If None, will return nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported activation function.
    """

    def __init__(self, func: str) -> None:
        super().__init__()

        if func == "relu":
            self.layer = nn.ReLU()
        elif func == "elu":
            self.layer = nn.ELU()
        elif func == "gelu":
            self.layer = nn.GELU()
        elif func == "gelu_borzoi":
            self.layer = nn.GELU(approximate = 'tanh')
        elif func == "gelu_enformer":
            self.layer = GELU()
        elif func == "softplus":
            self.layer = nn.Softplus()
        elif func == "exp":
            self.layer = torch.exp
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Pool(nn.Module):
    """
    A pooling layer.

    Args:
        func: Type of pooling function. Supported values are 'avg', 'max',
            or 'attn'. If None, will return nn.Identity.
        pool_size: The number of positions to pool together
        in_channels: Number of channels in the input. Only needeed for attention pooling.
        **kwargs: Additional arguments to pass to the pooling function.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(
        self,
        func: Optional[str],
        pool_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if func == "avg":
            self.layer = nn.AvgPool1d(kernel_size=pool_size, **kwargs)
        elif func == "max":
            self.layer = nn.MaxPool1d(kernel_size=pool_size, **kwargs)
        elif func == "attn":
            if in_channels is None:
                raise ValueError("The number of input channels must be provided.")
            self.layer = AttentionPool(dim=in_channels, pool_size=pool_size, **kwargs)
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class AdaptivePool(nn.Module):
    """
    An Adaptive Pooling layer. This layer does not have a defined pooling width but
    instead pools together all the values in the last axis.

    Args:
        func: Type of pooling function. Supported values are 'avg' or 'max'. If None,
            will return nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(self, func: Optional[str] = None) -> None:
        super().__init__()

        if func == "avg":
            self.layer = nn.AdaptiveAvgPool1d(1)
        elif func == "max":
            self.layer = nn.AdaptiveMaxPool1d(1)
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Norm(nn.Module):
    """
    A batch normalization or layer normalization layer.

    Args:
        func: Type of normalization function. Supported values are 'batch',
            'syncbatch', 'instance',  or 'layer'. If None, will return nn.Identity.
        in_dim: Number of features in the input tensor.
        **kwargs: Additional arguments to pass to the normalization function.
    """

    def __init__(
        self, func: Optional[str] = None, in_dim: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__()

        if func == "batch":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.BatchNorm1d(in_dim, **kwargs)

        elif func == "syncbatch":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.SyncBatchNorm(in_dim, **kwargs)

        elif func == "layer":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.LayerNorm(in_dim, **kwargs)

        elif func == "instance":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            # overwrite the defaults to make them consistant with batch norm
            kwargs = kwargs.copy()
            kwargs["affine"] = kwargs.get("affine", True)
            kwargs["track_running_stats"] = kwargs.get("track_running_stats", True)
            self.layer = nn.InstanceNorm1d(in_dim, **kwargs)

        elif func is None:
            self.layer = nn.Identity()

        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class ChannelTransform(nn.Module):
    """
    A convolutional layer to transform the number of channels in the input.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        if_equal: Whether to create layer if input and output channels are equal
        **kwargs: Additional arguments to pass to the convolutional layer.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 1, if_equal: bool = False, **kwargs
    ) -> None:
        super().__init__()
        if (in_channels == out_channels) and (not if_equal):
            self.layer = nn.Identity()
        else:
            self.layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same", **kwargs
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Dropout(nn.Module):
    """
    Optional dropout layer

    Args:
        p: Dropout probability. If this is set to 0, will return nn.Identity.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Crop(nn.Module):
    """
    Optional cropping layer.

    Args:
        crop_len: Number of positions to crop at each end of the input.
        receptive_field: Receptive field of the model to calculate crop_len.
            Only needed if crop_len is None.
    """

    def __init__(
        self, crop_len: int = 0, receptive_field: Optional[int] = None
    ) -> None:
        super().__init__()
        if crop_len == 0:
            self.layer = nn.Identity()
        else:
            if crop_len == "auto":
                assert (
                    receptive_field is not None
                ), "Receptive field must be provided for autocropping"
                # crop_len = int(np.ceil(receptive_field / 2))
                crop_len = int(receptive_field // 2)
            self.layer = nn.ConstantPad1d(-crop_len, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Attention(nn.Module):
    def __init__(
        self,
        in_len: int,
        key_len: int,
        value_len: int,
        n_heads: int,
        n_pos_features: int,
        pos_dropout: float = 0,
        attn_dropout: float = 0,
        device=None,
        dtype=None,
    ):
        """
        Multi-head Attention (MHA) layer. Modified from
        https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py

        Args:
            in_len: Length of the input
            key_len: Length of the key vectors
            value_len: Length of the value vectors.
            n_heads: Number of attention heads
            n_pos_features: Number of positional embedding features
            pos_dropout: Dropout probability in the positional embeddings
            attn_dropout: Dropout probability in the output layer
            device: Device for the layers.
            dtype: Data type for the layers.
        """
        super().__init__()

        # Save params
        self.in_len = in_len
        self.key_len = key_len
        self.value_len = value_len
        self.n_heads = n_heads
        self.n_pos_features = n_pos_features

        # Create linear layers
        self.to_q = nn.Linear(
            self.in_len,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_k = nn.Linear(
            self.in_len,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_v = nn.Linear(
            self.in_len,
            self.value_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_out = nn.Linear(
            self.value_len * self.n_heads, self.in_len, device=device, dtype=dtype
        )

        # relative positional encoding
        self.positional_embed = get_central_mask
        self.to_pos_k = nn.Linear(
            self.n_pos_features,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, self.n_heads, 1, self.key_len, device=device, dtype=dtype)
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, self.n_heads, 1, self.key_len, device=device, dtype=dtype)
        )

        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _get_pos_k(self, x):
        positions = self.positional_embed(x, out_channels=self.n_pos_features)
        positions = self.pos_dropout(positions)
        pos_k = self.to_pos_k(positions)
        pos_k = rearrange(pos_k, "n (h d) -> h n d", h=self.n_heads)
        return pos_k

    def get_attn_scores(self, x, return_v=False):
        # Q, K, V
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Get content embeddings
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v)
        )
        q = q / (self.key_len**0.5)

        # Content logits
        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        # Positional embeddings
        pos_k = self._get_pos_k(x)

        # Positional logits
        pos_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, pos_k)
        pos_logits = relative_shift(pos_logits)

        # Add content and positional embeddings
        logits = content_logits + pos_logits

        # Softmax
        attn = logits.softmax(dim=-1)

        if return_v:
            return self.attn_dropout(attn), v
        else:
            return self.attn_dropout(attn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # Get attention scores
        attn, v = self.get_attn_scores(x, return_v=True)

        # Output
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FlashAttention(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, dropout_p=0.0, device=None, dtype=None
    ):
        """
        Flash Attention layer with RoPE for positional encoding.

        Args:
            embed_dim: Number of channels
            n_heads: Number of attention heads
            dropout_p: Dropout probability for attention
            device: Device for the layers.
            dtype: Data type for the layers.
        """

        super().__init__()

        try:
            from flash_attn import flash_attn_qkvpacked_func
            from flash_attn.layers.rotary import RotaryEmbedding
        except ImportError:
            raise ImportError(
                "gReLU needs to be installed with flash-attn to use Flash Attention. \
                    Please see README for instructions."
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout_p = dropout_p

        # Create linear layers
        self.qkv = nn.Linear(
            self.embed_dim, self.embed_dim * 3, bias=False, device=device, dtype=dtype
        )
        self.out = nn.Linear(self.embed_dim, self.embed_dim, device=device, dtype=dtype)

        # positional encoding
        self.rotary_embed = RotaryEmbedding(self.head_dim, device=device)

        # no parameters, just an operation
        self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor
        """
        qkv = rearrange(
            self.qkv(x),
            "b l (qkv nheads headdim) -> b l qkv nheads headdim",
            qkv=3,
            nheads=self.n_heads,
            headdim=self.head_dim,
        )
        qkv = self.rotary_embed(qkv)
        out = rearrange(
            self.flash_attn_qkvpacked_func(qkv, self.dropout_p, window_size=(-1, -1)),
            "b l nheads headdim -> b l (nheads headdim)",
        )
        return self.out(out)
