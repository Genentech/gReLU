"""
Blocks composed of multiple layers.
"""

import warnings
from typing import List, Optional, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from grelu.model.layers import (
    Activation,
    Attention,
    ChannelTransform,
    Crop,
    Dropout,
    FlashAttention,
    Norm,
    Pool,
)


class LinearBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout.

    Args:
        in_len: Length of input
        out_len: Length of output
        act_func: Name of activation function
        dropout: Dropout probability
        norm: If True, apply layer normalization
        bias: If True, include bias term.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        act_func: str = "relu",
        dropout: float = 0.0,
        norm: bool = False,
        bias: bool = True,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.norm = Norm(
            func="layer" if norm else None, in_dim=in_len, dtype=dtype, device=device
        )
        self.linear = nn.Linear(in_len, out_len, bias=bias, dtype=dtype, device=device)
        self.dropout = Dropout(dropout)
        self.act = Activation(act_func)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional layer along with optional normalization,
    activation, dilation, dropout, residual connection, and pooling.
    The order of these operations can be specified, except
    for pooling, which always comes last.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        dilation: Dilation
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Pooling width
        dropout: Dropout probability
        norm: If True, apply normalization layer
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        residual: If True, apply residual connection
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNRA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, residual addition, activation. Pooling is not included
            as it is always performed last.
        return_pre_pool: If this is True and pool_func is not None, the final
            output will be a tuple (output after pooling, output_before_pooling).
            This is useful if the output before pooling is required by a later
            layer.
        dtype: Data type of the weights
        device: Device on which to store the weights
        **kwargs: Additional arguments to be passed to nn.Conv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
        dropout: float = 0.0,
        norm: bool = True,
        norm_type="batch",
        norm_kwargs=None,
        residual: bool = False,
        order: str = "CDNRA",
        bias: bool = True,
        return_pre_pool: bool = False,
        dtype=None,
        device=None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
            "R",
        ], "The string supplied in order must contain one occurrence each of A, C, D, N and R."
        self.order = order
        norm_kwargs = norm_kwargs or dict()

        # Create norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm(
                    norm_type,
                    in_dim=out_channels,
                    dtype=dtype,
                    device=device,
                    **norm_kwargs,
                )
            else:
                self.norm = Norm(
                    norm_type,
                    in_dim=in_channels,
                    dtype=dtype,
                    device=device,
                    **norm_kwargs,
                )
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.act = Activation(act_func)
        self.pool = Pool(func=pool_func, pool_size=pool_size, in_channels=out_channels)
        self.dropout = Dropout(dropout)
        self.residual = residual
        if self.residual:
            self.channel_transform = ChannelTransform(
                in_channels, out_channels, dtype=dtype, device=device
            )
        self.order = order
        assert (
            len(set(self.order).difference(set("CDNRA"))) == 0
        ), "The string supplied in order contains a non-recognized letter."
        self.return_pre_pool = return_pre_pool

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input data.
        """
        if self.residual:
            x_input = self.channel_transform(x)

        # Intermediate layers
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "R":
                if self.residual:
                    x = torch.add(x, x_input)
            elif name == "A":
                x = self.act(x)

        # Pool
        if self.return_pre_pool:
            return self.pool(x), x
        else:
            return self.pool(x)


class ChannelTransformBlock(nn.Module):
    """
    Convolutional layer with kernel size=1 along with optional normalization, activation
    and dropout

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        act_func: Name of the activation function
        dropout: Dropout probability
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, activation.
        if_equal: If True, create a layer even if the input and output channels are equal.
        device: Device on which to store the weights
        dtype: Data type of the weights
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
        act_func: str = "relu",
        dropout: float = 0.0,
        order: str = "CDNA",
        norm_type="batch",
        norm_kwargs=None,
        if_equal: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
        ], "The string supplied in order must contain one occurrence each of A, C, D and N."
        self.order = order
        norm_kwargs = norm_kwargs or dict()

        # Create batch norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm(
                    norm_type,
                    in_dim=out_channels,
                    dtype=dtype,
                    device=device,
                    **norm_kwargs,
                )
            else:
                self.norm = Norm(
                    "batch",
                    in_dim=in_channels,
                    dtype=dtype,
                    device=device,
                    **norm_kwargs,
                )
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = ChannelTransform(
            in_channels, out_channels, if_equal=if_equal, dtype=dtype, device=device
        )
        self.act = Activation(act_func)
        self.dropout = Dropout(dropout)
        self.order = order

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "A":
                x = self.act(x)
        return x


class Stem(nn.Module):
    """
    Convolutional layer followed by optional activation and pooling.
    Meant to take one-hot encoded DNA sequence as input

    Args:
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of pooling layer
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            4,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.act = Activation(act_func)
        self.pool = Pool(pool_func, pool_size=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SeparableConv(nn.Module):
    """
    Equivalent class to `tf.keras.layers.SeparableConv1D`

    Args:
        in_channels: Number of channels in the input
        kernel_size: Convolutional kernel width
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self, in_channels: int, kernel_size: int, dtype=None, device=None
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding="same",
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvTower(nn.Module):
    """
    A module that consists of multiple convolutional blocks and takes a one-hot encoded
    DNA sequence as input.

    Args:
        n_blocks: Number of convolutional blocks, including the stem
        stem_channels: Number of channels in the stem,
        stem_kernel_size: Kernel width for the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        dilation_init: Initial dilation
        dilation_mult: Factor by which to multiply the dilation in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        dropout: Dropout probability
        norm: If True, apply batch norm
        residual: If True, apply residual connection
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNRA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, residual addition, activation. Pooling is not included
            as it is always performed last.
        crop_len: Number of positions to crop at either end of the output
        dtype: Data type of the weights
        device: Device on which to store
    """

    def __init__(
        self,
        stem_channels: int,
        stem_kernel_size: int,
        n_blocks: int = 2,
        channel_init: int = 16,
        channel_mult: float = 1,
        kernel_size: int = 5,
        dilation_init: int = 1,
        dilation_mult: float = 1,
        act_func: str = "relu",
        norm: bool = False,
        pool_func: Optional[str] = None,
        pool_size: Optional[int] = None,
        residual: bool = False,
        dropout: float = 0.0,
        order: str = "CDNRA",
        crop_len: Union[int, str] = 0,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            Stem(
                stem_channels,
                stem_kernel_size,
                act_func=act_func,
                dtype=dtype,
                device=device,
            )
        )
        self.receptive_field = stem_kernel_size
        self.pool_factor = 1
        self.out_channels = stem_channels

        # Add the remaining n-1 blocks
        in_channels = stem_channels
        out_channels = channel_init
        dilation = dilation_init

        for i in range(1, n_blocks):
            # Add block
            self.blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    act_func=act_func,
                    norm=norm,
                    residual=residual,
                    pool_func=pool_func,
                    pool_size=pool_size,
                    dropout=dropout,
                    order=order,
                    dtype=dtype,
                    device=device,
                )
            )

            # Account for kernel width
            self.receptive_field += dilation * (kernel_size - 1)

            # Account for pooling
            if pool_func is not None:
                self.receptive_field *= pool_size
                self.pool_factor *= pool_size

            # Set final number of output channels
            if i == n_blocks - 1:
                self.out_channels = out_channels

            else:
                # Output channels of this block become the input channels of the next block
                in_channels = out_channels

                # Multiply output channels and dilation
                out_channels = int(out_channels * channel_mult)
                dilation = int(dilation * dilation_mult)

        # Cropping layer
        self.crop = Crop(crop_len, receptive_field=self.receptive_field)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks:
            x = block(x)
        x = self.crop(x)
        return x


class FeedForwardBlock(nn.Module):
    """
    2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

    Args:
        in_len: Length of the input tensor
        dropout: Dropout probability
        act_func: Name of the activation function
        kwargs: Additional arguments to be passed to the linear layers
    """

    def __init__(
        self,
        in_len: int,
        dropout: float = 0.0,
        act_func: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.dense1 = LinearBlock(
            in_len,
            in_len * 2,
            norm=True,
            dropout=dropout,
            act_func=act_func,
            bias=True,
            **kwargs,
        )
        self.dense2 = LinearBlock(
            in_len * 2,
            in_len,
            norm=False,
            dropout=dropout,
            act_func=None,
            bias=True,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class GRUBlock(nn.Module):
    """
    Stacked bidirectional GRU layers followed by a feed-forward network.

    Args:
        in_channels: The number of channels in the input
        n_layers: The number of GRU layers
        gru_hidden_size: Number of hidden elements in GRU layers
        dropout: Dropout probability
        act_func: Name of the activation function for feed-forward network
        norm: If True, include layer normalization in feed-forward network.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        act_func: str = "relu",
        norm: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=in_channels,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            num_layers=n_layers,
            dtype=dtype,
            device=device,
        )
        self.ffn = FeedForwardBlock(
            in_len=in_channels,
            dropout=dropout,
            act_func=act_func,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        x = self.gru(x)[0]
        # Combine output of forward and reverse GRU
        x = x[:, :, : self.gru.hidden_size] + x[:, :, self.gru.hidden_size :]
        x = self.ffn(x)
        x = rearrange(x, "b l t -> b t l")
        return x


class TransformerBlock(nn.Module):
    """
    A block containing a multi-head attention layer followed by a feed-forward
    network and residual connections.

    Args:
        in_len: Length of the input
        n_heads: Number of attention heads
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    flash_attn_warn = False

    def __init__(
        self,
        in_len: int,
        n_heads: int,
        attn_dropout: float,
        ff_dropout: float,
        flash_attn: bool,
        n_pos_features: Optional[int] = None,
        key_len: Optional[int] = None,
        value_len: Optional[int] = None,
        pos_dropout: Optional[float] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.norm = Norm("layer", in_len)

        if flash_attn:
            if (
                not (
                    n_pos_features is None
                    and key_len is None
                    and value_len is None
                    and pos_dropout is None
                )
                and not TransformerBlock.flash_attn_warn
            ):
                warnings.warn(
                    "WARNING: FlashAttention does not use pos_dropout, key_len, value_len, n_pos_features arguments. \
                        Ignore if you are loading a pre-trained model."
                )
                TransformerBlock.flash_attn_warn = True

            self.mha = FlashAttention(
                embed_dim=in_len,
                n_heads=n_heads,
                dropout_p=attn_dropout,
                dtype=dtype,
                device=device,
            )
        else:
            self.mha = Attention(
                in_len=in_len,
                n_heads=n_heads,
                n_pos_features=n_pos_features,
                key_len=key_len,
                value_len=value_len,
                pos_dropout=pos_dropout,
                attn_dropout=attn_dropout,
                dtype=dtype,
                device=device,
            )
        self.dropout = Dropout(ff_dropout)
        self.ffn = FeedForwardBlock(
            in_len=in_len,
            dropout=ff_dropout,
            act_func="relu",
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = torch.add(ffn_input, x)
        return x


class TransformerTower(nn.Module):
    """
    Multiple stacked transformer encoder layers.

    Args:
        in_channels: Number of channels in the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        in_channels: int,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_pos_features: int = 32,
        key_len: int = 64,
        value_len: int = 64,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    in_len=in_channels,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    flash_attn=flash_attn,
                    n_pos_features=n_pos_features,
                    key_len=key_len,
                    value_len=value_len,
                    pos_dropout=pos_dropout,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, "b l t -> b t l")
        return x


class UnetBlock(nn.Module):
    """
    Upsampling U-net block

    Args:
        in_channels: Number of channels in the input
        y_in_channels: Number of channels in the higher-resolution representation.
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        device: Device on which to store the weights
        dtype: Data type of the weights
    """

    def __init__(
        self,
        in_channels: int,
        y_in_channels: int,
        norm_type="batch",
        norm_kwargs=None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            in_channels,
            1,
            norm=True,
            act_func="gelu",
            order="NACDR",
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            dtype=dtype,
            device=device,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.channel_transform = ChannelTransformBlock(
            y_in_channels,
            in_channels,
            norm=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            act_func="gelu",
            order="NACD",
            if_equal=True,
            dtype=dtype,
            device=device,
        )
        self.sconv = SeparableConv(in_channels, 3, dtype=dtype, device=device)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.upsample(x)
        x = torch.add(x, self.channel_transform(y))
        x = self.sconv(x)
        return x


class UnetTower(nn.Module):
    """
    Upsampling U-net tower for the Borzoi model

    Args:
        in_channels: Number of channels in the input
        y_in_channels: Number of channels in the higher-resolution representations.
        n_blocks: Number of U-net blocks
        kwargs: Additional arguments to be passed to the U-net blocks
    """

    def __init__(
        self, in_channels: int, y_in_channels: List[int], n_blocks: int, **kwargs
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for y_c in y_in_channels:
            self.blocks.append(UnetBlock(in_channels, y_c, **kwargs))

    def forward(self, x: Tensor, ys: List[Tensor]) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            ys: Higher-resolution representations

        Returns:
            Output tensor
        """
        for b, y in zip(self.blocks, ys):
            x = b(x, y)
        return x
