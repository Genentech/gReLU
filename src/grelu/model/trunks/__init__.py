"""
Some general purpose model architectures.
"""

from typing import Optional, Union

from torch import Tensor, nn

from grelu.model.blocks import ConvTower, GRUBlock, TransformerTower


class ConvTrunk(nn.Module):
    """
    A fully convolutional trunk that optionally includes pooling,
    residual connections, batch normalization, or dilated convolutions.

    Args:
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_blocks: Number of convolutional blocks, not including the stem
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
    """

    def __init__(
        self,
        # Stem
        stem_channels: int = 64,
        stem_kernel_size: int = 15,
        # Conv
        n_conv: int = 2,
        channel_init: int = 64,
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
        # Crop
        crop_len: int = 0,
    ) -> None:
        super().__init__()

        self.conv_tower = ConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_blocks=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            dilation_init=dilation_init,
            dilation_mult=dilation_mult,
            act_func=act_func,
            norm=norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            dropout=dropout,
            order="CDNRA",
            crop_len=crop_len,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv_tower(x)
        return x


class DilatedConvTrunk(nn.Module):
    """
    A model architecture based on dilated convolutional layers with residual connections.
    Inspired by the ChromBPnet model architecture.

    Args:
        channels: Number of channels for all convolutional layers
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        dilation_mult: Factor by which to multiply the dilation in each block
        act_func: Name of the activation function
        crop_len: Number of positions to crop at either end of the output
    """

    def __init__(
        self,
        channels: int = 64,
        stem_kernel_size: int = 21,
        kernel_size: int = 3,
        dilation_mult: float = 2,
        act_func: str = "relu",
        n_conv: int = 8,
        crop_len: Union[str, int] = "auto",
    ) -> None:
        super().__init__()
        self.conv_tower = ConvTower(
            stem_channels=channels,
            stem_kernel_size=stem_kernel_size,
            n_blocks=n_conv,
            channel_init=channels,
            channel_mult=1,
            kernel_size=kernel_size,
            dilation_init=2,
            dilation_mult=dilation_mult,
            act_func=act_func,
            norm=False,
            pool_func=None,
            pool_size=None,
            residual=True,
            dropout=0.0,
            crop_len=crop_len,
            order="CDNRA",
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv_tower(x)
        return x


class ConvGRUTrunk(nn.Module):
    """
    A model consisting of a convolutional tower followed by a bidirectional GRU layer and optional pooling.

    Args:
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        conv_norm: If True, apply batch normalization in the convolutional layers.
        residual: If True, apply residual connections in the convolutional layers.
        crop_len: Number of positions to crop at either end of the output
        n_gru: Number of GRU layers
        dropout: Dropout for GRU and feed-forward layers
        gru_norm: If True, include layer normalization in feed-forward network.
    """

    def __init__(
        self,
        # Stem
        stem_channels: int = 16,
        stem_kernel_size: int = 15,
        # Conv
        n_conv: int = 2,
        channel_init: int = 16,
        channel_mult: float = 1,
        kernel_size: int = 5,
        act_func: str = "relu",
        conv_norm: bool = False,
        pool_func: Optional[str] = None,
        pool_size: Optional[int] = None,
        residual: bool = False,
        # Crop
        crop_len: int = 0,
        # GRU
        n_gru: int = 1,
        dropout: float = 0.0,
        gru_norm: bool = False,
    ):
        super().__init__()
        self.conv_tower = ConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_blocks=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            dilation_init=1,
            dilation_mult=1,
            act_func=act_func,
            norm=conv_norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            order="CDNRA",
            crop_len=crop_len,
        )

        self.gru_tower = GRUBlock(
            in_channels=self.conv_tower.out_channels,
            n_layers=n_gru,
            dropout=dropout,
            act_func=act_func,
            norm=gru_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv_tower(x)
        x = self.gru_tower(x)
        return x


class ConvTransformerTrunk(nn.Module):
    """
    A model consisting of a convolutional tower followed by a transformer encoder layer and optional pooling.

    Args:
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        conv_norm: If True, apply batch normalization in the convolutional layers.
        residual: If True, apply residual connections in the convolutional layers.
        crop_len: Number of positions to crop at either end of the output
        n_transformers: Number of transformer encoder layers
        n_heads: Number of heads in each multi-head attention layer
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
    """

    def __init__(
        self,
        # Stem
        stem_channels: int = 16,
        stem_kernel_size: int = 15,
        # Conv
        n_conv: int = 2,
        channel_init: int = 16,
        channel_mult: float = 1,
        kernel_size: int = 5,
        act_func: str = "relu",
        norm: bool = False,
        pool_func: Optional[str] = None,
        pool_size: Optional[int] = None,
        residual: bool = False,
        # Crop
        crop_len: int = 0,
        # Transformer
        n_transformers=1,
        key_len: int = 8,
        value_len: int = 8,
        n_heads: int = 1,
        n_pos_features: int = 4,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_tower = ConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_blocks=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            dilation_init=1,
            dilation_mult=1,
            act_func=act_func,
            norm=norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            order="CDNRA",
            crop_len=crop_len,
        )

        self.transformer_tower = TransformerTower(
            in_channels=self.conv_tower.out_channels,
            n_blocks=n_transformers,
            n_heads=n_heads,
            n_pos_features=n_pos_features,
            key_len=key_len,
            value_len=value_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv_tower(x)
        x = self.transformer_tower(x)
        return x
