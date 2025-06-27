"""
The Borzoi model architecture and its required classes.
"""

from enformer_pytorch.modeling_enformer import exponential_linspace_int
from torch import Tensor, nn

from grelu.model.blocks import ConvBlock, Stem, TransformerTower, UnetTower
from grelu.model.layers import Activation, Crop


class BorzoiConvTower(nn.Module):
    """
    Convolutional tower for the Borzoi model.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        out_channels: Number of channels in the output
        kernel_size: Width of the convolutional kernel
        n_blocks: Number of convolutional/pooling blocks, including the stem
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        stem_channels: int,
        stem_kernel_size: int,
        init_channels: int,
        out_channels: int,
        kernel_size: int,
        n_blocks: int,
        norm_type="batch",
        norm_kwargs=None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Empty list
        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            Stem(
                out_channels=stem_channels,
                kernel_size=stem_kernel_size,
                act_func=None,
                pool_func="max",
                pool_size=2,
                dtype=dtype,
                device=device,
            )
        )

        # Get number of channels for the remaining n_blocks-1 blocks
        self.filters = [stem_channels] + exponential_linspace_int(
            init_channels, out_channels, (n_blocks - 1), 32
        )

        for i in range(1, n_blocks):
            self.blocks.append(
                ConvBlock(
                    in_channels=self.filters[i - 1],
                    out_channels=self.filters[i],
                    kernel_size=kernel_size,
                    norm=True,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    act_func="gelu_borzoi",
                    order="NACDR",
                    pool_func="max",
                    pool_size=2,
                    return_pre_pool=(i > (n_blocks - 3)),
                    dtype=dtype,
                    device=device,
                )
            )
        assert len(self.blocks) == n_blocks

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks[:-2]:
            x = block(x)
        x, y1 = self.blocks[-2](x)
        x, y0 = self.blocks[-1](x)
        return x, y0, y1


class BorzoiTrunk(nn.Module):
    """
    Trunk consisting of conv, transformer and U-net layers for the Borzoi model.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        n_conv: Number of convolutional/pooling blocks, including the stem
        kernel_size: Width of the convolutional kernel
        channels: Number of channels in the output
        n_transformers: Number of transformer blocks
        key_len: Length of the key
        value_len: Length of the value
        pos_dropout: Dropout rate for positional embeddings
        attn_dropout: Dropout rate for attention
        n_heads: Number of attention heads
        n_pos_features: Number of positional features
        crop_len: Length of the crop
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        # Stem
        stem_channels: int,
        stem_kernel_size: int,
        # Conv tower
        init_channels: int,
        n_conv: int,
        kernel_size: int,
        channels: int,
        # Transformer tower
        n_transformers: int,
        key_len: int,
        value_len: int,
        pos_dropout: float,
        attn_dropout: float,
        ff_dropout: float,
        n_heads: int,
        n_pos_features: int,
        # Crop
        crop_len: int,
        flash_attn: bool,
        norm_type="batch",
        norm_kwargs=None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.conv_tower = BorzoiConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            init_channels=init_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            n_blocks=n_conv,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            dtype=dtype,
            device=device,
        )
        self.transformer_tower = TransformerTower(
            n_blocks=n_transformers,
            in_channels=channels,
            key_len=key_len,
            value_len=value_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_kwargs=norm_kwargs,
            n_heads=n_heads,
            n_pos_features=n_pos_features,
            flash_attn=flash_attn,
            dtype=dtype,
            device=device,
        )
        self.unet_tower = UnetTower(
            n_blocks=2,
            in_channels=channels,
            y_in_channels=[channels, self.conv_tower.filters[-2]],
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            dtype=dtype,
            device=device,
        )
        self.pointwise_conv = ConvBlock(
            in_channels=channels,
            out_channels=round(channels * 1.25),
            kernel_size=1,
            act_func="gelu_borzoi",
            dropout=0.1,
            norm=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            order="NACDR",
            device=device,
            dtype=dtype,
        )
        self.act = Activation("gelu_borzoi")
        self.crop = Crop(crop_len=crop_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x, y0, y1 = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.unet_tower(x, [y0, y1])
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x
