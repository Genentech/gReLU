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
    """

    def __init__(
        self,
        stem_channels: int,
        stem_kernel_size: int,
        init_channels: int,
        out_channels: int,
        kernel_size: int,
        n_blocks: int,
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
                    act_func="gelu",
                    order="NACDR",
                    pool_func="max",
                    pool_size=2,
                    return_pre_pool=(i > (n_blocks - 3)),
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
        n_heads: int,
        n_pos_features: int,
        # Crop
        crop_len: int,
    ) -> None:
        super().__init__()

        self.conv_tower = BorzoiConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            init_channels=init_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            n_blocks=n_conv,
        )
        self.transformer_tower = TransformerTower(
            n_blocks=n_transformers,
            in_channels=channels,
            key_len=key_len,
            value_len=value_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            n_heads=n_heads,
            n_pos_features=n_pos_features,
        )
        self.unet_tower = UnetTower(
            n_blocks=2,
            in_channels=channels,
            y_in_channels=[channels, self.conv_tower.filters[-2]],
        )
        self.pointwise_conv = ConvBlock(
            in_channels=channels,
            out_channels=round(channels * 1.25),
            kernel_size=1,
            act_func="gelu",
            dropout=0.1,
            norm=True,
            order="NACDR",
        )
        self.act = Activation("gelu")
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
