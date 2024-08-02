"""
The Enformer model architecture and its required classes
"""
import torch
from einops import rearrange
from enformer_pytorch.modeling_enformer import Attention, exponential_linspace_int
from torch import Tensor, nn

from grelu.model.blocks import ConvBlock, FeedForwardBlock
from grelu.model.layers import Activation, Crop, Dropout, Norm


class EnformerConvTower(nn.Module):
    """
    Args:
        n_blocks: Number of convolutional/pooling blocks including the stem.
        out_channels: Number of channels in the output
    """

    def __init__(
        self,
        n_blocks: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        half_dim = out_channels // 2

        # Empty list
        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(4, half_dim, 15, padding="same"),
                ConvBlock(
                    in_channels=half_dim,
                    out_channels=half_dim,
                    kernel_size=1,
                    act_func="gelu_enformer",
                    residual=True,
                    order="NACDR",
                    pool_func="attn",
                    pool_size=2,
                ),
            )
        )

        # List input and output channels for the remaining n_blocks - 1 blocks
        filters = [half_dim] + exponential_linspace_int(
            half_dim, out_channels, num=(n_blocks - 1), divisible_by=128
        )

        # Add the remaining n_blocks - 1 blocks
        for i in range(1, n_blocks):
            self.blocks.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels=filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=5,
                        act_func="gelu_enformer",
                        residual=False,
                        order="NACDR",
                    ),
                    ConvBlock(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=1,
                        act_func="gelu_enformer",
                        residual=True,
                        order="NACDR",
                        pool_func="attn",
                        pool_size=2,
                    ),
                )
            )

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
        return x


class EnformerTransformerBlock(nn.Module):
    """
    Transformer tower for enformer model

    Args:
        in_len: Length of the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
    """

    def __init__(
        self,
        in_len: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        self.norm = Norm("layer", in_len)
        self.mha = Attention(
            dim=in_len,
            heads=n_heads,
            dim_key=key_len,
            dim_value=in_len // n_heads,
            dropout=attn_dropout,
            pos_dropout=pos_dropout,
            num_rel_pos_features=in_len // n_heads,
            use_tf_gamma=False,
        )
        self.dropout = Dropout(ff_dropout)
        self.ffn = FeedForwardBlock(
            in_len=in_len,
            dropout=ff_dropout,
            act_func="relu",
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


class EnformerTransformerTower(nn.Module):
    """
    Transformer tower for enformer model

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
    """

    def __init__(
        self,
        in_channels: int,
        n_blocks: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EnformerTransformerBlock(
                    in_len=in_channels,
                    n_heads=n_heads,
                    key_len=key_len,
                    attn_dropout=attn_dropout,
                    pos_dropout=pos_dropout,
                    ff_dropout=ff_dropout,
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


class EnformerTrunk(nn.Module):
    """
    Enformer model architecture.

    Args:
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
    """

    def __init__(
        self,
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
    ) -> None:
        super().__init__()

        self.conv_tower = EnformerConvTower(n_blocks=n_conv, out_channels=channels)
        self.transformer_tower = EnformerTransformerTower(
            in_channels=channels,
            n_blocks=n_transformers,
            n_heads=n_heads,
            key_len=key_len,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
        )
        self.pointwise_conv = ConvBlock(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
            act_func="gelu_enformer",
            dropout=ff_dropout // 8,
            order="NACDR",
        )
        self.act = Activation("gelu_enformer")
        self.crop = Crop(crop_len)

    def forward(self, x):
        x = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x
