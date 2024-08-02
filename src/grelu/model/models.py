"""
Some general purpose model architectures.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import torch
from torch import Tensor, nn

from grelu.model.heads import ConvHead, MLPHead
from grelu.model.trunks import (
    ConvGRUTrunk,
    ConvTransformerTrunk,
    ConvTrunk,
    DilatedConvTrunk,
)
from grelu.model.trunks.borzoi import BorzoiTrunk
from grelu.model.trunks.enformer import EnformerTrunk
from grelu.model.trunks.explainn import ExplaiNNTrunk


class BaseModel(nn.Module):
    """
    Base model class
    """

    def __init__(self, embedding: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.embedding(x)
        x = self.head(x)
        return x


class ConvModel(BaseModel):
    """
    A fully convolutional model that optionally includes pooling,
    residual connections, batch normalization, or dilated convolutions.

    Args:
        n_tasks: Number of channels in the output
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
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
        crop_len: Number of positions to crop at either end of the output
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
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
        # Final pool
        final_pool_func: str = "avg",
    ) -> None:
        embedding = ConvTrunk(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_conv=n_conv,
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
            crop_len=crop_len,
        )
        super().__init__(
            embedding=embedding,
            head=ConvHead(
                n_tasks,
                embedding.conv_tower.out_channels,
                pool_func=final_pool_func,
                act_func=None,
                norm=False,
            ),
        )


class DilatedConvModel(BaseModel):
    """
    A model architecture based on dilated convolutional layers with residual connections.
    Inspired by the ChromBPnet model architecture.

    Args:
        n_tasks: Number of channels in the output
        channels: Number of channels for all convolutional layers
        stem_kernel_size: Kernel width for the stem
        n_blocks: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        dilation_mult: Factor by which to multiply the dilation in each block
        act_func: Name of the activation function
        crop_len: Number of positions to crop at either end of the output
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
        channels: int = 64,
        stem_kernel_size: int = 21,
        kernel_size: int = 3,
        dilation_mult: float = 2,
        act_func: str = "relu",
        n_conv: int = 8,
        crop_len: Union[str, int] = "auto",
        final_pool_func: str = "avg",
    ) -> None:
        super().__init__(
            embedding=DilatedConvTrunk(
                channels=channels,
                stem_kernel_size=stem_kernel_size,
                n_conv=n_conv,
                kernel_size=kernel_size,
                dilation_mult=dilation_mult,
                act_func=act_func,
                crop_len=crop_len,
            ),
            head=ConvHead(
                n_tasks,
                in_channels=channels,
                pool_func=final_pool_func,
                act_func=None,
                norm=False,
            ),
        )


class ConvGRUModel(BaseModel):
    """
    A model consisting of a convolutional tower followed by a bidirectional GRU layer and optional pooling.

    Args:
        n_tasks: Number of channels in the output
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
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
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
        # Final pool
        final_pool_func: str = "avg",
    ):
        embedding = ConvGRUTrunk(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_conv=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            act_func=act_func,
            conv_norm=conv_norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            crop_len=crop_len,
            n_gru=n_gru,
            dropout=dropout,
            gru_norm=gru_norm,
        )
        super().__init__(
            embedding=embedding,
            head=ConvHead(
                n_tasks,
                embedding.conv_tower.out_channels,
                pool_func=final_pool_func,
                act_func=None,
                norm=False,
            ),
        )


class ConvTransformerModel(BaseModel):
    """
    A model consisting of a convolutional tower followed by a transformer encoder layer and optional pooling.

    Args:
        n_tasks: Number of channels in the output
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling layers
        norm: If True, apply batch normalization in the convolutional layers.
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
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
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
        # Final pool
        final_pool_func: str = "avg",
    ):
        embedding = ConvTransformerTrunk(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_conv=n_conv,
            channel_init=channel_init,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
            act_func=act_func,
            norm=norm,
            pool_func=pool_func,
            pool_size=pool_size,
            residual=residual,
            crop_len=crop_len,
            n_transformers=n_transformers,
            n_heads=n_heads,
            n_pos_features=n_pos_features,
            key_len=key_len,
            value_len=value_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        super().__init__(
            embedding=embedding,
            head=ConvHead(
                n_tasks,
                embedding.conv_tower.out_channels,
                pool_func=final_pool_func,
                act_func=None,
                norm=False,
            ),
        )


class ConvMLPModel(BaseModel):
    """
    A convolutional tower followed by a Multi-head perceptron (MLP) layer.

    Args:
        n_tasks: Number of channels in the output
        seq_len: Input length
        stem_channels: Number of channels in the stem
        stem_kernel_size: Kernel width for the stem
        n_conv: Number of convolutional blocks, not including the stem
        kernel_size: Convolutional kernel width
        channel_init: Initial number of channels,
        channel_mult: Factor by which to multiply the number of channels in each block
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of the pooling
        conv_norm: If True, apply batch norm in the convolutional layers
        residual: If True, apply residual connection
        mlp_norm: If True, apply layer norm in the MLP layers
        mlp_hidden_size: A list containing the dimensions for each hidden layer of the MLP.
        dropout: Dropout probability for the MLP layers.
    """

    def __init__(
        self,
        seq_len: int,
        n_tasks: int,
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
        residual: bool = True,
        # MLP
        mlp_norm: bool = False,
        mlp_act_func: Optional[str] = "relu",
        mlp_hidden_size: List[int] = [8],
        dropout: float = 0.0,
    ) -> None:
        embedding = ConvTrunk(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            n_conv=n_conv,
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
            crop_len=0,
        )
        super().__init__(
            embedding=embedding,
            head=MLPHead(
                n_tasks=n_tasks,
                in_len=seq_len // embedding.conv_tower.pool_factor,
                in_channels=embedding.conv_tower.out_channels,
                hidden_size=mlp_hidden_size,
                norm=mlp_norm,
                act_func=mlp_act_func,
                dropout=dropout,
            ),
        )


class BorzoiModel(BaseModel):
    """
    Model consisting of Borzoi conv and transformer layers followed by U-net upsampling and optional pooling.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        channels: Number of channels in the output of the convolutional tower
        kernel_size: Width of the convolutional kernel
        n_conv: Number of convolutional/pooling blocks
        n_transformers: Number of stacked transformer blocks
        n_pos_features: Number of features in the positional embeddings
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the attention layer
        crop_len: Number of positions to crop at either end of the output
        head_act_func: Name of the activation function to use in the final layer
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
        # Stem
        stem_channels: int = 512,
        stem_kernel_size: int = 15,
        # Conv tower
        init_channels: int = 608,
        channels: int = 1536,
        n_conv: int = 7,
        kernel_size: int = 5,
        # Transformer tower
        n_transformers: int = 8,
        key_len: int = 64,
        value_len: int = 192,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        n_heads: int = 8,
        n_pos_features: int = 32,
        # Head
        crop_len: int = 16,
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = "avg",
    ) -> None:
        super().__init__(
            embedding=BorzoiTrunk(
                stem_channels=stem_channels,
                stem_kernel_size=stem_kernel_size,
                init_channels=init_channels,
                channels=channels,
                n_conv=n_conv,
                kernel_size=kernel_size,
                n_transformers=n_transformers,
                key_len=key_len,
                value_len=value_len,
                pos_dropout=pos_dropout,
                attn_dropout=attn_dropout,
                n_heads=n_heads,
                n_pos_features=n_pos_features,
                crop_len=crop_len,
            ),
            head=ConvHead(
                n_tasks,
                in_channels=round(channels * 1.25),
                norm=False,
                act_func=final_act_func,
                pool_func=final_pool_func,
            ),
        )


class BorzoiPretrainedModel(BaseModel):
    """
    Borzoi model with published weights (ported from Keras).
    """

    def __init__(
        self,
        n_tasks: int,
        # weights
        fold: int = 0,
        n_transformers: int = 8,
        # head
        crop_len=0,
        final_pool_func="avg",
    ):
        model = BorzoiModel(
            crop_len=crop_len,
            n_tasks=7611,
            stem_channels=512,
            stem_kernel_size=15,
            init_channels=608,
            n_conv=7,
            kernel_size=5,
            n_transformers=8,
            key_len=64,
            value_len=192,
            pos_dropout=0.0,
            attn_dropout=0.0,
            n_heads=8,
            n_pos_features=32,
            final_act_func=None,
            final_pool_func=None,
        )

        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact(
            f"human_state_dict_fold{fold}", project="borzoi", alias="latest"
        )
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / f"fold{fold}.h5")

        model.load_state_dict(state_dict)

        # Fix depth
        model.embedding.transformer_tower.blocks = (
            model.embedding.transformer_tower.blocks[:n_transformers]
        )

        # Change head
        head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func=final_pool_func)

        super().__init__(embedding=model.embedding, head=head)


class ExplaiNNModel(nn.Module):
    """
    The ExplaiNN model architecture.

    Args:
        n_tasks (int): number of outputs
        input_length (int): length of the input sequences
        channels (int): number of independent CNN units (default=300)
        kernel_size (int): size of each unit's conv. filter (default=19)
    """

    def __init__(
        self,
        n_tasks: int,
        in_len: int,
        channels=300,
        kernel_size=19,
    ):
        super().__init__(
            embedding=ExplaiNNTrunk(
                in_len=in_len, channels=channels, kernel_size=kernel_size
            ),
            head=ConvHead(n_tasks=n_tasks, in_channels=channels),
        )


class EnformerModel(BaseModel):
    """
    Enformer model architecture.

    Args:
        n_tasks: Number of tasks for the model to predict
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
        final_act_func: Name of the activation function to use in the final layer
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
    """

    def __init__(
        self,
        n_tasks: int,
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
        # Head
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = "avg",
    ) -> None:
        super().__init__(
            embedding=EnformerTrunk(
                n_conv=n_conv,
                channels=channels,
                n_transformers=n_transformers,
                n_heads=n_heads,
                key_len=key_len,
                attn_dropout=attn_dropout,
                pos_dropout=pos_dropout,
                ff_dropout=ff_dropout,
                crop_len=crop_len,
            ),
            head=ConvHead(
                n_tasks=n_tasks,
                in_channels=2 * channels,
                act_func=final_act_func,
                norm=False,
                pool_func=final_pool_func,
            ),
        )


class EnformerPretrainedModel(BaseModel):
    """
    Borzoi model with published weights (ported from Keras).
    """

    def __init__(
        self,
        n_tasks: int,
        n_transformers: int = 11,
        # head
        crop_len=0,
        final_pool_func="avg",
    ):
        model = EnformerModel(
            crop_len=crop_len,
            n_tasks=5313,
            channels=1536,
            n_transformers=11,
            n_heads=8,
            key_len=64,
            attn_dropout=0.05,
            pos_dropout=0.01,
            ff_dropout=0.4,
            final_act_func=None,
            final_pool_func=None,
        )

        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact("human_state_dict", project="enformer", alias="latest")
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / "human.h5")

        model.load_state_dict(state_dict)

        # Fix depth
        model.embedding.transformer_tower.blocks = (
            model.embedding.transformer_tower.blocks[:n_transformers]
        )

        # Change head
        head = ConvHead(n_tasks=n_tasks, in_channels=3072, pool_func=final_pool_func)

        super().__init__(embedding=model.embedding, head=head)
