from einops import rearrange, repeat
from torch import Tensor, nn

from grelu.model.layers import Activation, Dropout, Norm


class ExplaiNNConvBlock(nn.Module):
    """
    Convolutional block for the ExplaiNN model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        act_func: str,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
            ),
        )
        self.norm = (Norm("batch", out_channels, eps=1e-05, momentum=0.1, affine=True),)
        self.act = Activation(act_func)
        self.dropout = Dropout(dropout)
        self.pool = nn.MaxPool1d(7, 7)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = rearrange(x, "b t l -> b (t l) 1")
        return x


class ExplaiNNTrunk(nn.Module):
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
        in_len: int,
        channels=300,
        kernel_size=19,
    ):
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ExplaiNNConvBlock(
                in_channels=4 * in_len,
                out_channels=in_len,
                kernel_size=kernel_size,
                groups=channels,
                dropout=0.0,
                act_func="exp",
            )
        )
        self.blocks.append(
            ExplaiNNConvBlock(
                in_channels=int((in_len - kernel_size) / 7) * channels,
                out_channels=100 * channels,
                kernel_size=1,
                groups=channels,
                dropout=0.3,
                act_func="relu",
            )
        )
        self.blocks.append(
            ExplaiNNConvBlock(
                in_channels=100 * channels,
                out_channels=channels,
                kernel_size=1,
                groups=channels,
                dropout=0.0,
                act_func="relu",
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
        x = repeat(x, "b t l -> b (r t) l", r=self.channels)
        for block in self.blocks:
            x = block(x)
        return x
