grelu.model.trunks
==================

.. py:module:: grelu.model.trunks

.. autoapi-nested-parse::

   Some general purpose model architectures.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/grelu/model/trunks/borzoi/index
   /autoapi/grelu/model/trunks/enformer/index
   /autoapi/grelu/model/trunks/explainn/index


Classes
-------

.. autoapisummary::

   grelu.model.trunks.ConvTower
   grelu.model.trunks.GRUBlock
   grelu.model.trunks.TransformerTower
   grelu.model.trunks.ConvTrunk
   grelu.model.trunks.DilatedConvTrunk
   grelu.model.trunks.ConvGRUTrunk
   grelu.model.trunks.ConvTransformerTrunk


Package Contents
----------------

.. py:class:: ConvTower(stem_channels: int, stem_kernel_size: int, n_blocks: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, dilation_init: int = 1, dilation_mult: float = 1, act_func: str = 'relu', norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, dropout: float = 0.0, order: str = 'CDNRA', crop_len: Union[int, str] = 0)

   Bases: :py:obj:`torch.nn.Module`


   A module that consists of multiple convolutional blocks and takes a one-hot encoded
   DNA sequence as input.

   :param n_blocks: Number of convolutional blocks, including the stem
   :param stem_channels: Number of channels in the stem,
   :param stem_kernel_size: Kernel width for the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param dilation_init: Initial dilation
   :param dilation_mult: Factor by which to multiply the dilation in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling layers
   :param dropout: Dropout probability
   :param norm: If True, apply batch norm
   :param residual: If True, apply residual connection
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNRA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, residual addition, activation. Pooling is not included
                 as it is always performed last.
   :param crop_len: Number of positions to crop at either end of the output


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: GRUBlock(in_channels: int, n_layers: int = 1, dropout: float = 0.0, act_func: str = 'relu', norm: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Stacked bidirectional GRU layers followed by a feed-forward network.

   :param in_channels: The number of channels in the input
   :param n_layers: The number of GRU layers
   :param gru_hidden_size: Number of hidden elements in GRU layers
   :param dropout: Dropout probability
   :param act_func: Name of the activation function for feed-forward network
   :param norm: If True, include layer normalization in feed-forward network.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: TransformerTower(in_channels: int, n_blocks: int = 1, n_heads: int = 1, n_pos_features: int = 32, key_len: int = 64, value_len: int = 64, pos_dropout: float = 0.0, attn_dropout: float = 0.0, ff_dropout: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   Multiple stacked transformer encoder layers.

   :param in_channels: Number of channels in the input
   :param n_blocks: Number of stacked transformer blocks
   :param n_heads: Number of attention heads
   :param n_pos_features: Number of positional embedding features
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param attn_dropout: Dropout probability in the output layer
   :param ff_droppout: Dropout probability in the linear feed-forward layers


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvTrunk(stem_channels: int = 64, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 64, channel_mult: float = 1, kernel_size: int = 5, dilation_init: int = 1, dilation_mult: float = 1, act_func: str = 'relu', norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, dropout: float = 0.0, crop_len: int = 0)

   Bases: :py:obj:`torch.nn.Module`


   A fully convolutional trunk that optionally includes pooling,
   residual connections, batch normalization, or dilated convolutions.

   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_blocks: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param dilation_init: Initial dilation
   :param dilation_mult: Factor by which to multiply the dilation in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling layers
   :param dropout: Dropout probability
   :param norm: If True, apply batch norm
   :param residual: If True, apply residual connection
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNRA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, residual addition, activation. Pooling is not included
                 as it is always performed last.
   :param crop_len: Number of positions to crop at either end of the output


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: DilatedConvTrunk(channels: int = 64, stem_kernel_size: int = 21, kernel_size: int = 3, dilation_mult: float = 2, act_func: str = 'relu', n_conv: int = 8, crop_len: Union[str, int] = 'auto')

   Bases: :py:obj:`torch.nn.Module`


   A model architecture based on dilated convolutional layers with residual connections.
   Inspired by the ChromBPnet model architecture.

   :param channels: Number of channels for all convolutional layers
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param dilation_mult: Factor by which to multiply the dilation in each block
   :param act_func: Name of the activation function
   :param crop_len: Number of positions to crop at either end of the output


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvGRUTrunk(stem_channels: int = 16, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, act_func: str = 'relu', conv_norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, crop_len: int = 0, n_gru: int = 1, dropout: float = 0.0, gru_norm: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   A model consisting of a convolutional tower followed by a bidirectional GRU layer and optional pooling.

   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling layers
   :param conv_norm: If True, apply batch normalization in the convolutional layers.
   :param residual: If True, apply residual connections in the convolutional layers.
   :param crop_len: Number of positions to crop at either end of the output
   :param n_gru: Number of GRU layers
   :param dropout: Dropout for GRU and feed-forward layers
   :param gru_norm: If True, include layer normalization in feed-forward network.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvTransformerTrunk(stem_channels: int = 16, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, act_func: str = 'relu', norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, crop_len: int = 0, n_transformers=1, key_len: int = 8, value_len: int = 8, n_heads: int = 1, n_pos_features: int = 4, pos_dropout: float = 0.0, attn_dropout: float = 0.0, ff_dropout: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   A model consisting of a convolutional tower followed by a transformer encoder layer and optional pooling.

   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling layers
   :param conv_norm: If True, apply batch normalization in the convolutional layers.
   :param residual: If True, apply residual connections in the convolutional layers.
   :param crop_len: Number of positions to crop at either end of the output
   :param n_transformers: Number of transformer encoder layers
   :param n_heads: Number of heads in each multi-head attention layer
   :param n_pos_features: Number of positional embedding features
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param attn_dropout: Dropout probability in the output layer
   :param ff_droppout: Dropout probability in the linear feed-forward layers


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



