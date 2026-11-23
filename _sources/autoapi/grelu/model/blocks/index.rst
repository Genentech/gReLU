grelu.model.blocks
==================

.. py:module:: grelu.model.blocks

.. autoapi-nested-parse::

   `grelu.model.blocks` defines larger blocks that form part of the architecture
   of sequence-to-function deep learning models. Each such block is composed of
   multiple layers.



Classes
-------

.. autoapisummary::

   grelu.model.blocks.LinearBlock
   grelu.model.blocks.ConvBlock
   grelu.model.blocks.ChannelTransformBlock
   grelu.model.blocks.Stem
   grelu.model.blocks.SeparableConv
   grelu.model.blocks.ConvTower
   grelu.model.blocks.FeedForwardBlock
   grelu.model.blocks.GRUBlock
   grelu.model.blocks.TransformerBlock
   grelu.model.blocks.TransformerTower
   grelu.model.blocks.UnetBlock
   grelu.model.blocks.UnetTower


Module Contents
---------------

.. py:class:: LinearBlock(in_len: int, out_len: int, act_func: str = 'relu', dropout: float = 0.0, norm: bool = False, norm_kwargs: Optional[dict] = None, bias: bool = True, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Linear layer followed by optional normalization,
   activation and dropout.

   :param in_len: Length of input
   :param out_len: Length of output
   :param act_func: Name of activation function
   :param dropout: Dropout probability
   :param norm: If True, apply layer normalization
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
   :param bias: If True, include bias term.
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: norm


   .. py:attribute:: linear


   .. py:attribute:: dropout


   .. py:attribute:: act


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvBlock(in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, act_func: str = 'relu', pool_func: Optional[str] = None, pool_size: Optional[str] = None, dropout: float = 0.0, norm: bool = True, norm_type='batch', norm_kwargs: Optional[dict] = None, residual: bool = False, order: str = 'CDNRA', bias: bool = True, return_pre_pool: bool = False, dtype=None, device=None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer along with optional normalization,
   activation, dilation, dropout, residual connection, and pooling.
   The order of these operations can be specified, except
   for pooling, which always comes last.

   :param in_channels: Number of channels in the input
   :param out_channels: Number of channels in the output
   :param kernel_size: Convolutional kernel width
   :param dilation: Dilation
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Pooling width
   :param dropout: Dropout probability
   :param norm: If True, apply normalization layer
   :param norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
   :param norm_kwargs: Additional arguments to be passed to the normalization layer
   :param residual: If True, apply residual connection
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNRA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, residual addition, activation. Pooling is not included
                 as it is always performed last.
   :param return_pre_pool: If this is True and pool_func is not None, the final
                           output will be a tuple (output after pooling, output_before_pooling).
                           This is useful if the output before pooling is required by a later
                           layer.
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights
   :param \*\*kwargs: Additional arguments to be passed to nn.Conv1d


   .. py:attribute:: order
      :value: 'CDNRA'



   .. py:attribute:: conv


   .. py:attribute:: act


   .. py:attribute:: pool


   .. py:attribute:: dropout


   .. py:attribute:: residual
      :value: False



   .. py:attribute:: return_pre_pool
      :value: False



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:class:: ChannelTransformBlock(in_channels: int, out_channels: int, norm: bool = False, act_func: str = 'relu', dropout: float = 0.0, order: str = 'CDNA', norm_type='batch', norm_kwargs: Optional[dict] = None, if_equal: bool = False, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer with kernel size=1 along with optional normalization, activation
   and dropout

   :param in_channels: Number of channels in the input
   :param out_channels: Number of channels in the output
   :param act_func: Name of the activation function
   :param dropout: Dropout probability
   :param norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, activation.
   :param if_equal: If True, create a layer even if the input and output channels are equal.
   :param device: Device on which to store the weights
   :param dtype: Data type of the weights


   .. py:attribute:: order
      :value: 'CDNA'



   .. py:attribute:: conv


   .. py:attribute:: act


   .. py:attribute:: dropout


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Stem(out_channels: int, kernel_size: int, act_func: str = 'relu', pool_func: Optional[str] = None, pool_size: Optional[str] = None, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer followed by optional activation and pooling.
   Meant to take one-hot encoded DNA sequence as input

   :param out_channels: Number of channels in the output
   :param kernel_size: Convolutional kernel width
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of pooling layer
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: conv


   .. py:attribute:: act


   .. py:attribute:: pool


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: SeparableConv(in_channels: int, kernel_size: int, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Equivalent class to `tf.keras.layers.SeparableConv1D`

   :param in_channels: Number of channels in the input
   :param kernel_size: Convolutional kernel width
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: depthwise


   .. py:attribute:: pointwise


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvTower(stem_channels: int, stem_kernel_size: int, n_blocks: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, dilation_init: int = 1, dilation_mult: float = 1, act_func: str = 'relu', norm: bool = False, norm_kwargs: Optional[dict] = None, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, dropout: float = 0.0, order: str = 'CDNRA', crop_len: Union[int, str] = 0, dtype=None, device=None)

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
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param residual: If True, apply residual connection
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNRA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, residual addition, activation. Pooling is not included
                 as it is always performed last.
   :param crop_len: Number of positions to crop at either end of the output
   :param dtype: Data type of the weights
   :param device: Device on which to store


   .. py:attribute:: blocks


   .. py:attribute:: receptive_field


   .. py:attribute:: pool_factor
      :value: 1



   .. py:attribute:: out_channels


   .. py:attribute:: crop


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: FeedForwardBlock(in_len: int, dropout: float = 0.0, act_func: str = 'relu', norm_kwargs: Optional[dict] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

   :param in_len: Length of the input tensor
   :param dropout: Dropout probability
   :param act_func: Name of the activation function
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param \*\*kwargs: Additional arguments to be passed to the linear layers


   .. py:attribute:: dense1


   .. py:attribute:: dense2


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: GRUBlock(in_channels: int, n_layers: int = 1, dropout: float = 0.0, act_func: str = 'relu', norm: bool = False, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Stacked bidirectional GRU layers followed by a feed-forward network.

   :param in_channels: The number of channels in the input
   :param n_layers: The number of GRU layers
   :param gru_hidden_size: Number of hidden elements in GRU layers
   :param dropout: Dropout probability
   :param act_func: Name of the activation function for feed-forward network
   :param norm: If True, include layer normalization in feed-forward network.
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: gru


   .. py:attribute:: ffn


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: TransformerBlock(in_len: int, n_heads: int, attn_dropout: float, ff_dropout: float, flash_attn: bool, n_pos_features: Optional[int] = None, key_len: Optional[int] = None, value_len: Optional[int] = None, pos_dropout: Optional[float] = None, norm_kwargs: Optional[dict] = None, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   A block containing a multi-head attention layer followed by a feed-forward
   network and residual connections.

   :param in_len: Length of the input
   :param n_heads: Number of attention heads
   :param attn_dropout: Dropout probability in the output layer
   :param ff_droppout: Dropout probability in the linear feed-forward layers
   :param flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
                      pos_dropout and n_pos_features are ignored.
   :param n_pos_features: Number of positional embedding features
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: flash_attn_warn
      :value: False



   .. py:attribute:: norm


   .. py:attribute:: dropout


   .. py:attribute:: ffn


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: TransformerTower(in_channels: int, n_blocks: int = 1, n_heads: int = 1, n_pos_features: int = 32, key_len: int = 64, value_len: int = 64, pos_dropout: float = 0.0, attn_dropout: float = 0.0, ff_dropout: float = 0.0, norm_kwargs: Optional[dict] = None, flash_attn: bool = False, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Multiple stacked transformer encoder layers.

   :param in_channels: Number of channels in the input
   :param n_blocks: Number of stacked transformer blocks
   :param n_heads: Number of attention heads
   :param n_pos_features: Number of positional embedding features
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param attn_dropout: Dropout probability in the attention layer
   :param ff_dropout: Dropout probability in the feed-forward layers
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param flash_attn: If True, uses Flash Attention with Rotational Position Embeddings
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: UnetBlock(in_channels: int, y_in_channels: int, norm_type='batch', norm_kwargs: Optional[dict] = None, act_func='gelu_borzoi', dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Upsampling U-net block

   :param in_channels: Number of channels in the input
   :param y_in_channels: Number of channels in the higher-resolution representation.
   :param norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param dtype: Data type of the weights
   :param device: Device on which to store the weights


   .. py:attribute:: conv


   .. py:attribute:: upsample


   .. py:attribute:: channel_transform


   .. py:attribute:: sconv


   .. py:method:: forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: UnetTower(in_channels: int, y_in_channels: List[int], n_blocks: int, act_func: str = 'gelu_borzoi', **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Upsampling U-net tower for the Borzoi model

   :param in_channels: Number of channels in the input
   :param y_in_channels: Number of channels in the higher-resolution representations.
   :param n_blocks: Number of U-net blocks
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param kwargs: Additional arguments to be passed to the U-net blocks


   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor, ys: List[torch.Tensor]) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)
      :param ys: Higher-resolution representations

      :returns: Output tensor



