grelu.model.blocks
==================

.. py:module:: grelu.model.blocks

.. autoapi-nested-parse::

   Blocks composed of multiple layers.



Classes
-------

.. autoapisummary::

   grelu.model.blocks.Activation
   grelu.model.blocks.Attention
   grelu.model.blocks.ChannelTransform
   grelu.model.blocks.Crop
   grelu.model.blocks.Dropout
   grelu.model.blocks.Norm
   grelu.model.blocks.Pool
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

.. py:class:: Activation(func: str)

   Bases: :py:obj:`torch.nn.Module`


   A nonlinear activation layer.

   :param func: The type of activation function. Supported values are 'relu',
                'elu', 'softplus', 'gelu', 'gelu_enformer' and 'exp'. If None, will return nn.Identity.

   :raises NotImplementedError: If 'func' is not a supported activation function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Attention(in_len: int, key_len: int, value_len: int, n_heads: int, n_pos_features: int, pos_dropout: float = 0, attn_dropout: float = 0)

   Bases: :py:obj:`torch.nn.Module`


   .. py:method:: _get_pos_k(x)


   .. py:method:: get_attn_scores(x, return_v=False)


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ChannelTransform(in_channels: int, out_channels: int = 1, if_equal: bool = False, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A convolutional layer to transform the number of channels in the input.

   :param in_channels: Number of channels in the input
   :param out_channels: Number of channels in the output
   :param if_equal: Whether to create layer if input and output channels are equal
   :param \*\*kwargs: Additional arguments to pass to the convolutional layer.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Crop(crop_len: int = 0, receptive_field: Optional[int] = None)

   Bases: :py:obj:`torch.nn.Module`


   Optional cropping layer.

   :param crop_len: Number of positions to crop at each end of the input.
   :param receptive_field: Receptive field of the model to calculate crop_len.
                           Only needed if crop_len is None.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Dropout(p: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   Optional dropout layer

   :param p: Dropout probability. If this is set to 0, will return nn.Identity.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Norm(func: Optional[str] = None, in_dim: Optional[int] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A batch normalization or layer normalization layer.

   :param func: Type of normalization function. Supported values are 'batch' or 'layer'. If None,
                will return nn.Identity.
   :param in_dim: Number of features in the input tensor.
   :param \*\*kwargs: Additional arguments to pass to the normalization function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Pool(func: Optional[str], pool_size: Optional[int] = None, in_channels: Optional[int] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A pooling layer.

   :param func: Type of pooling function. Supported values are 'avg', 'max',
                or 'attn'. If None, will return nn.Identity.
   :param pool_size: The number of positions to pool together
   :param in_channels: Number of channels in the input. Only needeed for attention pooling.
   :param \*\*kwargs: Additional arguments to pass to the pooling function.

   :raises NotImplementedError: If 'func' is not a supported pooling function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: LinearBlock(in_len: int, out_len: int, act_func: str = 'relu', dropout: float = 0.0, norm: bool = False, bias: bool = True)

   Bases: :py:obj:`torch.nn.Module`


   Linear layer followed by optional normalization,
   activation and dropout.

   :param in_len: Length of input
   :param out_len: Length of output
   :param act_func: Name of activation function
   :param dropout: Dropout probability
   :param norm: If True, apply layer normalization
   :param bias: If True, include bias term.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvBlock(in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, act_func: str = 'relu', pool_func: Optional[str] = None, pool_size: Optional[str] = None, dropout: float = 0.0, norm: bool = True, residual: bool = False, order: str = 'CDNRA', bias: bool = True, return_pre_pool: bool = False, **kwargs)

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
   :param norm: If True, apply batch norm
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
   :param \*\*kwargs: Additional arguments to be passed to nn.Conv1d


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:class:: ChannelTransformBlock(in_channels: int, out_channels: int, norm: bool = False, act_func: str = 'relu', dropout: float = 0.0, order: str = 'CDNA', if_equal: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer with kernel size=1 along with optional normalization, activation
   and dropout

   :param in_channels: Number of channels in the input
   :param out_channels: Number of channels in the output
   :param act_func: Name of the activation function
   :param dropout: Dropout probability
   :param norm: If True, apply batch norm
   :param order: A string representing the order in which operations are
                 to be performed on the input. For example, "CDNA" means that the
                 operations will be performed in the order: convolution, dropout,
                 batch norm, activation.
   :param if_equal: If True, create a layer even if the input and output channels are equal.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Stem(out_channels: int, kernel_size: int, act_func: str = 'relu', pool_func: Optional[str] = None, pool_size: Optional[str] = None)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional layer followed by optional activation and pooling.
   Meant to take one-hot encoded DNA sequence as input

   :param out_channels: Number of channels in the output
   :param kernel_size: Convolutional kernel width
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of pooling layer


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: SeparableConv(in_channels: int, kernel_size: int)

   Bases: :py:obj:`torch.nn.Module`


   Equivalent class to `tf.keras.layers.SeparableConv1D`

   :param in_channels: Number of channels in the input
   :param kernel_size: Convolutional kernel width


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



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



.. py:class:: FeedForwardBlock(in_len: int, dropout: float = 0.0, act_func: str = 'relu')

   Bases: :py:obj:`torch.nn.Module`


   2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

   :param in_len: Length of the input tensor
   :param dropout: Dropout probability
   :param act_func: Name of the activation function


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



.. py:class:: TransformerBlock(in_len: int, n_heads: int, n_pos_features: int, key_len: int, value_len: int, pos_dropout: float, attn_dropout: float, ff_dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   A block containing a multi-head attention layer followed by a feed-forward
   network and residual connections.

   :param in_len: Length of the input
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



.. py:class:: UnetBlock(in_channels: int, y_in_channels: int)

   Bases: :py:obj:`torch.nn.Module`


   Upsampling U-net block

   :param in_channels: Number of channels in the input
   :param y_in_channels: Number of channels in the higher-resolution representation.


   .. py:method:: forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: UnetTower(in_channels: int, y_in_channels: List[int], n_blocks: int)

   Bases: :py:obj:`torch.nn.Module`


   Upsampling U-net tower for the Borzoi model

   :param in_channels: Number of channels in the input
   :param y_in_channels: Number of channels in the higher-resolution representations.
   :param n_blocks: Number of U-net blocks


   .. py:method:: forward(x: torch.Tensor, ys: List[torch.Tensor]) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)
      :param ys: Higher-resolution representations

      :returns: Output tensor



