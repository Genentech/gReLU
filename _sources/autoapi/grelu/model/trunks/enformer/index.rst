grelu.model.trunks.enformer
===========================

.. py:module:: grelu.model.trunks.enformer

.. autoapi-nested-parse::

   The Enformer model architecture and its required classes



Classes
-------

.. autoapisummary::

   grelu.model.trunks.enformer.ConvBlock
   grelu.model.trunks.enformer.FeedForwardBlock
   grelu.model.trunks.enformer.Activation
   grelu.model.trunks.enformer.Crop
   grelu.model.trunks.enformer.Dropout
   grelu.model.trunks.enformer.Norm
   grelu.model.trunks.enformer.EnformerConvTower
   grelu.model.trunks.enformer.EnformerTransformerBlock
   grelu.model.trunks.enformer.EnformerTransformerTower
   grelu.model.trunks.enformer.EnformerTrunk


Module Contents
---------------

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



.. py:class:: EnformerConvTower(n_blocks: int, out_channels: int)

   Bases: :py:obj:`torch.nn.Module`


   :param n_blocks: Number of convolutional/pooling blocks including the stem.
   :param out_channels: Number of channels in the output


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: EnformerTransformerBlock(in_len: int, n_heads: int, key_len: int, attn_dropout: float, pos_dropout: float, ff_dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   Transformer tower for enformer model

   :param in_len: Length of the input
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



.. py:class:: EnformerTransformerTower(in_channels: int, n_blocks: int, n_heads: int, key_len: int, attn_dropout: float, pos_dropout: float, ff_dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   Transformer tower for enformer model

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



.. py:class:: EnformerTrunk(n_conv: int = 7, channels: int = 1536, n_transformers: int = 11, n_heads: int = 8, key_len: int = 64, attn_dropout: float = 0.05, pos_dropout: float = 0.01, ff_dropout: float = 0.4, crop_len: int = 0)

   Bases: :py:obj:`torch.nn.Module`


   Enformer model architecture.

   :param n_conv: Number of convolutional/pooling blocks
   :param channels: Number of output channels for the convolutional tower
   :param n_transformers: Number of stacked transformer blocks
   :param n_heads: Number of attention heads
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param attn_dropout: Dropout probability in the output layer
   :param ff_droppout: Dropout probability in the linear feed-forward layers
   :param crop_len: Number of positions to crop at either end of the output


   .. py:method:: forward(x)


