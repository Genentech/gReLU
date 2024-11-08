grelu.model.layers
==================

.. py:module:: grelu.model.layers

.. autoapi-nested-parse::

   Commonly used layers to build deep learning models.



Classes
-------

.. autoapisummary::

   grelu.model.layers.Activation
   grelu.model.layers.Pool
   grelu.model.layers.AdaptivePool
   grelu.model.layers.Norm
   grelu.model.layers.ChannelTransform
   grelu.model.layers.Dropout
   grelu.model.layers.Crop
   grelu.model.layers.Attention
   grelu.model.layers.FlashAttention


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



.. py:class:: AdaptivePool(func: Optional[str] = None)

   Bases: :py:obj:`torch.nn.Module`


   An Adaptive Pooling layer. This layer does not have a defined pooling width but
   instead pools together all the values in the last axis.

   :param func: Type of pooling function. Supported values are 'avg' or 'max'. If None,
                will return nn.Identity.

   :raises NotImplementedError: If 'func' is not a supported pooling function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: Norm(func: Optional[str] = None, in_dim: Optional[int] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A batch normalization or layer normalization layer.

   :param func: Type of normalization function. Supported values are 'batch',
                'syncbatch', 'instance',  or 'layer'. If None, will return nn.Identity.
   :param in_dim: Number of features in the input tensor.
   :param \*\*kwargs: Additional arguments to pass to the normalization function.


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



.. py:class:: Dropout(p: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   Optional dropout layer

   :param p: Dropout probability. If this is set to 0, will return nn.Identity.


   .. py:attribute:: layer


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



.. py:class:: Attention(in_len: int, key_len: int, value_len: int, n_heads: int, n_pos_features: int, pos_dropout: float = 0, attn_dropout: float = 0, device=None, dtype=None)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: in_len


   .. py:attribute:: key_len


   .. py:attribute:: value_len


   .. py:attribute:: n_heads


   .. py:attribute:: n_pos_features


   .. py:attribute:: to_q


   .. py:attribute:: to_k


   .. py:attribute:: to_v


   .. py:attribute:: to_out


   .. py:attribute:: positional_embed


   .. py:attribute:: to_pos_k


   .. py:attribute:: rel_content_bias


   .. py:attribute:: rel_pos_bias


   .. py:attribute:: pos_dropout


   .. py:attribute:: attn_dropout


   .. py:method:: _get_pos_k(x)


   .. py:method:: get_attn_scores(x, return_v=False)


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: FlashAttention(embed_dim: int, n_heads: int, dropout_p=0.0, device=None, dtype=None)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: embed_dim


   .. py:attribute:: n_heads


   .. py:attribute:: head_dim


   .. py:attribute:: dropout_p


   .. py:attribute:: qkv


   .. py:attribute:: out


   .. py:attribute:: rotary_embed


   .. py:attribute:: flash_attn_qkvpacked_func


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (batch_size, seq_len, embed_dim)

      :returns: Output tensor



