grelu.model.trunks.borzoi
=========================

.. py:module:: grelu.model.trunks.borzoi

.. autoapi-nested-parse::

   The Borzoi model architecture and its required classes.



Classes
-------

.. autoapisummary::

   grelu.model.trunks.borzoi.ConvBlock
   grelu.model.trunks.borzoi.Stem
   grelu.model.trunks.borzoi.TransformerTower
   grelu.model.trunks.borzoi.UnetTower
   grelu.model.trunks.borzoi.Activation
   grelu.model.trunks.borzoi.Crop
   grelu.model.trunks.borzoi.BorzoiConvTower
   grelu.model.trunks.borzoi.BorzoiTrunk


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



.. py:class:: BorzoiConvTower(stem_channels: int, stem_kernel_size: int, init_channels: int, out_channels: int, kernel_size: int, n_blocks: int)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional tower for the Borzoi model.

   :param stem_channels: Number of channels in the first (stem) convolutional layer
   :param stem_kernel_size: Width of the convolutional kernel in the first (stem) convolutional layer
   :param init_channels: Number of channels in the first convolutional block after the stem
   :param out_channels: Number of channels in the output
   :param kernel_size: Width of the convolutional kernel
   :param n_blocks: Number of convolutional/pooling blocks, including the stem


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: BorzoiTrunk(stem_channels: int, stem_kernel_size: int, init_channels: int, n_conv: int, kernel_size: int, channels: int, n_transformers: int, key_len: int, value_len: int, pos_dropout: float, attn_dropout: float, n_heads: int, n_pos_features: int, crop_len: int)

   Bases: :py:obj:`torch.nn.Module`


   Trunk consisting of conv, transformer and U-net layers for the Borzoi model.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



