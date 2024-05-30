grelu.model.trunks.explainn
===========================

.. py:module:: grelu.model.trunks.explainn


Classes
-------

.. autoapisummary::

   grelu.model.trunks.explainn.Activation
   grelu.model.trunks.explainn.Dropout
   grelu.model.trunks.explainn.Norm
   grelu.model.trunks.explainn.ExplaiNNConvBlock
   grelu.model.trunks.explainn.ExplaiNNTrunk


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



.. py:class:: ExplaiNNConvBlock(in_channels: int, out_channels: int, kernel_size: int, groups: int, act_func: str, dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional block for the ExplaiNN model.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ExplaiNNTrunk(in_len: int, channels=300, kernel_size=19)

   Bases: :py:obj:`torch.nn.Module`


   The ExplaiNN model architecture.

   :param n_tasks: number of outputs
   :type n_tasks: int
   :param input_length: length of the input sequences
   :type input_length: int
   :param channels: number of independent CNN units (default=300)
   :type channels: int
   :param kernel_size: size of each unit's conv. filter (default=19)
   :type kernel_size: int


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



