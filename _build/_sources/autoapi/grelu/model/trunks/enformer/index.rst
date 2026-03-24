grelu.model.trunks.enformer
===========================

.. py:module:: grelu.model.trunks.enformer

.. autoapi-nested-parse::

   The Enformer model architecture and its required classes



Classes
-------

.. autoapisummary::

   grelu.model.trunks.enformer.EnformerConvTower
   grelu.model.trunks.enformer.EnformerTransformerBlock
   grelu.model.trunks.enformer.EnformerTransformerTower
   grelu.model.trunks.enformer.EnformerTrunk


Module Contents
---------------

.. py:class:: EnformerConvTower(n_blocks: int, out_channels: int, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   :param n_blocks: Number of convolutional/pooling blocks including the stem.
   :param out_channels: Number of channels in the output
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: EnformerTransformerBlock(in_len: int, n_heads: int, key_len: int, attn_dropout: float, pos_dropout: float, ff_dropout: float, dtype=None, device=None)

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
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: norm


   .. py:attribute:: mha


   .. py:attribute:: dropout


   .. py:attribute:: ffn


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: EnformerTransformerTower(in_channels: int, n_blocks: int, n_heads: int, key_len: int, attn_dropout: float, pos_dropout: float, ff_dropout: float, dtype=None, device=None)

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
   :param device: Device for the layers.
   :param dtype: Data type for the layers.


   .. py:attribute:: blocks


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: EnformerTrunk(n_conv: int = 7, channels: int = 1536, n_transformers: int = 11, n_heads: int = 8, key_len: int = 64, attn_dropout: float = 0.05, pos_dropout: float = 0.01, ff_dropout: float = 0.4, crop_len: int = 0, dtype=None, device=None)

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
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: conv_tower


   .. py:attribute:: transformer_tower


   .. py:attribute:: pointwise_conv


   .. py:attribute:: act


   .. py:attribute:: crop


   .. py:method:: forward(x)


