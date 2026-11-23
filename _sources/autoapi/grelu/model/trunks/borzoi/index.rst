grelu.model.trunks.borzoi
=========================

.. py:module:: grelu.model.trunks.borzoi

.. autoapi-nested-parse::

   The Borzoi model architecture and its required classes.



Classes
-------

.. autoapisummary::

   grelu.model.trunks.borzoi.BorzoiConvTower
   grelu.model.trunks.borzoi.BorzoiTrunk


Module Contents
---------------

.. py:class:: BorzoiConvTower(stem_channels: int, stem_kernel_size: int, init_channels: int, out_channels: int, kernel_size: int, n_blocks: int, norm_type='batch', norm_kwargs=None, act_func='gelu_borzoi', dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional tower for the Borzoi model.

   :param stem_channels: Number of channels in the first (stem) convolutional layer
   :param stem_kernel_size: Width of the convolutional kernel in the first (stem) convolutional layer
   :param init_channels: Number of channels in the first convolutional block after the stem
   :param out_channels: Number of channels in the output
   :param kernel_size: Width of the convolutional kernel
   :param n_blocks: Number of convolutional/pooling blocks, including the stem
   :param norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
   :param norm_kwargs: Additional arguments to be passed to the normalization layer
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: blocks


   .. py:attribute:: filters


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: BorzoiTrunk(stem_channels: int, stem_kernel_size: int, init_channels: int, n_conv: int, kernel_size: int, channels: int, n_transformers: int, key_len: int, value_len: int, pos_dropout: float, attn_dropout: float, ff_dropout: float, n_heads: int, n_pos_features: int, crop_len: int, flash_attn: bool, norm_type='batch', norm_kwargs=None, act_func='gelu_borzoi', dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   Trunk consisting of conv, transformer and U-net layers for the Borzoi model.

   :param stem_channels: Number of channels in the first (stem) convolutional layer
   :param stem_kernel_size: Width of the convolutional kernel in the first (stem) convolutional layer
   :param init_channels: Number of channels in the first convolutional block after the stem
   :param n_conv: Number of convolutional/pooling blocks, including the stem
   :param kernel_size: Width of the convolutional kernel
   :param channels: Number of channels in the output
   :param n_transformers: Number of transformer blocks
   :param key_len: Length of the key
   :param value_len: Length of the value
   :param pos_dropout: Dropout rate for positional embeddings
   :param attn_dropout: Dropout rate for attention
   :param n_heads: Number of attention heads
   :param n_pos_features: Number of positional features
   :param crop_len: Length of the crop
   :param flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
                      pos_dropout and n_pos_features are ignored.
   :param norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
   :param norm_kwargs: Additional arguments to be passed to the normalization layer
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: conv_tower


   .. py:attribute:: transformer_tower


   .. py:attribute:: unet_tower


   .. py:attribute:: pointwise_conv


   .. py:attribute:: act


   .. py:attribute:: crop


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



