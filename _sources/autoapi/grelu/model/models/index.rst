grelu.model.models
==================

.. py:module:: grelu.model.models

.. autoapi-nested-parse::

   `grelu.model.models` defines complete architectures for sequence-to-function
   deep learning models.

   All models inherit from the `BaseModel` class and are composed of `embedding`
   and `head` sections, which use classes defined in `grelu.model.trunks` and
   `grelu.model.heads` respectively.

   All models have a `forward` function (inherited from `BaseModel`) which takes
   as input a one-hot encoded sequence tensor of shape (N, 4, length) and return a
   tensor of shape (N, tasks, output_length).



Classes
-------

.. autoapisummary::

   grelu.model.models.BaseModel
   grelu.model.models.ConvModel
   grelu.model.models.DilatedConvModel
   grelu.model.models.ConvGRUModel
   grelu.model.models.ConvTransformerModel
   grelu.model.models.ConvMLPModel
   grelu.model.models.BorzoiModel
   grelu.model.models.BorzoiPretrainedModel
   grelu.model.models.ExplaiNNModel
   grelu.model.models.EnformerModel
   grelu.model.models.EnformerPretrainedModel


Module Contents
---------------

.. py:class:: BaseModel(embedding: torch.nn.Module, head: torch.nn.Module)

   Bases: :py:obj:`torch.nn.Module`


   Base model class


   .. py:attribute:: embedding


   .. py:attribute:: head


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ConvModel(n_tasks: int, stem_channels: int = 64, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 64, channel_mult: float = 1, kernel_size: int = 5, dilation_init: int = 1, dilation_mult: float = 1, act_func: str = 'relu', norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, dropout: float = 0.0, crop_len: int = 0, final_pool_func: str = 'avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   A fully convolutional model that optionally includes pooling,
   residual connections, batch normalization, or dilated convolutions.

   :param n_tasks: Number of channels in the output
   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
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
   :param crop_len: Number of positions to crop at either end of the output
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: DilatedConvModel(n_tasks: int, channels: int = 64, stem_kernel_size: int = 21, kernel_size: int = 3, dilation_mult: float = 2, act_func: str = 'relu', n_conv: int = 8, crop_len: Union[str, int] = 'auto', final_pool_func: str = 'avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   A model architecture based on dilated convolutional layers with residual connections.
   Inspired by the ChromBPnet model architecture.

   :param n_tasks: Number of channels in the output
   :param channels: Number of channels for all convolutional layers
   :param stem_kernel_size: Kernel width for the stem
   :param n_blocks: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param dilation_mult: Factor by which to multiply the dilation in each block
   :param act_func: Name of the activation function
   :param crop_len: Number of positions to crop at either end of the output
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: ConvGRUModel(n_tasks: int, stem_channels: int = 16, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, act_func: str = 'relu', conv_norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, crop_len: int = 0, n_gru: int = 1, dropout: float = 0.0, gru_norm: bool = False, final_pool_func: str = 'avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   A model consisting of a convolutional tower followed by a bidirectional GRU layer and optional pooling.

   :param n_tasks: Number of channels in the output
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
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: ConvTransformerModel(n_tasks: int, stem_channels: int = 16, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, act_func: str = 'relu', norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = False, crop_len: int = 0, n_transformers=1, key_len: int = 8, value_len: int = 8, n_heads: int = 1, n_pos_features: int = 4, pos_dropout: float = 0.0, attn_dropout: float = 0.0, ff_dropout: float = 0.0, final_pool_func: str = 'avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   A model consisting of a convolutional tower followed by a transformer encoder layer and optional pooling.

   :param n_tasks: Number of channels in the output
   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling layers
   :param norm: If True, apply batch normalization in the convolutional layers.
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
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: ConvMLPModel(seq_len: int, n_tasks: int, stem_channels: int = 16, stem_kernel_size: int = 15, n_conv: int = 2, channel_init: int = 16, channel_mult: float = 1, kernel_size: int = 5, act_func: str = 'relu', conv_norm: bool = False, pool_func: Optional[str] = None, pool_size: Optional[int] = None, residual: bool = True, mlp_norm: bool = False, mlp_act_func: Optional[str] = 'relu', mlp_hidden_size: List[int] = [8], dropout: float = 0.0, dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   A convolutional tower followed by a Multi-head perceptron (MLP) layer.

   :param n_tasks: Number of channels in the output
   :param seq_len: Input length
   :param stem_channels: Number of channels in the stem
   :param stem_kernel_size: Kernel width for the stem
   :param n_conv: Number of convolutional blocks, not including the stem
   :param kernel_size: Convolutional kernel width
   :param channel_init: Initial number of channels,
   :param channel_mult: Factor by which to multiply the number of channels in each block
   :param act_func: Name of the activation function
   :param pool_func: Name of the pooling function
   :param pool_size: Width of the pooling
   :param conv_norm: If True, apply batch norm in the convolutional layers
   :param residual: If True, apply residual connection
   :param mlp_norm: If True, apply layer norm in the MLP layers
   :param mlp_hidden_size: A list containing the dimensions for each hidden layer of the MLP.
   :param dropout: Dropout probability for the MLP layers.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: BorzoiModel(n_tasks: int, stem_channels: int = 512, stem_kernel_size: int = 15, init_channels: int = 608, channels: int = 1536, n_conv: int = 7, kernel_size: int = 5, n_transformers: int = 8, key_len: int = 64, value_len: int = 192, pos_dropout: float = 0.01, attn_dropout: float = 0.05, ff_dropout: float = 0.2, norm_kwargs: Optional[dict] = None, n_heads: int = 8, n_pos_features: int = 32, crop_len: int = 16, act_func: str = 'gelu_borzoi', final_act_func: Optional[str] = None, final_pool_func: Optional[str] = 'avg', flash_attn=False, dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   Model consisting of Borzoi conv and transformer layers followed by U-net upsampling and optional pooling.

   :param stem_channels: Number of channels in the first (stem) convolutional layer
   :param stem_kernel_size: Width of the convolutional kernel in the first (stem) convolutional layer
   :param init_channels: Number of channels in the first convolutional block after the stem
   :param channels: Number of channels in the output of the convolutional tower
   :param kernel_size: Width of the convolutional kernel
   :param n_conv: Number of convolutional/pooling blocks
   :param n_transformers: Number of stacked transformer blocks
   :param n_pos_features: Number of features in the positional embeddings
   :param n_heads: Number of attention heads
   :param key_len: Length of the key vectors
   :param value_len: Length of the value vectors.
   :param pos_dropout: Dropout probability in the positional embeddings
   :param attn_dropout: Dropout probability in the attention layer
   :param crop_len: Number of positions to crop at either end of the output
   :param head_act_func: Name of the activation function to use in the final layer
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
                      pos_dropout and n_pos_features are ignored.
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers.
                       Defaults to {"eps": 0.001}.
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: BorzoiPretrainedModel(n_tasks: int, fold: int = 0, n_transformers: int = 8, crop_len=0, act_func='gelu_borzoi', norm_kwargs: Optional[dict] = None, final_pool_func='avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   Borzoi model with published weights (ported from Keras).

   :param n_tasks: Number of tasks for the model to predict
   :param fold: Which fold of the model to load (default=0)
   :param n_transformers: Number of transformer blocks to use (default=8)
   :param crop_len: Number of positions to crop at either end of the output (default=0)
   :param act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
                    tanh approximation (different from PyTorch's default GELU implementation).
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers.
                       Defaults to {"eps": 0.001}.
   :param final_pool_func: Name of the pooling function to apply to the final output (default="avg")
   :param dtype: Data type for the layers
   :param device: Device for the layers


.. py:class:: ExplaiNNModel(n_tasks: int, in_len: int, channels=300, kernel_size=19, dtype=None, device=None)

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
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: EnformerModel(n_tasks: int, n_conv: int = 7, channels: int = 1536, n_transformers: int = 11, n_heads: int = 8, key_len: int = 64, attn_dropout: float = 0.05, pos_dropout: float = 0.01, ff_dropout: float = 0.4, crop_len: int = 0, final_act_func: Optional[str] = None, final_pool_func: Optional[str] = 'avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   Enformer model architecture.

   :param n_tasks: Number of tasks for the model to predict
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
   :param final_act_func: Name of the activation function to use in the final layer
   :param final_pool_func: Name of the pooling function to apply to the final output.
                           If None, no pooling will be applied at the end.
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


.. py:class:: EnformerPretrainedModel(n_tasks: int, n_transformers: int = 11, crop_len=0, final_pool_func='avg', dtype=None, device=None)

   Bases: :py:obj:`BaseModel`


   Borzoi model with published weights (ported from Keras).


