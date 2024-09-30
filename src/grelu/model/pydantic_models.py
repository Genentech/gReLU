from typing import List, Optional, Union

from pydantic import BaseModel, Field

# Example usaglse
# config = ConvModelConfig(n_tasks=10)
# model = ConvModel(**config.dict())


class BaseModelConfig(BaseModel):
    n_tasks: int
    crop_len: int = 0
    final_pool_func: str = "avg"


class ConvModelConfig(BaseModelConfig):
    stem_channels: int = 64
    stem_kernel_size: int = 15
    n_conv: int = 2
    channel_init: int = 64
    channel_mult: float = 1
    kernel_size: int = 5
    dilation_init: int = 1
    dilation_mult: float = 1
    act_func: str = "relu"
    norm: bool = False
    pool_func: Optional[str] = None
    pool_size: Optional[int] = None
    residual: bool = False
    dropout: float = 0.0


class DilatedConvModelConfig(BaseModelConfig):
    channels: int = 64
    stem_kernel_size: int = 21
    kernel_size: int = 3
    dilation_mult: float = 2
    act_func: str = "relu"
    n_conv: int = 8
    crop_len: Union[str, int] = "auto"


class ConvGRUModelConfig(BaseModelConfig):
    stem_channels: int = 16
    stem_kernel_size: int = 15
    n_conv: int = 2
    channel_init: int = 16
    channel_mult: float = 1
    kernel_size: int = 5
    act_func: str = "relu"
    conv_norm: bool = False
    pool_func: Optional[str] = None
    pool_size: Optional[int] = None
    residual: bool = False
    n_gru: int = 1
    dropout: float = 0.0
    gru_norm: bool = False


class ConvTransformerModelConfig(BaseModelConfig):
    stem_channels: int = 16
    stem_kernel_size: int = 15
    n_conv: int = 2
    channel_init: int = 16
    channel_mult: float = 1
    kernel_size: int = 5
    act_func: str = "relu"
    norm: bool = False
    pool_func: Optional[str] = None
    pool_size: Optional[int] = None
    residual: bool = False
    n_transformers: int = 1
    key_len: int = 8
    value_len: int = 8
    n_heads: int = 1
    n_pos_features: int = 4
    pos_dropout: float = 0.0
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0


class ConvMLPModelConfig(BaseModelConfig):
    seq_len: int
    stem_channels: int = 16
    stem_kernel_size: int = 15
    n_conv: int = 2
    channel_init: int = 16
    channel_mult: float = 1
    kernel_size: int = 5
    act_func: str = "relu"
    conv_norm: bool = False
    pool_func: Optional[str] = None
    pool_size: Optional[int] = None
    residual: bool = True
    mlp_norm: bool = False
    mlp_act_func: Optional[str] = "relu"
    mlp_hidden_size: List[int] = Field(default_factory=lambda: [8])
    dropout: float = 0.0


class BorzoiModelConfig(BaseModelConfig):
    stem_channels: int = 512
    stem_kernel_size: int = 15
    init_channels: int = 608
    channels: int = 1536
    n_conv: int = 7
    kernel_size: int = 5
    n_transformers: int = 8
    key_len: int = 64
    value_len: int = 192
    pos_dropout: float = 0.0
    attn_dropout: float = 0.0
    n_heads: int = 8
    n_pos_features: int = 32
    final_act_func: Optional[str] = None


class BorzoiPretrainedModelConfig(BaseModelConfig):
    fold: int = 0
    n_transformers: int = 8


class ExplaiNNModelConfig(BaseModel):
    n_tasks: int
    in_len: int
    channels: int = 300
    kernel_size: int = 19


class EnformerModelConfig(BaseModelConfig):
    n_conv: int = 7
    channels: int = 1536
    n_transformers: int = 11
    n_heads: int = 8
    key_len: int = 64
    attn_dropout: float = 0.05
    pos_dropout: float = 0.01
    ff_dropout: float = 0.4
    final_act_func: Optional[str] = None


class EnformerPretrainedModelConfig(BaseModelConfig):
    n_transformers: int = 11


def grelu_models(model_name: Optional[str] = None) -> None:
    """
    Display information about all available models with a pydantic definition.

    If no model name is provided, this function prints the names of all available models.
    If a model name is provided and it exists, this function prints the model's name and its fields,
    including their types and default values. If the model name does not exist, an error message is printed.

    Args:
        model_name (Optional[str]): The name of the model to display information about. If None,
                                    information about all available models is displayed.

    Returns:
        None
    """
    models = {cls.__name__: cls for cls in BaseModelConfig.__subclasses__()}
    if model_name is None:
        print("Available models:")
        for name in models:
            print(f"- {name}")
    elif model_name in models:
        model = models[model_name]
        print(f"Model: {model_name}")
        print("Fields:")
        for name, field in model.__fields__.items():
            field_type = field.annotation
            default = field.default if field.default != ... else "Required"
            print(f"- {name}: type={field_type}, default={default}")
    else:
        print(f"Model '{model_name}' not found.")


...
