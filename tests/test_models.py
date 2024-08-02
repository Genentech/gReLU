import torch

import wandb
from grelu.model.models import (
    BorzoiModel,
    BorzoiPretrainedModel,
    ConvGRUModel,
    ConvMLPModel,
    ConvModel,
    ConvTransformerModel,
    DilatedConvModel,
    EnformerModel,
    EnformerPretrainedModel,
)
from grelu.resources import DEFAULT_WANDB_HOST
from grelu.sequence.format import convert_input_type

inputs = convert_input_type(["A" * 128], "one_hot")


try:
    wandb.login(host=DEFAULT_WANDB_HOST, anonymous="never", timeout=0)
except wandb.errors.UsageError:  # login anonymously if not logged in already
    wandb.login(host=DEFAULT_WANDB_HOST, relogin=True, anonymous="must", timeout=0)


# Test a fully convolutional model with residual connections and autocropping
def test_conv_model():
    model = ConvModel(
        n_tasks=5,
        n_conv=3,
        stem_channels=8,
        stem_kernel_size=21,
        channel_init=16,
        channel_mult=1.2,
        kernel_size=3,
        act_func="relu",
        residual=True,
        crop_len="auto",
        final_pool_func="avg",
    ).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 3

    # Check the conv tower params
    assert model.embedding.conv_tower.pool_factor == 1
    assert model.embedding.conv_tower.receptive_field == 25

    # Check embedding shape
    emb = model.embedding(inputs)
    assert emb.shape == (1, 19, 104)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)


def test_dilatedconvmodel():
    inputs = torch.rand(1, 4, 2114)

    # Build the model
    model = DilatedConvModel(
        n_tasks=1,
        n_conv=9,
        crop_len="auto",
        final_pool_func=None,
        stem_kernel_size=21,
        kernel_size=3,
        dilation_mult=2,
    ).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 9

    # Check the conv tower params
    assert model.embedding.conv_tower.pool_factor == 1
    assert model.embedding.conv_tower.receptive_field == 1041

    # Check embedding shape
    emb = model.embedding(inputs)
    assert emb.shape == (1, 64, 1074)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 1, 1074)


def test_convgrumodel():
    model = ConvGRUModel(
        n_tasks=5,
        n_conv=3,
        stem_channels=16,
        channel_init=16,
        channel_mult=1,
        stem_kernel_size=15,
        kernel_size=3,
        act_func="gelu",
        pool_func="max",
        pool_size=2,
        residual=False,
        crop_len=0,
        n_gru=1,
        dropout=0.1,
        final_pool_func="avg",
    ).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 3

    # Check the conv tower params
    assert model.embedding.conv_tower.pool_factor == 4

    # Check intermediate shapes
    x = model.embedding.conv_tower(inputs)
    assert x.shape == (1, 16, 32)
    x = model.embedding.gru_tower(x)
    assert x.shape == (1, 16, 32)
    x = model.head(x)
    assert x.shape == (1, 5, 1)

    # Check embedding shape
    emb = model.embedding(inputs)
    assert emb.shape == (1, 16, 32)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)


def test_convtransformermodel():
    model = ConvTransformerModel(
        n_tasks=5,
        n_conv=3,
        stem_channels=16,
        channel_init=16,
        channel_mult=1,
        stem_kernel_size=15,
        kernel_size=3,
        act_func="gelu",
        norm=False,
        pool_func="max",
        pool_size=2,
        residual=True,
        crop_len=10,
        n_transformers=2,
        n_heads=4,
        n_pos_features=32,
        key_len=32,
        value_len=32,
        final_pool_func=None,
    ).eval()
    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 3

    # Check the conv tower params
    assert model.embedding.conv_tower.pool_factor == 4

    # Check intermediate shapes
    x = model.embedding.conv_tower(inputs)
    assert x.shape == (1, 16, 12)
    x = model.embedding.transformer_tower(x)
    assert x.shape == (1, 16, 12)
    x = model.head(x)
    assert x.shape == (1, 5, 12)

    # Check embedding shape
    emb = model.embedding(inputs)
    assert emb.shape == (1, 16, 12)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 12)


def test_convmlpmodel():
    # Build the model
    model = ConvMLPModel(
        seq_len=128,
        n_tasks=5,
        n_conv=3,
        stem_channels=4,
        stem_kernel_size=3,
        channel_init=8,
        channel_mult=1,
        kernel_size=3,
        act_func="elu",
        conv_norm=True,
        pool_func="attn",
        pool_size=2,
        residual=False,
        mlp_hidden_size=[7, 3],
        mlp_norm=True,
    ).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 3

    # Check the conv tower params
    assert model.embedding.conv_tower.pool_factor == 4

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)

    # Check embedding shape
    emb = model.embedding(inputs)
    assert emb.shape == (1, 8, 32)


def test_enformer():
    # Build the model
    model = EnformerModel(n_tasks=5, crop_len=5, n_transformers=1, n_conv=3).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 3

    # Check embedding shape
    pred = model.embedding(inputs)
    assert pred.shape == (1, 3072, 6)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)


def test_pretrained_enformer():
    # Build the model
    model = EnformerPretrainedModel(n_tasks=5, n_transformers=0, crop_len=0).eval()

    # Check embedding shape
    pred = model.embedding(inputs)
    assert pred.shape == (1, 3072, 1)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)


def test_borzoi():
    # Build the model
    model = BorzoiModel(n_tasks=5, crop_len=5, n_transformers=1, n_conv=4).eval()

    # Check the number of blocks
    assert len(model.embedding.conv_tower.blocks) == 4

    # Check embedding shape
    pred = model.embedding(inputs)
    assert pred.shape == (1, 1920, 22)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)


def test_pretrained_borzoi():
    # Build the model
    model = BorzoiPretrainedModel(
        n_tasks=5, n_transformers=0, crop_len=0, fold=0
    ).eval()

    # Check embedding shape
    pred = model.embedding(inputs)
    assert pred.shape == (1, 1920, 4)

    # Check prediction shape
    pred = model(inputs)
    assert pred.shape == (1, 5, 1)
