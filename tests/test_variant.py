import os
import warnings

import pandas as pd
from torch import Tensor, nn

from grelu.lightning import LightningModel
from grelu.variant import check_reference, filter_variants, predict_variant_effects

cwd = os.path.realpath(os.path.dirname(__file__))


def test_filter_variants():
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 6,
            "pos": range(1, 7),
            "ref": ["A", "AAAA", "T", "A", "-", "C"],
            "alt": ["C", "-", "GG", "-", "G", "N"],
        }
    )
    assert filter_variants(
        df, max_insert_len=3, max_del_len=3, standard_bases=True
    ).equals(df.iloc[[0, 2, 3, 4], :])
    assert filter_variants(
        df, max_insert_len=0, max_del_len=0, standard_bases=True
    ).equals(df.iloc[[0], :])


variant_file = os.path.join(cwd, "files", "test_variants.txt")
variants = pd.read_table(variant_file, usecols=(0, 1, 2))
variants["ref"] = variants.variation.apply(lambda x: x.split(">")[0])
variants["alt"] = variants.variation.apply(lambda x: x.split(">")[1].split(",")[0])
variants = variants[["chrom", "pos", "ref", "alt"]]


def test_check_reference():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_reference(variants)


def test_predict_variant_effects():
    model = LightningModel(
        model_params={
            "model_type": "ConvModel",
            "n_tasks": 2,
            "n_conv": 0,
            "stem_channels": 2,
            "stem_kernel_size": 1,
            "act_func": None,
            "norm": False,
            "final_pool_func": "avg",
        },
        train_params={
            "task": "regression",
            "loss": "MSE",
        },
    )

    weight = Tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]]).unsqueeze(2)
    bias = Tensor([0.0, 0.0])
    model.model.embedding.conv_tower.blocks[0].conv.bias = nn.Parameter(bias)
    model.model.embedding.conv_tower.blocks[0].conv.weight = nn.Parameter(weight)

    effects = predict_variant_effects(
        variants, model, seq_len=3, compare_func="divide", return_ad=False
    )
    assert effects.shape == (31, 2, 1)
