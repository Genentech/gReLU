import os

import numpy as np
import pandas as pd
from torch import Tensor, nn

from grelu.interpret.motifs import (
    motifs_to_strings,
    run_tomtom,
    scan_sequences,
    trim_pwm,
)
from grelu.interpret.score import ISM_predict, get_attention_scores, get_attributions
from grelu.interpret.simulate import (
    marginalize_pattern_spacing,
    marginalize_patterns,
    shuffle_tiles,
)
from grelu.lightning import LightningModel
from grelu.sequence.utils import generate_random_sequences

cwd = os.path.realpath(os.path.dirname(__file__))
meme_file = os.path.join(cwd, "files", "test.meme")


def test_motifs_to_strings(motifs=meme_file):
    assert motifs_to_strings(motifs, sample=False) == ["CACGTG", "TGCGTG"]
    assert motifs_to_strings(motifs, names=["MA0004.1 Arnt"], sample=False) == [
        "CACGTG"
    ]
    assert motifs_to_strings(
        motifs, names=["MA0006.1 Ahr::Arnt"], sample=True, rng=np.random.RandomState(0)
    ) == ["CGCGTG"]


def test_trim_pwm():
    pwm = (
        np.log2(
            np.array(
                [
                    [0.25, 0.5, 0.97, 0.01, 0.01],
                    [0.25, 0.2, 0.01, 0.97, 0.97],
                    [0.24, 0.2, 0.01, 0.01, 0.01],
                    [0.26, 0.1, 0.01, 0.01, 0.01],
                ]
            )
        )
        + 2
    )
    assert np.all(trim_pwm(pwm, trim_threshold=0.3) == pwm[:, 2:])
    assert np.all(trim_pwm(pwm, trim_threshold=0.01) == pwm[:, 1:])


# Create test model
model = LightningModel(
    model_params={
        "model_type": "ConvModel",
        "n_tasks": 1,
        "n_conv": 0,
        "stem_channels": 1,
        "stem_kernel_size": 1,
        "act_func": None,
        "norm": False,
    },
    train_params={
        "task": "regression",
        "loss": "mse",
    },
)

weight = Tensor([[4, -2, 0, 0]]).unsqueeze(2)
bias = Tensor([0])
model.model.embedding.conv_tower.blocks[0].conv.bias = nn.Parameter(bias)
model.model.embedding.conv_tower.blocks[0].conv.weight = nn.Parameter(weight)
model.data_params = {"tasks": {"name": ["task"]}}
assert model.get_task_idxs("task") == 0


def test_marginalize_patterns():
    seqs = ["CATACGTGAGGC", "AGGAGGCCAAAG"]

    # Simple case
    preds_before, preds_after = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=1,
        seed=0,
        compare_func=None,
    )
    assert preds_before.shape == (2, 1, 1, 1, 1)
    assert np.allclose(preds_before.squeeze(), [0.5, 1.3333334])
    assert preds_after.shape == (2, 1, 1, 1, 1)
    assert np.allclose(
        preds_after.squeeze(),
        [0.8333333, 1.6666666],
    )
    # Multiple shuffles
    preds_before, preds_after = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=3,
        seed=0,
        compare_func=None,
    )
    assert preds_before.shape == (2, 3, 1, 1, 1)
    assert np.allclose(
        preds_before.squeeze(), [[0.5, 0.5, 0.5], [1.3333334, 1.3333334, 1.3333334]]
    )
    assert preds_after.shape == (2, 3, 1, 1, 1)
    assert np.allclose(
        preds_after.squeeze(),
        [[0.8333333, 0.5, 0.8333333], [1.6666666, 1.3333334, 1.6666666]],
    )
    # Multiple shuffles + rc augmentation
    preds_before, preds_after = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=3,
        rc=True,
        seed=0,
        compare_func=None,
    )
    assert preds_before.shape == (2, 3, 1, 1, 1)
    assert np.allclose(preds_before.squeeze(), [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])
    assert preds_after.shape == (2, 3, 1, 1, 1)
    assert np.allclose(preds_after.squeeze(), [[0.25, 0.25, 0.25], [0.5, 0.25, 0.5]])

    # Multiple shuffles + rc augmentation + compare_func
    preds = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=3,
        seed=0,
        rc=True,
        compare_func="subtract",
    )
    assert preds.shape == (2, 3, 1, 1, 1)
    assert np.allclose(preds.squeeze(), [[0.0, 0.0, 0.0], [0.25, 0.0, 0.25]], atol=1e-5)


def test_ISM_predict():

    # Single sequence
    seq = "AA"
    expected_preds = np.array([[4.0, 1.0, 2.0, 2.0], [4.0, 1.0, 2.0, 2.0]]).T
    preds = ISM_predict(seq, model, compare_func=None)
    assert np.allclose(preds.values, expected_preds)
    preds = ISM_predict(seq, model, compare_func="log2FC")
    assert np.allclose(preds.values, np.log2(expected_preds / 4))

    # Multiple sequences
    seqs = ["AAA", "CCC"]
    expected_preds = np.expand_dims(
        np.array(
            [
                [
                    [4.0, 2.0, 2.6666667, 2.6666667],
                    [4.0, 2.0, 2.6666667, 2.6666667],
                    [4.0, 2.0, 2.6666667, 2.6666667],
                ],
                [
                    [0.0, -2.0, -1.3333334, -1.3333334],
                    [0.0, -2.0, -1.3333334, -1.3333334],
                    [0.0, -2.0, -1.3333334, -1.3333334],
                ],
            ]
        ),
        (3, 4),
    )
    preds = ISM_predict(seqs, model, compare_func=None, return_df=False)
    assert np.allclose(preds, expected_preds)
    preds = ISM_predict(seqs, model, compare_func="log2FC", return_df=False)
    assert np.allclose(
        preds, np.log2(np.stack([expected_preds[0] / 4, -expected_preds[1] / 2]))
    )


def test_get_attributions():
    seq = generate_random_sequences(n=1, seq_len=50, seed=0, output_format="strings")[0]
    for hypothetical in [True, False]:
        attrs = get_attributions(
            model, seq, hypothetical=hypothetical, n_shuffles=10, method="deepshap"
        )
        assert attrs.shape == (1, 4, 50)
    for method in ["saliency", "inputxgradient", "integratedgradients"]:
        attrs = get_attributions(model, seq, method=method)
        assert attrs.shape == (1, 4, 50)


def test_get_attention_scores():
    model = LightningModel(
        model_params={
            "model_type": "ConvTransformerModel",
            "n_tasks": 1,
            "n_conv": 0,
            "n_transformers": 2,
            "stem_channels": 8,
            "stem_kernel_size": 1,
            "n_heads": 2,
            "n_pos_features": 4,
        },
        train_params={
            "task": "regression",
            "loss": "poisson",
        },
    )
    attn = get_attention_scores(model, ["GGG"], block_idx=0)
    assert attn.shape == (2, 3, 3)
    attn = get_attention_scores(model, ["GGG"], block_idx=None)
    assert attn.shape == (2, 2, 3, 3)


def test_scan_sequences():
    seqs = ["TCACGTGAA", "CACGCAGGA", "CCTGCGTGA"]

    # No reverse complement
    out = scan_sequences(seqs, motifs=meme_file, rc=False, pthresh=1e-3)
    expected = pd.DataFrame({
        'motif': ['MA0004.1 Arnt', 'MA0006.1 Ahr::Arnt'],
     'sequence': ['0', '2'],
     'seq_idx': [0, 2],
     'start': [1, 2],
     'end': [7, 8],
     'strand': ['+', '+'],
     'score': [11.60498046875, 10.691319823265076],
     'p-value': [0.000244140625, 0.000244140625],
     'matched_seq': ['CACGTG', 'TGCGTG']
    })
    assert out.equals(expected)

    # Allow reverse complement
    out = scan_sequences(seqs, motifs=meme_file, rc=True, pthresh=1e-3)

    expected = pd.DataFrame({
        'motif': ['MA0004.1 Arnt', 'MA0004.1 Arnt','MA0006.1 Ahr::Arnt', 'MA0006.1 Ahr::Arnt'],
     'sequence': ['0', '0', '1', '2'],
     'seq_idx': [0, 0, 1, 2],
     'start': [1, 1, 0, 2],
     'end': [7, 7, 6, 8],
     'strand': ['+', '-', '-', '+'],
     'score': [11.60498046875, 11.60498046875, 10.691319823265076, 10.691319823265076],
     'p-value': [0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625],
     'matched_seq': ['CACGTG', 'CACGTG', 'CACGCA', 'TGCGTG']
    })

    assert out.equals(expected)

    # Reverse complement with attributions
    attrs = get_attributions(model, seqs, method="inputxgradient")
    out = scan_sequences(seqs, motifs=meme_file, rc=True, pthresh=1e-3, attrs=attrs)
    expected = pd.DataFrame({
    'motif': ['MA0004.1 Arnt', 'MA0004.1 Arnt', 'MA0006.1 Ahr::Arnt', 'MA0006.1 Ahr::Arnt'],
     'sequence': ['0', '0', '1', '2'],
     'seq_idx': [0, 0, 1, 2],
     'start': [1, 1, 0, 2],
     'end': [7, 7, 6, 8],
     'strand': ['+', '-', '-', '+'],
     'score': [11.60498046875, 11.60498046875, 10.691319823265076, 10.691319823265076],
     'p-value': [0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625],
     'matched_seq': ['CACGTG', 'CACGTG', 'CACGCA', 'TGCGTG'],
     'site_attr_score': np.float32([0.0, 0.0, 0.009259258396923542, -0.009259259328246117]),
     'motif_attr_score': [0.003703703731298441, 0.0, 0.0, -0.03549381507926434]
    })
    assert out.equals(expected)


def test_run_tomtom():

    motifs = {
        "MA0004.1 Arnt": np.array(
            [
                [0.2, 0.95, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.05, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
        "MA0006.1 Ahr::Arnt": np.array(
            [
                [0.125, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.333333, 0.0, 0.958333, 0.0, 0.0, 0.0],
                [0.083333, 0.958333, 0.0, 0.958333, 0.0, 1.0],
                [0.458333, 0.041667, 0.041667, 0.041667, 1.0, 0.0],
            ]
        ),
    }
    df = run_tomtom(motifs=motifs, meme_file=meme_file)
    assert df.Query_ID.tolist() == [
        "MA0004.1 Arnt",
        "MA0004.1 Arnt",
        "MA0006.1 Ahr::Arnt",
        "MA0006.1 Ahr::Arnt",
    ]
    assert df.Target_ID.tolist() == [
        "MA0004.1 Arnt",
        "MA0006.1 Ahr::Arnt",
        "MA0004.1 Arnt",
        "MA0006.1 Ahr::Arnt",
    ]
    assert np.allclose(df.Optimal_offset, [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(
        df["p-value"],
        [
            1.339591458093814e-06,
            0.013090962390804095,
            0.017318747323072148,
            3.348979505934935e-07,
        ],
        rtol=1e-3,
    )
    assert np.allclose(
        df["E-value"],
        [
            2.679182916187628e-06,
            0.02618192478160819,
            0.034637494646144296,
            6.69795901186987e-07,
        ],
        rtol=1e-3,
    )
    assert np.allclose(
        df["q-value"],
        [
            2.679182916187628e-06,
            0.017318747323072148,
            0.017318747323072148,
            1.339591802373974e-06,
        ],
        rtol=1e-3,
    )
    assert np.allclose(df.Overlap, [6.0, 6.0, 6.0, 6.0])
    assert df.Query_consensus.tolist() == ["CACGTG", "CACGTG", "TGCGTG", "TGCGTG"]
    assert df.Target_consensus.tolist() == ["CACGTG", "TGCGTG", "CACGTG", "TGCGTG"]
    assert df.Orientation.tolist() == ["+", "+", "+", "+"]


def test_marginalize_pattern_spacing():

    seqs = ["CATACGTGAGGC", "AGGAGGCCAAAG"]

    preds, distances = marginalize_pattern_spacing(
        model,
        fixed_pattern="A",
        moving_pattern="CCC",
        seqs=seqs,
        n_shuffles=3,
        seed=0,
        stride=3,
        compare_func="subtract",
    )
    assert preds.shape == (2, 3, 3, 1, 1)
    expected_preds = np.array(
        [
            [[-0.5, -5 / 6, -2 / 3], [-0.5, -5 / 6, -1 / 3], [-0.5, -5 / 6, -1 / 3]],
            [
                [-3 / 2, -2 / 3, -2 / 3],
                [-5 / 6, -1 / 6, -7 / 6],
                [-7 / 6, -1 / 3, -2 / 3],
            ],
        ]
    )
    assert np.allclose(preds.squeeze(), expected_preds)
    assert distances == [-5, 1, 4]


def test_shuffle_tiles():
    seqs = ["CATACGTGAGGC", "AGGAGGCCAAAG"]
    before_preds, after_preds, positions = shuffle_tiles(
        model=model,
        seqs=seqs,
        tile_len=8,
        stride=4,
        n_shuffles=3,
        seed=0,
        compare_func=None,
    )
    assert positions.equals(pd.DataFrame({"start": [0, 4], "end": [8, 12]}))
    assert before_preds.shape == (2, 1, 1, 1, 1)
    assert after_preds.shape == (2, 2, 3, 1, 1)
    assert np.allclose(before_preds.squeeze(), np.array([0.5, 4 / 3]))
    assert np.allclose(
        after_preds.squeeze(), np.repeat([0.5, 4 / 3], 6).reshape(2, 2, 3)
    )
