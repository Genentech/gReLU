import os

import numpy as np
from torch import Tensor, nn

from grelu.interpret.motifs import (
    marginalize_patterns,
    motifs_to_strings,
    scan_sequences,
    trim_pwm,
)
from grelu.interpret.score import ISM_predict, get_attention_scores, get_attributions
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
    preds_before, preds_after = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=3,
        seed=0,
        compare_func=None,
    )
    assert preds_before.shape == (2, 3, 1)
    assert np.allclose(
        preds_before.squeeze(), [[0.5, 0.5, 0.5], [1.3333334, 1.3333334, 1.3333334]]
    )
    assert preds_after.shape == (2, 3, 1)
    assert np.allclose(
        preds_after.squeeze(),
        [[0.5, 0.8333333, 0.8333333], [1.3333334, 1.6666666, 1.6666666]],
    )


def test_ISM_predict():
    seq = "AA"
    expected_preds = np.array([[4.0, 1.0, 2.0, 2.0], [4.0, 1.0, 2.0, 2.0]]).T
    preds = ISM_predict(seq, model, compare_func=None)
    assert np.allclose(preds.values, expected_preds)
    preds = ISM_predict(seq, model, compare_func="log2FC")
    assert np.allclose(preds.values, np.log2(expected_preds / 4))


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
    seqs = ["TCACGTGA", "CCTGCGTGA", "CACGCAGG"]
    out = scan_sequences(seqs, motifs=meme_file, rc=False, pthresh=1e-3)
    assert out.motif.tolist() == ["MA0004.1 Arnt", "MA0006.1 Ahr::Arnt"]
    assert out.sequence.tolist() == ["0", "1"]
    assert out.start.tolist() == [1, 2]
    assert out.end.tolist() == [7, 8]
    assert out.strand.tolist() == ["+", "+"]
    assert out.matched_seq.tolist() == ["CACGTG", "TGCGTG"]

    out = scan_sequences(seqs, motifs=meme_file, rc=True, pthresh=1e-3)
    assert out.motif.tolist() == [
        "MA0004.1 Arnt",
        "MA0004.1 Arnt",
        "MA0006.1 Ahr::Arnt",
        "MA0006.1 Ahr::Arnt",
    ]
    assert out.sequence.tolist() == ["0", "0", "1", "2"]
    assert out.start.tolist() == [1, 1, 2, 0]
    assert out.end.tolist() == [7, 7, 8, 6]
    assert out.strand.tolist() == ["+", "-", "+", "-"]
    assert out.matched_seq.tolist() == ["CACGTG", "CACGTG", "TGCGTG", "CACGCA"]
