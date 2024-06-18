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
    assert motifs_to_strings(motifs, names=["Arnt"], sample=False) == ["CACGTG"]
    assert motifs_to_strings(
        motifs, names=["Ahr::Arnt"], sample=True, rng=np.random.RandomState(0)
    ) == ["CGCGTG"]


def test_trim_pwm():
    pwm = (
        np.log2(
            np.array(
                [
                    [0.25, 0.25, 0.24, 0.26],
                    [0.5, 0.2, 0.2, 0.1],
                    [0.97, 0.01, 0.01, 0.01],
                    [0.01, 0.97, 0.01, 0.01],
                    [0.01, 0.97, 0.01, 0.01],
                ]
            )
        )
        + 2
    )
    assert np.all(trim_pwm(pwm, trim_threshold=0.3, padding=0) == pwm[2:, :])
    assert np.all(trim_pwm(pwm, trim_threshold=0.01, padding=0) == pwm[1:, :])
    assert np.all(trim_pwm(pwm, trim_threshold=0.01, padding=4) == pwm)


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
    seqs = ["ACTGT", "GATCC"]
    preds_before, preds_after = marginalize_patterns(
        model,
        patterns=["A"],
        seqs=seqs,
        n_shuffles=3,
        seed=0,
        compare_func=None,
    )
    assert preds_before.shape == (2, 3, 1)
    assert np.allclose(preds_before.squeeze(), [[0.4, 0.4, 0.4], [0, 0, 0]])
    assert preds_after.shape == (2, 3, 1)
    assert np.allclose(preds_after.squeeze(), [[1.2, 1.2, 1.2], [0.8, 0.8, 0.8]])


def test_ISM_predict():
    seq = "AA"
    expected_preds = np.array([[4.0, 1.0, 2.0, 2.0], [4.0, 1.0, 2.0, 2.0]]).T
    preds = ISM_predict(seq, model, compare_func=None)
    assert np.allclose(preds.values, expected_preds)
    preds = ISM_predict(seq, model, compare_func="log2FC")
    assert np.allclose(preds.values, np.log2(expected_preds / 4))


def test_get_attributions():
    seq = generate_random_sequences(n=1, seq_len=50, seed=0, output_format="strings")[0]
    attrs = get_attributions(model, seq, hypothetical=False, n_shuffles=10)
    assert attrs.shape == (1, 4, 50)
    attrs = get_attributions(model, seq, hypothetical=True, n_shuffles=10)
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
    out = scan_sequences(seqs, motifs=meme_file, rc=False)
    assert out.motif.tolist() == ["Arnt", "Ahr::Arnt"]
    assert out.sequence.tolist() == ["0", "1"]
    out = scan_sequences(seqs, motifs=meme_file, rc=True)
    assert out.motif.tolist() == ["Arnt", "Arnt", "Ahr::Arnt", "Ahr::Arnt"]
    assert out.sequence.tolist() == ["0", "0", "1", "2"]
