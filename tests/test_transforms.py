import os

import numpy as np
import torch

from grelu.transforms.label_transforms import LabelTransform
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from grelu.transforms.seq_transforms import MotifScore, PatternScore

label = np.expand_dims(
    np.array([[-1, 101, 30], [3, 0.3, 1000]], dtype=np.float32), 2
)  # 2, 3, 1
seqs = ["CAATCGGGAA", "AACGCGCTT", "CTCGTTTCTA"]

cwd = os.path.realpath(os.path.dirname(__file__))
meme_file = os.path.join(cwd, "files", "test.meme")

preds = torch.rand(2, 4, 6)


def test_label_transform():
    # Threshold + log
    t = LabelTransform(min_clip=0, max_clip=100, transform_func=np.log)
    assert np.allclose(t(label).squeeze(), np.log([[0, 100, 30], [3, 0.3, 100]]))
    # Threshold only
    t = LabelTransform(min_clip=0, max_clip=100, transform_func=None)
    assert np.allclose(t(label).squeeze(), np.array([[0, 100, 30], [3, 0.3, 100]]))


def test_pattern_score():
    t = PatternScore(patterns=["AA", "TT"], weights=[1, -0.5])
    assert np.allclose(t(seqs), np.array([2, 0.5, -1]))


def test_motif_score():
    t = MotifScore(meme_file=meme_file, weights=[-1, 0.5], rc=False)
    assert t(seqs) == [0, 0, 0]
    assert t(["CCCACGTGAA", "AATGCGTGGG"]) == [-1, 0.5]


def test_aggregate():
    t = Aggregate(
        tasks=[0, 1], positions=[3, 4], length_aggfunc="sum", task_aggfunc="sum"
    )
    expected = preds[:, [0, 1], :][:, :, [3, 4]].sum(axis=(1, 2), keepdims=True)
    assert torch.allclose(t(preds), expected)
    assert np.allclose(t.compute(preds.numpy()), expected.numpy())

    t = Aggregate(tasks=[0, 1], length_aggfunc=None, task_aggfunc="mean")
    expected = preds[:, [0, 1], :].mean(1, keepdims=True)
    assert torch.allclose(t(preds), expected)
    assert np.allclose(t.compute(preds.numpy()), expected.numpy())

    t = Aggregate(task_aggfunc=None, length_aggfunc="sum")
    expected = preds.sum(2, keepdims=True)
    assert torch.allclose(t(preds), expected)
    assert np.allclose(t.compute(preds.numpy()), expected.numpy())

    t = Aggregate(task_aggfunc=None, length_aggfunc="sum", weight=-1)
    expected = -1 * preds.sum(2, keepdims=True)
    assert np.allclose(t.compute(preds.numpy()), expected.numpy())
    assert torch.allclose(t(preds), expected)


def test_specificity():
    # No weight
    t = Specificity(
        on_tasks=[0],
        off_tasks=[1, 2],
        on_aggfunc="mean",
        off_aggfunc="max",
        compare_func="divide",
    )
    expected_on = preds[:, [0], :].numpy().sum(2, keepdims=True)
    expected_off = (
        preds[:, [1, 2], :].numpy().sum(2, keepdims=True).max(1, keepdims=True)
    )
    out = t(preds).numpy()
    assert np.allclose(out, np.divide(expected_on, expected_off))

    # Constant weight
    t = Specificity(
        on_tasks=[0],
        off_tasks=[1, 2],
        on_aggfunc="mean",
        off_aggfunc="max",
        compare_func="divide",
        off_weight=2,
    )
    expected_on = preds[:, [0], :].numpy().sum(2, keepdims=True)
    expected_off = (
        preds[:, [1, 2], :].numpy().sum(2, keepdims=True).max(1, keepdims=True)
    )
    expected_off = expected_off * 2
    out = t(preds).numpy()
    assert np.allclose(out, np.divide(expected_on, expected_off))


def test_specificity_threshold():
    # Thresholded weight
    t = Specificity(
        on_tasks=[0],
        off_tasks=[1],
        on_aggfunc="mean",
        off_aggfunc="max",
        compare_func="divide",
        off_weight=2,
        off_thresh=1,
        positions=[1, 2],
    )
    preds = torch.Tensor(
        [
            [0.1, 0.2, 1, 1.2],
            [0, -1, 10, 10],
        ]
    ).unsqueeze(
        0
    )  # 1, 2, 4

    expected_on = np.expand_dims(np.expand_dims(np.array([1.2]), 0), 2)
    expected_off = np.expand_dims(np.expand_dims(np.array([18]), 0), 2)
    out = t(preds).numpy()
    assert np.allclose(out, np.divide(expected_on, expected_off))

    preds = torch.Tensor(
        [
            [0.1, 0.2, 1, 1.2],
            [0, -1, 1.1, 10],
        ]
    ).unsqueeze(
        0
    )  # 1, 2, 4

    expected_on = np.expand_dims(np.expand_dims(np.array([1.2]), 0), 2)
    expected_off = np.expand_dims(np.expand_dims(np.array([0.1]), 0), 2)
    out = t(preds).numpy()
    assert np.allclose(out, np.divide(expected_on, expected_off))
