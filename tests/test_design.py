import numpy as np
import pandas as pd
from torch import Tensor, nn

from grelu.design import evolve
from grelu.lightning import LightningModel
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from grelu.transforms.seq_transforms import PatternScore

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
        "loss": "mse",
        "print_val": False,
        "logger": None,
        "max_epochs": 1,
        "batch_size": 2,
        "num_workers": 1,
        "devices": "cpu",
    },
    data_params={},
)

weight = Tensor([[1, 2, 0, 0], [0, 0, -2, -1]]).unsqueeze(2)
bias = Tensor([0, 0])
model.model.embedding.conv_tower.blocks[0].conv.bias = nn.Parameter(bias)
model.model.embedding.conv_tower.blocks[0].conv.weight = nn.Parameter(weight)

model.data_params["tasks"] = {"name": ["label1", "label2"]}


def test_task_idxs():
    assert np.all(model.data_params["tasks"]["name"] == ["label1", "label2"])
    assert model.get_task_idxs("label1") == 0
    assert model.get_task_idxs("label2") == 1
    assert model.get_task_idxs(["label2", "label1"]) == [1, 0]
    assert model.get_task_idxs(0) == 0
    assert model.get_task_idxs(1) == 1
    assert model.get_task_idxs([1, 0]) == [1, 0]


seqs = ["AT", "TG"]


def test_evolve_1():
    # Single starting sequence, one task

    output = evolve(
        [seqs[0]],
        model,
        max_iter=2,
        prediction_transform=Aggregate(tasks=["label1"], model=model),
        devices="cpu",
        num_workers=1,
    )

    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 13
    assert np.all(
        output.columns
        == [
            "iter",
            "start_seq",
            "best_in_iter",
            "prediction_score",
            "seq_score",
            "total_score",
            "seq",
            "position",
            "allele",
            "label1",
        ]
    )
    assert np.all(output["iter"] == [0] + [1] * 6 + [2] * 6)
    assert output["seq"].iloc[0] == seqs[0]


def test_evolve_2():
    # Single starting sequence, two tasks
    output = evolve(
        [seqs[0]],
        model,
        max_iter=2,
        devices="cpu",
        num_workers=1,
        prediction_transform=Specificity(
            on_tasks=["label1"], off_tasks=["label2"], model=model
        ),
    )

    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 13

    # Check output values
    assert np.all(output["iter"] == [0] + [1] * 6 + [2] * 6)
    assert output["seq"].iloc[0] == seqs[0]


def test_evolve_3():
    # Two start sequences, two tasks, independent
    output = evolve(
        seqs,
        model,
        max_iter=2,
        devices="cpu",
        num_workers=1,
        for_each=True,
        prediction_transform=Specificity(
            on_tasks=["label1"], off_tasks=["label2"], model=model
        ),
    )

    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 26  # 2 + (2*6) + (2*6)

    # Check output values
    assert np.all(output["iter"] == [0] * 2 + [1] * 12 + [2] * 12)
    assert np.all(output.start_seq == [0, 1] + [0] * 6 + [1] * 6 + [0] * 6 + [1] * 6)
    assert len(output[output.best_in_iter]) == 6


def test_evolve_4():
    # Two start sequences, two tasks, single run
    output = evolve(
        seqs,
        model,
        max_iter=2,
        devices="cpu",
        num_workers=1,
        prediction_transform=Specificity(
            on_tasks=["label2"], off_tasks=["label1"], model=model
        ),
        for_each=False,
    )

    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 14  # 2 + 6 + 6

    # Check output values
    assert np.all(output["iter"] == [0] * 2 + [1] * 6 + [2] * 6)
    assert len(output[output.best_in_iter]) == 3
    assert output.seq[(output.iter == 2) & (output.best_in_iter)].tolist() == ["CA"]


def test_evolve_5():
    # Two start sequences, two tasks, single run
    output = evolve(
        seqs,
        model,
        max_iter=2,
        devices="cpu",
        num_workers=1,
        prediction_transform=Specificity(
            on_tasks=["label2"], off_tasks=["label1"], model=model
        ),
        seq_transform=PatternScore(patterns=["CA"], weights=[-1]),
        for_each=False,
    )

    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 14  # 2 + 6 + 6

    # Check output values
    assert np.all(output["iter"] == [0] * 2 + [1] * 6 + [2] * 6)
    assert len(output[output.best_in_iter]) == 3
    assert output.seq[(output.iter == 2) & (output.best_in_iter)].tolist() != ["CA"]


def test_evolve_6():
    output = evolve(
        ["GCCT"],
        model,
        max_iter=1,
        prediction_transform=Aggregate(tasks=["label1"], model=model),
        devices="cpu",
        num_workers=1,
        method="pattern",
        patterns=["AA"],
    )
    # Check output format
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 4

    # Check output values
    assert np.all(output["iter"] == [0] + [1] * 3)
    assert len(output[output.best_in_iter]) == 2
    assert output.seq.tolist() == ["GCCT", "AACT", "GAAT", "GCAA"]
