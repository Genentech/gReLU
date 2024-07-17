import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn, optim

from grelu.data.dataset import DFSeqDataset, SeqDataset
from grelu.lightning import LightningModel, LightningModelEnsemble
from grelu.sequence.format import strings_to_one_hot
from grelu.transforms.prediction_transforms import Aggregate, Specificity


def generate_model(
    task, loss, n_tasks, class_weights=None, pos_weight=None, final_pool_func="avg"
):
    model = LightningModel(
        model_params={
            "model_type": "ConvModel",
            "n_tasks": n_tasks,
            "n_conv": 0,
            "stem_channels": n_tasks,
            "stem_kernel_size": 1,
            "act_func": None,
            "norm": False,
            "final_pool_func": final_pool_func,
        },
        train_params={
            "task": task,
            "loss": loss,
            "print_val": False,
            "logger": None,
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 1,
            "devices": "cpu",
            "class_weights": class_weights,
            "pos_weight": pos_weight,
        },
        data_params={},
    )

    if n_tasks == 1:
        weight = Tensor([[1, 0, 0, 0]]).unsqueeze(2)
        bias = Tensor([0])

    elif n_tasks == 2:
        weight = Tensor([[1, 0, 0, 0], [0, 0, 0, -1]]).unsqueeze(2)
        bias = Tensor([0, 0])

    elif n_tasks == 3:
        weight = Tensor([[1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0]]).unsqueeze(2)
        bias = Tensor([0, 0, 0])

    model.model.embedding.conv_tower.blocks[0].conv.bias = nn.Parameter(bias)
    model.model.embedding.conv_tower.blocks[0].conv.weight = nn.Parameter(weight)

    return model


# Build models
multitask_reg_model = generate_model(task="regression", loss="poisson", n_tasks=2)
single_task_reg_model = generate_model(task="regression", loss="MSE", n_tasks=1)
multitask_bin_model = generate_model(
    task="binary", loss="bce", n_tasks=2, pos_weight=[10.0, 1.0]
)
single_task_bin_model = generate_model(task="binary", loss="bce", n_tasks=1)
multicla_model = generate_model(
    task="multiclass", loss="ce", n_tasks=3, class_weights=[1.0, 2.0, 3.0]
)
multitask_profile_model = generate_model(
    task="regression", loss="poisson", n_tasks=2, final_pool_func=None
)

# Create inputs
strings = ["AAG", "CGA", "TTT"]
one_hot = strings_to_one_hot(strings)
labels_reg = Tensor([[1.0, 0.5], [0.5, 1.0], [1.0, 0.0]]).unsqueeze(2)
labels_bin = Tensor([[0], [1], [0]]).unsqueeze(2)
labels_multicla = Tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).unsqueeze(2)
multitask_bin_labels = (
    nn.functional.one_hot(labels_bin.type(torch.long), num_classes=2)
    .squeeze()
    .unsqueeze(2)
    .type(torch.float)
)
udataset = SeqDataset(strings[:2])
udataset_aug = SeqDataset(strings[:2], rc=True, max_seq_shift=1)
ldataset = DFSeqDataset(pd.DataFrame({"seq": strings, "label": 1.0}), rc=True)


def test_lightning_model_input():
    assert torch.allclose(single_task_reg_model.format_input(one_hot[0]), one_hot[[0]])
    assert torch.allclose(single_task_reg_model.format_input(one_hot), one_hot)
    assert torch.allclose(
        single_task_reg_model.format_input((one_hot, labels_reg)), one_hot
    )


def test_lightning_model_devices():
    # Devices
    assert single_task_reg_model.parse_devices("cpu") == ("cpu", "auto")
    assert single_task_reg_model.parse_devices(0) == ("gpu", [0])
    assert single_task_reg_model.parse_devices([0]) == ("gpu", [0])
    assert single_task_reg_model.parse_devices([0, 1]) == ("gpu", [0, 1])


def test_lightning_model_optimizer():
    # test the configure_optimizers method
    optimizer = multitask_reg_model.configure_optimizers()
    assert isinstance(optimizer, optim.Optimizer)


def test_lightning_model_results():
    # Regression, poisson loss, multitask
    loss = multitask_reg_model.validation_step((one_hot, labels_reg), 0).detach()
    preds = multitask_reg_model.model(one_hot)
    expected_loss = torch.tensor(0.97964877)
    expected_mse = Tensor([[0.5667739, 0.1284451]])
    expected_pearson = Tensor([[0.09491341, 0.8660254]])
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(
        multitask_reg_model.val_metrics["mse"].compute(), expected_mse
    )
    assert torch.allclose(
        multitask_reg_model.val_metrics["pearson"].compute(), expected_pearson
    )

    # Regression, MSE loss, single task
    loss = single_task_reg_model.validation_step(
        (one_hot, labels_reg[:, [0], :]), 0
    ).detach()
    preds = single_task_reg_model(one_hot)
    expected_mse = torch.tensor(0.37962961196899414)
    expected_pearson = torch.tensor(0.0)
    assert torch.allclose(loss, expected_mse)
    assert torch.allclose(
        single_task_reg_model.val_metrics["mse"].compute(), expected_mse
    )
    assert torch.allclose(
        single_task_reg_model.val_metrics["pearson"].compute(), expected_pearson
    )

    # Binary classification, multitask with loss weights
    loss = multitask_bin_model.validation_step(
        (one_hot, multitask_bin_labels), 0
    ).detach()
    expected_loss = torch.tensor(2.2747278213500977)
    assert torch.allclose(loss, expected_loss)

    logits = multitask_bin_model(one_hot, logits=True)
    expected_preds = nn.functional.sigmoid(logits)
    preds = multitask_bin_model(one_hot)
    assert torch.allclose(preds, expected_preds)

    expected_f1 = Tensor([0.8, 0.666667])
    expected_ap = Tensor([0.833333, 0.5])
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(
        multitask_bin_model.val_metrics["accuracy"].compute(), Tensor([1 / 3, 2 / 3])
    )
    assert torch.allclose(
        multitask_bin_model.val_metrics["best_f1"].compute(), expected_f1
    )
    assert torch.allclose(
        multitask_bin_model.val_metrics["avgprec"].compute(), expected_ap
    )

    # Binary classification, single task
    loss = single_task_bin_model.validation_step((one_hot, labels_bin), 0).detach()
    preds = single_task_bin_model(one_hot)
    expected_loss = torch.tensor(0.7714965)
    expected_f1 = torch.tensor(2 / 3)
    expected_ap = torch.tensor(1 / 2)
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(
        single_task_bin_model.val_metrics["accuracy"].compute(), torch.tensor(2 / 3)
    )
    assert torch.allclose(
        single_task_bin_model.val_metrics["best_f1"].compute(), expected_f1
    )
    assert torch.allclose(
        single_task_bin_model.val_metrics["avgprec"].compute(), expected_ap
    )

    # Multiclass classification with class weights
    loss = multicla_model.validation_step((one_hot, labels_multicla), 0).detach()
    expected_loss = torch.tensor(2.911)
    assert torch.allclose(loss, expected_loss, rtol=1e-3)
    logits = multicla_model(one_hot, logits=True)  # N, n_tasks, 1
    expected_preds = nn.functional.softmax(logits, 1)
    preds = multicla_model(one_hot)
    assert torch.allclose(preds, expected_preds)


def test_lightning_model_predict_on_dataset():
    preds = single_task_reg_model.predict_on_dataset(dataset=udataset, devices="cpu")
    assert preds.shape == (2, 1, 1)

    preds_aug = multitask_profile_model.predict_on_dataset(
        dataset=udataset_aug, devices="cpu"
    )
    assert preds_aug.shape == (2, 2, 3)

    preds_aug = multitask_profile_model.predict_on_dataset(
        dataset=udataset_aug, devices="cpu", augment_aggfunc=None
    )
    assert preds_aug.shape == (2, 6, 2, 3)


def test_lightning_model_predict_on_seqs():
    assert np.allclose(
        single_task_reg_model.predict_on_seqs(strings[0]),
        single_task_reg_model(one_hot[[0]]).detach().numpy(),
    )
    assert np.allclose(
        single_task_reg_model.predict_on_seqs(strings),
        single_task_reg_model(one_hot).detach().numpy(),
    )


def test_lightning_model_transform():
    # predict
    orig_preds = multitask_profile_model.predict_on_dataset(udataset)
    assert orig_preds.shape == (2, 2, 3)

    # Add aggregate transform
    t = Aggregate(tasks=1, positions=[1, 2], length_aggfunc="sum")
    multitask_profile_model.add_transform(t)
    preds = multitask_profile_model.predict_on_dataset(udataset)
    assert preds.shape == (2, 1, 1)

    # Remove
    multitask_profile_model.reset_transform()
    preds = multitask_profile_model.predict_on_dataset(udataset)
    assert np.allclose(preds, orig_preds)

    # Add specificity transform
    t = Specificity(
        on_tasks=0,
        off_tasks=1,
        positions=[1, 2],
        length_aggfunc="sum",
        compare_func="subtract",
    )
    multitask_profile_model.add_transform(t)
    preds = multitask_profile_model.predict_on_dataset(udataset)
    assert preds.shape == (2, 1, 1)

    # Remove
    multitask_profile_model.reset_transform()
    preds = multitask_profile_model.predict_on_dataset(udataset)
    assert preds.shape == orig_preds.shape


def test_lightning_model_embed_on_dataset():
    preds = single_task_reg_model.embed_on_dataset(dataset=udataset, device="cpu")
    assert preds.shape == (2, 1, 3)


def test_lightning_model_train_on_dataset():
    _ = single_task_reg_model.train_on_dataset(
        train_dataset=ldataset,
        val_dataset=ldataset,
    )
    assert single_task_reg_model.data_params["tasks"] == {"name": ["label"]}
    assert single_task_reg_model.data_params["train_max_pair_shift"] == 0
    assert single_task_reg_model.data_params["train_max_seq_shift"] == 0
    assert single_task_reg_model.data_params["train_n_augmented"] == 2
    assert single_task_reg_model.data_params["train_n_seqs"] == 3
    assert single_task_reg_model.data_params["train_n_tasks"] == 1
    assert single_task_reg_model.data_params["train_rc"]
    assert single_task_reg_model.data_params["train_seq_len"] == 3
    assert single_task_reg_model.data_params["val_max_pair_shift"] == 0
    assert single_task_reg_model.data_params["val_max_seq_shift"] == 0
    assert single_task_reg_model.data_params["val_n_augmented"] == 2
    assert single_task_reg_model.data_params["val_n_seqs"] == 3
    assert single_task_reg_model.data_params["val_n_tasks"] == 1
    assert single_task_reg_model.data_params["val_rc"]
    assert single_task_reg_model.data_params["val_seq_len"] == 3


def test_lightning_model_test_on_dataset():
    metrics = single_task_reg_model.test_on_dataset(dataset=ldataset, devices="cpu")
    assert metrics.index == ["label"]
    assert np.all(metrics.columns == ["test_mse", "test_pearson"])


def test_lightning_model_finetune():
    # Finetune only the head
    multitask_reg_model = generate_model(task="regression", loss="poisson", n_tasks=2)
    assert multitask_reg_model.model_params["n_tasks"] == 2
    multitask_reg_model.tune_on_dataset(
        ldataset, ldataset, final_pool_func="avg", freeze_embedding=True
    )
    assert torch.allclose(
        multitask_reg_model.model.embedding.conv_tower.blocks[0].conv.bias,
        nn.Parameter(Tensor([0.0, 0.0])),
    )
    assert multitask_reg_model.model_params["n_tasks"] == 1

    multitask_reg_model = generate_model(task="regression", loss="poisson", n_tasks=2)
    # Fine tune whole model
    multitask_reg_model.tune_on_dataset(
        ldataset, ldataset, final_pool_func="avg", freeze_embedding=False
    )
    assert not torch.allclose(
        multitask_reg_model.model.embedding.conv_tower.blocks[0].conv.bias,
        nn.Parameter(Tensor([0.0, 0.0])),
    )
    assert multitask_reg_model.model_params["n_tasks"] == 1


def test_lightning_model_ensemble():
    # Make individual models
    model0 = generate_model(task="binary", loss="bce", n_tasks=2)
    model0.data_params["tasks"] = {"name": ["a", "b"]}
    model1 = generate_model(task="binary", loss="bce", n_tasks=2)
    model1.data_params["tasks"] = {"name": ["c", "b"]}

    # Combine
    model = LightningModelEnsemble([model0, model1])

    # Test tasks
    assert np.all(
        model.data_params["tasks"]["name"]
        == ["model0_a", "model0_b", "model1_c", "model1_b"]
    )

    # Test get_task_idxs
    assert np.all(
        model.get_task_idxs(["model0_b", "model1_b", "model1_c", "model0_a"])
        == [1, 3, 2, 0]
    )

    # test predict
    udataset = SeqDataset(["AAG", "CGA"])
    preds = model.predict_on_dataset(dataset=udataset, devices="cpu")
    assert preds.shape == (2, 4, 1)


bin_model = generate_model(task="binary", loss="bce", n_tasks=2)
bin_model.model_params["crop_len"] = 0
bin_model.data_params["train_bin_size"] = 2

crop_model = generate_model(task="binary", loss="bce", n_tasks=2)
crop_model.model_params["crop_len"] = 3
crop_model.data_params["train_bin_size"] = 1

crop_bin_model = generate_model(task="binary", loss="bce", n_tasks=2)
crop_bin_model.model_params["crop_len"] = 3
crop_bin_model.data_params["train_bin_size"] = 2


def test_input_coord_to_output_bin():
    assert bin_model.input_coord_to_output_bin(input_coord=6) == 3
    assert bin_model.input_coord_to_output_bin(input_coord=7) == 3
    assert bin_model.input_coord_to_output_bin(input_coord=8) == 4
    assert bin_model.input_coord_to_output_bin(input_coord=8, start_pos=1) == 3

    assert crop_model.input_coord_to_output_bin(input_coord=6) == 3
    assert crop_model.input_coord_to_output_bin(input_coord=7) == 4
    assert crop_model.input_coord_to_output_bin(input_coord=8) == 5
    assert crop_model.input_coord_to_output_bin(input_coord=8, start_pos=1) == 4

    assert crop_bin_model.input_coord_to_output_bin(input_coord=6) == 0
    assert crop_bin_model.input_coord_to_output_bin(input_coord=7) == 0
    assert crop_bin_model.input_coord_to_output_bin(input_coord=8) == 1
    assert crop_bin_model.input_coord_to_output_bin(input_coord=8, start_pos=1) == 0


def test_output_bin_to_input_coord():
    assert bin_model.output_bin_to_input_coord(output_bin=1) == 2
    assert bin_model.output_bin_to_input_coord(output_bin=1, return_pos="end") == 4
    assert bin_model.output_bin_to_input_coord(output_bin=1, start_pos=1) == 3
    assert (
        bin_model.output_bin_to_input_coord(output_bin=1, return_pos="end", start_pos=1)
        == 5
    )

    assert crop_model.output_bin_to_input_coord(output_bin=1) == 4
    assert crop_model.output_bin_to_input_coord(output_bin=1, return_pos="end") == 5
    assert crop_model.output_bin_to_input_coord(output_bin=1, start_pos=1) == 5
    assert (
        crop_model.output_bin_to_input_coord(
            output_bin=1, return_pos="end", start_pos=1
        )
        == 6
    )

    assert crop_bin_model.output_bin_to_input_coord(output_bin=1) == 8
    assert (
        crop_bin_model.output_bin_to_input_coord(output_bin=1, return_pos="end") == 10
    )
    assert crop_bin_model.output_bin_to_input_coord(output_bin=1, start_pos=1) == 9
    assert (
        crop_bin_model.output_bin_to_input_coord(
            output_bin=1, return_pos="end", start_pos=1
        )
        == 11
    )


def test_input_intervals_to_output_intervals():
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [14]})

    output = bin_model.input_intervals_to_output_intervals(intervals=intervals)
    assert output.equals(intervals)
    output = crop_model.input_intervals_to_output_intervals(intervals=intervals)
    assert output.equals(pd.DataFrame({"chrom": ["chr1"], "start": [3], "end": [11]}))
    output = crop_bin_model.input_intervals_to_output_intervals(intervals=intervals)
    assert output.equals(pd.DataFrame({"chrom": ["chr1"], "start": [6], "end": [8]}))


def test_input_intervals_to_output_bins():
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [6], "end": [14]})

    output = bin_model.input_intervals_to_output_bins(intervals=intervals)
    assert output.equals(pd.DataFrame({"start": [3], "end": [8]}))
    output = crop_model.input_intervals_to_output_bins(intervals=intervals)
    assert output.equals(pd.DataFrame({"start": [3], "end": [12]}))
    output = crop_bin_model.input_intervals_to_output_bins(intervals=intervals)
    assert output.equals(pd.DataFrame({"start": [0], "end": [5]}))
