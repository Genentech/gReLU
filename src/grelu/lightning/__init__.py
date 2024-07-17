"""
The LightningModel class.
"""

import warnings
from collections import defaultdict
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, AveragePrecision, MetricCollection

import grelu.model.models
from grelu.data.dataset import (
    ISMDataset,
    LabeledSeqDataset,
    MotifScanDataset,
    PatternMarginalizeDataset,
    VariantDataset,
    VariantMarginalizeDataset,
)
from grelu.lightning.losses import PoissonMultinomialLoss
from grelu.lightning.metrics import MSE, BestF1, PearsonCorrCoef
from grelu.model.heads import ConvHead
from grelu.sequence.format import strings_to_one_hot
from grelu.utils import get_aggfunc, get_compare_func, make_list

default_train_params = {
    "task": "binary",  # binary, multiclass, or regression
    "lr": 1e-4,
    "optimizer": "adam",
    "batch_size": 512,
    "num_workers": 1,
    "devices": "cpu",
    "logger": None,
    "save_dir": ".",
    "max_epochs": 1,
    "checkpoint": True,
    "loss": "bce",
    "clip": 0,
    "pos_weight": None,
    "class_weights": None,
    "total_weight": None,
}


class LightningModel(pl.LightningModule):
    """
    Wrapper for predictive sequence models

    Args:
        model_params: Dictionary of parameters specifying model architecture
        train_params: Dictionary specifying training parameters
        data_params: Dictionary specifying parameters of the training data.
            This is empty by default and will be filled at the time of
            training.
    """

    def __init__(
        self, model_params: dict, train_params: dict = {}, data_params: dict = {}
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        # Add default training parameters
        for key in default_train_params.keys():
            if key not in train_params:
                train_params[key] = default_train_params[key]
            if key in ["loss", "task", "optimizer"]:
                train_params[key] = train_params[key].lower()

        # Save params
        self.model_params = model_params
        self.train_params = train_params
        self.data_params = data_params

        # Build model
        self.build_model()

        # Set up loss function
        self.initialize_loss()

        # Set up activation function
        self.initialize_activation()

        # Inititalize metrics
        self.initialize_metrics()

        # Initialize prediction transform
        self.reset_transform()

    def build_model(self) -> None:
        """
        Build a model from parameter dictionary
        """
        model_type = self.model_params["model_type"]
        if hasattr(grelu.model.models, model_type):
            self.model = getattr(grelu.model.models, model_type)
        else:
            raise Exception("Unknown model type")

        self.model = self.model(
            **{k: v for k, v in self.model_params.items() if k not in ["model_type"]}
        )

    def initialize_loss(self) -> None:
        """
        Create the specified loss function.
        """
        # Losses always accept logits i.e. pre-activation values
        # Regression loss: Poisson or MSE
        if self.train_params["task"] == "regression":
            if self.train_params["loss"] == "poisson":
                self.loss = nn.PoissonNLLLoss(log_input=True, full=True)
            elif self.train_params["loss"] == "poisson_multinomial":
                self.loss = PoissonMultinomialLoss(
                    total_weight=self.train_params["total_weight"], log_input=True
                )
            elif self.train_params["loss"] == "mse":
                self.loss = nn.MSELoss()
            else:
                raise Exception("Regression losses: poisson, poisson_multinomial, MSE")

        # Binary: Binary cross-entropy loss
        elif self.train_params["task"] == "binary":
            if self.train_params["loss"] == "bce":
                pos_weight = self.train_params["pos_weight"]
                if pos_weight is not None:
                    pos_weight = Tensor(pos_weight).unsqueeze(1)
                self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                raise Exception("Binary losses: bce")

        # Multiclass: Cross-entropy loss
        elif self.train_params["task"] == "multiclass":
            weight = self.train_params["class_weights"]
            if weight is not None:
                weight = Tensor(weight)
            self.loss = nn.CrossEntropyLoss(weight=weight)
        else:
            raise Exception("Task is not allowed")

    def initialize_activation(self) -> None:
        """
        Add a task-specific activation function to the model.
        """
        # Regression
        if self.train_params["task"] == "regression":
            if self.train_params["loss"] == "poisson":
                self.activation = torch.exp
            elif self.train_params["loss"] == "mse":
                self.activation = nn.Identity()
            else:
                raise Exception("Regression losses: poisson, MSE")

        # Binary
        elif self.train_params["task"] == "binary":
            self.activation = torch.sigmoid

        # Multiclass
        elif self.train_params["task"] == "multiclass":
            self.activation = nn.Softmax(dim=1)
        else:
            raise Exception("Defined tasks: regression, binary and multiclass")

    def initialize_metrics(self):
        """
        Initialize the appropriate metrics for the given task.
        """
        # Regression:
        if self.train_params["task"] == "regression":
            metrics = MetricCollection(
                {
                    "mse": MSE(num_outputs=self.model.head.n_tasks, average=False),
                    "pearson": PearsonCorrCoef(
                        num_outputs=self.model.head.n_tasks, average=False
                    ),
                }
            )

        # Binary classification
        elif self.train_params["task"] == "binary":
            if self.model.head.n_tasks == 1:
                metrics = MetricCollection(
                    {
                        "accuracy": Accuracy("binary", average=None),
                        "avgprec": AveragePrecision("binary", average=None),
                        "auroc": AUROC("binary", average=None),
                        "best_f1": BestF1(
                            num_labels=self.model.head.n_tasks, average=False
                        ),
                    }
                )
            else:
                metrics = MetricCollection(
                    {
                        "accuracy": Accuracy(
                            task="multilabel",
                            num_labels=self.model.head.n_tasks,
                            average=None,
                        ),
                        "avgprec": AveragePrecision(
                            "multilabel",
                            num_labels=self.model.head.n_tasks,
                            average=None,
                        ),
                        "auroc": AUROC(
                            "multilabel",
                            num_labels=self.model.head.n_tasks,
                            average=None,
                        ),
                        "best_f1": BestF1(
                            num_labels=self.model.head.n_tasks, average=False
                        ),
                    }
                )

        # Multiclass
        elif self.train_params["task"] == "multiclass":
            metrics = MetricCollection(
                {
                    "accuracy": Accuracy(
                        task="multiclass",
                        num_classes=self.model.head.n_tasks,
                        average=None,
                    ),
                    "avgprec": AveragePrecision(
                        "multiclass", num_classes=self.model.head.n_tasks, average=None
                    ),
                    "auroc": AUROC(
                        "multiclass", num_classes=self.model.head.n_tasks, average=None
                    ),
                }
            )
        else:
            raise Exception("Defined tasks: regression, binary and multiclass")

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_losses = []
        self.test_losses = []

    def update_metrics(self, metrics: dict, y_hat: Tensor, y: Tensor) -> None:
        """
        Update metrics after each pass
        """
        if self.train_params["task"] == "binary":
            metrics.update(y_hat, y.type(torch.long))
        elif self.train_params["task"] == "multiclass":
            metrics.update(y_hat, y.type(torch.long).argmax(axis=1))
        else:
            metrics.update(y_hat, y)

    def format_input(self, x: Union[Tuple[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Extract the one-hot encoded sequence from the input
        """
        # if x is a tuple of sequence, label, return the sequence
        if isinstance(x, Tensor):
            if x.ndim == 3:
                return x
            else:
                return x.unsqueeze(0)
        elif isinstance(x, Tuple):
            return x[0]
        else:
            raise Exception("Cannot perform forward pass on the given input format.")

    def forward(
        self,
        x: Union[Tuple[Tensor, Tensor], Tensor, str, List[str]],
        logits: bool = False,
    ) -> Tensor:
        """
        Forward pass
        """
        # Format the input as a one-hot encoded tensor
        x = self.format_input(x)

        # Run the model
        x = self.model(x)

        # forward() produces prediction (e.g. post-activation)
        # unless logits=True, which is used in loss functions
        if not logits:
            x = self.activation(x)

        # Apply transform
        x = self.transform(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        self.log(
            "train_loss",
            loss,
            logger=self.train_params["logger"] is not None,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        y_hat = self.activation(logits)
        self.log(
            "val_loss",
            loss,
            logger=self.train_params["logger"] is not None,
            on_step=False,
            on_epoch=True,
        )
        self.update_metrics(self.val_metrics, y_hat, y)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Calculate metrics for entire validation set
        """
        # Compute metrics
        val_metrics = self.val_metrics.compute()
        mean_val_metrics = {k: v.mean() for k, v in val_metrics.items()}
        # Compute loss
        losses = torch.stack(self.val_losses)
        mean_losses = torch.mean(losses)
        # Log or print
        if self.train_params["logger"] is None:
            print(mean_val_metrics)
            print(f"validation loss: {mean_losses}")
        else:
            self.log_dict(mean_val_metrics)
            self.log("val_loss", mean_losses)

        self.val_metrics.reset()
        self.val_losses = []

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Calculate metrics after a single test step
        """
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        y_hat = self.activation(logits)
        self.log("test_loss", loss, logger=True, on_step=False, on_epoch=True)
        self.update_metrics(self.test_metrics, y_hat, y)
        self.test_losses.append(loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Calculate metrics for entire test set
        """
        self.computed_test_metrics = self.test_metrics.compute()
        self.log_dict({k: v.mean() for k, v in self.computed_test_metrics.items()})
        losses = torch.stack(self.test_losses)
        self.log("test_loss", torch.mean(losses))

        self.test_metrics.reset()
        self.test_losses = []

    def configure_optimizers(self) -> None:
        """
        Configure oprimizer for training
        """
        if self.train_params["optimizer"] == "adam":
            return optim.Adam(self.parameters(), lr=self.train_params["lr"])
        elif self.train_params["optimizer"] == "sgd":
            return optim.SGD(
                self.parameters(), lr=self.train_params["lr"], momentum=0.9
            )
        else:
            raise Exception("Unknown optimizer")

    def count_params(self) -> int:
        """
        Number of gradient enabled parameters in the model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def parse_devices(
        self, devices: Union[str, int, List[int]]
    ) -> Tuple[str, Union[str, List[int]]]:
        """
        Parses the devices argument and returns a tuple of accelerator and devices.

        Args:
            devices: Either "cpu" or an integer or list of integers representing the indices
                of the GPUs for training.

        Returns:
            A tuple of accelerator and devices.
        """
        if devices == "cpu":
            accelerator = "cpu"
            devices = "auto"
        else:
            accelerator = "gpu"
            devices = make_list(devices)
        return accelerator, devices

    def parse_logger(self) -> str:
        """
        Parses the name of the logger supplied in train_params.
        """
        if "name" not in self.train_params:
            self.train_params["name"] = datetime.now().strftime("%Y_%d_%m_%H_%M")
        if self.train_params["logger"] == "wandb":
            logger = WandbLogger(
                name=self.train_params["name"],
                log_model=True,
                save_dir=self.train_params["save_dir"],
            )
        elif self.train_params["logger"] == "csv":
            logger = CSVLogger(
                name=self.train_params["name"], save_dir=self.train_params["save_dir"]
            )
        elif self.train_params["logger"] is None:
            logger = None
        else:
            raise NotImplementedError
        return logger

    def add_transform(self, prediction_transform: Callable) -> None:
        """
        Add a prediction transform
        """
        if prediction_transform is not None:
            self.transform = prediction_transform

    def reset_transform(self) -> None:
        """
        Remove a prediction transform
        """
        self.transform = nn.Identity()

    def make_train_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for training
        """
        assert isinstance(dataset, LabeledSeqDataset)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=True,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_test_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for validation and testing
        """
        assert isinstance(dataset, LabeledSeqDataset)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_predict_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for prediction
        """
        if isinstance(dataset, LabeledSeqDataset):
            dataset.predict = True
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def train_on_dataset(
        self,
        train_dataset: Callable,
        val_dataset: Callable,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Train model and optionally log metrics to wandb.

        Args:
            train_dataset (Dataset): Dataset object that yields training examples
            val_dataset (Dataset) : Dataset object that yields training examples
            checkpoint_path (str): Path to model checkpoint from which to resume training.
                The optimizer will be set to its checkpointed state.

        Returns:
            PyTorch Lightning Trainer
        """
        torch.set_float32_matmul_precision("medium")

        # Checkpointing
        if self.train_params["checkpoint"] is True:
            checkpoint_callbacks = [
                ModelCheckpoint(monitor="val_loss", mode="min", save_last=True)
            ]
        elif isinstance(self.train_params["checkpoint"], dict):
            checkpoint_callbacks = [ModelCheckpoint(**self.train_params["checkpoint"])]
        else:
            raise Exception("Checkpoint type must be a bool or dict")

        # Get device
        accelerator, devices = self.parse_devices(self.train_params["devices"])

        # Set up logging
        logger = self.parse_logger()

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.train_params["max_epochs"],
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=checkpoint_callbacks,
            default_root_dir=self.train_params["save_dir"],
            gradient_clip_val=self.train_params["clip"],
        )

        # Make dataloaders
        train_dataloader = self.make_train_loader(train_dataset)
        val_dataloader = self.make_test_loader(val_dataset)

        if checkpoint_path is None:
            # First validation pass
            trainer.validate(model=self, dataloaders=val_dataloader)
            self.val_metrics.reset()

        # Add data parameters
        self.data_params["tasks"] = train_dataset.tasks.reset_index(
            names="name"
        ).to_dict(orient="list")

        for attr, value in self._get_dataset_attrs(train_dataset):
            self.data_params["train_" + attr] = value

        for attr, value in self._get_dataset_attrs(val_dataset):
            self.data_params["val_" + attr] = value

        # Training
        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=checkpoint_path,
        )
        return trainer

    def _get_dataset_attrs(self, dataset: Callable) -> None:
        """
        Read data parameters from a dataset object
        """
        for attr in dir(dataset):
            if not attr.startswith("_") and not attr.isupper():
                value = getattr(dataset, attr)
                if (
                    (attr == "chroms")
                    or (isinstance(value, str))
                    or (isinstance(value, int))
                    or (isinstance(value, float))
                    or (value is None)
                ):
                    yield attr, value

    def change_head(
        self,
        n_tasks: int,
        final_pool_func: str,
    ) -> None:
        """
        Build a new head with the desired number of tasks
        """
        in_channels = self.model.head.in_channels
        self.model.head = ConvHead(
            n_tasks=n_tasks,
            in_channels=in_channels,
            act_func=None,
            pool_func=final_pool_func,
            norm=False,
        )
        self.model_params["n_tasks"] = n_tasks
        self.model_params["final_pool_func"] = final_pool_func

        self.initialize_metrics()

    def tune_on_dataset(
        self,
        train_dataset: Callable,
        val_dataset: Callable,
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = None,
        freeze_embedding: bool = False,
    ):
        """
        Fine-tune a pretrained model on a new dataset.

        Args:
            train_dataset: Dataset object that yields training examples
            val_dataset: Dataset object that yields training examples
            final_act_func: Name of the final activation layer
            final_pool_func: Name of the final pooling layer
            freeze_embedding: If True, all the embedding layers of the pretrained
                model will be frozen and only the head will be trained.

        Returns:
            PyTorch Lightning Trainer
        """
        # Move train data parameters
        self.base_data_params = self.data_params.copy()
        self.data_params = {}

        # Make new model head
        self.change_head(
            n_tasks=train_dataset.n_tasks,
            final_pool_func=final_pool_func,
        )

        # Freeze the model embedding layers if needed
        if freeze_embedding:
            for param in self.model.embedding.parameters():
                param.requires_grad = False

        # Train new model
        return self.train_on_dataset(train_dataset, val_dataset)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["hyper_parameters"]["data_params"] = self.data_params

    def predict_on_seqs(
        self,
        x: Union[str, List[str]],
        device: Union[str, int] = "cpu",
    ) -> np.ndarray:
        """
        A simple function to return model predictions directly
        on a batch of a single batch of sequences in string
        format.

        Args:
            x: DNA sequences as a string or list of strings.
            device: Index of the device to use

        Returns:
            A numpy array of predictions.
        """
        x = strings_to_one_hot(x, add_batch_axis=True)
        x = x.to(device)
        self.model = self.model.eval().to(device)
        preds = self.forward(x).detach().cpu().numpy()
        self.model = self.model.cpu()
        return preds

    def predict_on_dataset(
        self,
        dataset: Callable,
        devices: Union[int, str, List[int]] = "cpu",
        num_workers: int = 1,
        batch_size: int = 256,
        augment_aggfunc: Union[str, Callable] = "mean",
        compare_func: Optional[Union[str, Callable]] = None,
        return_df: bool = False,
    ):
        """
        Predict for a dataset of sequences or variants

        Args:
            dataset: Dataset object that yields one-hot encoded sequences
            devices: Device IDs to use
            num_workers: Number of workers for data loader
            batch_size: Batch size for data loader
            augment_aggfunc: Return the average prediction across all augmented
                versions of a sequence
            compare_func: Return the alt/ref difference for variants
            return_df: Return the predictions as a Pandas dataframe

        Returns:
            Model predictions as a numpy array or dataframe
        """
        torch.set_float32_matmul_precision("medium")
        dataloader = self.make_predict_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        accelerator, devices = self.parse_devices(devices)
        trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=None)

        # Predict
        preds = torch.concat(trainer.predict(self, dataloader))

        # Reshape predictions
        preds = rearrange(
            preds,
            "(b n a) t l -> b n a t l",
            n=dataset.n_augmented,
            a=dataset.n_alleles,
        )

        # Convert predictions to numpy array
        preds = preds.detach().cpu().numpy()

        # ISM or Motif Scanning
        if (isinstance(dataset, ISMDataset)) or (isinstance(dataset, MotifScanDataset)):
            return preds

        else:
            # Flip predictions for reverse complemented sequences
            if (dataset.rc) and (preds.shape[-1] > 1):
                preds[:, dataset.n_augmented // 2 :, :, :, :] = np.flip(
                    preds[:, dataset.n_augmented // 2 :, :, :, :], axis=-1
                )

            # Compare the predictions for two alleles
            if (
                (isinstance(dataset, VariantDataset))
                or (isinstance(dataset, VariantMarginalizeDataset))
                or (isinstance(dataset, PatternMarginalizeDataset))
            ):
                if compare_func is not None:
                    assert preds.shape[2] == 2
                    preds = get_compare_func(compare_func)(
                        preds[:, :, 1, :, :], preds[:, :, 0, :, :]
                    )  # BNTL

                # Combine predictions for augmented sequences
                if augment_aggfunc is not None:
                    preds = get_aggfunc(augment_aggfunc)(preds, axis=1)  # B T L

                return preds

            else:
                # Regular sequences
                preds = preds.squeeze(2)  # B N T L
                if augment_aggfunc is not None:
                    preds = get_aggfunc(augment_aggfunc)(preds, axis=1)  # B T L
                elif preds.shape[1] == 1:
                    preds = preds.squeeze(1)

                # Make dataframe
                if return_df:
                    if (preds.ndim == 3) and (preds.shape[-1] == 1):
                        preds = pd.DataFrame(
                            preds.squeeze(-1), columns=self.data_params["tasks"]["name"]
                        )
                    else:
                        warnings.warn(
                            "Cannot produce dataframe output."
                            + "Either output length > 1 or augmented sequences are not aggregated."
                        )

            return preds

    def test_on_dataset(
        self,
        dataset: Callable,
        devices: Union[str, int, List[int]] = "cpu",
        num_workers: int = 1,
        batch_size: int = 256,
    ):
        """
        Run test loop for a dataset

        Args:
            dataset: Dataset object that yields one-hot encoded sequences
            devices: Device IDs to use for inference
            num_workers: Number of workers for data loader
            batch_size: Batch size for data loader

        Returns:
            Dataframe containing all calculated metrics on the test set.
        """
        torch.set_float32_matmul_precision("medium")
        dataloader = self.make_test_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        accelerator, devices = self.parse_devices(devices)
        trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=None)
        self.test_metrics.reset()
        trainer.test(model=self, dataloaders=dataloader, verbose=True)

        metric_dict = {
            k: v.detach().cpu().numpy() for k, v in self.computed_test_metrics.items()
        }
        self.test_metrics.reset()
        return pd.DataFrame(metric_dict, index=self.data_params["tasks"]["name"])

    def embed_on_dataset(
        self,
        dataset: Callable,
        device: Union[str, int] = "cpu",
        num_workers: int = 1,
        batch_size: int = 256,
    ):
        """
        Return embeddings for a dataset of sequences

        Args:
            dataset: Dataset object that yields one-hot encoded sequences
            device: Device ID to use
            num_workers: Number of workers for data loader
            batch_size: Batch size for data loader

        Returns:
            Numpy array of shape (B, T, L) containing embeddings.
        """
        torch.set_float32_matmul_precision("medium")

        # Make dataloader
        dataloader = self.make_predict_loader(
            dataset, num_workers=num_workers, batch_size=batch_size
        )

        # Get device
        if isinstance(device, list):
            device = device[0]
            warnings.warn(
                f"embed_on_dataset currently only uses a single GPU: Using {device}"
            )
        if isinstance(device, str):
            try:
                device = int(device)
            except Exception:
                pass
        device = torch.device(device)

        # Move model to device
        orig_device = self.device
        self.to(device)

        # Get embeddings
        preds = []
        self.model = self.model.eval()
        for batch in iter(dataloader):
            batch = batch.to(device)
            preds.append(self.model.embedding(batch).detach().cpu())

        # Return to original device
        self.to(orig_device)
        return torch.vstack(preds).numpy()

    def get_task_idxs(
        self,
        tasks: Union[int, str, List[int], List[str]],
        key: str = "name",
        invert: bool = False,
    ) -> Union[int, List[int]]:
        """
        Given a task name or metadata entry, get the task index
        If integers are provided, return them unchanged

        Args:
            tasks: A string corresponding to a task name or metadata entry,
                or an integer indicating the index of a task, or a list of strings/integers
            key: key to model.data_params["tasks"] in which the relevant task data is
                stored. "name" will be used by default.
            invert: Get indices for all tasks except those listed in tasks

        Returns:
            The index or indices of the corresponding task(s) in the model's
            output.
        """
        # If a string is provided, extract the index
        if isinstance(tasks, str):
            return self.data_params["tasks"][key].index(tasks)
        # If an integer is provided, return it as the index
        elif isinstance(tasks, int):
            return tasks
        # If a list is provided, return teh index for each element
        elif isinstance(tasks, list):
            return [self.get_task_idxs(task) for task in tasks]
        else:
            raise TypeError("Input must be a list, string or integer")
        if invert:
            return [
                i
                for i in range(self.model_params["n_tasks"])
                if i not in make_list(tasks)
            ]

    def input_coord_to_output_bin(
        self,
        input_coord: int,
        start_pos: int = 0,
    ) -> int:
        """
        Given the position of a base in the input, get the index of the corresponding bin
        in the model's prediction.

        Args:
            input_coord: Genomic coordinate of the input position
            start_pos: Genomic coordinate of the first base in the input sequence

        Returns:
            Index of the output bin containing the given position.

        """
        output_bin = (input_coord - start_pos) / self.data_params[
            "train_bin_size"
        ] - self.model_params["crop_len"]
        return int(np.floor(output_bin))

    def output_bin_to_input_coord(
        self,
        output_bin: int,
        return_pos: str = "start",
        start_pos: int = 0,
    ) -> int:
        """
        Given the index of a bin in the output, get its corresponding
        start or end coordinate.

        Args:
            output_bin: Index of the bin in the model's output
            return_pos: "start" or "end"
            start_pos: Genomic coordinate of the first base in the input sequence

        Returns:
            Genomic coordinate corresponding to the start (if return_pos = start)
            or end (if return_pos=end) of the bin.

        """
        start = (output_bin + self.model_params["crop_len"]) * self.data_params[
            "train_bin_size"
        ]
        if return_pos == "start":
            return start + start_pos
        elif return_pos == "end":
            return start + self.data_params["train_bin_size"] + start_pos
        else:
            raise NotImplementedError

    def input_intervals_to_output_intervals(
        self,
        intervals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Given a dataframe containing intervals corresponding to the
        input sequences, return a dataframe containing intervals corresponding
        to the model output.

        Args:
            intervals: A dataframe of genomic intervals

        Returns:
            A dataframe containing the genomic intervals corresponding
            to the model output from each input interval.
        """
        output_intervals = intervals.copy()
        crop_coords = self.model_params["crop_len"] * self.data_params["train_bin_size"]
        output_intervals["start"] = intervals.start + crop_coords
        output_intervals["end"] = intervals.end - crop_coords
        return output_intervals

    def input_intervals_to_output_bins(
        self, intervals: pd.DataFrame, start_pos: int = 0
    ) -> None:
        """
        Given a dataframe of genomic intervals, add columns indicating
        the indices of output bins that overlap the start and end of each interval.

        Args:
            intervals: A dataframe of genomic intervals
            start_pos: The start position of the sequence input to the model.

        Returns:start and end indices of the output bins corresponding
            to each input interval.
        """
        return pd.DataFrame(
            {
                "start": intervals.start.apply(
                    self.input_coord_to_output_bin, args=(start_pos,)
                ),
                "end": intervals.end.apply(
                    self.input_coord_to_output_bin, args=(start_pos,)
                )
                + 1,
            }
        )


class LightningModelEnsemble(pl.LightningModule):
    """
    Combine multiple LightningModel objects into a single object.
    When predict_on_dataset is used, it will return the concatenated
    predictions from all the models in the order in which they were supplied.

    Args:
        models (list): A list of multiple LightningModel objects
        model_names (list): A name for each model. This will be prefixed
            to the names of the individual tasks predicted by the model.
            If not supplied, the models will be named "model0", "model1", etc.
    """

    def __init__(self, models: list, model_names: Optional[List[str]] = None) -> None:
        super().__init__()

        # Save models
        self.models = [model for model in models]

        # Save model names
        self.model_names = model_names or [f"model{i}" for i in range(len(self.models))]

        # Save their task metadata
        self.model_params = {
            "n_tasks": sum([model.model_params["n_tasks"] for model in self.models])
        }
        self.data_params = {"tasks": defaultdict(list)}
        self._combine_tasks()

    def _combine_tasks(self) -> None:
        """
        Combine the task metadata of all the sub-models into self.data_params["tasks"]
        """
        # List all available keys
        task_keys = set(self.models[0].data_params["tasks"].keys())
        for model in self.models[1:]:
            task_keys = task_keys.intersection(model.data_params["tasks"].keys())

        # Collect task metadata for each key
        for key in task_keys:
            for name, model in zip(self.model_names, self.models):
                self.data_params["tasks"][key].extend(
                    ["_".join([name, x]) for x in model.data_params["tasks"][key]]
                )
            assert (
                len(set(self.data_params["tasks"][key])) == self.model_params["n_tasks"]
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward Pass.
        """
        return torch.cat([model(x) for model in self.models], axis=1)  # B, T, L

    def predict_on_dataset(self, dataset: Callable, **kwargs) -> np.ndarray:
        """
        This will return the concatenated predictions from all the
        constituent models, in the order in which they were supplied.
        Predictions will be concatenated along the task axis.
        """
        return np.concatenate(
            [model.predict_on_dataset(dataset, **kwargs) for model in self.models],
            axis=-2,
        )

    def get_task_idxs(
        self, tasks: Union[str, int, List[str], List[int]], key: str = "name"
    ) -> Union[int, List[int]]:
        """
        Return the task index given the name of the task. Note that task
        names should be supplied with a prefix indicating the model number,
        so for instance if you want the predictions from the second model
        on astrocytes, the task name would be "{name of second model}_astrocytes".
        If model names were not supplied to __init__, the task name would
        be "model1_astrocytes".

        Args:
            tasks: A string corresponding to a task name or metadata entry,
                or an integer indicating the index of a task, or a list of strings/integers
            key: key to model.data_params["tasks"] in which the relevant task data is
                stored. "name" will be used by default.

        Returns: An integer or list of integers representing the indices of the
            tasks in the model output.
        """
        # If a string is provided, extract the index
        if isinstance(tasks, str):
            return self.data_params["tasks"][key].index(tasks)

        # If an integer is provided, return it as the index
        elif isinstance(tasks, int):
            return tasks

        # If a list is provided, return the index for each element
        elif isinstance(tasks, list):
            return [self.get_task_idxs(task) for task in tasks]
        else:
            raise TypeError("Input must be a list, string or integer")
