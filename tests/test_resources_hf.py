"""
Integration tests for HuggingFace-based model zoo functions.

These tests call the real HuggingFace API.
"""

import os
import pytest


def test_list_models():
    """list_models() returns model repos from collection."""
    from grelu.resources import list_models

    models = list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert all(m.endswith("-model") for m in models)


def test_list_datasets():
    """list_datasets() returns dataset repos from collection."""
    from grelu.resources import list_datasets

    datasets = list_datasets()

    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert all(d.endswith("-data") for d in datasets)


def test_download_model():
    """download_model() downloads file and returns path."""
    from grelu.resources import download_model

    path = download_model(repo_id="Genentech/test-model", filename="model.ckpt")

    assert isinstance(path, str)
    assert os.path.exists(path)
    assert path.endswith("model.ckpt")


def test_download_dataset():
    """download_dataset() downloads file and returns path."""
    from grelu.resources import download_dataset

    path = download_dataset(repo_id="Genentech/test-data", filename="data.h5ad")

    assert isinstance(path, str)
    assert os.path.exists(path)
    assert path.endswith("data.h5ad")


def test_load_model():
    """load_model() downloads and loads a LightningModel."""
    from grelu.resources import load_model
    from grelu.lightning import LightningModel

    model = load_model(repo_id="Genentech/test-model", filename="model.ckpt")

    assert isinstance(model, LightningModel)


def test_get_model_info():
    """get_model_info() returns metadata including files list."""
    from grelu.resources import get_model_info

    info = get_model_info("Genentech/test-model")

    assert "id" in info
    assert "files" in info
    assert "model.ckpt" in info["files"]


def test_get_dataset_info():
    """get_dataset_info() returns metadata including files list."""
    from grelu.resources import get_dataset_info

    info = get_dataset_info("Genentech/test-data")

    assert "id" in info
    assert "files" in info
    assert "data.h5ad" in info["files"]


def test_get_datasets_by_model():
    """get_datasets_by_model() returns linked datasets from model card."""
    from grelu.resources import get_datasets_by_model

    datasets = get_datasets_by_model("Genentech/human-atac-catlas-model")

    assert isinstance(datasets, list)
    assert "Genentech/human-atac-catlas-data" in datasets


def test_get_base_models():
    """get_base_models() returns base models from model card."""
    from grelu.resources import get_base_models

    base_models = get_base_models("Genentech/human-atac-catlas-model")

    assert isinstance(base_models, list)
    assert "Genentech/enformer-model" in base_models


def test_get_models_by_dataset():
    """get_models_by_dataset() returns models that use the dataset."""
    from grelu.resources import get_models_by_dataset

    models = get_models_by_dataset("Genentech/human-atac-catlas-data")

    assert isinstance(models, list)
    assert "Genentech/human-atac-catlas-model" in models


def test_deprecation_errors():
    """Deprecated functions raise DeprecationError with guidance."""
    from grelu.resources import (
        DeprecationError,
        projects, artifacts, models, datasets, runs, get_artifact
    )

    deprecated_funcs = [projects, artifacts, models, datasets, runs, get_artifact]

    for func in deprecated_funcs:
        with pytest.raises(DeprecationError):
            func()


def test_load_model_old_api_raises_deprecation():
    """load_model() with old kwargs raises DeprecationError."""
    from grelu.resources import load_model, DeprecationError

    with pytest.raises(DeprecationError) as exc_info:
        load_model(project="some-project", model_name="some-model")

    assert "HuggingFace" in str(exc_info.value)
    assert "repo_id" in str(exc_info.value)
