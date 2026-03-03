# HuggingFace Model Zoo Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate gReLU model zoo from wandb to HuggingFace as default backend with new API.

**Architecture:** New HuggingFace-native API in `grelu.resources`, legacy wandb functions moved to `grelu.resources.wandb`. Pretrained models (Borzoi, Enformer) updated to download from HF.

**Tech Stack:** huggingface_hub, pytorch-lightning, pytest

---

## Task 1: Add huggingface_hub Dependency

**Files:**
- Modify: `setup.cfg:50-53`

**Step 1: Add huggingface_hub to install_requires**

In `setup.cfg`, add `huggingface_hub` after the `wandb` line:

```
install_requires =
    importlib-metadata
    importlib-resources
    wandb >= 0.14
    huggingface_hub
    numpy
```

**Step 2: Verify the change**

Run: `grep -A5 "install_requires" setup.cfg`
Expected: Should show `huggingface_hub` in the list

**Step 3: Commit**

```bash
git add setup.cfg
git commit -m "feat: add huggingface_hub dependency"
```

---

## Task 2: Create utils.py with Shared Resource Utilities

**Files:**
- Create: `src/grelu/resources/utils.py`

**Step 1: Create utils.py with meme and blacklist functions**

```python
"""
Utility functions for accessing resource files bundled with gReLU.
"""

import os
import importlib_resources


def get_meme_file_path(meme_motif_db: str) -> str:
    """
    Return the path to a MEME file.

    Args:
        meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
            Current name options are "hocomoco_v12", "hocomoco_v13", and "consensus".

    Returns:
        Path to the specified MEME file.
    """
    if meme_motif_db == "hocomoco_v13":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "H13CORE_meme_format.meme"
        )
    elif meme_motif_db == "hocomoco_v12":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "H12CORE_meme_format.meme"
        )
    elif meme_motif_db == "consensus":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "jaspar_2024_consensus.meme"
        )
    elif meme_motif_db == 'jaspar':
        raise Exception("'jaspar' can no longer be supplied as a meme file name. Please see the function grelu.io.motifs.get_jaspar to load motifs from the JASPAR database.")
    if os.path.isfile(meme_motif_db):
        return str(meme_motif_db)
    else:
        raise Exception(f"{meme_motif_db} is not a valid file.")


def get_blacklist_file(genome: str) -> str:
    """
    Return the path to a blacklist file

    Args:
        genome: Name of a genome whose blacklist file is included with gReLU.
            Current name options are "hg19", "hg38" and "mm10".

    Returns:
        Path to the specified blacklist file.
    """
    blacklist = (
        importlib_resources.files("grelu")
        / "resources"
        / "blacklists"
        / "encode"
        / f"{genome}-blacklist.v2.bed"
    )
    assert blacklist.exists()
    return str(blacklist)
```

**Step 2: Verify file created**

Run: `ls -la src/grelu/resources/utils.py`
Expected: File exists

**Step 3: Commit**

```bash
git add src/grelu/resources/utils.py
git commit -m "refactor: extract shared utils to resources/utils.py"
```

---

## Task 3: Create wandb.py with Legacy Functions

**Files:**
- Create: `src/grelu/resources/wandb.py`

**Step 1: Create wandb.py with all wandb functions**

```python
"""
Legacy functions for accessing the gReLU model zoo on Weights & Biases (wandb).

Note: This module is deprecated. Use grelu.resources for HuggingFace-based access.
"""

from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import wandb
from grelu.lightning import LightningModel

DEFAULT_WANDB_ENTITY = 'grelu'
DEFAULT_WANDB_HOST = 'https://api.wandb.ai'


def _check_wandb(host: str = DEFAULT_WANDB_HOST) -> None:
    """
    Check that the user is logged into Weights and Biases

    Args:
        host: URL of the Weights & Biases host
    """
    try:
        wandb.login(host=host, anonymous="allow")
    except Exception as _:
        try:
            wandb.login(host=host, anonymous="must", timeout=0)
        except Exception as e:
            raise RuntimeError(f'Weights & Biases (wandb) is not configured, see {host}/authorize') from e


def projects(host: str = DEFAULT_WANDB_HOST) -> List[str]:
    """
    List all projects in the model zoo

    Args:
        host: URL of the Weights & Biases host

    Returns:
        List of project names
    """
    _check_wandb(host=host)

    api = wandb.Api()
    projects = api.projects(DEFAULT_WANDB_ENTITY)
    return [p.name for p in projects]


def artifacts(project: str, host: str = DEFAULT_WANDB_HOST, type_is: Optional[str] = None, type_contains: Optional[str] = None) -> List[str]:
    """
    List all artifacts associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host
        type_is: Return only artifacts with this type
        type_contains: Return only artifacts whose type contains this string

    Returns:
        List of artifact names
    """
    _check_wandb(host)
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'

    api = wandb.Api()
    if type_is is not None:
        types = [x.name for x in api.artifact_types(project_path) if type_is == x.name]
    elif type_contains is not None:
        types = [x.name for x in api.artifact_types(project_path) if type_contains in x.name]
    else:
        types = [x.name for x in api.artifact_types(project_path)]

    assert len(types) > 0, 'Artifact not found'

    coll = [api.artifact_type(art_type, project_path) for art_type in types]
    arts = [art.name for arts in coll for art in arts.collections()]
    return arts


def models(project: str, host: str = DEFAULT_WANDB_HOST) -> List[str]:
    """
    List all models associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host

    Returns:
        List of model names
    """
    return artifacts(project, host=host, type_contains='model')


def datasets(project: str, host: str = DEFAULT_WANDB_HOST) -> List[str]:
    """
    List all datasets associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host

    Returns:
        List of dataset names
    """
    return artifacts(project, host=host, type_contains='dataset')


def runs(project: str, host: str = DEFAULT_WANDB_HOST, field: str = 'id', filters: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    List attributes of all runs associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host
        field: Field to return from the run metadata
        filters: Dictionary of filters to pass to `api.runs`

    Returns:
        List of run attributes
    """
    _check_wandb(host=host)
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'

    api = wandb.Api()
    return [getattr(run, field) for run in api.runs(project_path, filters=filters)]


def get_artifact(name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest'):
    """
    Retrieve an artifact associated with a project in the model zoo

    Args:
        name: Name of the artifact
        project: Name of the project containing the artifact
        host: URL of the Weights & Biases host
        alias: Alias of the artifact

    Returns:
        The specific artifact
    """
    _check_wandb(host=host)
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'

    api = wandb.Api()
    return api.artifact(f'{project_path}/{name}:{alias}')


def get_dataset_by_model(model_name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest') -> List[str]:
    """
    List all datasets associated with a model in the model zoo

    Args:
        model_name: Name of the model
        project: Name of the project containing the model
        host: URL of the Weights & Biases host
        alias: Alias of the model artifact

    Returns:
        A list containing the names of all datasets linked to the model
    """
    art = get_artifact(model_name, project, host=host, alias=alias)
    run = art.logged_by()
    return [x.name for x in run.used_artifacts()]


def get_model_by_dataset(dataset_name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest') -> List[str]:
    """
    List all models associated with a dataset in the model zoo

    Args:
        dataset_name: Name of the dataset
        project: Name of the project containing the dataset
        host: URL of the Weights & Biases host
        alias: Alias of the dataset artifact

    Returns:
        A list containing the names of all models linked to the dataset
    """
    art = get_artifact(dataset_name, project, host=host, alias=alias)
    runs = art.used_by()
    assert len(runs) > 0
    return [x.name for x in runs[0].logged_artifacts()]


def load_model(
    project: str, model_name: str, device: Union[str, int] = 'cpu', host: str = DEFAULT_WANDB_HOST, alias: str = 'latest', checkpoint_file: str = 'model.ckpt'
) -> LightningModel:
    """
    Download and load a model from the model zoo

    Args:
        project: Name of the project containing the model
        model_name: Name of the model
        device: Device index on which to load the model.
        host: URL of the Weights & Biases host
        alias: Alias of the model artifact
        checkpoint_file: Name of the checkpoint file contained in the model artifact

    Returns:
        A LightningModel object
    """
    art = get_artifact(model_name, project, host=host, alias=alias)

    with TemporaryDirectory() as d:
        art.download(d)
        model = LightningModel.load_from_checkpoint(Path(d) / checkpoint_file, map_location=device)

    return model
```

**Step 2: Verify file created**

Run: `python -c "from grelu.resources import wandb; print(dir(wandb))"`
Expected: Should list all functions without import errors

**Step 3: Commit**

```bash
git add src/grelu/resources/wandb.py
git commit -m "refactor: move wandb functions to resources/wandb.py"
```

---

## Task 4: Write Failing Tests for HuggingFace Functions

**Files:**
- Create: `tests/test_resources_hf.py`

**Step 1: Write tests for new HuggingFace API**

```python
"""
Tests for HuggingFace-based model zoo functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestListModels:
    """Tests for list_models function."""

    @patch('grelu.resources.HfApi')
    def test_list_models_returns_model_repos(self, mock_hf_api):
        from grelu.resources import list_models

        # Setup mock
        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = [
            Mock(item_id="Genentech/human-atac-catlas-model"),
            Mock(item_id="Genentech/borzoi-model"),
        ]
        mock_api.get_collection.return_value = mock_collection

        result = list_models()

        assert "Genentech/human-atac-catlas-model" in result
        assert "Genentech/borzoi-model" in result

    @patch('grelu.resources.HfApi')
    def test_list_models_filters_to_models_only(self, mock_hf_api):
        from grelu.resources import list_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = [
            Mock(item_id="Genentech/human-atac-catlas-model"),
            Mock(item_id="Genentech/human-atac-catlas-data"),
        ]
        mock_api.get_collection.return_value = mock_collection

        result = list_models()

        assert "Genentech/human-atac-catlas-model" in result
        assert "Genentech/human-atac-catlas-data" not in result


class TestListDatasets:
    """Tests for list_datasets function."""

    @patch('grelu.resources.HfApi')
    def test_list_datasets_returns_dataset_repos(self, mock_hf_api):
        from grelu.resources import list_datasets

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = [
            Mock(item_id="Genentech/human-atac-catlas-data"),
            Mock(item_id="Genentech/borzoi-data"),
        ]
        mock_api.get_collection.return_value = mock_collection

        result = list_datasets()

        assert "Genentech/human-atac-catlas-data" in result
        assert "Genentech/borzoi-data" in result


class TestDownloadModel:
    """Tests for download_model function."""

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_returns_path(self, mock_download):
        from grelu.resources import download_model

        mock_download.return_value = "/path/to/model.ckpt"

        result = download_model(repo_id="Genentech/test-model")

        assert result == "/path/to/model.ckpt"
        mock_download.assert_called_once_with(
            repo_id="Genentech/test-model",
            filename="model.ckpt",
        )

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_custom_filename(self, mock_download):
        from grelu.resources import download_model

        mock_download.return_value = "/path/to/custom.ckpt"

        result = download_model(repo_id="Genentech/test-model", filename="custom.ckpt")

        mock_download.assert_called_once_with(
            repo_id="Genentech/test-model",
            filename="custom.ckpt",
        )


class TestDownloadDataset:
    """Tests for download_dataset function."""

    @patch('grelu.resources.hf_hub_download')
    def test_download_dataset_returns_path(self, mock_download):
        from grelu.resources import download_dataset

        mock_download.return_value = "/path/to/data.h5ad"

        result = download_dataset(repo_id="Genentech/test-data")

        assert result == "/path/to/data.h5ad"
        mock_download.assert_called_once_with(
            repo_id="Genentech/test-data",
            filename="data.h5ad",
            repo_type="dataset",
        )


class TestLoadModel:
    """Tests for load_model function."""

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_downloads_and_loads(self, mock_download, mock_lightning):
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/model.ckpt"
        mock_model = Mock()
        mock_lightning.load_from_checkpoint.return_value = mock_model

        result = load_model(repo_id="Genentech/test-model")

        assert result == mock_model
        mock_download.assert_called_once()
        mock_lightning.load_from_checkpoint.assert_called_once_with(
            "/path/to/model.ckpt", map_location="cpu"
        )


class TestGetDatasetsByModel:
    """Tests for get_datasets_by_model function."""

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_parses_metadata(self, mock_hf_api):
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.datasets = ["Genentech/human-atac-catlas-data"]
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/human-atac-catlas-model")

        assert result == ["Genentech/human-atac-catlas-data"]

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_returns_empty_if_no_datasets(self, mock_hf_api):
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = None
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/some-model")

        assert result == []


class TestGetBaseModels:
    """Tests for get_base_models function."""

    @patch('grelu.resources.HfApi')
    def test_get_base_models_parses_metadata(self, mock_hf_api):
        from grelu.resources import get_base_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.base_model = ["Genentech/enformer-model"]
        mock_api.model_info.return_value = mock_info

        result = get_base_models(repo_id="Genentech/human-atac-catlas-model")

        assert result == ["Genentech/enformer-model"]


class TestUtilityFunctions:
    """Tests for utility functions (meme, blacklist)."""

    def test_get_blacklist_file_hg38(self):
        from grelu.resources import get_blacklist_file

        result = get_blacklist_file("hg38")
        assert "hg38" in result
        assert result.endswith(".bed")

    def test_get_meme_file_path_hocomoco_v12(self):
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("hocomoco_v12")
        assert result.endswith(".meme")

    def test_get_meme_file_path_hocomoco_v13(self):
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("hocomoco_v13")
        assert result.endswith(".meme")

    def test_get_meme_file_path_consensus(self):
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("consensus")
        assert result.endswith(".meme")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resources_hf.py -v 2>&1 | head -50`
Expected: ImportError or AttributeError (functions don't exist yet)

**Step 3: Commit**

```bash
git add tests/test_resources_hf.py
git commit -m "test: add failing tests for HuggingFace resource functions"
```

---

## Task 5: Implement HuggingFace API in resources/__init__.py

**Files:**
- Modify: `src/grelu/resources/__init__.py` (complete rewrite)

**Step 1: Rewrite __init__.py with HuggingFace API**

```python
"""
`grelu.resources` contains functions to access the gReLU model zoo on HuggingFace,
as well as resource files bundled with gReLU.

For legacy wandb access, use `grelu.resources.wandb`.
"""

from typing import List, Dict, Any, Union

from huggingface_hub import hf_hub_download, HfApi

from grelu.lightning import LightningModel
from grelu.resources.utils import get_meme_file_path, get_blacklist_file

# Re-export utility functions
__all__ = [
    # Utility functions
    "get_meme_file_path",
    "get_blacklist_file",
    # HuggingFace functions
    "list_models",
    "list_datasets",
    "download_model",
    "download_dataset",
    "load_model",
    "get_datasets_by_model",
    "get_base_models",
    "get_models_by_dataset",
    "get_model_info",
    "get_dataset_info",
]

DEFAULT_HF_COLLECTION = "Genentech/grelu-model-zoo-67b0a87e19442de9c2c6bd61"


def list_models() -> List[str]:
    """
    List all model repo IDs in the gReLU model zoo collection.

    Returns:
        List of model repository IDs (e.g., ["Genentech/human-atac-catlas-model", ...])
    """
    api = HfApi()
    collection = api.get_collection(DEFAULT_HF_COLLECTION)
    return [item.item_id for item in collection.items if item.item_id.endswith("-model")]


def list_datasets() -> List[str]:
    """
    List all dataset repo IDs in the gReLU model zoo collection.

    Returns:
        List of dataset repository IDs (e.g., ["Genentech/human-atac-catlas-data", ...])
    """
    api = HfApi()
    collection = api.get_collection(DEFAULT_HF_COLLECTION)
    return [item.item_id for item in collection.items if item.item_id.endswith("-data")]


def download_model(repo_id: str, filename: str = "model.ckpt") -> str:
    """
    Download a model checkpoint file from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-model")
        filename: Name of the checkpoint file to download (default: "model.ckpt")

    Returns:
        Local path to the downloaded file
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download_dataset(repo_id: str, filename: str = "data.h5ad") -> str:
    """
    Download a dataset file from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-data")
        filename: Name of the dataset file to download (default: "data.h5ad")

    Returns:
        Local path to the downloaded file
    """
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def load_model(
    repo_id: str,
    filename: str = "model.ckpt",
    device: Union[str, int] = "cpu",
) -> LightningModel:
    """
    Download and load a model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-model")
        filename: Name of the checkpoint file (default: "model.ckpt")
        device: Device to load the model on (default: "cpu")

    Returns:
        A LightningModel object
    """
    path = download_model(repo_id=repo_id, filename=filename)
    return LightningModel.load_from_checkpoint(path, map_location=device)


def get_model_info(repo_id: str) -> Dict[str, Any]:
    """
    Get full model card metadata from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Dictionary containing model metadata
    """
    api = HfApi()
    info = api.model_info(repo_id)
    return {
        "id": info.id,
        "tags": info.tags,
        "card_data": info.card_data.__dict__ if info.card_data else {},
        "downloads": info.downloads,
        "last_modified": info.last_modified,
    }


def get_dataset_info(repo_id: str) -> Dict[str, Any]:
    """
    Get full dataset card metadata from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Dictionary containing dataset metadata
    """
    api = HfApi()
    info = api.dataset_info(repo_id)
    return {
        "id": info.id,
        "tags": info.tags,
        "card_data": info.card_data.__dict__ if info.card_data else {},
        "downloads": info.downloads,
        "last_modified": info.last_modified,
    }


def get_datasets_by_model(repo_id: str) -> List[str]:
    """
    Get datasets linked to a model (from 'datasets' field in model card).

    Args:
        repo_id: HuggingFace model repository ID

    Returns:
        List of dataset repository IDs linked to this model
    """
    api = HfApi()
    info = api.model_info(repo_id)
    if info.card_data and hasattr(info.card_data, 'datasets') and info.card_data.datasets:
        return list(info.card_data.datasets)
    return []


def get_base_models(repo_id: str) -> List[str]:
    """
    Get base models this model was fine-tuned from (from 'base_model' field).

    Args:
        repo_id: HuggingFace model repository ID

    Returns:
        List of base model repository IDs
    """
    api = HfApi()
    info = api.model_info(repo_id)
    if info.card_data and hasattr(info.card_data, 'base_model') and info.card_data.base_model:
        base_model = info.card_data.base_model
        if isinstance(base_model, str):
            return [base_model]
        return list(base_model)
    return []


def get_models_by_dataset(repo_id: str) -> List[str]:
    """
    Get models trained on a dataset (searches collection models).

    Args:
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of model repository IDs that use this dataset
    """
    models = list_models()
    result = []
    for model_id in models:
        datasets = get_datasets_by_model(model_id)
        if repo_id in datasets:
            result.append(model_id)
    return result
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_resources_hf.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/grelu/resources/__init__.py
git commit -m "feat: implement HuggingFace API for model zoo access"
```

---

## Task 6: Update Pretrained Models to Use HuggingFace

**Files:**
- Modify: `src/grelu/model/models.py:630-641` (BorzoiPretrainedModel)
- Modify: `src/grelu/model/models.py:796-805` (EnformerPretrainedModel)

**Step 1: Update BorzoiPretrainedModel**

Find the section around line 630 and replace:

```python
        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact(
            f"human_state_dict_fold{fold}", project="borzoi", alias="latest"
        )
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / f"fold{fold}.h5")
```

With:

```python
        # Load state dict from HuggingFace
        from grelu.resources import download_model

        path = download_model(
            repo_id="Genentech/borzoi-model",
            filename=f"human_state_dict_rep{fold}.h5",
        )
        state_dict = torch.load(path)
```

**Step 2: Update EnformerPretrainedModel**

Find the section around line 796 and replace:

```python
        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact("human_state_dict", project="enformer", alias="latest")
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / "human.h5")
```

With:

```python
        # Load state dict from HuggingFace
        from grelu.resources import download_model

        path = download_model(
            repo_id="Genentech/enformer-model",
            filename="human_state_dict.h5",
        )
        state_dict = torch.load(path)
```

**Step 3: Verify syntax**

Run: `python -c "import grelu.model.models; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add src/grelu/model/models.py
git commit -m "feat: update pretrained models to download from HuggingFace"
```

---

## Task 7: Mark Existing wandb Tests with pytest Marker

**Files:**
- Modify: `tests/test_resources.py`
- Modify: `setup.cfg`

**Step 1: Update test_resources.py with wandb marker**

Replace the entire file with:

```python
import pytest

from grelu.resources import get_blacklist_file, get_meme_file_path


def test_resources():
    """Test utility functions for bundled resource files."""
    assert "hg38" in get_blacklist_file("hg38")
    assert get_meme_file_path("hocomoco_v12")
    assert get_meme_file_path("hocomoco_v13")
    assert get_meme_file_path("consensus")


@pytest.mark.wandb
def test_wandb_resources():
    """Test legacy wandb functions - skipped by default."""
    from grelu.resources.wandb import (
        DEFAULT_WANDB_HOST,
        artifacts,
        datasets,
        get_dataset_by_model,
        models,
        projects,
    )
    import wandb

    try:
        wandb.login(host=DEFAULT_WANDB_HOST, anonymous="never", timeout=0)
    except wandb.errors.UsageError:
        wandb.login(host=DEFAULT_WANDB_HOST, relogin=True, anonymous="must", timeout=0)

    assert len(projects()) > 0

    assert artifacts("model-zoo-test")
    assert models("model-zoo-test")
    assert datasets("model-zoo-test")

    assert get_dataset_by_model("somemodel", "model-zoo-test")
```

**Step 2: Update setup.cfg to exclude wandb tests by default**

In setup.cfg, find the `[tool:pytest]` section and update `addopts`:

```ini
[tool:pytest]
addopts =
    --cov grelu --cov-report term-missing
    --verbose
    -m "not wandb"
```

Also add markers configuration:

```ini
markers =
    wandb: marks tests that require wandb authentication (deselect with '-m "not wandb"')
```

**Step 3: Verify wandb tests are skipped**

Run: `pytest tests/test_resources.py -v`
Expected: test_resources PASS, test_wandb_resources SKIPPED (or not collected)

**Step 4: Commit**

```bash
git add tests/test_resources.py setup.cfg
git commit -m "test: mark wandb tests with pytest marker, skip by default"
```

---

## Task 8: Update README with Breaking Changes Notice

**Files:**
- Modify: `README.md:9-11`

**Step 1: Replace existing notice with breaking changes notice**

Find and replace lines 9-11 (the current notice section) with:

```markdown
## Breaking Changes in v2.0

**Model Zoo Migration:** The gReLU model zoo has moved from Weights & Biases to HuggingFace. The `grelu.resources` API has changed:

```python
# Old API (wandb) - still available at grelu.resources.wandb
grelu.resources.load_model(project="human-atac-catlas", model_name="model")

# New API (HuggingFace)
grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
```

See the [Model Zoo Tutorial](docs/tutorials/6_model_zoo.ipynb) for updated usage.
```

**Step 2: Verify README renders correctly**

Run: `head -30 README.md`
Expected: Breaking changes notice visible at top

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add breaking changes notice for HuggingFace migration"
```

---

## Task 9: Rewrite Model Zoo Tutorial

**Files:**
- Modify: `docs/tutorials/6_model_zoo.ipynb` (complete rewrite)

**Step 1: Create new model zoo tutorial content**

This task requires rewriting the Jupyter notebook. Create new content:

- Title: "Querying the gReLU model zoo on HuggingFace"
- Sections:
  1. Introduction (link to https://huggingface.co/collections/Genentech/grelu-model-zoo)
  2. List available models and datasets
  3. Download and load a model
  4. Download a dataset
  5. Query model metadata (lineage)
  6. Legacy wandb access (brief section)

Key code cells to include:

```python
import grelu.resources

# List all models
grelu.resources.list_models()

# List all datasets
grelu.resources.list_datasets()

# Load a model
model = grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")

# Download a dataset
dataset_path = grelu.resources.download_dataset(repo_id="Genentech/human-atac-catlas-data")

# Get linked datasets
grelu.resources.get_datasets_by_model(repo_id="Genentech/human-atac-catlas-model")

# Legacy wandb access
from grelu.resources import wandb
wandb.projects()
```

**Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('docs/tutorials/6_model_zoo.ipynb'))"`
Expected: No error

**Step 3: Commit**

```bash
git add docs/tutorials/6_model_zoo.ipynb
git commit -m "docs: rewrite model zoo tutorial for HuggingFace API"
```

---

## Task 10: Update Other Tutorials

**Files:**
- Modify: `docs/tutorials/1_inference.ipynb`
- Modify: `docs/tutorials/2_finetune.ipynb`
- Modify: `docs/tutorials/3_train.ipynb`
- Modify: `docs/tutorials/4_design.ipynb`
- Modify: `docs/tutorials/5_variant.ipynb`
- Modify: `docs/tutorials/7_simulations.ipynb`

**Step 1: Update each tutorial**

For each tutorial, find and replace the old API calls:

| Old Code | New Code |
|----------|----------|
| `grelu.resources.load_model(project="X", model_name="Y")` | `grelu.resources.load_model(repo_id="Genentech/X-model")` |
| `grelu.resources.get_artifact(name="dataset", project="X")` | `grelu.resources.download_dataset(repo_id="Genentech/X-data")` |
| `grelu.resources.get_artifact(project="X", name="Y")` | `grelu.resources.download_model(repo_id="Genentech/X-model", filename="Y")` or `download_dataset` |

Specific changes needed:

**1_inference.ipynb:**
```python
# Old
model = grelu.resources.load_model(project="borzoi", model_name="model")
# New
model = grelu.resources.load_model(repo_id="Genentech/borzoi-model")
```

**2_finetune.ipynb:**
```python
# Old
artifact = grelu.resources.get_artifact(name="dataset", project="binary_atac_cell_lines")
# New
dataset_path = grelu.resources.download_dataset(repo_id="Genentech/binary-atac-tutorial-data")
```

**3_train.ipynb:**
```python
# Old
fragment_file_dir = grelu.resources.get_artifact(project='microglia-scatac-tutorial', name='fragment_file').download()
peak_file_dir = grelu.resources.get_artifact(project='microglia-scatac-tutorial', name='peak_file').download()
# New
fragment_file_path = grelu.resources.download_dataset(repo_id="Genentech/microglia-scatac-tutorial-data", filename="fragment_file.tsv.gz")
peak_file_path = grelu.resources.download_dataset(repo_id="Genentech/microglia-scatac-tutorial-data", filename="peaks.bed")
```

**4_design.ipynb:**
```python
# Old
model = grelu.resources.load_model(project='human-atac-catlas', model_name="model")
# New
model = grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
```

**5_variant.ipynb:**
```python
# Old
model = grelu.resources.load_model(project='human-atac-catlas', model_name='model')
variant_dir = grelu.resources.get_artifact(project='alzheimers-variant-tutorial', name='variant_files').download()
# New
model = grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
variant_dir = grelu.resources.download_dataset(repo_id="Genentech/alzheimers-variant-tutorial-data")
```

**7_simulations.ipynb:**
```python
# Old
catlas = grelu.resources.load_model(project="human-atac-catlas", model_name="model")
enformer = grelu.resources.load_model(project="enformer", model_name="model")
# New
catlas = grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
enformer = grelu.resources.load_model(repo_id="Genentech/enformer-model")
```

**Step 2: Verify notebooks are valid**

Run: `for f in docs/tutorials/*.ipynb; do python -c "import json; json.load(open('$f'))" && echo "$f OK"; done`
Expected: All notebooks OK

**Step 3: Commit**

```bash
git add docs/tutorials/*.ipynb
git commit -m "docs: update tutorials for new HuggingFace API"
```

---

## Task 11: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --ignore=tests/test_resources.py::test_wandb_resources`
Expected: All tests PASS

**Step 2: Run import check**

Run: `python -c "import grelu; import grelu.resources; import grelu.resources.wandb; print('All imports OK')"`
Expected: All imports OK

**Step 3: Final commit with all changes verified**

```bash
git log --oneline -10
```
Expected: See all commits from this implementation

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add huggingface_hub dependency | setup.cfg |
| 2 | Create utils.py | src/grelu/resources/utils.py |
| 3 | Create wandb.py | src/grelu/resources/wandb.py |
| 4 | Write failing tests | tests/test_resources_hf.py |
| 5 | Implement HuggingFace API | src/grelu/resources/__init__.py |
| 6 | Update pretrained models | src/grelu/model/models.py |
| 7 | Mark wandb tests | tests/test_resources.py, setup.cfg |
| 8 | Update README | README.md |
| 9 | Rewrite model zoo tutorial | docs/tutorials/6_model_zoo.ipynb |
| 10 | Update other tutorials | docs/tutorials/*.ipynb |
| 11 | Run full test suite | (verification) |
