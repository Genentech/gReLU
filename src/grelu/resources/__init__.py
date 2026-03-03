"""
`grelu.resources` contains functions to access the gReLU model zoo on HuggingFace,
as well as resource files bundled with gReLU.

For legacy wandb access, use `grelu.resources.wandb`.
"""

from typing import List, Dict, Any, Union

from huggingface_hub import hf_hub_download, HfApi

from grelu.lightning import LightningModel
from grelu.resources.utils import get_meme_file_path, get_blacklist_file


class DeprecationError(Exception):
    """Raised when deprecated API is used."""
    pass


# Re-export utility functions
__all__ = [
    # Exception
    "DeprecationError",
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


# === Deprecation stubs for old API ===
# These provide helpful error messages for users migrating from wandb API


def projects(*args, **kwargs):
    """Deprecated: Use list_models() or list_datasets() for HuggingFace."""
    raise DeprecationError(
        "grelu.resources.projects() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.list_models() or list_datasets()\n"
        "  - Legacy (wandb): use grelu.resources.wandb.projects()"
    )


def artifacts(*args, **kwargs):
    """Deprecated: Use list_models() or list_datasets() for HuggingFace."""
    raise DeprecationError(
        "grelu.resources.artifacts() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.list_models() or list_datasets()\n"
        "  - Legacy (wandb): use grelu.resources.wandb.artifacts()"
    )


def models(*args, **kwargs):
    """Deprecated: Use list_models() for HuggingFace."""
    raise DeprecationError(
        "grelu.resources.models() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.list_models()\n"
        "  - Legacy (wandb): use grelu.resources.wandb.models()"
    )


def datasets(*args, **kwargs):
    """Deprecated: Use list_datasets() for HuggingFace."""
    raise DeprecationError(
        "grelu.resources.datasets() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.list_datasets()\n"
        "  - Legacy (wandb): use grelu.resources.wandb.datasets()"
    )


def runs(*args, **kwargs):
    """Deprecated: Use get_model_info() for HuggingFace metadata."""
    raise DeprecationError(
        "grelu.resources.runs() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.get_model_info() for metadata\n"
        "  - Legacy (wandb): use grelu.resources.wandb.runs()"
    )


def get_artifact(*args, **kwargs):
    """Deprecated: Use download_model() or download_dataset() for HuggingFace."""
    raise DeprecationError(
        "grelu.resources.get_artifact() has been replaced.\n"
        "  - New (HuggingFace): use grelu.resources.download_model() or download_dataset()\n"
        "  - Legacy (wandb): use grelu.resources.wandb.get_artifact()"
    )
