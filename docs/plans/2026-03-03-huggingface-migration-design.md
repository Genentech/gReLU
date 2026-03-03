# HuggingFace Model Zoo Migration Design

**Date:** 2026-03-03
**Status:** Approved

## Overview

Migrate the gReLU model zoo from Weights & Biases (wandb) to HuggingFace as the default backend. This is a breaking API change that introduces a new HuggingFace-native API while preserving wandb functionality in a legacy submodule.

## Background

- wandb deprecated anonymous downloads (https://github.com/wandb/wandb/pull/10909)
- Model zoo has been copied to HuggingFace: https://huggingface.co/collections/Genentech/grelu-model-zoo
- All models and datasets are public on HuggingFace (no authentication required)

## Design Decisions

### 1. New HuggingFace-Native API (Breaking Change)

Rather than mapping the old wandb API to HuggingFace, we introduce a new API where users provide full HuggingFace repo IDs:

```python
# Old API (wandb)
grelu.resources.load_model(project="human-atac-catlas", model_name="model")

# New API (HuggingFace)
grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
```

**Rationale:** Cleaner design, allows users to load from any HuggingFace repo (not just Genentech), no brittle naming convention mapping.

### 2. Module Organization

```
src/grelu/resources/
├── __init__.py      # New HF-native API (default)
├── wandb.py         # Legacy wandb functions
└── utils.py         # Shared utilities (meme files, blacklists)
```

- `grelu.resources` = HuggingFace (default)
- `grelu.resources.wandb` = legacy wandb access

### 3. No Hardcoded Organization

Users provide full repo IDs like `"Genentech/human-atac-catlas-model"` or `"MyOrg/my-custom-model"`. This allows flexibility for users with their own HuggingFace repos.

### 4. huggingface_hub as Required Dependency

Add `huggingface_hub` to `install_requires` in setup.cfg. Keep `wandb` for the legacy submodule.

## New API Surface

### `grelu.resources` (HuggingFace - default)

```python
# Constants
DEFAULT_HF_COLLECTION = "Genentech/grelu-model-zoo"

# Listing functions
def list_models() -> List[str]
    """List all model repo IDs in the gReLU collection."""

def list_datasets() -> List[str]
    """List all dataset repo IDs in the gReLU collection."""

# Download/Load functions
def load_model(repo_id: str, filename: str = "model.ckpt", device: str = "cpu") -> LightningModel
    """Download and load a model from HuggingFace."""

def download_model(repo_id: str, filename: str = "model.ckpt") -> str
    """Download a model checkpoint file, return local path."""

def download_dataset(repo_id: str, filename: str = "data.h5ad") -> str
    """Download a dataset file, return local path."""

# Lineage functions (using model card metadata)
def get_datasets_by_model(repo_id: str) -> List[str]
    """Get datasets linked to a model (from 'datasets' field in model card)."""

def get_base_models(repo_id: str) -> List[str]
    """Get base models this was fine-tuned from (from 'base_model' field)."""

def get_models_by_dataset(repo_id: str) -> List[str]
    """Get models trained on a dataset (searches collection models)."""

def get_model_info(repo_id: str) -> Dict
    """Get full model card metadata."""

def get_dataset_info(repo_id: str) -> Dict
    """Get full dataset card metadata."""
```

### `grelu.resources.wandb` (legacy)

All existing functions moved here unchanged:
- `projects()`, `artifacts()`, `models()`, `datasets()`, `runs()`
- `get_artifact()`, `load_model()`, `get_dataset_by_model()`, `get_model_by_dataset()`

## Pretrained Models Update

`BorzoiPretrainedModel` and `EnformerPretrainedModel` in `src/grelu/model/models.py` will use the new HuggingFace download:

```python
# Before (wandb)
from grelu.resources import get_artifact
art = get_artifact(f"human_state_dict_fold{fold}", project="borzoi", alias="latest")
with TemporaryDirectory() as d:
    art.download(d)
    state_dict = torch.load(Path(d) / f"fold{fold}.h5")

# After (HuggingFace)
from grelu.resources import download_model
path = download_model(
    repo_id="Genentech/borzoi-model",
    filename=f"human_state_dict_rep{fold}.h5"
)
state_dict = torch.load(path)
```

**Note:** Files use `rep` naming on HuggingFace (e.g., `human_state_dict_rep0.h5`).

## Caching

`huggingface_hub` provides automatic caching in `~/.cache/huggingface/hub/`:
- Files are not re-downloaded if already cached and unchanged
- Users can force re-download with `force_download=True`
- Users can work offline with `local_files_only=True`

This is an improvement over the current wandb code which downloads to a TemporaryDirectory each time.

## Error Handling

- **Authentication:** Not required for public repos. Clear error if user tries private repo.
- **File not found:** Raise `ValueError` listing available files.
- **Network errors:** Let `huggingface_hub` exceptions propagate.
- **Lineage edge cases:** Return empty list if metadata fields not present.

## Documentation Updates

### README.md

Add breaking changes notice at top:

```markdown
## Breaking Changes in v2.0

**Model Zoo Migration:** The gReLU model zoo has moved from Weights & Biases to HuggingFace.
The `grelu.resources` API has changed:

# Old API (wandb) - still available at grelu.resources.wandb
grelu.resources.load_model(project="human-atac-catlas", model_name="model")

# New API (HuggingFace)
grelu.resources.load_model(repo_id="Genentech/human-atac-catlas-model")
```

Remove/update the existing migration notice at line 10.

### Tutorials

- `docs/tutorials/6_model_zoo.ipynb` - Complete rewrite for new API
- `docs/tutorials/1_inference.ipynb` - Update any `load_model` calls
- Scan other tutorials for `grelu.resources` usage

## Testing Strategy

### New tests for HuggingFace functions

```python
# Unit tests (mock HF API)
def test_list_models_returns_list()
def test_list_datasets_returns_list()
def test_load_model_downloads_and_loads()
def test_download_model_returns_path()
def test_download_dataset_returns_path()
def test_get_datasets_by_model_parses_metadata()
def test_get_models_by_dataset_searches_collection()
def test_download_nonexistent_file_raises_error()

# Integration tests (optional, hit real HF API)
def test_load_real_model_from_hf()
def test_download_real_dataset_from_hf()
```

### wandb tests

- Mark existing wandb tests with `@pytest.mark.wandb`
- Exclude by default in pytest config: `addopts = -m "not wandb"`
- Can run explicitly with `pytest -m wandb`

## Files Changed

| File | Change |
|------|--------|
| `src/grelu/resources/__init__.py` | Rewrite with new HF API |
| `src/grelu/resources/wandb.py` | New file - move all wandb functions here |
| `src/grelu/resources/utils.py` | New file - shared utils (meme files, blacklists) |
| `src/grelu/model/models.py` | Update BorzoiPretrainedModel, EnformerPretrainedModel |
| `setup.cfg` | Add `huggingface_hub` dependency |
| `README.md` | Add breaking changes notice, update existing notice |
| `docs/tutorials/6_model_zoo.ipynb` | Complete rewrite for new API |
| `docs/tutorials/1_inference.ipynb` | Update `load_model` calls if any |
| `tests/test_resources.py` | New tests for HF functions |
| `tests/` (existing wandb tests) | Add `@pytest.mark.wandb` marker |
| `setup.cfg` [tool:pytest] | Add `-m "not wandb"` to addopts |
