# Deprecation Messages for API Transition Design

**Date:** 2026-03-03
**Status:** Approved

## Overview

Add helpful error messages when users try to use the old wandb-style API, guiding them to either the new HuggingFace API or the legacy wandb submodule.

## Approach

Use stub functions that raise `DeprecationError` with clear migration instructions.

## Functions to Handle

**Removed functions** (need stub functions):
- `projects()`
- `artifacts()`
- `models()`
- `datasets()`
- `runs()`
- `get_artifact()`

**Existing function with changed signature** (detect old kwargs):
- `load_model()` - old: `load_model(project=, model_name=)`, new: `load_model(repo_id=)`

## Error Message Format

Pattern: Always "New (HuggingFace)" first, then "Legacy (wandb)" second.

```python
# For projects()
"grelu.resources.projects() has been replaced.\n"
"  - New (HuggingFace): use grelu.resources.list_models() or list_datasets()\n"
"  - Legacy (wandb): use grelu.resources.wandb.projects()"

# For load_model() with old kwargs
"grelu.resources.load_model() API has changed.\n"
"  - New (HuggingFace): load_model(repo_id='Genentech/X-model')\n"
"  - Legacy (wandb): use grelu.resources.wandb.load_model(project='X', model_name='Y')"

# For get_artifact()
"grelu.resources.get_artifact() has been replaced.\n"
"  - New (HuggingFace): use grelu.resources.download_model() or download_dataset()\n"
"  - Legacy (wandb): use grelu.resources.wandb.get_artifact()"

# For artifacts()
"grelu.resources.artifacts() has been replaced.\n"
"  - New (HuggingFace): use grelu.resources.list_models() or list_datasets()\n"
"  - Legacy (wandb): use grelu.resources.wandb.artifacts()"

# For models()
"grelu.resources.models() has been replaced.\n"
"  - New (HuggingFace): use grelu.resources.list_models()\n"
"  - Legacy (wandb): use grelu.resources.wandb.models()"

# For datasets()
"grelu.resources.datasets() has been replaced.\n"
"  - New (HuggingFace): use grelu.resources.list_datasets()\n"
"  - Legacy (wandb): use grelu.resources.wandb.datasets()"

# For runs()
"grelu.resources.runs() has been replaced.\n"
"  - New (HuggingFace): no direct equivalent (use get_model_info() for metadata)\n"
"  - Legacy (wandb): use grelu.resources.wandb.runs()"
```

## Implementation Structure

**File:** `src/grelu/resources/__init__.py`

1. Add custom exception class:
```python
class DeprecationError(Exception):
    """Raised when deprecated API is used."""
    pass
```

2. Add stub functions for removed functions:
```python
def projects(*args, **kwargs):
    raise DeprecationError(...)
```

3. Modify `load_model()` to detect old kwargs:
```python
def load_model(
    repo_id: str = None,
    filename: str = "model.ckpt",
    device: Union[str, int] = "cpu",
    # Deprecated kwargs for detection
    project: str = None,
    model_name: str = None,
) -> LightningModel:
    if project is not None or model_name is not None:
        raise DeprecationError(...)
    if repo_id is None:
        raise ValueError("repo_id is required")
    # ... rest of function
```

## Testing

Add tests to `tests/test_resources_hf.py`:

```python
class TestDeprecationErrors:
    def test_projects_raises_deprecation_error(self):
        from grelu.resources import projects, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            projects()
        assert "list_models()" in str(exc_info.value)
        assert "grelu.resources.wandb.projects()" in str(exc_info.value)

    def test_load_model_with_old_kwargs_raises_deprecation_error(self):
        from grelu.resources import load_model, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            load_model(project="human-atac-catlas", model_name="model")
        assert "repo_id=" in str(exc_info.value)

    def test_get_artifact_raises_deprecation_error(self):
        from grelu.resources import get_artifact, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            get_artifact(name="dataset", project="test")
        assert "download_model()" in str(exc_info.value) or "download_dataset()" in str(exc_info.value)
```

## Files Changed

| File | Change |
|------|--------|
| `src/grelu/resources/__init__.py` | Add DeprecationError class, stub functions, modify load_model() |
| `tests/test_resources_hf.py` | Add deprecation error tests |
