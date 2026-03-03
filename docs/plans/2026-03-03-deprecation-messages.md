# Deprecation Messages Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add helpful error messages when users try old wandb-style API, guiding them to new HuggingFace API or legacy wandb submodule.

**Architecture:** Add DeprecationError class and stub functions in resources/__init__.py. Modify load_model() to detect old kwargs.

**Tech Stack:** Python, pytest

---

## Task 1: Write Failing Tests for Deprecation Errors

**Files:**
- Modify: `tests/test_resources_hf.py`

**Step 1: Add deprecation error tests**

Add this class at the end of the file:

```python
class TestDeprecationErrors:
    """Tests for deprecation error messages."""

    def test_projects_raises_deprecation_error(self):
        from grelu.resources import projects, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            projects()
        assert "list_models()" in str(exc_info.value)
        assert "grelu.resources.wandb.projects()" in str(exc_info.value)

    def test_artifacts_raises_deprecation_error(self):
        from grelu.resources import artifacts, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            artifacts("test-project")
        assert "grelu.resources.wandb.artifacts()" in str(exc_info.value)

    def test_models_raises_deprecation_error(self):
        from grelu.resources import models, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            models("test-project")
        assert "list_models()" in str(exc_info.value)

    def test_datasets_raises_deprecation_error(self):
        from grelu.resources import datasets, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            datasets("test-project")
        assert "list_datasets()" in str(exc_info.value)

    def test_runs_raises_deprecation_error(self):
        from grelu.resources import runs, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            runs("test-project")
        assert "grelu.resources.wandb.runs()" in str(exc_info.value)

    def test_get_artifact_raises_deprecation_error(self):
        from grelu.resources import get_artifact, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            get_artifact(name="dataset", project="test")
        assert "download_model()" in str(exc_info.value) or "download_dataset()" in str(exc_info.value)

    def test_load_model_with_project_kwarg_raises_deprecation_error(self):
        from grelu.resources import load_model, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            load_model(project="human-atac-catlas", model_name="model")
        assert "repo_id=" in str(exc_info.value)
        assert "grelu.resources.wandb.load_model" in str(exc_info.value)

    def test_load_model_with_only_project_raises_deprecation_error(self):
        from grelu.resources import load_model, DeprecationError
        with pytest.raises(DeprecationError) as exc_info:
            load_model(project="human-atac-catlas")
        assert "repo_id=" in str(exc_info.value)
```

**Step 2: Run tests to verify they fail**

Run: `ml Miniforge3 && micromamba activate grelu-test && pytest tests/test_resources_hf.py::TestDeprecationErrors -v 2>&1 | head -30`
Expected: FAIL (DeprecationError not defined, functions not found)

**Step 3: Commit**

```bash
git add tests/test_resources_hf.py
git commit -m "test: add failing tests for deprecation error messages"
```

---

## Task 2: Implement DeprecationError and Stub Functions

**Files:**
- Modify: `src/grelu/resources/__init__.py`

**Step 1: Add DeprecationError class after imports**

Add after the imports section (around line 10):

```python
class DeprecationError(Exception):
    """Raised when deprecated API is used."""
    pass
```

**Step 2: Add DeprecationError to __all__**

Update `__all__` to include "DeprecationError":

```python
__all__ = [
    # Exception
    "DeprecationError",
    # Utility functions
    "get_meme_file_path",
    ...
]
```

**Step 3: Add stub functions at end of file**

Add after all existing functions:

```python
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
```

**Step 4: Run tests to verify stub functions work**

Run: `ml Miniforge3 && micromamba activate grelu-test && pytest tests/test_resources_hf.py::TestDeprecationErrors::test_projects_raises_deprecation_error -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/grelu/resources/__init__.py
git commit -m "feat: add deprecation stubs for removed wandb functions"
```

---

## Task 3: Modify load_model() to Detect Old Kwargs

**Files:**
- Modify: `src/grelu/resources/__init__.py`

**Step 1: Update load_model() signature and add detection**

Find the existing `load_model()` function and replace it with:

```python
def load_model(
    repo_id: str = None,
    filename: str = "model.ckpt",
    device: Union[str, int] = "cpu",
    # Deprecated kwargs - kept for helpful error messages
    project: str = None,
    model_name: str = None,
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
    # Detect old API usage
    if project is not None or model_name is not None:
        raise DeprecationError(
            "grelu.resources.load_model() API has changed.\n"
            "  - New (HuggingFace): load_model(repo_id='Genentech/X-model')\n"
            "  - Legacy (wandb): use grelu.resources.wandb.load_model(project='X', model_name='Y')"
        )

    if repo_id is None:
        raise ValueError("repo_id is required. Example: load_model(repo_id='Genentech/human-atac-catlas-model')")

    path = download_model(repo_id=repo_id, filename=filename)
    return LightningModel.load_from_checkpoint(path, map_location=device)
```

**Step 2: Run all deprecation tests**

Run: `ml Miniforge3 && micromamba activate grelu-test && pytest tests/test_resources_hf.py::TestDeprecationErrors -v`
Expected: All PASS

**Step 3: Run full test suite to ensure no regressions**

Run: `ml Miniforge3 && micromamba activate grelu-test && pytest tests/test_resources_hf.py -v 2>&1 | tail -20`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/grelu/resources/__init__.py
git commit -m "feat: add deprecation detection to load_model() for old kwargs"
```

---

## Task 4: Verify and Final Commit

**Step 1: Test the user experience manually**

Run: `ml Miniforge3 && micromamba activate grelu-test && python -c "import grelu.resources; grelu.resources.projects()" 2>&1`
Expected: DeprecationError with helpful message

Run: `ml Miniforge3 && micromamba activate grelu-test && python -c "import grelu.resources; grelu.resources.load_model(project='test', model_name='model')" 2>&1`
Expected: DeprecationError with helpful message

**Step 2: Verify correct usage still works**

Run: `ml Miniforge3 && micromamba activate grelu-test && python -c "from grelu.resources import list_models; print('list_models works')"`
Expected: "list_models works"

**Step 3: Check git log**

Run: `git log --oneline -5`
Expected: See deprecation-related commits

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Write failing tests | tests/test_resources_hf.py |
| 2 | Add DeprecationError and stubs | src/grelu/resources/__init__.py |
| 3 | Modify load_model() detection | src/grelu/resources/__init__.py |
| 4 | Verify user experience | (manual testing) |
