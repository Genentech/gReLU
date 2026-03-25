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
