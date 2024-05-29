from grelu.resources import DEFAULT_WANDB_HOST  # get_model_by_dataset,
from grelu.resources import (
    artifacts,
    datasets,
    get_blacklist_file,
    get_dataset_by_model,
    get_meme_file_path,
    models,
    projects,
)


def test_resources():
    assert "hg38" in get_blacklist_file("hg38")
    assert get_meme_file_path("jaspar")

    import wandb

    try:
        wandb.login(host=DEFAULT_WANDB_HOST, anonymous="never", timeout=0)
    except wandb.errors.UsageError:  # login anonymously if not logged in already
        wandb.login(host=DEFAULT_WANDB_HOST, relogin=True, anonymous="must", timeout=0)

    assert len(projects()) > 0

    assert artifacts("model-zoo-test")
    assert models("model-zoo-test")
    assert datasets("model-zoo-test")

    # assert get_model_by_dataset("somedataset", "model-zoo-test")
    assert get_dataset_by_model("somemodel", "model-zoo-test")
