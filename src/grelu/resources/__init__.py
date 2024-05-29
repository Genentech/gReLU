import os
import importlib_resources
from tempfile import TemporaryDirectory
from pathlib import Path

import wandb
from grelu.lightning import LightningModel

DEFAULT_WANDB_ENTITY = 'grelu'
DEFAULT_WANDB_HOST = 'https://api.wandb.ai'


def get_meme_file_path(meme_motif_db):
    """
    Return the path to a MEME file.

    Args:
        meme_motif_db (str): Path to a MEME file or the name of a MEME file included with gReLU.
            Current name options are "jaspar" and "consensus".

    Returns:
        (str): Path to the specified MEME file.
    """
    if meme_motif_db == "jaspar":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "JASPAR2022_CORE_non-redundant_pfms_meme.txt"
        )
    elif meme_motif_db == "consensus":
        meme_motif_db = (
            importlib_resources.files("grelu")
            / "resources"
            / "meme"
            / "jvierstra_consensus_pwms.meme"
        )
    if os.path.isfile(meme_motif_db):
        return str(meme_motif_db)
    else:
        raise Exception(f"{meme_motif_db} is not a valid file.")


def get_default_config_file():
    config = importlib_resources.files("grelu") / "resources" / "default_config.yaml"
    assert config.exists()
    return str(config)


def get_blacklist_file(genome):
    blacklist = (
        importlib_resources.files("grelu")
        / "resources"
        / "blacklists"
        / "encode"
        / f"{genome}-blacklist.v2.bed"
    )
    assert blacklist.exists()
    return str(blacklist)


def _check_wandb(host=DEFAULT_WANDB_HOST):
    assert wandb.login(host=host), f'Weights & Biases (wandb) is not configured, see {DEFAULT_WANDB_HOST}/authorize'


def projects(host=DEFAULT_WANDB_HOST):
    _check_wandb(host=host)
    
    api = wandb.Api()
    projects = api.projects(DEFAULT_WANDB_ENTITY)
    return [p.name for p in projects]


def artifacts(project, host=DEFAULT_WANDB_HOST, type_is=None, type_contains=None):
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


def models(project, host=DEFAULT_WANDB_HOST):
    return artifacts(project, host=host, type_contains='model')


def datasets(project, host=DEFAULT_WANDB_HOST):
    return artifacts(project, host=host, type_contains='dataset')


def runs(project, host=DEFAULT_WANDB_HOST, field='id', filters=None):
    _check_wandb(host=host)
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'
    
    api = wandb.Api()
    return [getattr(run, field) for run in api.runs(project_path, filters=filters)]


def get_artifact(name, project, alias='latest'):
    _check_wandb()
    project_path = f'{DEFAULT_WANDB_ENTITY}/{project}'
    
    api = wandb.Api()    
    return api.artifact(f'{project_path}/{name}:{alias}')


def get_dataset_by_model(model_name, project, alias='latest'):
    art = get_artifact(model_name, project, alias=alias)
    run = art.logged_by()
    return [x.name for x in run.used_artifacts()]


def get_model_by_dataset(dataset_name, project, alias='latest'):
    art = get_artifact(dataset_name, project, alias=alias)
    runs = art.used_by()
    assert len(runs) > 0
    return [x.name for x in runs[0].logged_artifacts()]


def load_model(project, model_name, alias='latest', checkpoint_file='model.ckpt'):

    art = get_artifact(model_name, project, alias=alias)

    with TemporaryDirectory() as d:
        art.download(d)
        model = LightningModel.load_from_checkpoint(Path(d) / checkpoint_file)

    return model
