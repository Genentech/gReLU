import os
import importlib_resources
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import wandb
from grelu.lightning import LightningModel

DEFAULT_WANDB_ENTITY = 'grelu'
DEFAULT_WANDB_HOST = 'https://api.wandb.ai'


def get_meme_file_path(meme_motif_db: str) -> str:
    """
    Return the path to a MEME file.

    Args:
        meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
            Current name options are "jaspar" and "consensus".

    Returns:
        Path to the specified MEME file.
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


def _check_wandb(host:str=DEFAULT_WANDB_HOST) -> None:
    """
    Check that the user is logged into Weights and Biases

    Args:
        host: URL of the Weights & Biases host
    """
    assert wandb.login(host=host, anonymous="allow"), f'Weights & Biases (wandb) is not configured, see {host}/authorize'


def projects(host: str=DEFAULT_WANDB_HOST) -> List[str]:
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


def artifacts(project: str, host: str=DEFAULT_WANDB_HOST, type_is: Optional[str]=None, type_contains: Optional[str]=None) -> List[str]:
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


def models(project:str, host:str=DEFAULT_WANDB_HOST) -> List[str]:
    """
    List all models associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host

    Returns:
        List of model names
    """
    return artifacts(project, host=host, type_contains='model')


def datasets(project:str, host:str=DEFAULT_WANDB_HOST) -> List[str]:
    """
    List all datasets associated with a project in the model zoo

    Args:
        project: Name of the project to search
        host: URL of the Weights & Biases host

    Returns:
        List of dataset names
    """
    return artifacts(project, host=host, type_contains='dataset')


def runs(project:str, host:str=DEFAULT_WANDB_HOST, field:str='id', filters: Optional[Dict[str, Any]]=None) -> List[str]:
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


def get_artifact(name:str, project:str, host:str=DEFAULT_WANDB_HOST, alias:str='latest'):
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


def get_dataset_by_model(model_name:str, project:str, host:str=DEFAULT_WANDB_HOST, alias:str='latest') -> List[str]:
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


def get_model_by_dataset(dataset_name:str, project:str, host:str=DEFAULT_WANDB_HOST, alias:str='latest') -> List[str]:
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
  project:str, model_name:str, device:Union[str, int]='cpu', host:str=DEFAULT_WANDB_HOST, alias:str='latest', checkpoint_file:str='model.ckpt'
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
