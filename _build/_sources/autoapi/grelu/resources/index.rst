grelu.resources
===============

.. py:module:: grelu.resources

.. autoapi-nested-parse::

   `grelu.resources` contains additional data files that can be used by gReLU functions.
   It also contains functions to load these files as well as files stored externally,
   such as model checkpoints and datasets in the model zoo on huggingface.

   For legacy wandb access, use `grelu.resources.wandb`.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/grelu/resources/utils/index
   /autoapi/grelu/resources/wandb/index


Exceptions
----------

.. autoapisummary::

   grelu.resources.DeprecationError


Functions
---------

.. autoapisummary::

   grelu.resources.get_meme_file_path
   grelu.resources.get_blacklist_file
   grelu.resources.list_models
   grelu.resources.list_datasets
   grelu.resources.download_model
   grelu.resources.download_dataset
   grelu.resources.load_model
   grelu.resources.get_model_info
   grelu.resources.get_dataset_info
   grelu.resources.get_datasets_by_model
   grelu.resources.get_base_models
   grelu.resources.get_models_by_dataset


Package Contents
----------------

.. py:function:: get_meme_file_path(meme_motif_db: str) -> str

   Return the path to a MEME file.

   :param meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
                         Current name options are "hocomoco_v12", "hocomoco_v13", and "consensus".

   :returns: Path to the specified MEME file.


.. py:function:: get_blacklist_file(genome: str) -> str

   Return the path to a blacklist file

   :param genome: Name of a genome whose blacklist file is included with gReLU.
                  Current name options are "hg19", "hg38" and "mm10".

   :returns: Path to the specified blacklist file.


.. py:exception:: DeprecationError

   Bases: :py:obj:`Exception`


   Raised when deprecated API is used.


.. py:function:: list_models() -> List[str]

   List all model repo IDs in the gReLU model zoo collection.

   :returns: List of model repository IDs (e.g., ["Genentech/human-atac-catlas-model", ...])


.. py:function:: list_datasets() -> List[str]

   List all dataset repo IDs in the gReLU model zoo collection.

   :returns: List of dataset repository IDs (e.g., ["Genentech/human-atac-catlas-data", ...])


.. py:function:: download_model(repo_id: str, filename: str = 'model.ckpt', **kwargs) -> str

   Download a model checkpoint file from HuggingFace.

   :param repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-model")
   :param filename: Name of the checkpoint file to download (default: "model.ckpt")
   :param \*\*kwargs: Additional arguments to pass to hf_hub_download

   :returns: Local path to the downloaded file


.. py:function:: download_dataset(repo_id: str, filename: str = 'data.h5ad', **kwargs) -> str

   Download a dataset file from HuggingFace.

   :param repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-data")
   :param filename: Name of the dataset file to download (default: "data.h5ad")
   :param \*\*kwargs: Additional arguments to pass to hf_hub_download

   :returns: Local path to the downloaded file


.. py:function:: load_model(repo_id: Optional[str] = None, filename: str = 'model.ckpt', device: Union[str, int] = 'cpu', project: Optional[str] = None, model_name: Optional[str] = None) -> grelu.lightning.LightningModel

   Download and load a model from HuggingFace.

   :param repo_id: HuggingFace repository ID (e.g., "Genentech/human-atac-catlas-model")
   :param filename: Name of the checkpoint file (default: "model.ckpt")
   :param device: Device to load the model on (default: "cpu")

   :returns: A LightningModel object


.. py:function:: get_model_info(repo_id: str) -> Dict[str, Any]

   Get full model card metadata from HuggingFace.

   :param repo_id: HuggingFace repository ID

   :returns: Dictionary containing model metadata including list of files


.. py:function:: get_dataset_info(repo_id: str) -> Dict[str, Any]

   Get full dataset card metadata from HuggingFace.

   :param repo_id: HuggingFace repository ID

   :returns: Dictionary containing dataset metadata including list of files


.. py:function:: get_datasets_by_model(repo_id: str) -> List[str]

   Get datasets linked to a model (from 'datasets' field in model card).

   :param repo_id: HuggingFace model repository ID

   :returns: List of dataset repository IDs linked to this model


.. py:function:: get_base_models(repo_id: str) -> List[str]

   Get base models this model was fine-tuned from (from 'base_model' field).

   :param repo_id: HuggingFace model repository ID

   :returns: List of base model repository IDs


.. py:function:: get_models_by_dataset(repo_id: str) -> List[str]

   Get models trained on a dataset (searches collection models).

   :param repo_id: HuggingFace dataset repository ID

   :returns: List of model repository IDs that use this dataset


