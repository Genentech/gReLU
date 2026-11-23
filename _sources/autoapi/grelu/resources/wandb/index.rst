grelu.resources.wandb
=====================

.. py:module:: grelu.resources.wandb

.. autoapi-nested-parse::

   Legacy functions for accessing the gReLU model zoo on Weights & Biases (wandb).

   Note: This module is deprecated. Use grelu.resources for HuggingFace-based access.



Attributes
----------

.. autoapisummary::

   grelu.resources.wandb.DEFAULT_WANDB_ENTITY
   grelu.resources.wandb.DEFAULT_WANDB_HOST


Functions
---------

.. autoapisummary::

   grelu.resources.wandb.projects
   grelu.resources.wandb.artifacts
   grelu.resources.wandb.models
   grelu.resources.wandb.datasets
   grelu.resources.wandb.runs
   grelu.resources.wandb.get_artifact
   grelu.resources.wandb.get_dataset_by_model
   grelu.resources.wandb.get_model_by_dataset
   grelu.resources.wandb.load_model


Module Contents
---------------

.. py:data:: DEFAULT_WANDB_ENTITY
   :value: 'grelu'


.. py:data:: DEFAULT_WANDB_HOST
   :value: 'https://api.wandb.ai'


.. py:function:: projects(host: str = DEFAULT_WANDB_HOST) -> List[str]

   List all projects in the model zoo

   :param host: URL of the Weights & Biases host

   :returns: List of project names


.. py:function:: artifacts(project: str, host: str = DEFAULT_WANDB_HOST, type_is: Optional[str] = None, type_contains: Optional[str] = None) -> List[str]

   List all artifacts associated with a project in the model zoo

   :param project: Name of the project to search
   :param host: URL of the Weights & Biases host
   :param type_is: Return only artifacts with this type
   :param type_contains: Return only artifacts whose type contains this string

   :returns: List of artifact names


.. py:function:: models(project: str, host: str = DEFAULT_WANDB_HOST) -> List[str]

   List all models associated with a project in the model zoo

   :param project: Name of the project to search
   :param host: URL of the Weights & Biases host

   :returns: List of model names


.. py:function:: datasets(project: str, host: str = DEFAULT_WANDB_HOST) -> List[str]

   List all datasets associated with a project in the model zoo

   :param project: Name of the project to search
   :param host: URL of the Weights & Biases host

   :returns: List of dataset names


.. py:function:: runs(project: str, host: str = DEFAULT_WANDB_HOST, field: str = 'id', filters: Optional[Dict[str, Any]] = None) -> List[str]

   List attributes of all runs associated with a project in the model zoo

   :param project: Name of the project to search
   :param host: URL of the Weights & Biases host
   :param field: Field to return from the run metadata
   :param filters: Dictionary of filters to pass to `api.runs`

   :returns: List of run attributes


.. py:function:: get_artifact(name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest')

   Retrieve an artifact associated with a project in the model zoo

   :param name: Name of the artifact
   :param project: Name of the project containing the artifact
   :param host: URL of the Weights & Biases host
   :param alias: Alias of the artifact

   :returns: The specific artifact


.. py:function:: get_dataset_by_model(model_name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest') -> List[str]

   List all datasets associated with a model in the model zoo

   :param model_name: Name of the model
   :param project: Name of the project containing the model
   :param host: URL of the Weights & Biases host
   :param alias: Alias of the model artifact

   :returns: A list containing the names of all datasets linked to the model


.. py:function:: get_model_by_dataset(dataset_name: str, project: str, host: str = DEFAULT_WANDB_HOST, alias: str = 'latest') -> List[str]

   List all models associated with a dataset in the model zoo

   :param dataset_name: Name of the dataset
   :param project: Name of the project containing the dataset
   :param host: URL of the Weights & Biases host
   :param alias: Alias of the dataset artifact

   :returns: A list containing the names of all models linked to the dataset


.. py:function:: load_model(project: str, model_name: str, device: Union[str, int] = 'cpu', host: str = DEFAULT_WANDB_HOST, alias: str = 'latest', checkpoint_file: str = 'model.ckpt', weights_only: bool = False) -> grelu.lightning.LightningModel

   Download and load a model from the model zoo

   :param project: Name of the project containing the model
   :param model_name: Name of the model
   :param device: Device index on which to load the model.
   :param host: URL of the Weights & Biases host
   :param alias: Alias of the model artifact
   :param checkpoint_file: Name of the checkpoint file contained in the model artifact
   :param weights_only: Whether to only load weights (default: False)

   :returns: A LightningModel object


