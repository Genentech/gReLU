grelu.resources
===============

.. py:module:: grelu.resources


Attributes
----------

.. autoapisummary::

   grelu.resources.DEFAULT_WANDB_ENTITY
   grelu.resources.DEFAULT_WANDB_HOST


Classes
-------

.. autoapisummary::

   grelu.resources.LightningModel


Functions
---------

.. autoapisummary::

   grelu.resources.get_meme_file_path
   grelu.resources.get_default_config_file
   grelu.resources.get_blacklist_file
   grelu.resources._check_wandb
   grelu.resources.projects
   grelu.resources.artifacts
   grelu.resources.models
   grelu.resources.datasets
   grelu.resources.runs
   grelu.resources.get_artifact
   grelu.resources.get_dataset_by_model
   grelu.resources.get_model_by_dataset
   grelu.resources.load_model


Package Contents
----------------

.. py:class:: LightningModel(model_params: dict, train_params: dict = {}, data_params: dict = {})

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   Wrapper for predictive sequence models

   :param model_params: Dictionary of parameters specifying model architecture
   :param train_params: Dictionary specifying training parameters
   :param data_params: Dictionary specifying parameters of the training data.
                       This is empty by default and will be filled at the time of
                       training.


   .. py:method:: build_model() -> None

      Build a model from parameter dictionary



   .. py:method:: initialize_loss() -> None

      Create the specified loss function.



   .. py:method:: initialize_activation() -> None

      Add a task-specific activation function to the model.



   .. py:method:: initialize_metrics()

      Initialize the appropriate metrics for the given task.



   .. py:method:: update_metrics(metrics: dict, y_hat: torch.Tensor, y: torch.Tensor) -> None

      Update metrics after each pass



   .. py:method:: format_input(x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor

      Extract the one-hot encoded sequence from the input



   .. py:method:: forward(x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, str, List[str]], logits: bool = False) -> torch.Tensor

      Forward pass



   .. py:method:: training_step(batch: torch.Tensor, batch_idx: int) -> torch.Tensor


   .. py:method:: validation_step(batch: torch.Tensor, batch_idx: int) -> torch.Tensor


   .. py:method:: on_validation_epoch_end()

      Calculate metrics for entire validation set



   .. py:method:: test_step(batch: torch.Tensor, batch_idx: int) -> torch.Tensor

      Calculate metrics after a single test step



   .. py:method:: on_test_epoch_end() -> None

      Calculate metrics for entire test set



   .. py:method:: configure_optimizers() -> None

      Configure oprimizer for training



   .. py:method:: count_params() -> int

      Number of gradient enabled parameters in the model



   .. py:method:: parse_devices(devices: Union[str, int, List[int]]) -> Tuple[str, Union[str, List[int]]]

      Parses the devices argument and returns a tuple of accelerator and devices.

      :param devices: Either "cpu" or an integer or list of integers representing the indices
                      of the GPUs for training.

      :returns: A tuple of accelerator and devices.



   .. py:method:: parse_logger() -> str

      Parses the name of the logger supplied in train_params.



   .. py:method:: add_transform(prediction_transform: Callable) -> None

      Add a prediction transform



   .. py:method:: reset_transform() -> None

      Remove a prediction transform



   .. py:method:: make_train_loader(dataset: Callable, batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> Callable

      Make dataloader for training



   .. py:method:: make_test_loader(dataset: Callable, batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> Callable

      Make dataloader for validation and testing



   .. py:method:: make_predict_loader(dataset: Callable, batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> Callable

      Make dataloader for prediction



   .. py:method:: train_on_dataset(train_dataset: Callable, val_dataset: Callable, checkpoint_path: Optional[str] = None)

      Train model and optionally log metrics to wandb.

      :param train_dataset: Dataset object that yields training examples
      :type train_dataset: Dataset
      :param val_dataset: Dataset object that yields training examples
      :type val_dataset: Dataset
      :param checkpoint_path: Path to model checkpoint from which to resume training.
                              The optimizer will be set to its checkpointed state.
      :type checkpoint_path: str

      :returns: PyTorch Lightning Trainer



   .. py:method:: _get_dataset_attrs(dataset: Callable) -> None

      Read data parameters from a dataset object



   .. py:method:: change_head(n_tasks: int, final_pool_func: str) -> None

      Build a new head with the desired number of tasks



   .. py:method:: tune_on_dataset(train_dataset: Callable, val_dataset: Callable, final_act_func: Optional[str] = None, final_pool_func: Optional[str] = None, freeze_embedding: bool = False)

      Fine-tune a pretrained model on a new dataset.

      :param train_dataset: Dataset object that yields training examples
      :param val_dataset: Dataset object that yields training examples
      :param final_act_func: Name of the final activation layer
      :param final_pool_func: Name of the final pooling layer
      :param freeze_embedding: If True, all the embedding layers of the pretrained
                               model will be frozen and only the head will be trained.

      :returns: PyTorch Lightning Trainer



   .. py:method:: on_save_checkpoint(checkpoint: dict) -> None


   .. py:method:: predict_on_seqs(x: Union[str, List[str]], device: Union[str, int] = 'cpu') -> numpy.ndarray

      A simple function to return model predictions directly
      on a batch of a single batch of sequences in string
      format.

      :param x: DNA sequences as a string or list of strings.
      :param device: Index of the device to use

      :returns: A numpy array of predictions.



   .. py:method:: predict_on_dataset(dataset: Callable, devices: Union[int, str, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 256, augment_aggfunc: Union[str, Callable] = 'mean', compare_func: Optional[Union[str, Callable]] = None, return_df: bool = False)

      Predict for a dataset of sequences or variants

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param devices: Device IDs to use
      :param num_workers: Number of workers for data loader
      :param batch_size: Batch size for data loader
      :param augment_aggfunc: Return the average prediction across all augmented
                              versions of a sequence
      :param compare_func: Return the alt/ref difference for variants
      :param return_df: Return the predictions as a Pandas dataframe

      :returns: Model predictions as a numpy array or dataframe



   .. py:method:: test_on_dataset(dataset: Callable, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 256)

      Run test loop for a dataset

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param devices: Device IDs to use for inference
      :param num_workers: Number of workers for data loader
      :param batch_size: Batch size for data loader

      :returns: Dataframe containing all calculated metrics on the test set.



   .. py:method:: embed_on_dataset(dataset: Callable, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 256)

      Return embeddings for a dataset of sequences

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param devices: Device IDs to use
      :param num_workers: Number of workers for data loader
      :param batch_size: Batch size for data loader

      :returns: Numpy array of shape (B, T, L) containing embeddings.



   .. py:method:: get_task_idxs(tasks: Union[int, str, List[int], List[str]], key: str = 'name', invert: bool = False) -> Union[int, List[int]]

      Given a task name or metadata entry, get the task index
      If integers are provided, return them unchanged

      :param tasks: A string corresponding to a task name or metadata entry,
                    or an integer indicating the index of a task, or a list of strings/integers
      :param key: key to model.data_params["tasks"] in which the relevant task data is
                  stored. "name" will be used by default.
      :param invert: Get indices for all tasks except those listed in tasks

      :returns: The index or indices of the corresponding task(s) in the model's
                output.



   .. py:method:: input_coord_to_output_bin(input_coord: int, start_pos: int = 0) -> int

      Given the position of a base in the input, get the index of the corresponding bin
      in the model's prediction.

      :param input_coord: Genomic coordinate of the input position
      :param start_pos: Genomic coordinate of the first base in the input sequence

      :returns: Index of the output bin containing the given position.



   .. py:method:: output_bin_to_input_coord(output_bin: int, return_pos: str = 'start', start_pos: int = 0) -> int

      Given the index of a bin in the output, get its corresponding
      start or end coordinate.

      :param output_bin: Index of the bin in the model's output
      :param return_pos: "start" or "end"
      :param start_pos: Genomic coordinate of the first base in the input sequence

      :returns: Genomic coordinate corresponding to the start (if return_pos = start)
                or end (if return_pos=end) of the bin.



   .. py:method:: input_intervals_to_output_intervals(intervals: pandas.DataFrame) -> pandas.DataFrame

      Given a dataframe containing intervals corresponding to the
      input sequences, return a dataframe containing intervals corresponding
      to the model output.

      :param intervals: A dataframe of genomic intervals

      :returns: A dataframe containing the genomic intervals corresponding
                to the model output from each input interval.



   .. py:method:: input_intervals_to_output_bins(intervals: pandas.DataFrame, start_pos: int = 0) -> None

      Given a dataframe of genomic intervals, add columns indicating
      the indices of output bins that overlap the start and end of each interval.

      :param intervals: A dataframe of genomic intervals
      :param start_pos: The start position of the sequence input to the model.

      Returns:start and end indices of the output bins corresponding
          to each input interval.



.. py:data:: DEFAULT_WANDB_ENTITY
   :value: 'grelu'


.. py:data:: DEFAULT_WANDB_HOST
   :value: 'https://api.wandb.ai'


.. py:function:: get_meme_file_path(meme_motif_db)

   Return the path to a MEME file.

   :param meme_motif_db: Path to a MEME file or the name of a MEME file included with gReLU.
                         Current name options are "jaspar" and "consensus".
   :type meme_motif_db: str

   :returns: Path to the specified MEME file.
   :rtype: (str)


.. py:function:: get_default_config_file()

.. py:function:: get_blacklist_file(genome)

.. py:function:: _check_wandb(host=DEFAULT_WANDB_HOST)

.. py:function:: projects(host=DEFAULT_WANDB_HOST)

.. py:function:: artifacts(project, host=DEFAULT_WANDB_HOST, type_is=None, type_contains=None)

.. py:function:: models(project, host=DEFAULT_WANDB_HOST)

.. py:function:: datasets(project, host=DEFAULT_WANDB_HOST)

.. py:function:: runs(project, host=DEFAULT_WANDB_HOST, field='id', filters=None)

.. py:function:: get_artifact(name, project, alias='latest')

.. py:function:: get_dataset_by_model(model_name, project, alias='latest')

.. py:function:: get_model_by_dataset(dataset_name, project, alias='latest')

.. py:function:: load_model(project, model_name, alias='latest', checkpoint_file='model.ckpt')

