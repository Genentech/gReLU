grelu.lightning
===============

.. py:module:: grelu.lightning

.. autoapi-nested-parse::

   `grelu.lightning` contains LightningModel class, which inherits from
   `pytorch_lightning.LightningModule`. This class wraps sequence-to-function
   models and allows them to be used with downstream functions provided by
   pytorch lightning, including training, inference, validation, testing and
   fine-tuning.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/grelu/lightning/losses/index
   /autoapi/grelu/lightning/metrics/index


Attributes
----------

.. autoapisummary::

   grelu.lightning.default_train_params


Classes
-------

.. autoapisummary::

   grelu.lightning.ISMDataset
   grelu.lightning.LabeledSeqDataset
   grelu.lightning.MotifScanDataset
   grelu.lightning.PatternMarginalizeDataset
   grelu.lightning.SeqDataset
   grelu.lightning.SpacingMarginalizeDataset
   grelu.lightning.TilingShuffleDataset
   grelu.lightning.VariantDataset
   grelu.lightning.VariantMarginalizeDataset
   grelu.lightning.PoissonMultinomialLoss
   grelu.lightning.MSE
   grelu.lightning.BestF1
   grelu.lightning.PearsonCorrCoef
   grelu.lightning.ConvHead
   grelu.lightning.LightningModel
   grelu.lightning.LightningModelEnsemble


Functions
---------

.. autoapisummary::

   grelu.lightning.strings_to_one_hot
   grelu.lightning.get_aggfunc
   grelu.lightning.make_list


Package Contents
----------------

.. py:class:: ISMDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], genome: Optional[str] = None, drop_ref: bool = False, positions: Optional[List[int]] = None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to perform In silico mutagenesis (ISM)

   :param seqs: DNA sequences as intervals, strings, indices or one-hot.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param drop_ref: If True, the base that already exists at each position
                    will not be included in the returned sequences.
   :param positions: List of positions to mutate. If None, all positions
                     will be mutated.


   .. py:attribute:: positions
      :value: None



   .. py:attribute:: genome
      :value: None



   .. py:attribute:: drop_ref
      :value: False



   .. py:attribute:: n_alleles
      :value: 4



   .. py:attribute:: n_seqs


   .. py:attribute:: seq_len


   .. py:attribute:: n_positions


   .. py:method:: _load_seqs(seqs) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int, return_compressed=False) -> torch.Tensor


.. py:class:: LabeledSeqDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], labels: numpy.ndarray, tasks: Optional[Union[Sequence, pandas.DataFrame]] = None, seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, label_len: Optional[int] = None, max_pair_shift: int = 0, label_aggfunc: Optional[Union[str, Callable]] = None, bin_size: Optional[int] = None, min_label_clip: Optional[int] = None, max_label_clip: Optional[int] = None, label_transform_func: Optional[Union[str, Callable]] = None, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   A general Dataset class for DNA sequences and labels. All sequences and
   labels will be stored in memory.

   :param seqs: DNA sequences as intervals, strings, indices or one-hot.
   :param labels: A numpy array of shape (B, T, L) containing the labels.
   :param tasks: A list of task names or a pandas dataframe containing task information.
                 If a dataframe is supplied, the row indices should be the task names.
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If False,
              they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation. This
                         is normally a small value (< 10). If 0, sequences will not be augmented by shifting.
   :param label_len: Uniform expected length (in base pairs) for output labels
   :param max_pair_shift: Maximum number of bases to shift both the sequence and label for
                          augmentation. If 0, sequence and label pairs will not be augmented by shifting.
   :param label_aggfunc: Function to aggregate the labels over bin_size.
   :param bin_size: Number of bases to aggregate in the label. Only used if label_aggfunc is not None.
                    If None, it will be taken as equal to label_len.
   :param min_label_clip: Minimum value for label
   :param max_label_clip: Maximum value for label
   :param label_transform_func: Function to transform label values.
   :param seed: Random seed for reproducibility
   :param augment_mode: "random" or "serial"


   .. py:attribute:: end
      :value: 'both'



   .. py:attribute:: genome
      :value: None



   .. py:attribute:: min_label_clip
      :value: None



   .. py:attribute:: max_label_clip
      :value: None



   .. py:attribute:: label_transform_func
      :value: None



   .. py:attribute:: seq_len


   .. py:attribute:: label_len


   .. py:attribute:: label_aggfunc
      :value: None



   .. py:attribute:: bin_size
      :value: None



   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: max_pair_shift
      :value: 0



   .. py:attribute:: padded_seq_len


   .. py:attribute:: padded_label_len


   .. py:attribute:: n_seqs


   .. py:attribute:: n_tasks


   .. py:attribute:: label_transform


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: n_alleles
      :value: 1



   .. py:attribute:: predict
      :value: False



   .. py:method:: _load_seqs(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray]) -> None


   .. py:method:: _load_tasks(tasks: Union[pandas.DataFrame, List]) -> None


   .. py:method:: _load_labels(labels: numpy.ndarray) -> None


   .. py:method:: __len__() -> int


   .. py:method:: get_labels() -> numpy.ndarray

      Return the labels as a numpy array of shape (B, T, L). This does not
      account for data augmentation.



   .. py:method:: __getitem__(idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


.. py:class:: MotifScanDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], motifs: List[str], genome: Optional[str] = None, positions: Optional[List[int]] = None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to perform in silico motif scanning by inserting a motif
   at each position of a sequence.

   :param seqs: Background DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param motifs: A list of subsequences to insert into the background sequences.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param positions: List of positions at which to insert the motif. If None, all positions
                     will be mutated.


   .. py:attribute:: positions
      :value: None



   .. py:attribute:: genome
      :value: None



   .. py:attribute:: motifs


   .. py:attribute:: max_motif_len


   .. py:attribute:: n_alleles


   .. py:attribute:: n_seqs


   .. py:attribute:: seq_len


   .. py:attribute:: n_positions


   .. py:method:: _load_seqs(seqs)


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int, return_compressed=False) -> torch.Tensor


.. py:class:: PatternMarginalizeDataset(seqs: Union[List[str], pandas.DataFrame, numpy.ndarray], patterns: List[str], n_shuffles: int = 1, genome: Optional[str] = None, seed: Optional[int] = None, rc: bool = False, max_seq_shift: int = 0)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to marginalize the effect of given sequence patterns
   across shuffled background sequences. All sequences are stored in memory.

   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param patterns: List of alleles or motif sequences to insert into the background sequences.
   :param n_shuffles: Number of times to shuffle each background sequence to
                      generate a background distribution.
   :param genome: The name of the genome from which to read sequences. Only used if genomic
                  intervals are supplied.
   :param seed: Seed for random number generator
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value. If 0, sequences will not be augmented by shifting.


   .. py:attribute:: genome
      :value: None



   .. py:attribute:: seed
      :value: None



   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: n_shuffles
      :value: 1



   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: bg
      :value: None



   .. py:attribute:: curr_seq_idx
      :value: None



   .. py:method:: _load_alleles(patterns: List[str]) -> None


   .. py:method:: _load_seqs(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray]) -> None

      Make the background sequences



   .. py:method:: __update__(idx: int) -> None

      Update the current background



   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: SeqDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], seq_len: Optional[int] = None, genome: Optional[str] = None, end: str = 'both', rc: bool = False, max_seq_shift: int = 0, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to cycle through unlabeled sequences for inference. All sequences
   are stored in memory.

   :param seqs: DNA sequences
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences. Only needed if
                  genomic intervals are supplied.
   :param end: Which end of the sequence to resize if necessary. Supported values are "left",
               "right" and "both".
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not be
                         augmented by shifting.


   .. py:attribute:: end
      :value: 'both'



   .. py:attribute:: genome
      :value: None



   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: seq_len


   .. py:attribute:: padded_seq_len


   .. py:attribute:: n_seqs


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: n_alleles
      :value: 1



   .. py:method:: _load_seqs(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray]) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: SpacingMarginalizeDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], fixed_pattern: str, moving_pattern: str, stride: int = 1, genome: Optional[str] = None, n_shuffles: int = 1, rc: bool = False, seed: int = 0)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset class to perform pairwise motif distance analysis. One motif
   is inserted at a fixed position in shuffled sequences and the second motif is
   inserted at variable distances from the first.

   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param fixed_pattern: A subsequence to insert in the center of each background sequence.
   :param moving_pattern: A subsequence to insert into the background sequences at
                          different distances from `fixed_motif`.
   :param stride: Number of bases by which to shift the moving pattern.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param n_shuffles: Number of times to shuffle each sequence in `seqs`, to
                      generate a background distribution.
   :param seed: Seed for random number generator


   .. py:attribute:: stride
      :value: 1



   .. py:attribute:: genome
      :value: None



   .. py:attribute:: n_shuffles
      :value: 1



   .. py:attribute:: seed
      :value: 0



   .. py:attribute:: rc
      :value: False



   .. py:attribute:: n_alleles


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: bg
      :value: None



   .. py:attribute:: curr_seq_idx
      :value: None



   .. py:method:: _load_seqs(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray]) -> None

      Make the background sequences



   .. py:method:: _load_patterns(fixed_pattern: str, moving_pattern: str) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __update__(idx: int) -> None

      Update the current background



   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: TilingShuffleDataset(seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], tile_len: int, stride: Optional[int] = None, protect_center: Optional[int] = None, genome: Optional[str] = None, n_shuffles: int = 1, seed: int = 0)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset class to perform regulatory element discovery by shuffling tiles along
   the input sequences.

   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param tile_len: Length of tile to shuffle.
   :param stride: Distance between the start positions of successive tiles. If None,
                  tiles will be non-overlapping
   :param protect_center: Length of central region to protect
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param n_shuffles: Number of times to shuffle each tile.
   :param seed: Seed for random number generator


   .. py:attribute:: genome
      :value: None



   .. py:attribute:: tile_len


   .. py:attribute:: protect_center
      :value: None



   .. py:attribute:: n_shuffles
      :value: 1



   .. py:attribute:: seed
      :value: 0



   .. py:attribute:: stride


   .. py:attribute:: positions


   .. py:attribute:: n_positions


   .. py:method:: _load_seqs(seqs: Union[pandas.DataFrame, List[str], numpy.ndarray]) -> None

      Make the background sequences



   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: VariantDataset(variants: pandas.DataFrame, seq_len: int, genome: Optional[str] = None, rc: bool = False, max_seq_shift: int = 0, seed: Optional[int] = None, augment_mode: str = 'serial')

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset class to perform inference on sequence variants.

   :param variants: pd.DataFrame with columns "chrom", "pos", "ref", "alt".
   :param seq_len: Uniform expected length (in base pairs) for output sequences
   :param genome: The name of the genome from which to read sequences.
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not
                         be augmented by shifting.


   .. py:attribute:: genome
      :value: None



   .. py:attribute:: seq_len


   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: n_alleles
      :value: 2



   .. py:attribute:: n_seqs


   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:method:: _load_alleles(variants: pandas.DataFrame) -> None


   .. py:method:: _load_seqs(variants: pandas.DataFrame) -> None


   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: VariantMarginalizeDataset(variants: pandas.DataFrame, genome: str, seq_len: int, seed: Optional[int] = None, rc: bool = False, max_seq_shift: int = 0, n_shuffles: int = 100)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Dataset to marginalize the effect of given variants
   across shuffled background sequences. All sequences are stored
   in memory.

   :param variants: A dataframe of sequence variants
   :param genome: The name of the genome from which to read sequences. Only used if genomic
                  intervals are supplied.
   :param seed: Seed for random number generator
   :param rc: If True, sequences will be augmented by reverse complementation. If
              False, they will not be reverse complemented.
   :param max_seq_shift: Maximum number of bases to shift the sequence for augmentation.
                         This is normally a small value (< 10). If 0, sequences will not
                         be augmented by shifting.
   :param n_shuffles: Number of times to shuffle each background sequence to
                      generate a background distribution.


   .. py:attribute:: genome


   .. py:attribute:: seed
      :value: None



   .. py:attribute:: seq_len


   .. py:attribute:: rc
      :value: False



   .. py:attribute:: max_seq_shift
      :value: 0



   .. py:attribute:: n_shuffles
      :value: 100



   .. py:attribute:: augmenter


   .. py:attribute:: n_augmented


   .. py:attribute:: bg
      :value: None



   .. py:attribute:: curr_seq_idx
      :value: None



   .. py:method:: _load_alleles(variants: pandas.DataFrame) -> None

      Load the alleles to substitute into the background



   .. py:method:: _load_seqs(variants: pandas.DataFrame) -> None

      Load sequences surrounding the variant position



   .. py:method:: __update__(idx: int) -> None

      Update the current background



   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> torch.Tensor


.. py:class:: PoissonMultinomialLoss(total_weight: float = 1, eps: float = 1e-07, log_input: bool = True, reduction: str = 'mean', multinomial_axis: str = 'length')

   Bases: :py:obj:`torch.nn.Module`


   Possion decomposition with multinomial specificity term.

   :param total_weight: Weight of the Poisson total term.
   :param eps: Added small value to avoid log(0). Only needed if log_input = False.
   :param log_input: If True, the input is transformed with torch.exp to produce predicted
                     counts. Otherwise, the input is assumed to already represent predicted
                     counts.
   :param multinomial_axis: Either "length" or "task", representing the axis along which the
                            multinomial distribution should be calculated.
   :param reduction: "mean" or "none".


   .. py:attribute:: eps
      :value: 1e-07



   .. py:attribute:: total_weight
      :value: 1



   .. py:attribute:: log_input
      :value: True



   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor

      Loss computation

      :param input: Tensor of shape (B, T, L)
      :param target: Tensor of shape (B, T, L)

      :returns: Loss value



.. py:class:: MSE(num_outputs: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the MSE for each task.

   :param num_outputs: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Predictions of shape (N, n_tasks, L)
       target: Ground truth labels (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the MSE


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


.. py:class:: BestF1(num_labels: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the best F1 score for each task.

   :param num_labels: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Probabilities of shape (N, n_tasks, L)
       target: Ground truth labels of shape (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the best F1 score


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


.. py:class:: PearsonCorrCoef(num_outputs: int = 1, average: bool = True)

   Bases: :py:obj:`torchmetrics.Metric`


   Metric class to calculate the Pearson correlation coefficient for each task.

   :param num_outputs: Number of tasks
   :param average: If true, return the average metric across tasks.
                   Otherwise, return a separate value for each task

   As input to forward and update the metric accepts the following input:
       preds: Predictions of shape (N, n_tasks, L)
       target: Ground truth labels of shape (N, n_tasks, L)

   As output of forward and compute the metric returns the following output:
       output: A tensor with the Pearson coefficient.


   .. py:attribute:: pearson


   .. py:attribute:: average
      :value: True



   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None


   .. py:method:: compute() -> torch.Tensor


   .. py:method:: reset() -> None


.. py:class:: ConvHead(n_tasks: int, in_channels: int, act_func: Optional[str] = None, pool_func: Optional[str] = None, norm: bool = False, norm_kwargs: Optional[dict] = None, dtype=None, device=None)

   Bases: :py:obj:`torch.nn.Module`


   A 1x1 Conv layer that transforms the the number of channels in the input and then
   optionally pools along the length axis.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the convolutional layer
   :param pool_func: Pooling function.
   :param norm: If True, batch normalization will be included.
   :param norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
   :param dtype: Data type for the layers.
   :param device: Device for the layers.


   .. py:attribute:: n_tasks


   .. py:attribute:: in_channels


   .. py:attribute:: act_func
      :value: None



   .. py:attribute:: pool_func
      :value: None



   .. py:attribute:: norm
      :value: False



   .. py:attribute:: channel_transform


   .. py:attribute:: pool


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:function:: strings_to_one_hot(strings: Union[str, List[str]], add_batch_axis: bool = False) -> torch.Tensor

   Convert a list of DNA sequences to one-hot encoded format.

   :param seqs: A DNA sequence or a list of DNA sequences.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.

   :returns: The one-hot encoded DNA sequence(s).

   :raises AssertionError: If the input sequences are not of the same length,
   :raises or if the input is not a string or a list of strings.:


.. py:function:: get_aggfunc(func: Optional[Union[str, Callable]], tensor: bool = False) -> Callable

   Return a function to aggregate values.

   :param func: A function or the name of a function. Supported names
                are "max", "min", "mean", and "sum". If a function is supplied, it
                will be returned unchanged.
   :param tensor: If True, it is assumed that the inputs will be torch tensors.
                  If False, it is assumed that the inputs will be numpy arrays.

   :returns: The desired function.

   :raises NotImplementedError: If the input is neither a function nor
       a supported function name.


.. py:function:: make_list(x: Optional[Union[pandas.Series, numpy.ndarray, torch.Tensor, Sequence, int, float, str]]) -> list

   Convert various kinds of inputs into a list

   :param x: An input value or sequence of values.

   :returns: The input values in list format.


.. py:data:: default_train_params

.. py:class:: LightningModel(model_params: dict, train_params: dict = {})

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   Wrapper for predictive sequence models

   :param model_params: Dictionary of parameters specifying model architecture
   :param train_params: Dictionary specifying training parameters


   .. py:attribute:: model_params


   .. py:attribute:: train_params


   .. py:attribute:: data_params


   .. py:attribute:: performance


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


   .. py:method:: on_load_checkpoint(checkpoint: dict) -> None


   .. py:method:: predict_on_seqs(x: Union[str, List[str]], device: Union[str, int] = 'cpu') -> numpy.ndarray

      A simple function to return model predictions directly
      on a batch of a single batch of sequences in string
      format.

      :param x: DNA sequences as a string or list of strings.
      :param device: Index of the device to use

      :returns: A numpy array of predictions.



   .. py:method:: predict_on_dataset(dataset: Callable, devices: Union[int, str, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 256, augment_aggfunc: Union[str, Callable] = 'mean', return_df: bool = False, precision: Optional[str] = None)

      Predict for a dataset of sequences or variants

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param devices: Device IDs to use
      :param num_workers: Number of workers for data loader
      :param batch_size: Batch size for data loader
      :param augment_aggfunc: Return the average prediction across all augmented
                              versions of a sequence
      :param return_df: Return the predictions as a Pandas dataframe
      :param precision: Precision of the trainer e.g. '32' or 'bf16-mixed'.

      :returns: Model predictions as a numpy array or dataframe



   .. py:method:: test_on_dataset(dataset: Callable, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 256, precision: Optional[str] = None, write_path: Optional[str] = None)

      Run test loop for a dataset

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param devices: Device IDs to use for inference
      :param num_workers: Number of workers for data loader
      :param batch_size: Batch size for data loader
      :param precision: Precision of the trainer e.g. '32' or 'bf16-mixed'.
      :param write_path: Path to write a new model checkpoint containing
                         test data parameters and performance.

      :returns: Dataframe containing all calculated metrics on the test set.



   .. py:method:: embed_on_dataset(dataset: Callable, device: Union[str, int] = 'cpu', num_workers: int = 1, batch_size: int = 256)

      Return embeddings for a dataset of sequences

      :param dataset: Dataset object that yields one-hot encoded sequences
      :param device: Device ID to use
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



.. py:class:: LightningModelEnsemble(models: list, model_names: Optional[List[str]] = None)

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   Combine multiple LightningModel objects into a single object.
   When predict_on_dataset is used, it will return the concatenated
   predictions from all the models in the order in which they were supplied.

   :param models: A list of multiple LightningModel objects
   :type models: list
   :param model_names: A name for each model. This will be prefixed
                       to the names of the individual tasks predicted by the model.
                       If not supplied, the models will be named "model0", "model1", etc.
   :type model_names: list


   .. py:attribute:: models


   .. py:attribute:: model_names


   .. py:attribute:: model_params


   .. py:attribute:: data_params


   .. py:method:: _combine_tasks() -> None

      Combine the task metadata of all the sub-models into self.data_params["tasks"]



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward Pass.



   .. py:method:: predict_on_dataset(dataset: Callable, **kwargs) -> numpy.ndarray

      This will return the concatenated predictions from all the
      constituent models, in the order in which they were supplied.
      Predictions will be concatenated along the task axis.



   .. py:method:: get_task_idxs(tasks: Union[str, int, List[str], List[int]], key: str = 'name') -> Union[int, List[int]]

      Return the task index given the name of the task. Note that task
      names should be supplied with a prefix indicating the model number,
      so for instance if you want the predictions from the second model
      on astrocytes, the task name would be "{name of second model}_astrocytes".
      If model names were not supplied to __init__, the task name would
      be "model1_astrocytes".

      :param tasks: A string corresponding to a task name or metadata entry,
                    or an integer indicating the index of a task, or a list of strings/integers
      :param key: key to model.data_params["tasks"] in which the relevant task data is
                  stored. "name" will be used by default.

      Returns: An integer or list of integers representing the indices of the
          tasks in the model output.



