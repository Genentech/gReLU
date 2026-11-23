grelu.interpret.simulate
========================

.. py:module:: grelu.interpret.simulate


Functions
---------

.. autoapisummary::

   grelu.interpret.simulate.marginalize_patterns
   grelu.interpret.simulate.marginalize_pattern_spacing
   grelu.interpret.simulate.shuffle_tiles


Module Contents
---------------

.. py:function:: marginalize_patterns(model: Callable, patterns: Union[str, List[str]], seqs: Union[pandas.DataFrame, List[str], numpy.ndarray], genome: Optional[str] = None, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, n_shuffles: int = 0, seed: Optional[int] = None, prediction_transform: Optional[Callable] = None, rc: bool = False, compare_func: Optional[Union[str, Callable]] = None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]

   Runs a marginalization experiment.

       Given a model, a pattern (short sequence) to insert, and a set of background
       sequences, get the predictions from the model before and after
       inserting the patterns into the dinucleotide-shuffled background sequences.

   :param model: trained model of class `grelu.lightning.LightningModel`
   :param patterns: a sequence or list of sequences to insert
   :param seqs: background sequences
   :param genome: Name of the genome to use if genomic intervals are supplied
   :param devices: Index of device on which to run inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param seed: Random seed
   :param prediction_transform: A module to transform the model output
   :param rc: If True, augment by reverse complementation
   :param compare_func: Function to compare the predictions with and without the
                        pattern. Options are "divide" or "subtract". If not provided, the
                        predictions before and after pattern insertion will be returned.

   :returns: The predictions from the background sequences
             preds_after: The predictions after inserting the pattern into
                 the background sequences.
   :rtype: preds_before


.. py:function:: marginalize_pattern_spacing(model: Callable, seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], fixed_pattern: str, moving_pattern: str, genome: Optional[str] = None, stride: int = 1, n_shuffles: int = 1, rc: bool = False, seed: int = 0, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, prediction_transform: Optional[Callable] = None, compare_func: Optional[Union[str, Callable]] = None) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]

   Runs a marginalization experiment to predict the impact of the spacing between
   two patterns (sub-sequences).
   Given a model and a set of background sequences, dinucleotide-shuffles the sequences,
   inserts the fixed pattern into the center of each shuffled sequence, then gets the
   predictions from the model on inserting the moving pattern at different distances from
   the fixed pattern.
   :param model: trained model of class `grelu.lightning.LightningModel`
   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param fixed_pattern: A subsequence to insert in the center of each background sequence.
   :param moving_pattern: A subsequence to insert into the background sequences at
                          different distances from `fixed_motif`.
   :param stride: Number of bases by which to shift the moving pattern.
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param n_shuffles: Number of times to shuffle each sequence in `seqs`, to
                      generate a background distribution.
   :param rc: If True, augment by reverse complementation
   :param seed: Seed for random number generator
   :param devices: Index of device on which to run inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param prediction_transform: A module to transform the model output
   :param compare_func: Function to compare the predictions with and without the moving
                        pattern. Options are "divide" or "subtract". If not provided, the predictions
                        without the moving pattern will be returned separately.

   :returns: The predictions from the background sequences
             preds_after: The predictions after inserting the pattern into
                 the background sequences.
             distances: A list containing the distance of the moving pattern from the fixed
                 pattern. Distances are the number of bases between the end of one motif and the
                 start of the other. Negative values indicate that the moving pattern is to the
                 left of the fixed pattern.
   :rtype: preds_before


.. py:function:: shuffle_tiles(model: Callable, seqs: Union[str, Sequence, pandas.DataFrame, numpy.ndarray], tile_len: int, stride: Optional[int] = None, protect_center: Optional[int] = None, n_shuffles: int = 1, seed: int = 0, genome: Optional[str] = None, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, prediction_transform: Optional[Callable] = None, compare_func: Optional[Union[str, Callable]] = None) -> Union[pandas.DataFrame, Tuple[numpy.ndarray, pandas.DataFrame]]

   Dataset class to perform regulatory element discovery by shuffling tiles along
   the input sequences.
   :param model: trained model of class `grelu.lightning.LightningModel`
   :param seqs: DNA sequences as intervals, strings, integer encoded or one-hot encoded.
   :param tile_len: Length of tile to shuffle.
   :param stride: Distance between the start positions of successive tiles.
   :param protect_center: Length of central region to protect
   :param n_shuffles: Number of times to shuffle each tile.
   :param seed: Seed for random number generator
   :param genome: The name of the genome from which to read sequences. This
                  is only needed if genomic intervals are supplied in `seqs`.
   :param deviced: Index of device on which to run inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param prediction_transform: A module to transform the model output
   :param compare_func: Function to compare the predictions after and before shuffling each
                        tile. Options are "divide" or "subtract". If not provided, the predictions
                        before and after shuffling will be returned separately.

   :returns: Model predictions on the original sequences.
             after_preds: Model predictions on the sequences with shuffled tiles.
             tiles: Dataframe containing the coordinates of the tiles that were shuffled.
   :rtype: before_preds


