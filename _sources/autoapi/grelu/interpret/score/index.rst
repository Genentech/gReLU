grelu.interpret.score
=====================

.. py:module:: grelu.interpret.score

.. autoapi-nested-parse::

   Functions related to scoring the importance of individual DNA bases.



Functions
---------

.. autoapisummary::

   grelu.interpret.score.convert_input_type
   grelu.interpret.score.ISM_predict
   grelu.interpret.score.get_attributions
   grelu.interpret.score.run_modisco
   grelu.interpret.score.get_attention_scores


Module Contents
---------------

.. py:function:: convert_input_type(inputs: Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor], output_type: str = 'indices', genome: Optional[str] = None, add_batch_axis: bool = False) -> Union[pandas.DataFrame, str, List[str], numpy.ndarray, torch.Tensor]

   Convert input DNA sequence data into the desired format.

   :param inputs: DNA sequence(s) in one of the following formats: intervals, strings, indices, or one-hot encoded.
   :param output_type: The desired output format.
   :param genome: The name of the genome to use if genomic intervals are provided.
   :param add_batch_axis: If True, a batch axis will be included in the output for single
                          sequences. If False, the output for a single sequence will be a 2-dimensional
                          tensor.

   :returns: The converted DNA sequence(s) in the desired format.

   :raises ValueError: If the conversion is not possible between the input and output formats.


.. py:function:: ISM_predict(seqs: Union[pandas.DataFrame, numpy.ndarray, str], model: Callable, genome: Optional[str] = None, prediction_transform: Optional[Callable] = None, start_pos: int = 0, end_pos: Optional[int] = None, compare_func: Optional[Union[str, Callable]] = None, devices: Union[str, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, return_df: bool = True) -> Union[numpy.array, pandas.DataFrame]

   Predicts the importance scores of each nucleotide position in a given DNA sequence
   using the In Silico Mutagenesis (ISM) method.

   :param seqs: Input DNA sequences as genomic intervals, strings, or integer-encoded form.
   :param genome: Name of the genome to use if a genomic interval is supplied.
   :param model: A pre-trained deep learning model
   :param prediction_transform: A module to transform the model output
   :param start_pos: Index of the position to start applying ISM
   :param end_pos: Index of the position to stop applying ISM
   :param compare_func: A function or name of a function to compare the predictions for mutated
                        and reference sequences. Allowed names are "divide", "subtract" and "log2FC".
                        If not provided, the raw predictions for both mutant and reference sequences will
                        be returned.
   :param devices: Indices of the devices on which to run inference
   :param num_workers: number of workers for inference
   :param batch_size: batch size for model inference
   :param return_df: If True, the ISM results will be returned as a dataframe. Otherwise, they
                     will be returned as a Numpy array.

   :returns: A numpy array of the predicted scores for each nucleotide position (if return_df = False)
             or a pandas dataframe with A, C, G, and T as row labels and the bases at each position
             of the sequence as column labels  (if return_df = True).


.. py:function:: get_attributions(model, seqs: Union[pandas.DataFrame, numpy.array, List[str]], genome: Optional[str] = None, prediction_transform: Optional[Callable] = None, device: Union[str, int] = 'cpu', method: str = 'deepshap', hypothetical: bool = False, n_shuffles: int = 20, seed=None) -> numpy.array

   Get per-nucleotide importance scores for sequences using Captum.

   :param model: A trained deep learning model
   :param seqs: input DNA sequences as genomic intervals, strings, or integer-encoded form.
   :param genome: Name of the genome to use if a genomic interval is supplied.
   :param prediction_transform: A module to transform the model output
   :param devices: Indices of the devices to use for inference
   :param method: One of "deepshap", "saliency", "inputxgradient" or "integratedgradients"
   :param hypothetical: whether to calculate hypothetical importance scores
                        set to True to obtain input for tf-modisco, False otherwise
   :param n_shuffles: Number of times to dinucleotide shuffle sequence
   :param seed: Random seed

   :returns: Per-nucleotide importance scores as numpy array of shape (B, 4, L).


.. py:function:: run_modisco(model, seqs: Union[pandas.DataFrame, numpy.array, List[str]], genome: Optional[str] = None, prediction_transform: Optional[Callable] = None, window: int = None, meme_file: str = None, out_dir: str = 'outputs', devices: Union[str, int] = 'cpu', num_workers: int = 1, batch_size: int = 64, n_shuffles: int = 10, seed=None, method: str = 'deepshap', **kwargs)

   Run TF-Modisco to get relevant motifs for a set of inputs, and optionally score the
   motifs against a reference set of motifs using TOMTOM

   :param model: A trained deep learning model
   :param seqs: Input DNA sequences as genomic intervals, strings, or integer-encoded form.
   :param genome: Name of the genome to use. Only used if genomic intervals are provided.
   :param prediction_transform: A module to transform the model output
   :param window: Sequence length over which to consider attributions
   :param meme_file: Path to a MEME file containing reference motifs for TOMTOM.
   :param out_dir: Output directory
   :param devices: Indices of devices to use for model inference
   :param num_workers: Number of workers to use for model inference
   :param batch_size: Batch size to use for model inference
   :param n_shuffles: Number of times to shuffle the background sequences for deepshap.
   :param seed: Random seed
   :param method: Either "deepshap" or "ism".
   :param \*\*kwargs: Additional arguments to pass to TF-Modisco.

   :raises NotImplementedError: if the method is neither "deepshap" nor "ism"


.. py:function:: get_attention_scores(model, seqs: Union[pandas.DataFrame, str, numpy.ndarray, torch.Tensor], block_idx: Optional[int] = None, genome: Optional[str] = None) -> numpy.ndarray

   Get the attention scores from a model's transformer layers, for a given input sequence.

   :param model: A trained deep learning model
   :param seq: Input sequences as genoic intervals, strings or in index or one-hot encoded format.
   :param block_idx: Index of the transformer layer to use, ranging from 0 to n_transformers-1.
                     If None, attention scores from all transformer layers will be returned.
   :param genome: Name of the genome to use if genomic intervals are supplied.

   :returns: Numpy array of shape (Layers, Heads, L, L) if block_idx is None or (Heads, L, L) otherwise.


