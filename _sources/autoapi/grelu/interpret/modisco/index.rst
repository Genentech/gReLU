grelu.interpret.modisco
=======================

.. py:module:: grelu.interpret.modisco

.. autoapi-nested-parse::

   `grelu.interpret.modisco` contains functions that enable the user to run TF-MoDISco
   (Shrikumar et al. 2018) on trained models. Many of the functions here are based on
   https://github.com/jmschrei/tfmodisco-lite.



Functions
---------

.. autoapisummary::

   grelu.interpret.modisco._ism_attrs
   grelu.interpret.modisco._add_tomtom_to_modisco_report
   grelu.interpret.modisco._tomtom_on_modisco
   grelu.interpret.modisco.run_modisco


Module Contents
---------------

.. py:function:: _ism_attrs(model, seqs: List[str], one_hot: torch.tensor, prediction_transform: Optional[Callable], start: int, end: int, devices: Union[str, int], num_workers: int, batch_size: int, genome: str)

   Perform ISM and format the results for TF-Modisco.


.. py:function:: _add_tomtom_to_modisco_report(modisco_dir: str, tomtom_results: pandas.DataFrame, meme_file: str, top_n_matches: int) -> None

   Modified from https://github.com/jmschrei/tfmodisco-lite/blob/3c6e38f/modiscolite/report.py#L245


.. py:function:: _tomtom_on_modisco(out_dir: str, h5_file: str, meme_file: str, top_n_matches: int = 10, trim_threshold: float = 0.3)

   Run tomtom on motifs in a modisco report


.. py:function:: run_modisco(model, seqs: Union[pandas.DataFrame, numpy.array, List[str]], genome: Optional[str] = None, prediction_transform: Optional[Callable] = None, window: int = None, meme_file: str = None, out_dir: str = 'outputs', devices: Union[str, int] = 'cpu', num_workers: int = 1, batch_size: int = 64, n_shuffles: int = 10, seed=None, method: str = 'deepshap', correct_grad: bool = False, **kwargs) -> None

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
   :param method: Either "deepshap", "saliency" or "ism".
   :param correct_grad: If True, gradients will be corrected using the method of Majdandzic et al.
                        (PMID: 37161475). Only used with method='saliency'.
   :param \*\*kwargs: Additional arguments to pass to TF-Modisco.

   :raises NotImplementedError: if the method is neither "deepshap" nor "ism"


