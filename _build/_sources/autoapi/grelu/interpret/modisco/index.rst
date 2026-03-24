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

   grelu.interpret.modisco.run_modisco


Module Contents
---------------

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


