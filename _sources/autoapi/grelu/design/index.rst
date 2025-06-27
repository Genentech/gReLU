grelu.design
============

.. py:module:: grelu.design

.. autoapi-nested-parse::

   `grelu.design` contains methods to design novel DNA sequences
   using trained sequence-to-function deep learning models.



Functions
---------

.. autoapisummary::

   grelu.design.evolve
   grelu.design.ledidi


Module Contents
---------------

.. py:function:: evolve(seqs: Union[List[str], pandas.DataFrame], model: grelu.lightning.LightningModel, method: str = 'ism', patterns: Optional[List[str]] = None, prediction_transform: Optional[torch.nn.Module] = None, seq_transform: Optional[torch.nn.Module] = None, max_iter: int = 10, positions: List[int] = None, devices: Union[str, int, List[int]] = 'cpu', num_workers: int = 1, batch_size: int = 64, genome: Optional[str] = None, for_each: bool = True, return_seqs: str = 'all', return_preds: bool = True, verbose: bool = True) -> pandas.DataFrame

   Sequence design by greedy directed evolution

   :param seqs: a set of DNA sequences as strings or genomic intervals
   :param model: LightningModel object containing a trained deep learning model
   :param method: Either "ism" or "pattern".
   :param patterns: A list of subsequences to try inserting into the starting sequence.
   :param prediction_transform: A module to transform the model output
   :param seq_transform: A module to asign scores to sequences
   :param max_iter: Number of iterations
   :param positions: Positions to mutate. If None, all positions will be mutated
   :param devices: Device(s) for inference
   :param num_workers: Number of workers for inference
   :param batch_size: Batch size for inference
   :param genome: genome to use if intervals are provided as starting sequences
   :param for_each: If multiple start sequences are provided, perform directed
                    evolution independently from each one
   :param return_seqs: "all", "best" or "none".
   :param return_preds: If True, return all the individual model predictions in addition to the
                        model prediction score.
   :param verbose: Print status after each iteration

   :returns: A dataframe containing directed evolution results


.. py:function:: ledidi(seq: str, model: Callable, prediction_transform: Optional[torch.nn.Module] = None, max_iter: int = 20000, positions: Optional[List[int]] = None, devices: Union[str, int] = 'cpu', num_workers: int = 1, **kwargs)

   Sequence design with Ledidi

   :param seq: an initial DNA sequence as a string.
   :param model: A trained LightningModel object
   :param prediction_transform: A module to transform the model output
   :param max_iter: Number of iterations
   :param positions: Positions to mutate. If None, all positions will be mutated
   :param targets: List of targets for each loss function
   :param devices: Index of device to use for inference
   :param num_workers: Number of workers for inference
   :param \*\*kwargs: Other arguments to pass on to Ledidi

   :returns: Output DNA sequence(s) as strings.


