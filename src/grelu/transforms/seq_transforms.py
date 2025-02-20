"""
`grelu.transform.seq_transforms` contains classes to assign each sequence a score
based on its content.

All classes must define the `forward` function, which should take as input DNA
sequences as a list of strings, and return a numpy array containing a scalar value
for each sequence.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import regex

from grelu.interpret.motifs import scan_sequences


class PatternScore:
    """
    A class that returns a weighted score based on the number of occurrences of given subsequences.

    Args:
        patterns: List of subsequences
        weights: List of weights for each subsequence. If None, all patterns will receive a weight of 1.
    """

    def __init__(self, patterns: List[str], weights: List[float]) -> None:
        self.patterns = patterns
        self.weights = weights
        if self.weights is not None:
            assert len(self.weights) == len(self.patterns)

    def forward(self, seqs: List[str]) -> List[float]:
        """
        Compute scores.

        Args:
            seqs: A list of input sequences as strings.
        """
        counts = np.array(
            [
                [
                    len(regex.findall(pattern, seq, overlapped=True))
                    for pattern in self.patterns
                ]
                for seq in seqs
            ]
        )  # N, P
        if self.weights is None:
            return np.sum(counts, 1)
        else:
            return np.dot(counts, self.weights)

    def __call__(self, seqs: List[str]) -> List[float]:
        return self.forward(seqs)


class MotifScore:
    """
    A scorer that returns a weighted score based on the number of occurrences of given subsequences.

    Args:
        motifs: Either the path to a MEME file, or a dictionary whose values are numpy arrays of shape (4, L).
        names: List of names of motifs to read from the meme file. If None, all motifs will be read
            from the file.
        weights: List of weights for each motif. If None, all motifs will receive a weight of 1.
        pthresh: p-value cutoff to define binding sites
        rc: Whether to scan the sequence reverse complement as well
    """

    def __init__(
        self,
        motifs: Union[str, Dict[str, np.ndarray]] = None,
        names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        pthresh: float = 1e-3,
        rc: bool = True,
    ) -> None:

        # Load motifs
        if isinstance(motifs, str):
            from grelu.io.motifs import read_meme_file

            motifs = read_meme_file(motifs, names=names)
        self.motifs = motifs

        # Save weights
        if weights is None:
            self.weights = weights
        else:
            motif_names = list(self.motifs.keys())
            assert len(weights) == len(motif_names)
            self.weights = {m: w for m, w in zip(motif_names, weights)}

        # Store other params
        self.pthresh = pthresh
        self.rc = rc

    def forward(self, seqs: List[str]) -> List[float]:
        """
        Compute scores.

        Args:
            seqs: A list of input sequences as strings.
        """
        # Scan sequences
        sites = scan_sequences(
            seqs, motifs=self.motifs, pthresh=self.pthresh, rc=self.rc
        )

        # If no sites are present, return a score of 0 for each sequence
        if len(sites) == 0:
            return [0] * len(seqs)

        else:
            # Count the number of sites per sequence and motif
            n_sites = (
                sites[["sequence", "motif"]].value_counts().reset_index(name="count")
            )

            # Calculate weighted score for each sequence
            if self.weights is None:
                return len(n_sites)
            else:
                n_sites["weight"] = n_sites["motif"].map(self.weights)
                return n_sites[["count", "weight"]].product(1).tolist()

    def __call__(self, seqs: List[str]) -> List[float]:
        return self.forward(seqs)
