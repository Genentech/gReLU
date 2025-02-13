import numpy as np
import pyBigWig
import tiledb
import torch
from tqdm import tqdm

from grelu.sequence.format import intervals_to_strings, strings_to_indices


def _write_chrom_sequence(args):
    """Simple function to extract sequences and write to tiledb."""
    genome, chrom_data, uri = args
    data = intervals_to_strings(intervals=chrom_data, genome=genome)
    data = strings_to_indices(data)
    with tiledb.open(uri, "w") as tiledb_fp:
        tiledb_fp[chrom_data.start_idx : chrom_data.end_idx] = data


def _write_chrom_cov(args):
    """Simple function to extract coverage from a single bigWig file and write to tiledb."""
    chrom_data, bw_file, task_idx, uri = args
    with pyBigWig.open(bw_file, "r") as bw:
        data = bw.values(chrom_data.chrom, chrom_data.start, chrom_data.end, numpy=True)

    data = np.nan_to_num(data).astype(np.float32)
    with tiledb.open(uri, "w") as tiledb_fp:
        tiledb_fp[task_idx, chrom_data.start_idx : chrom_data.end_idx] = data


def worker_init_fn(worker_id):
    torch.utils.data.get_worker_info().dataset.open_tiledb()
