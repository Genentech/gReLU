import numpy as np
import tiledb
from grelu.sequence.format import strings_to_indices, intervals_to_strings
import torch
from tqdm import tqdm
import os
import shutil
from multiprocessing import Pool
from genomicarrays import buildutils_tiledb_array as uta
import pyBigWig


def _write_chrom_sequence(args):
    """Simple function to extract sequences and write to tiledb."""
    genome, chrom_data = args
    data = intervals_to_strings(intervals=chrom_data, genome=genome)
    data = strings_to_indices(data)
    with tiledb.open(chrom_data.uri, "w") as tiledb_fp:
        tiledb_fp[0, chrom_data.start: chrom_data.end] = data


def _write_chrom_cov(args):
    """Simple function to extract coverage from a single bigWig file and write to tiledb."""
    chrom_data, bw_file, task_idx = args
    with pyBigWig.open(bw_file, 'r') as bw:
        data = bw.values(chrom_data.chrom, chrom_data.start, chrom_data.end, numpy=True)
        data = np.nan_to_num(data)
    with tiledb.open(chrom_data.uri, "w") as tiledb_fp:
        tiledb_fp[task_idx, chrom_data.start: chrom_data.end] = data.astype(np.float32)
        

def create_tiledb_array(
    tiledb_uri_path: str,
    x_dim_length: int = None,
    y_dim_length: int = None,
    z_dim_length: int = None,
    x_dim_name: str = "x_idx",
    y_dim_name: str = "y_idx",
    z_dim_name: int = "z_idx",
    x_dim_dtype: np.dtype = np.uint32,
    y_dim_dtype: np.dtype = np.uint32,
    z_dim_dtype: np.dtype = np.uint32,
    x_dim_tile: int = None,
    y_dim_tile = None,
    z_dim_tile = None,
    matrix_dim_dtype: np.dtype = np.uint32,
):
    
    xdim = tiledb.Dim(name=x_dim_name, domain=(0, x_dim_length - 1), dtype=x_dim_dtype, tile=x_dim_tile)
    ydim = tiledb.Dim(name=y_dim_name, domain=(0, y_dim_length - 1), dtype=y_dim_dtype, tile=y_dim_tile)
    if z_dim_length is None:
        dom = tiledb.Domain(xdim, ydim)
    else:
        zdim = tiledb.Dim(name=z_dim_name, domain=(0, z_dim_length - 1), dtype=z_dim_dtype, tile=z_dim_tile)
        dom = tiledb.Domain(xdim, ydim, zdim)

    tdb_attr = tiledb.Attr(name='data', dtype=matrix_dim_dtype, filters=tiledb.FilterList([tiledb.GzipFilter()]))

    schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=[tdb_attr], cell_order="row-major", tile_order="row-major")

    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.Array.create(tiledb_uri_path, schema)
    tdbfile = tiledb.open(tiledb_uri_path, "w")
    tdbfile.close()


def worker_init_fn(worker_id):
    torch.utils.data.get_worker_info().dataset.open_tiledb()