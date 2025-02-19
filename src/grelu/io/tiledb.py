"""
This submodule contains functions related to reading and writing genomic data into TileDB arrays.
"""
import numpy as np
import tiledb
import torch
from tqdm import tqdm
import os
import multiprocessing
from multiprocessing import Pool
from grelu.sequence.format import strings_to_indices, intervals_to_strings
from grelu.io.bigwig import read_bigwig


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

    xdim = tiledb.Dim(
        name=x_dim_name, domain=(0, x_dim_length - 1), dtype=x_dim_dtype, tile=x_dim_tile)
    ydim = tiledb.Dim(
        name=y_dim_name, domain=(0, y_dim_length - 1), dtype=y_dim_dtype, tile=y_dim_tile)
    if z_dim_length is None:
        dom = tiledb.Domain(xdim, ydim)
    else:
        zdim = tiledb.Dim(
            name=z_dim_name, domain=(0, z_dim_length - 1), dtype=z_dim_dtype, tile=z_dim_tile)
        dom = tiledb.Domain(xdim, ydim, zdim)

    tdb_attr = tiledb.Attr(name='data', dtype=matrix_dim_dtype, 
                           filters=tiledb.FilterList([tiledb.GzipFilter()]))
    schema = tiledb.ArraySchema(
        domain=dom, sparse=False, attrs=[tdb_attr], cell_order="row-major", tile_order="row-major")

    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.Array.create(tiledb_uri_path, schema)
    tdbfile = tiledb.open(tiledb_uri_path, "w")
    tdbfile.close()


def _write_params(str_params, int_params, uri):

    attributes = [tiledb.Attr(dtype=np.int32)] * len(int_params.keys()) + [tiledb.Attr(dtype="U256")] * len(str_params.keys())

    # Create the attribute list, with dtype specified for each attribute
    attributes = [tiledb.Attr(name=k, dtype=np.int32) for k in int_params.keys()] + [tiledb.Attr(name=k, dtype="U256") for k in str_params.keys()]

    domain = tiledb.Domain(
        tiledb.Dim(name="unit_id", domain=(0, 0), tile=1, dtype=np.int64)
    )
    schema = tiledb.ArraySchema(domain=domain, attrs=attributes, sparse=True)
    tiledb.SparseArray.create(uri, schema)

    arr_dict = dict()
    for k, v in int_params.items():
        arr_dict[k] = np.array([v], dtype=np.int32)
    for k, v in str_params.items():
        arr_dict[k] = np.array([v], dtype="U256")

    with tiledb.open(uri, mode="w") as arr:
        arr[np.array([0], dtype=np.int64)] = arr_dict


def _chunk_intervals(intervals, chunk_size):
    chunk_start_idxs = range(0, len(intervals), chunk_size)
    chunks = [intervals[i:i+chunk_size] for i in chunk_start_idxs]
    return zip(chunk_start_idxs, chunks)


def _write_seqs(intervals, chunk_size, genome, uri, num_threads):
    options = [(chunk, idx, genome, uri) for idx, chunk in _chunk_intervals(intervals, chunk_size)]
    if num_threads > 1:
        with Pool(num_threads) as p:
            p.map(_write_chunk_seqs, options)
    else:
        for opt in tqdm(options):
            _write_chunk_seqs(opt)

    cfg = tiledb.Config()
    cfg["sm.consolidation.step_min_frags"] = 1
    cfg["sm.consolidation.step_max_frags"] = 200
    tiledb.consolidate(uri, config=cfg)
    tiledb.vacuum(uri)


def _write_cov(intervals, chunk_size, tasks, uri, bin_size, aggfunc, num_threads):
    for idx, chunk in tqdm(_chunk_intervals(intervals, chunk_size)):
        options = [(chunk, idx, row.bigwig_path, row.task_idx, uri, bin_size, aggfunc) for row in tasks.itertuples()]
        if num_threads > 1:
            with Pool(num_threads) as p:
                p.map(_write_chunk_cov, options)
        else:
            for opt in tqdm(options):
                _write_chunk_cov(opt)

    cfg = tiledb.Config()
    cfg["sm.consolidation.step_min_frags"] = 1
    cfg["sm.consolidation.step_max_frags"] = 200
    tiledb.consolidate(uri, config=cfg)
    tiledb.vacuum(uri)


def _write_chunk_seqs(args):
    chunk, chunk_start_idx, genome, uri = args
    data = intervals_to_strings(chunk, genome=genome)
    data = strings_to_indices(data) # B, 4, L
    with tiledb.open(uri, 'w') as tiledb_fp:
        tiledb_fp[chunk_start_idx:chunk_start_idx+len(chunk), :] = data


def _write_chunk_cov(args):
    chunk, chunk_start_idx, bw_file, task_idx, uri, bin_size, aggfunc = args
    data = read_bigwig(chunk, bw_file, bin_size, aggfunc).squeeze(1)
    with tiledb.open(uri, 'w') as tiledb_fp:
        tiledb_fp[chunk_start_idx:chunk_start_idx+len(chunk), task_idx, :] = data


def bigwigs_to_tiledb(tdb_path, intervals, seq_len, label_len, max_seq_shift, max_pair_shift, bin_size, aggfunc, genome, tasks, num_threads, chunk_size):
    from grelu.sequence.utils import resize
    if not os.path.exists(tdb_path):
        os.mkdir(tdb_path)

    task_uri = f"{tdb_path}/tasks"
    intervals_uri = f"{tdb_path}/intervals"
    seq_uri = f"{tdb_path}/sequences"
    label_uri = f"{tdb_path}/labels"
    params_uri = f"{tdb_path}/params"

    assert max_pair_shift % bin_size == 0

    int_params = {
        'n_seqs': len(intervals),
        'seq_len': seq_len,
        'n_tasks': len(tasks),
        'label_len':label_len, 
        'label_bins': label_len//bin_size,
        'max_seq_shift':max_seq_shift, 
        'max_pair_shift':max_pair_shift,
        'bin_size':bin_size,
        'padded_seq_len': seq_len + (2 * max_seq_shift) + (2 * max_pair_shift),
        'padded_label_len': label_len + (2 * max_pair_shift),
        'padded_label_bins': (label_len + (2 * max_pair_shift))//bin_size,
    }

    str_params = {
        'aggfunc':aggfunc, 
        'genome':genome,
    }
    _write_params(str_params, int_params, params_uri)    

    # Write task dataframe
    tiledb.from_pandas(task_uri, tasks)

    # Write intervals dataframe
    intervals = resize(intervals, seq_len=int_params['padded_seq_len'], end='both')
    tiledb.from_pandas(intervals_uri, intervals)

    # Set up multiprocessing
    if num_threads > 1:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # Create empty array for sequence
    create_tiledb_array(
        seq_uri,
        x_dim_length=int_params['n_seqs'], 
        y_dim_length=int_params['padded_seq_len'], 
        x_dim_tile=1, y_dim_tile=int_params['padded_seq_len'], matrix_dim_dtype = np.int8)

    print("Writing DNA sequences")
    _write_seqs(intervals, chunk_size, genome, seq_uri, num_threads)

    # Create empty array for labels
    create_tiledb_array(
        label_uri,
        x_dim_length=int_params['n_seqs'], 
        y_dim_length=int_params['n_tasks'], 
        z_dim_length=int_params['padded_label_bins'], 
        x_dim_tile=1, y_dim_tile=int_params['n_tasks'], 
        z_dim_tile=int_params['padded_label_bins'], matrix_dim_dtype=np.float32)
    
    print("Writing coverage from BigWig files")
    intervals = resize(intervals, int_params['padded_label_len'])
    _write_cov(intervals, chunk_size, tasks, label_uri, bin_size, aggfunc, num_threads)
