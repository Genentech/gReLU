import os

import anndata
import numpy as np
import pandas as pd
import torch

from grelu.data.dataset import (
    AnnDataSeqDataset,
    BigWigSeqDataset,
    DFSeqDataset,
    ISMDataset,
    MotifScanDataset,
    PatternMarginalizeDataset,
    SeqDataset,
    VariantDataset,
    VariantMarginalizeDataset,
)
from grelu.sequence.format import convert_input_type

cwd = os.path.realpath(os.path.dirname(__file__))


# Test loading sequences from dataframe

seq_df = pd.DataFrame(
    {
        "seq": ["AAA", "CCCGT"],
        "label1": [0, 1],
        "label2": [2, 3],
        "label3": ["T1", "T2"],
    }
)
seq_df.label3 = seq_df.label3.astype(str)


def test_dfseqdataset_seqs_no_aug():
    # test mode
    ds = DFSeqDataset(seq_df.iloc[:, :3], rc=False, end="right", augment_mode="serial")
    assert (
        (not ds.rc)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 5)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.labels.shape == (2, 2, 1))
        and (len(ds) == 2)
        and not (ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["AAANN", "CCCGT"]
    assert ys.shape == (2, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[0, 2], [1, 3]])

    # predict mode
    ds = DFSeqDataset(
        seq_df.iloc[:, :3], rc=False, seq_len=3, end="left", augment_mode="serial"
    )
    ds.predict = True
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["AAA", "CGT"]

    # random augmentation
    ds = DFSeqDataset(
        seq_df.iloc[:, :3],
        rc=False,
        seq_len=3,
        end="left",
        augment_mode="random",
        seed=0,
    )
    ds.predict = True
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["AAA", "CGT"]


def test_dfseqdataset_seqs_aug():
    # rc=True, pad_end='both', n_tasks=1

    # test mode
    ds = DFSeqDataset(seq_df.iloc[:, :2], rc=True, end="both", augment_mode="serial")
    assert (
        (ds.rc)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 5)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (np.all(ds.tasks.index == ["label1"]))
        and ds.labels.shape == (2, 1, 1)
        and (len(ds) == 4)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["NAAAN", "NTTTN", "CCCGT", "ACGGG"]
    assert ys.shape == (4, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [0, 0, 1, 1])

    # predict mode
    ds.predict = True
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["NAAAN", "NTTTN", "CCCGT", "ACGGG"]

    # train mode
    ds = DFSeqDataset(
        seq_df.iloc[:, :2], rc=True, end="both", seed=0, augment_mode="random"
    )
    assert len(ds) == 2
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["NAAAN", "ACGGG"]
    assert ys.shape == (2, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [0, 1])


def test_dfseqdataset_seqs_multiclass():
    # test mode
    ds = DFSeqDataset(
        seq_df.iloc[:, [0, 3]], rc=True, end="both", augment_mode="serial"
    )
    assert (
        (ds.rc)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 5)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (np.all(ds.tasks.index == ["T1", "T2"]))
        and ds.labels.shape == (2, 2, 1)
        and (len(ds) == 4)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["NAAAN", "NTTTN", "CCCGT", "ACGGG"]
    assert ys.shape == (4, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[1, 0], [1, 0], [0, 1], [0, 1]])


# Test loading genomic intervals from dataframe


interval_df = pd.DataFrame(
    {
        "chrom": ["chr1"] * 2,
        "start": [1e6, 2e6],
        "end": [1e6 + 2, 2e6 + 2],
        "label1": [0, 1],
        "label2": [1, 1],
        "label3": ["T1", "T2"],
    }
)
interval_df.start = interval_df.start.astype(int)
interval_df.end = interval_df.end.astype(int)
interval_df.label3 = interval_df.label3.astype(str)


def test_dfseqdataset_intervals_no_aug():
    # No augmentation
    ds = DFSeqDataset(
        df=interval_df.iloc[:, :4],
        rc=False,
        max_seq_shift=0,
        genome="hg38",
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["label1"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GT", "TA"]
    assert ys.shape == (2, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [0, 1])


def test_dfseqdataset_intervals_aug():
    # Augmentation: rc

    # Test mode
    ds = DFSeqDataset(interval_df.iloc[:, :4], genome="hg38", rc=True, max_seq_shift=0)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (np.all(ds.tasks.index == ["label1"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 4)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GT", "AC", "TA", "TA"]
    assert ys.shape == (4, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [0, 0, 1, 1])

    # Augmentation: sequence shift, multitask
    ds = DFSeqDataset(
        interval_df.iloc[:, :5],
        rc=False,
        max_seq_shift=1,
        genome="hg38",
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 3)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 6)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GG", "GT", "TG", "CT", "TA", "AA"]
    assert ys.shape == (6, 2, 1)
    assert np.allclose(
        ys.squeeze().numpy(), [[0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1]]
    )

    # Predict mode
    ds.predict = True
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GG", "GT", "TG", "CT", "TA", "AA"]

    # Train mode
    ds = DFSeqDataset(
        interval_df.iloc[:, :5],
        rc=False,
        max_seq_shift=1,
        augment_mode="random",
        seed=0,
        genome="hg38",
    )
    assert len(ds) == 2
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GG", "TA"]
    assert ys.shape == (2, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[0, 1], [1, 1]])

    # Augmentation: rc + seq shift
    ds = DFSeqDataset(interval_df.iloc[:, :4], genome="hg38", rc=True, max_seq_shift=1)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 6)
        and (np.all(ds.tasks.index == ["label1"]))
        and (ds.chroms == ["chr1"])
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == [
        "GG",
        "GT",
        "TG",
        "CC",
        "AC",
        "CA",
        "CT",
        "TA",
        "AA",
        "AG",
        "TA",
        "TT",
    ]
    assert ys.shape == (12, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Test predict mode
    ds.predict = True
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == [
        "GG",
        "GT",
        "TG",
        "CC",
        "AC",
        "CA",
        "CT",
        "TA",
        "AA",
        "AG",
        "TA",
        "TT",
    ]


def test_dfseqdataset_intervals_multiclass():
    # Multiclass, no augmentation
    ds = DFSeqDataset(
        interval_df.iloc[:, [0, 1, 2, 5]],
        rc=False,
        max_seq_shift=0,
        genome="hg38",
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["T1", "T2"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GT", "TA"]
    assert ys.shape == (2, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[1, 0], [0, 1]])


# Test anndata loader
ad = anndata.AnnData(
    X=interval_df.iloc[:, [3, 4]].values.T,
    var=interval_df.iloc[:, [0, 1, 2, 5]],
    obs=pd.DataFrame(
        {"cell_type": [interval_df.columns[3], interval_df.columns[4]]}
    ).set_index("cell_type"),
    dtype=np.float32,
)


# No augmentation
def test_anndata_dataset_no_aug():
    ds = AnnDataSeqDataset(ad, genome="hg38", rc=False, max_seq_shift=0)
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (not ds.predict)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GT", "TA"]
    assert ys.shape == (2, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[0, 1], [1, 1]])

    # No augmentation, predict
    ds = AnnDataSeqDataset(ad, genome="hg38", rc=False, max_seq_shift=0)
    ds.predict = True
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.predict)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GT", "TA"]


def test_anndata_dataset_aug():
    # Augmentation: rc
    ds = AnnDataSeqDataset(ad, genome="hg38", rc=True, max_seq_shift=0)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 0)
        and (not ds.predict)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
        and (not ds.predict)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GT", "AC", "TA", "TA"]
    assert ys.shape == (4, 2, 1)
    assert np.allclose(ys.squeeze().numpy(), [[0, 1], [0, 1], [1, 1], [1, 1]])

    # Augmentation: sequence shift
    ds = AnnDataSeqDataset(ad, genome="hg38", rc=False, max_seq_shift=1)
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 1)
        and (not ds.predict)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 3)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 6)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == ["GG", "GT", "TG", "CT", "TA", "AA"]
    assert ys.shape == (6, 2, 1)
    assert np.allclose(
        ys.squeeze().numpy(), [[0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1]]
    )

    # augmentation: rc + seq shift
    ds = AnnDataSeqDataset(ad, genome="hg38", rc=True, max_seq_shift=1)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 1)
        and (not ds.predict)
        and (ds.n_tasks == 2)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 6)
        and (np.all(ds.tasks.index == ["label1", "label2"]))
        and (ds.chroms == ["chr1"])
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    xs = [convert_input_type(x, "strings") for x in xs]
    ys = torch.stack(ys)
    assert xs == [
        "GG",
        "GT",
        "TG",
        "CC",
        "AC",
        "CA",
        "CT",
        "TA",
        "AA",
        "AG",
        "TA",
        "TT",
    ]
    assert ys.shape == (12, 2, 1)
    assert np.allclose(
        ys.squeeze().numpy(),
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
    )


# Test bigwig loader


bw_intervals = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [1, 2], "end": [7, 8]})
bw_file = os.path.join(cwd, "files", "test.bw")


def test_bigwig_dataset_no_aug():
    # Simple
    ds = BigWigSeqDataset(
        intervals=bw_intervals, bw_files=[bw_file], genome="hg38", label_aggfunc=None
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.max_pair_shift == 0)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 6)
        and (ds.label_len == 6)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["test"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    ys = torch.stack(ys)
    assert ys.shape == (2, 1, 6)
    assert np.allclose(ys.squeeze().numpy(), [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])

    # mean label, label_len different from seq_len
    ds = BigWigSeqDataset(
        intervals=bw_intervals,
        bw_files=[bw_file],
        label_aggfunc="mean",
        seq_len=6,
        label_len=2,
        genome="hg38",
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.max_pair_shift == 0)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 6)
        and (ds.label_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["test"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
        and (ds.bin_size == 2)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    ys = torch.stack(ys)
    assert ys.shape == (2, 1, 1)
    assert np.allclose(ys.squeeze().numpy(), [3.5, 4.5])

    # sum with bin size = 2, transform
    ds = BigWigSeqDataset(
        intervals=bw_intervals,
        bw_files=[bw_file],
        label_aggfunc="sum",
        seq_len=6,
        label_len=4,
        min_label_clip=6,
        max_label_clip=10,
        bin_size=2,
        genome="hg38",
    )
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.max_pair_shift == 0)
        and (ds.n_tasks == 1)
        and (ds.seq_len == 6)
        and (ds.label_len == 4)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (np.all(ds.tasks.index == ["test"]))
        and (ds.chroms == ["chr1"])
        and (len(ds) == 2)
        and (ds.bin_size == 2)
    )
    xs, ys = list(zip(*[ds[i] for i in range(len(ds))]))
    ys = torch.stack(ys)
    assert ys.shape == (2, 1, 2)
    assert np.allclose(ys.squeeze().numpy(), [[6, 9], [7, 10]])


# Test unlabeled sequence dataset


def test_unlabeled_dataset_no_aug():
    ds = SeqDataset(interval_df.iloc[:, :3], genome="hg38", rc=False, max_seq_shift=0)
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (len(ds) == 2)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GT", "TA"]


def test_unlabeled_dataset_aug():
    # Augmentation: rc
    ds = SeqDataset(interval_df.iloc[:, :3], genome="hg38", rc=True, max_seq_shift=0)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (len(ds) == 4)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GT", "AC", "TA", "TA"]

    # Augmentation: sequence shift
    ds = SeqDataset(interval_df.iloc[:, :3], genome="hg38", rc=False, max_seq_shift=1)
    assert (
        (not ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 3)
        and (len(ds) == 6)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GG", "GT", "TG", "CT", "TA", "AA"]

    # Augmentation: rc + seq_shift
    ds = SeqDataset(interval_df.iloc[:, :3], genome="hg38", rc=True, max_seq_shift=1)
    assert (
        (ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.seq_len == 2)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 6)
        and (len(ds) == 12)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == [
        "GG",
        "GT",
        "TG",
        "CC",
        "AC",
        "CA",
        "CT",
        "TA",
        "AA",
        "AG",
        "TA",
        "TT",
    ]


# Test variant dataset

variant_file = os.path.join(cwd, "files", "test_variants.txt")
variants = pd.read_table(variant_file, usecols=(0, 1, 2)).iloc[:2, :]
variants["ref"] = variants.variation.apply(lambda x: x.split(">")[0])
variants["alt"] = variants.variation.apply(lambda x: x.split(">")[1].split(",")[0])
variants = variants[["chrom", "pos", "ref", "alt"]]


def test_variant_dataset_no_aug():
    # rc=False, max_seq_shift=0
    ds = VariantDataset(variants, genome="hg38", seq_len=4, rc=False, max_seq_shift=0)
    assert (
        (ds.seq_len == 4)
        and (not ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 1)
        and (ds.n_alleles == 2)
        and (len(ds) == 4)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["CGTG", "CATG", "GGCC", "GACC"]


def test_variant_dataset_aug():
    # rc=True, max_seq_shift=0
    ds = VariantDataset(variants, genome="hg38", seq_len=4, rc=True, max_seq_shift=0)
    assert (
        (ds.seq_len == 4)
        and (ds.rc)
        and (ds.max_seq_shift == 0)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 2)
        and (ds.n_alleles == 2)
        and (len(ds) == 8)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["CGTG", "CATG", "CACG", "CATG", "GGCC", "GACC", "GGCC", "GGTC"]

    # rc=False, max_seq_shift=1
    ds = VariantDataset(variants, genome="hg38", seq_len=4, rc=False, max_seq_shift=1)
    assert (
        (ds.seq_len == 4)
        and (not ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 3)
        and (ds.n_alleles == 2)
        and (len(ds) == 12)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == [
        "ACGT",
        "ACAT",
        "CGTG",
        "CATG",
        "GTGA",
        "ATGA",
        "AGGC",
        "AGAC",
        "GGCC",
        "GACC",
        "GCCA",
        "ACCA",
    ]

    # rc=True, max seq shift=1
    ds = VariantDataset(variants, genome="hg38", seq_len=4, rc=True, max_seq_shift=1)
    assert (
        (ds.seq_len == 4)
        and (ds.rc)
        and (ds.max_seq_shift == 1)
        and (ds.n_seqs == 2)
        and (ds.n_augmented == 6)
        and (ds.n_alleles == 2)
        and (len(ds) == 24)
    )
    xs = [ds[i] for i in range(len(ds))]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == [
        "ACGT",
        "ACAT",
        "CGTG",
        "CATG",
        "GTGA",
        "ATGA",
        "ACGT",
        "ATGT",
        "CACG",
        "CATG",
        "TCAC",
        "TCAT",
        "AGGC",
        "AGAC",
        "GGCC",
        "GACC",
        "GCCA",
        "ACCA",
        "GCCT",
        "GTCT",
        "GGCC",
        "GGTC",
        "TGGC",
        "TGGT",
    ]


# Test ISM dataset


def test_ism_dataset():
    # drop_ref=False
    ds = ISMDataset(["CC"], drop_ref=False)
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == ["AC", "CC", "GC", "TC", "CA", "CC", "CG", "CT"]

    # drop_ref=True
    ds = ISMDataset(["CC"], drop_ref=True)
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == ["AC", "GC", "TC", "CA", "CG", "CT"]

    # specify positions
    ds = ISMDataset(["CC"], drop_ref=False, positions=[0])
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == ["AC", "CC", "GC", "TC"]


# Test marginalization dataset


def test_marginalize_dataset_variants():
    # Marginalize variants
    ds = VariantMarginalizeDataset(
        variants=variants, genome="hg38", seq_len=6, n_shuffles=2, seed=0
    )
    assert (
        (ds.n_shuffles == 2)
        and (ds.seq_len == 6)
        and (ds.n_seqs == 2)
        and (ds.ref.shape == (2, 1))
        and (ds.alt.shape == (2, 1))
        and (len(ds) == 8)
        and (ds.n_augmented == 2)
    )
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == [
        "ACGTGA",
        "ACATGA",
        "ACGTGA",
        "ACATGA",
        "AGGCCA",
        "AGACCA",
        "AGGCCA",
        "AGACCA",
    ]


def test_marginalize_dataset_motifs():
    # Marginalize motifs
    ds = PatternMarginalizeDataset(
        seqs=["ACCTACACT"], patterns=["AAA"], n_shuffles=2, seed=0
    )
    assert (
        (ds.n_shuffles == 2)
        and (ds.n_seqs == 1)
        and (ds.alleles.shape == (1, 3))
        and (len(ds) == 4)
        and (ds.n_augmented == 2)
        and (ds.n_alleles == 2)
    )

    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == ["ACACCGACG", "ACAAAAACG", "ACACGACCG", "ACAAAACCG"]


# Test Motif scanning dataset


def test_motifscan_dataset():
    ds = MotifScanDataset(
        seqs=["ACCTACACT"],
        motifs=["AAA"],
        positions=None,
    )
    assert (ds.seqs.shape == (1, 9)) and (len(ds) == 7)
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == [
        "AAATACACT",
        "AAAAACACT",
        "ACAAACACT",
        "ACCAAAACT",
        "ACCTAAACT",
        "ACCTAAAAT",
        "ACCTACAAA",
    ]

    ds = MotifScanDataset(
        seqs=["ACCTACACT"],
        motifs=["AAA"],
        positions=[2, 3, 4],
    )
    assert (ds.seqs.shape == (1, 9)) and (len(ds) == 3)
    xs = [convert_input_type(ds[i], "strings") for i in range(len(ds))]
    assert xs == ["ACAAACACT", "ACCAAAACT", "ACCTAAACT"]
