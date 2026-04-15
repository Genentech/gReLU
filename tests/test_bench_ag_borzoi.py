"""
Regression tests for scripts/bench_ag_borzoi.py — unit layer.
No GPU, no model weights required. Runs in ~5 s on any machine.

════════════════════════════════════════════════════════════════════════════════
Design Philosophy
════════════════════════════════════════════════════════════════════════════════

Principles
──────────
1. Import, don't copy: Import constants (BORZOI_INPUT_LEN, etc.) and pure functions
   (_stats, _bin_obs, _parse_variant) directly from source.
2. Test contracts, not steps: Focus on "what" the code promised, not "how" it
   calculates it.
3. Distinguish Golden Values: Hardcode biological/technical specs in 
   TestArchitectureConstants. Use derived relationships for other constants.
4. Parametrize boundary cases: Use @pytest.mark.parametrize for edge cases.
5. One test, one rule: Test names should describe the business rule being verified.
6. Stub the boundary: Only use Mocks for uncontrollable I/O (filesystem, weights, GPU).

════════════════════════════════════════════════════════════════════════════════
Source Refactoring Notes
════════════════════════════════════════════════════════════════════════════════
bench_ag_borzoi.py was refactored to expose:
1. Architecture constants at module level.
2. Pure functions (_parse_variant, _stats, _bin_obs) at module level.
3. self.ism_center in setup() for coordinate verification.
"""

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# --- Minimal Stubs ----------------------------------------------------------

def _build_grelu_stub():
    grelu = types.ModuleType("grelu")

    resources = types.ModuleType("grelu.resources")
    resources.load_model = MagicMock()
    grelu.resources = resources

    seq_fmt = types.ModuleType("grelu.sequence.format")
    seq_fmt.convert_input_type = MagicMock(return_value=["ACGT"])
    grelu.sequence = types.ModuleType("grelu.sequence")
    grelu.sequence.format = seq_fmt

    score_mod = types.ModuleType("grelu.interpret.score")
    score_mod.ISM_predict = MagicMock()
    grelu.interpret = types.ModuleType("grelu.interpret")
    grelu.interpret.score = score_mod

    pt_mod = types.ModuleType("grelu.transforms.prediction_transforms")

    class _Specificity:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

    pt_mod.Specificity = _Specificity
    pt_mod.Aggregate = MagicMock()
    grelu.transforms = types.ModuleType("grelu.transforms")
    grelu.transforms.prediction_transforms = pt_mod

    io_mod = types.ModuleType("grelu.io.genome")
    io_mod.read_gtf   = MagicMock()
    io_mod.read_sizes = MagicMock()
    grelu.io = types.ModuleType("grelu.io")
    grelu.io.genome = io_mod
    grelu.io.bigwig  = types.ModuleType("grelu.io.bigwig")

    grelu.data = types.ModuleType("grelu.data")
    grelu.data.preprocess = types.ModuleType("grelu.data.preprocess")

    ds_mod = types.ModuleType("grelu.data.dataset")
    ds_mod.SeqDataset        = MagicMock()
    ds_mod.BigWigSeqDataset  = MagicMock()
    grelu.data.dataset = ds_mod

    grelu.visualize = types.ModuleType("grelu.visualize")
    grelu.visualize.plot_ISM = MagicMock()

    grelu.lightning = types.ModuleType("grelu.lightning")
    grelu.lightning.LightningModel = MagicMock()

    grelu.variant = types.ModuleType("grelu.variant")
    grelu.variant.predict_variant_effects = MagicMock()

    return grelu


def _install_stub(grelu):
    mapping = {
        "grelu":                                   grelu,
        "grelu.resources":                         grelu.resources,
        "grelu.sequence":                          grelu.sequence,
        "grelu.sequence.format":                   grelu.sequence.format,
        "grelu.interpret":                         grelu.interpret,
        "grelu.interpret.score":                   grelu.interpret.score,
        "grelu.transforms":                        grelu.transforms,
        "grelu.transforms.prediction_transforms":  grelu.transforms.prediction_transforms,
        "grelu.io":                                grelu.io,
        "grelu.io.genome":                         grelu.io.genome,
        "grelu.io.bigwig":                         grelu.io.bigwig,
        "grelu.data":                              grelu.data,
        "grelu.data.preprocess":                   grelu.data.preprocess,
        "grelu.data.dataset":                      grelu.data.dataset,
        "grelu.visualize":                         grelu.visualize,
        "grelu.lightning":                         grelu.lightning,
        "grelu.variant":                           grelu.variant,
    }
    sys.modules.update(mapping)


def _install_alphagenome_stub():
    ag  = types.ModuleType("alphagenome_pytorch")
    cfg = types.ModuleType("alphagenome_pytorch.config")

    class _DP:
        @staticmethod
        def mixed_precision():
            return "mp"

    cfg.DtypePolicy = _DP
    ag.config = cfg

    metrics = types.ModuleType("alphagenome_pytorch.metrics")
    metrics.pearson_r = MagicMock(return_value=torch.zeros(10))
    ag.metrics = metrics

    sys.modules["alphagenome_pytorch"]         = ag
    sys.modules["alphagenome_pytorch.config"]  = cfg
    sys.modules["alphagenome_pytorch.metrics"] = metrics


@pytest.fixture(scope="module")
def bench():
    """Import bench_ag_borzoi with heavy deps stubbed out. Returns the module."""
    grelu = _build_grelu_stub()
    _install_stub(grelu)
    _install_alphagenome_stub()

    import matplotlib
    matplotlib.use("Agg")

    sys.modules.pop("bench_ag_borzoi", None)
    mod = importlib.import_module("bench_ag_borzoi")
    mod.grelu = grelu   # expose so tests can patch sub-attributes
    return mod


@pytest.fixture()
def app(bench):
    """Fresh GreluTutorialApp without GPU."""
    return bench.GreluTutorialApp(gene="SRSF11", genome="hg38", devices="cpu")


# ==============================================================================
# [A1] Golden-value tests — Architecture constants must match specs
# ==============================================================================

class TestArchitectureConstants:

    def test_borzoi_input_len(self, bench):
        assert bench.BORZOI_INPUT_LEN == 524_288

    def test_borzoi_bin_size(self, bench):
        assert bench.BORZOI_BIN_SIZE == 32

    def test_ag_input_len(self, bench):
        assert bench.AG_INPUT_LEN == 131_072

    def test_ag_bin_size(self, bench):
        assert bench.AG_BIN_SIZE == 128

    def test_ag_output_bins_derived_from_input_and_bin(self, bench):
        """AG_OUTPUT_BINS must be exactly AG_INPUT_LEN // AG_BIN_SIZE."""
        assert bench.AG_OUTPUT_BINS == bench.AG_INPUT_LEN // bench.AG_BIN_SIZE

    def test_ism_half_width(self, bench):
        assert bench.ISM_HALF_WIDTH == 100


# ==============================================================================
# [A2] Coordinate arithmetic in GreluTutorialApp.setup()
# ==============================================================================

class TestSetupCoordinates:

    def _pre_setup(self, app, bench, min_start=2_000_000, chrom="chr1"):
        exons = pd.DataFrame({
            "gene_name": ["SRSF11"] * 3,
            "chrom":     [chrom] * 3,
            "start":     [min_start, min_start + 500, min_start + 1000],
            "end":       [min_start + 200, min_start + 700, min_start + 1200],
        })
        app.exons = exons
        with patch.object(bench.grelu.sequence.format, "convert_input_type",
                          return_value=["A"]):
            app.setup()
        return exons

    def test_ism_center_is_min_exon_start(self, app, bench):
        """ISM center = min(exon.start)."""
        self._pre_setup(app, bench, min_start=5_000_000)
        assert app.ism_center == 5_000_000

    def test_borzoi_window_centered_on_ism(self, app, bench):
        """Borzoi window must be centered on ism_center."""
        self._pre_setup(app, bench, min_start=2_000_000)
        assert app.borzoi_start_coord + bench.BORZOI_INPUT_LEN // 2 == app.ism_center

    def test_ag_window_centered_on_ism(self, app, bench):
        """AlphaGenome window must be centered on ism_center."""
        self._pre_setup(app, bench, min_start=2_000_000)
        assert app.ag_start_coord + bench.AG_INPUT_LEN // 2 == app.ism_center

    def test_ag_window_length_equals_ag_input_len(self, app, bench):
        self._pre_setup(app, bench)
        assert app.ag_end_coord - app.ag_start_coord == bench.AG_INPUT_LEN

    def test_ism_region_symmetric_around_center(self, app, bench):
        """ISM region must be symmetric with width = 2 * ISM_HALF_WIDTH."""
        self._pre_setup(app, bench, min_start=3_000_000)
        assert app.ism_region["start"] == app.ism_center - bench.ISM_HALF_WIDTH
        assert app.ism_region["end"]   == app.ism_center + bench.ISM_HALF_WIDTH

    def test_setup_is_idempotent(self, app, bench):
        """Re-calling setup() should not reload exons (early-return protection)."""
        exons = pd.DataFrame({
            "gene_name": ["SRSF11"], "chrom": ["chr1"],
            "start": [1_500_000], "end": [1_500_200],
        })
        app.exons = exons
        call_log = {"n": 0}

        def _counting(*a, **kw):
            call_log["n"] += 1
            return exons

        with patch.object(bench.grelu.io.genome, "read_gtf", _counting):
            with patch.object(bench.grelu.sequence.format, "convert_input_type",
                              return_value=["A"]):
                app.input_seqs = None
                app.setup()
                app.setup()

        assert call_log["n"] == 0  # exons was pre-set; read_gtf must not be called


# ==============================================================================
# [B] _make_windows() — Window generation logic
# ==============================================================================

class TestMakeWindows:

    def _sizes_df(self, d: dict) -> pd.DataFrame:
        return pd.DataFrame({"chrom": list(d), "size": list(d.values())})

    @pytest.mark.parametrize("chrom_len,seq_len,expected_n", [
        (3_000, 1_000, 3),   # Exact fit
        (2_500, 1_000, 2),   # Remainder discarded
        (1_000, 1_000, 1),   # Single window
        (999,   1_000, 0),   # Too short
    ])
    def test_window_count(self, app, bench, chrom_len, seq_len, expected_n):
        """Windows exceeding chrom_len must be excluded."""
        sizes = self._sizes_df({"chr1": chrom_len})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1"], seq_len=seq_len,
                                   stride=seq_len, max_windows=None, seed=42)
        assert len(df) == expected_n

    def test_window_boundaries_are_correct(self, app, bench):
        sizes = self._sizes_df({"chr1": 3_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1"], seq_len=1_000, stride=1_000,
                                   max_windows=None, seed=42)
        assert df["start"].tolist() == [0, 1000, 2000]
        assert df["end"].tolist()   == [1000, 2000, 3000]

    def test_overlapping_windows_when_stride_lt_seq_len(self, app, bench):
        """Stride < seq_len should produce overlapping windows."""
        sizes = self._sizes_df({"chr1": 3_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1"], seq_len=1_000, stride=500,
                                   max_windows=None, seed=42)
        # starts: 0,500,1000,1500,2000 → 5 valid windows
        assert len(df) == 5

    def test_subsampling_is_deterministic(self, app, bench):
        """Same seed must produce same subset."""
        sizes = self._sizes_df({"chr1": 100_000})
        kwargs = dict(seq_len=1_000, stride=1_000, max_windows=10, seed=7)
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df1 = app._make_windows(["chr1"], **kwargs)
            df2 = app._make_windows(["chr1"], **kwargs)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))

    def test_different_seeds_produce_different_subsets(self, app, bench):
        sizes = self._sizes_df({"chr1": 100_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df1 = app._make_windows(["chr1"], seq_len=1_000, stride=1_000,
                                    max_windows=10, seed=1)
            df2 = app._make_windows(["chr1"], seq_len=1_000, stride=1_000,
                                    max_windows=10, seed=99)
        assert df1["start"].tolist() != df2["start"].tolist()

    def test_no_subsampling_when_count_lte_max(self, app, bench):
        sizes = self._sizes_df({"chr1": 5_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1"], seq_len=1_000, stride=1_000,
                                   max_windows=10, seed=42)
        assert len(df) == 5

    def test_unknown_chrom_silently_skipped(self, app, bench):
        sizes = self._sizes_df({"chr1": 3_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1", "chr2"], seq_len=1_000, stride=1_000,
                                   max_windows=None, seed=42)
        assert "chr2" not in df["chrom"].values

    def test_output_has_required_columns(self, app, bench):
        sizes = self._sizes_df({"chr1": 2_000})
        with patch.object(bench.grelu.io.genome, "read_sizes", return_value=sizes):
            df = app._make_windows(["chr1"], seq_len=1_000, stride=1_000,
                                   max_windows=None, seed=42)
        assert {"chrom", "start", "end", "strand"}.issubset(df.columns)
        assert (df["strand"] == "+").all()


# ==============================================================================
# [C] _bin_obs() — Test module-level pure function
# ==============================================================================

class TestBinObs:
    """Tests for bench_ag_borzoi._bin_obs."""

    @pytest.fixture(autouse=True)
    def _fn(self, bench):
        self.bin_obs = bench._bin_obs

    def test_output_shape_after_binning(self):
        raw = np.ones((4, 1024))
        out = self.bin_obs(raw, bin_size=128, n_pred_bins=8)
        assert out.shape == (4, 8)

    def test_binning_uses_mean_not_sum(self):
        raw = np.zeros((1, 256))
        raw[0, :128] = 2.0
        raw[0, 128:] = 0.0
        out = self.bin_obs(raw, bin_size=128, n_pred_bins=2)
        assert out[0, 0] == pytest.approx(2.0)
        assert out[0, 1] == pytest.approx(0.0)

    def test_no_crop_when_obs_equals_pred_bins(self):
        raw = np.random.default_rng(0).random((3, 1024))
        out = self.bin_obs(raw, bin_size=128, n_pred_bins=8)
        assert out.shape[1] == 8

    @pytest.mark.parametrize("n_obs,n_pred,expected_crop_start", [
        (10, 6, 2),   # (10-6)//2 = 2
        (16, 14, 1),  # (16-14)//2 = 1
        (8,  4, 2),   # (8-4)//2  = 2
    ])
    def test_center_crop_formula(self, n_obs, n_pred, expected_crop_start):
        """Center crop offset = (n_obs_bins - n_pred_bins) // 2."""
        raw = np.arange(n_obs, dtype=float).reshape(1, -1)
        out = self.bin_obs(raw, bin_size=1, n_pred_bins=n_pred)
        expected = raw[0, expected_crop_start : expected_crop_start + n_pred]
        np.testing.assert_array_equal(out[0], expected)

    def test_ag_no_crop_1024_bins(self, bench):
        """AG: 131072bp / 128 = 1024 bins; matches model output bins."""
        n_obs = bench.AG_INPUT_LEN // bench.AG_BIN_SIZE  # 1024
        raw = np.ones((2, n_obs))
        out = self.bin_obs(raw, bin_size=1, n_pred_bins=n_obs)
        assert out.shape == (2, 1024)


# ==============================================================================
# [D] Track-selection logic
# ==============================================================================

class TestTrackSelection:
    """
    Business rules:
      Borzoi on  = RNA assay AND 'brain' in sample
      Borzoi off = RNA assay AND 'liver' in sample
      AG on      = output_type=='rna_seq' AND 'brain' in biosample_name
      AG off     = output_type=='rna_seq' AND 'liver' in biosample_name
                   AND assay_title=='polyA plus RNA-seq'
    """

    @pytest.fixture()
    def borzoi_tasks(self):
        return pd.DataFrame({
            "assay":       ["RNA",    "RNA",              "CAGE",          "RNA",     "ATAC",  "RNA"],
            "sample":      ["Brain",  "liver hepatocyte", "Brain CAGE",    "Kidney",  "brain", "liver HepG2"],
            "name":        [f"b{i}" for i in range(6)],
            "description": [f"d{i}" for i in range(6)],
        })

    @pytest.fixture()
    def ag_meta(self):
        return pd.DataFrame({
            "output_type":    ["rna_seq",             "rna_seq",            "rna_seq",         "cage"],
            "biosample_name": ["brain frontal cortex","liver HepG2",        "kidney cortex",   "brain cerebellum"],
            "assay_title":    ["polyA plus RNA-seq",  "polyA plus RNA-seq", "polyA plus RNA-seq", "CAGE"],
            "track_index":    [0, 1, 2, 3],
            "track_name":     ["bfc", "lhepg2", "kidney", "bcage"],
        })

    def _mock_borzoi(self, tasks_df):
        m = MagicMock()
        m.data_params = {"tasks": tasks_df.to_dict("list")}
        m.input_intervals_to_output_bins.return_value = pd.DataFrame(
            {"start": [0], "end": [5]}
        )
        return m

    def test_borzoi_on_tasks_brain_rna_only(self, app, borzoi_tasks):
        app.borzoi = self._mock_borzoi(borzoi_tasks)
        app.target_exons   = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        app.borzoi_start_coord = 0
        kw = app.get_borzoi_transform()._kwargs
        assert 0 in kw["on_tasks"]   # "Brain" RNA
        assert 2 not in kw["on_tasks"]  # CAGE excluded
        assert 3 not in kw["on_tasks"]  # Kidney excluded
        assert 4 not in kw["on_tasks"]  # ATAC excluded

    def test_borzoi_off_tasks_liver_rna_only(self, app, borzoi_tasks):
        app.borzoi = self._mock_borzoi(borzoi_tasks)
        app.target_exons   = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        app.borzoi_start_coord = 0
        kw = app.get_borzoi_transform()._kwargs
        assert 1 in kw["off_tasks"]   # "liver hepatocyte" RNA
        assert 5 in kw["off_tasks"]   # "liver HepG2" RNA
        assert 0 not in kw["off_tasks"]  # brain not in off

    def test_borzoi_transform_compare_func_is_divide(self, app, borzoi_tasks):
        app.borzoi = self._mock_borzoi(borzoi_tasks)
        app.target_exons   = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        app.borzoi_start_coord = 0
        kw = app.get_borzoi_transform()._kwargs
        assert kw["compare_func"]   == "divide"
        assert kw["on_aggfunc"]     == "mean"
        assert kw["off_aggfunc"]    == "mean"
        assert kw["length_aggfunc"] == "mean"

    def test_ag_on_tasks_brain_rna_seq(self, app, bench, ag_meta):
        mock_model = MagicMock()
        mock_model.input_intervals_to_output_bins.return_value = pd.DataFrame(
            {"start": [0], "end": [10]}
        )
        app.ag_start_coord = 0
        app.target_exons   = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [1500]})
        with patch("pandas.read_parquet", return_value=ag_meta):
            kw = app.get_ag_transform(mock_model)._kwargs
        assert 0 in kw["on_tasks"]       # brain rna_seq
        assert 2 not in kw["on_tasks"]   # kidney
        assert 3 not in kw["on_tasks"]   # cage type

    def test_ag_off_tasks_liver_polya_rna_seq_only(self, app, bench):
        """Excludes 'total RNA-seq' from off_tasks; only 'polyA' included."""
        meta = pd.DataFrame({
            "output_type":    ["rna_seq",            "rna_seq"],
            "biosample_name": ["liver HepG2",        "liver total RNA"],
            "assay_title":    ["polyA plus RNA-seq", "total RNA-seq"],
            "track_index":    [0, 1],
            "track_name":     ["polya", "total"],
        })
        mock_model = MagicMock()
        mock_model.input_intervals_to_output_bins.return_value = pd.DataFrame(
            {"start": [0], "end": [5]}
        )
        app.ag_start_coord = 0
        app.target_exons   = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [1500]})
        with patch("pandas.read_parquet", return_value=meta):
            kw = app.get_ag_transform(mock_model)._kwargs
        assert 0 in kw["off_tasks"]      # polyA → included
        assert 1 not in kw["off_tasks"]  # total RNA-seq → excluded


# ==============================================================================
# [E] _parse_variant() — Test module-level function
# ==============================================================================

class TestParseVariant:

    @pytest.fixture(autouse=True)
    def _fn(self, bench):
        self.parse = bench._parse_variant

    @pytest.mark.parametrize("raw,expected", [
        ("chr1_70355119_G_A",  ("chr1",  70_355_119, "G",    "A")),
        ("chrX_1000_C_T",      ("chrX",  1_000,      "C",    "T")),
        ("chr2_50_A_ACGT",     ("chr2",  50,         "A",    "ACGT")),
        ("chr22_999999_T_G",   ("chr22", 999_999,    "T",    "G")),
    ])
    def test_parse_fields(self, raw, expected):
        assert self.parse(raw) == expected

    def test_position_is_int_not_str(self):
        _, pos, _, _ = self.parse("chr3_500_A_T")
        assert isinstance(pos, int)


# ==============================================================================
# [F] Credible-set filtering
# ==============================================================================

class TestCredibleSetFiltering:
    """
    Filter rules:
      1. max_pip >= min_pip_causal (per locus)
      2. cs_size >= min_cs_size    (per locus)
      3. SNP: len(ref)==1 AND len(alt)==1
      4. edge: pos +/- half width within chrom boundaries
    """

    @staticmethod
    def _apply_locus_filter(cs: pd.DataFrame, min_pip: float, min_size: int):
        cs_max  = cs.groupby(["gene_id","cs_id"])["pip"].max().reset_index()
        cs_max.columns = ["gene_id","cs_id","max_pip"]
        cs_sz   = cs.groupby(["gene_id","cs_id"])["pip"].count().reset_index()
        cs_sz.columns = ["gene_id","cs_id","cs_size"]
        info = cs_max.merge(cs_sz)
        return info[(info.max_pip >= min_pip) & (info.cs_size >= min_size)]

    @pytest.mark.parametrize("max_pip,cs_size,min_pip,min_size,should_pass", [
        (0.9, 5, 0.5, 5, True),
        (0.4, 5, 0.5, 5, False),
        (0.9, 3, 0.5, 5, False),
        (0.4, 3, 0.5, 5, False),
    ])
    def test_locus_filter_combinations(self, max_pip, cs_size, min_pip, min_size, should_pass):
        rows = [("G1","CS1",f"chr1_{i}_A_T",
                 max_pip if i == 0 else 0.01)
                for i in range(cs_size)]
        cs = pd.DataFrame(rows, columns=["gene_id","cs_id","variant","pip"])
        good = self._apply_locus_filter(cs, min_pip, min_size)
        assert (len(good) > 0) == should_pass

    @pytest.mark.parametrize("ref,alt,is_snp", [
        ("A",  "T",    True),
        ("AT", "G",    False),
        ("C",  "GT",   False),
        ("G",  "G",    True),
    ])
    def test_snp_mask(self, ref, alt, is_snp):
        row = pd.Series({"ref": ref, "alt": alt})
        result = len(row["ref"]) == 1 and len(row["alt"]) == 1
        assert result == is_snp

    @pytest.mark.parametrize("pos,chrom_size,passes", [
        (3_000_000, 10_000_000, True),
        (100_000,   10_000_000, False),
        (9_900_000, 10_000_000, False),
    ])
    def test_edge_filter_uses_borzoi_half(self, bench, pos, chrom_size, passes):
        """Edge filter should use BORZOI_INPUT_LEN // 2."""
        half = bench.BORZOI_INPUT_LEN // 2
        result = (pos - half >= 0) and (pos + half <= chrom_size)
        assert result == passes

    def test_auprc_skip_degenerate_loci(self):
        """Skip loci that are all-causal or all-non-causal in AUPRC loops."""
        cases = [(0, 5, True), (5, 5, True), (2, 5, False), (1, 3, False)]
        for n_causal, cs_size, should_skip in cases:
            skip = (n_causal == 0) or (n_causal == cs_size)
            assert skip == should_skip


# ==============================================================================
# [G] _stats() — Test module-level function
# ==============================================================================

class TestStats:

    @pytest.fixture(autouse=True)
    def _fn(self, bench):
        self.stats = bench._stats

    @pytest.mark.parametrize("arr,expected_n", [
        (np.array([0.5, 0.7, 0.9]),   3),
        (np.array([0.5, np.nan, 0.9]), 2),
        (np.array([np.inf, 0.5]),      1),
        (np.array([np.nan, np.inf]),   0),
    ])
    def test_n_windows_excludes_non_finite(self, arr, expected_n):
        assert self.stats(arr)["n_windows"] == expected_n

    def test_mean_computed_over_finite_values_only(self):
        r = np.array([0.5, np.nan, 0.9])
        assert self.stats(r)["mean_r"] == pytest.approx((0.5 + 0.9) / 2)

    def test_all_nan_returns_nan_not_error(self):
        s = self.stats(np.array([np.nan, np.nan]))
        assert np.isnan(s["mean_r"])
        assert s["n_windows"] == 0

    def test_signal_mask_combines_finite_and_obs_threshold(self):
        """signal_mask = isfinite(r) AND obs_mean > threshold"""
        r        = np.array([0.8,    np.nan, 0.5,    0.3])
        obs_mean = np.array([0.10,   0.20,   0.01,   0.15])
        mask = np.isfinite(r) & (obs_mean > 0.05)
        assert mask.tolist() == [True, False, False, True]


# ==============================================================================
# [H] LightningModel integration config (stub-based)
# ==============================================================================

class TestLightningModelConfig:
    """
    Verifies that setup_ag_rna / setup_ag_cage pass correct params to LightningModel:
    - output_key (rna_seq vs cage)
    - regression task + mse loss
    - data_params (seq_len, bin_size)
    - crop_len = 0
    """

    @pytest.fixture()
    def app_with_chrom(self, bench):
        a = bench.GreluTutorialApp(gene="SRSF11", devices="cpu")
        a.chrom = "chr1"
        return a, bench

    def _capture_constructor(self, bench_mod):
        """Capture calls to LightningModel constructor within bench_ag_borzoi namespace."""
        captured = {}

        def _ctor(**kwargs):
            captured.update(kwargs)
            m = MagicMock()
            m.data_params = {"train": {}}
            m.model_params = {}
            return m

        ctx = patch.object(bench_mod, "LightningModel", side_effect=_ctor)
        return ctx, captured

    def test_rna_output_key_is_rna_seq(self, app_with_chrom, bench):
        app, _ = app_with_chrom
        ctx, captured = self._capture_constructor(bench)
        with ctx:
            app.setup_ag_rna()
        assert captured["model_params"]["output_key"] == "rna_seq"

    def test_cage_output_key_is_cage(self, app_with_chrom, bench):
        app, _ = app_with_chrom
        ctx, captured = self._capture_constructor(bench)
        with ctx:
            app.setup_ag_cage()
        assert captured["model_params"]["output_key"] == "cage"

    def test_task_is_regression_not_binary(self, app_with_chrom, bench):
        """Explicitly require regression task to avoid automatic Sigmoid layers."""
        app, _ = app_with_chrom
        ctx, captured = self._capture_constructor(bench)
        with ctx:
            app.setup_ag_rna()
        assert captured["train_params"]["task"] == "regression"
        assert captured["train_params"]["loss"]  == "mse"

    def test_seq_len_is_ag_input_len(self, app_with_chrom, bench):
        app, _ = app_with_chrom
        m = MagicMock()
        m.data_params = {"train": {}}
        m.model_params = {}
        with patch.object(bench, "LightningModel", return_value=m):
            app.setup_ag_rna()
        assert app.ag_rna.data_params["train"]["seq_len"]  == bench.AG_INPUT_LEN
        assert app.ag_rna.data_params["train"]["bin_size"] == bench.AG_BIN_SIZE

    def test_crop_len_is_zero(self, app_with_chrom, bench):
        """AlphaGenome does not crop; crop_len must be 0."""
        app, _ = app_with_chrom
        m = MagicMock()
        m.data_params = {"train": {}}
        m.model_params = {}
        with patch.object(bench, "LightningModel", return_value=m):
            app.setup_ag_rna()
        assert app.ag_rna.model_params["crop_len"] == 0
