"""
AlphaGenome inference integration tests.
Requires CUDA + model weights; auto-skips when either is absent.

════════════════════════════════════════════════════════════════════════════════
Design Philosophy
════════════════════════════════════════════════════════════════════════════════

Unit vs Integration Tests
──────────────────────────
- test_bench_ag_borzoi.py (Unit): Verifies logic with mock data. Fast, no GPU required.
- This file (Integration): Verifies real system behavior with weights and GPU.

What to Test here?
──────────────────
Focus on properties that require a live model:
- Shape Contracts: Actual output tensor shapes (tracks, bins).
- Numerical Validity: No NaN/Inf; RNA non-negativity (softplus).
- Determinism: Identical outputs for identical inputs in eval mode.
- DDP Consistency: Multi-GPU (devices=[0,1]) results match single-GPU.
- Memory Management: No leaks after _cleanup().
- Routing: output_key correctly routes to RNA vs CAGE heads.

Principles
──────────
- Test observable contracts, not implementation details.
- Use source constants (AG_INPUT_LEN, etc.) directly to ensure synchronization.
- Use class-scoped fixtures for models to save time and avoid CUDA fragmentation.

════════════════════════════════════════════════════════════════════════════════
Skip Strategy (CI Friendly)
════════════════════════════════════════════════════════════════════════════════
- @requires_cuda: Skips if no GPU found.
- @requires_weights: Skips if local weight cache is empty.
- @requires_2gpus: Skips multi-GPU tests if < 2 GPUs available.

════════════════════════════════════════════════════════════════════════════════
Module Isolation
════════════════════════════════════════════════════════════════════════════════
The _clear_stubs fixture ensures that MagicMocks installed by unit tests in 
sys.modules are cleared before running integration tests, forcing a real import.
"""

import gc
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# --- Weight / GPU Detection -------------------------------------------------
_WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_fold_0.safetensors"
)
_AG_META_PATH = (
    REPO_ROOT / "src/alphagenome_pytorch/src/alphagenome_pytorch/data"
    / "track_metadata_human.parquet"
)
_WEIGHTS_AVAILABLE = os.path.exists(_WEIGHTS_PATH)
_META_AVAILABLE    = _AG_META_PATH.exists()
_CUDA_AVAILABLE    = torch.cuda.is_available()
_N_GPUS            = torch.cuda.device_count() if _CUDA_AVAILABLE else 0

requires_weights = pytest.mark.skipif(
    not _WEIGHTS_AVAILABLE,
    reason=f"AG weights not found at {_WEIGHTS_PATH}",
)
requires_meta = pytest.mark.skipif(
    not _META_AVAILABLE,
    reason=f"AG track metadata not found at {_AG_META_PATH}",
)
requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA device required for inference tests",
)
requires_2gpus = pytest.mark.skipif(
    _N_GPUS < 2,
    reason=f"Multi-GPU test requires ≥2 GPUs; found {_N_GPUS}",
)


# --- Stub Isolation Fixture -------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _clear_stubs():
    """
    Remove grelu/alphagenome_pytorch stubs installed by unit tests to allow 
    real model loading. Restores them after tests to avoid side effects.
    """
    _prefixes = ("grelu", "alphagenome_pytorch")
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if any(k == p or k.startswith(p + ".") for p in _prefixes)
    }
    yield
    sys.modules.update(saved)


# --- Shared Helpers ---------------------------------------------------------

def _random_dna(length: int, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(["A", "C", "G", "T"], size=length))


def _load_ag_model(output_key: str = "rna_seq"):
    """
    Instantiate model with parameters synchronized with bench_ag_borzoi.
    """
    from alphagenome_pytorch.config import DtypePolicy
    from grelu.lightning import LightningModel
    from bench_ag_borzoi import AG_INPUT_LEN, AG_BIN_SIZE

    model = LightningModel(
        model_params={
            "model_type":   "AlphaGenomeModel",
            "output_key":   output_key,
            "weights_path": _WEIGHTS_PATH,
            "dtype_policy": DtypePolicy.mixed_precision(),
            "resolution":   128,
        },
        train_params={"task": "regression", "loss": "mse"},
    )
    model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
    model.model_params["crop_len"] = 0
    return model


def _track_count(output_key: str) -> int:
    meta = pd.read_parquet(_AG_META_PATH)
    return int((meta.output_type == output_key).sum())


def _free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==============================================================================
# [1] Single GPU Inference
# ==============================================================================

@requires_cuda
@requires_weights
@requires_meta
class TestSingleGPUInference:
    """
    Verifies AlphaGenome inference via gReLU LightningModel on a single GPU.
    Uses class-scoped fixtures to avoid reloading weights for every test.
    """

    @pytest.fixture(scope="class")
    def rna_model(self):
        model = _load_ag_model("rna_seq")
        yield model
        _free_model(model)

    @pytest.fixture(scope="class")
    def cage_model(self):
        model = _load_ag_model("cage")
        yield model
        _free_model(model)

    @pytest.fixture(scope="class")
    def test_seq(self):
        return _random_dna(131_072, seed=42)

    # --- Shape Contracts ----------------------------------------------------

    def test_rna_output_shape(self, rna_model, test_seq):
        """RNA output shape = (1, n_rna_tracks, 1024)."""
        preds = rna_model.predict_on_seqs(test_seq, device=0)
        n_tracks = _track_count("rna_seq")
        assert preds.shape == (1, n_tracks, 1024), (
            f"Expected (1, {n_tracks}, 1024), got {preds.shape}"
        )

    def test_cage_output_shape(self, cage_model, test_seq):
        """CAGE output shape = (1, n_cage_tracks, 1024)."""
        preds = cage_model.predict_on_seqs(test_seq, device=0)
        n_tracks = _track_count("cage")
        assert preds.shape == (1, n_tracks, 1024)

    def test_output_bin_count_matches_ag_spec(self, rna_model, test_seq):
        """131072 / 128 = 1024 bins; matches AG_OUTPUT_BINS constant."""
        from bench_ag_borzoi import AG_OUTPUT_BINS
        preds = rna_model.predict_on_seqs(test_seq, device=0)
        assert preds.shape[2] == AG_OUTPUT_BINS

    # --- Numerical Validity -------------------------------------------------

    def test_rna_predictions_are_finite(self, rna_model, test_seq):
        preds = rna_model.predict_on_seqs(test_seq, device=0)
        assert np.all(np.isfinite(preds)), "RNA predictions contain NaN or Inf"

    def test_rna_predictions_are_nonnegative(self, rna_model, test_seq):
        """RNA-seq head uses softplus; output must be non-negative."""
        preds = rna_model.predict_on_seqs(test_seq, device=0)
        assert np.all(preds >= 0), f"Got negative RNA predictions: min={preds.min():.4f}"

    def test_cage_predictions_are_finite(self, cage_model, test_seq):
        preds = cage_model.predict_on_seqs(test_seq, device=0)
        assert np.all(np.isfinite(preds))

    # --- Data Types ---------------------------------------------------------

    def test_output_dtype_is_float(self, rna_model, test_seq):
        preds = rna_model.predict_on_seqs(test_seq, device=0)
        assert preds.dtype in (np.float32, np.float64), (
            f"Expected float32/64, got {preds.dtype}"
        )

    # --- Determinism --------------------------------------------------------

    def test_deterministic_with_same_sequence(self, rna_model, test_seq):
        """Identical inputs must produce identical outputs in eval mode."""
        p1 = rna_model.predict_on_seqs(test_seq, device=0)
        p2 = rna_model.predict_on_seqs(test_seq, device=0)
        np.testing.assert_array_equal(p1, p2)

    def test_different_sequences_give_different_predictions(self, rna_model):
        """Model should not collapse to a constant function."""
        seq1 = _random_dna(131_072, seed=1)
        seq2 = _random_dna(131_072, seed=2)
        p1 = rna_model.predict_on_seqs(seq1, device=0)
        p2 = rna_model.predict_on_seqs(seq2, device=0)
        assert not np.allclose(p1, p2, atol=1e-6), (
            "Two different seqs gave identical predictions"
        )

    # --- Routing Verification -----------------------------------------------

    def test_rna_and_cage_track_counts_differ(self, rna_model, cage_model, test_seq):
        """Verify output_key routes to correct heads with different track counts."""
        rna_preds  = rna_model.predict_on_seqs(test_seq, device=0)
        cage_preds = cage_model.predict_on_seqs(test_seq, device=0)
        assert rna_preds.shape[1] != cage_preds.shape[1], (
            "RNA and CAGE should have different track counts"
        )

    # --- GPU Memory Cleanup -------------------------------------------------

    def test_gpu_memory_released_after_cleanup(self, test_seq):
        """GPU memory should return to baseline (< 50 MB leak) after explicit cleanup."""
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(0)

        model = _load_ag_model("rna_seq")
        _ = model.predict_on_seqs(test_seq, device=0)
        _free_model(model)

        torch.cuda.synchronize()
        leaked_mb = (torch.cuda.memory_allocated(0) - mem_before) / 1024 ** 2
        assert leaked_mb < 50, (
            f"Possible GPU memory leak: {leaked_mb:.1f} MB not released after cleanup"
        )


# ==============================================================================
# [2] Multi-GPU Consistency
# ==============================================================================

@requires_cuda
@requires_2gpus
@requires_weights
@requires_meta
class TestMultiGPUConsistency:
    """
    Verifies that multi-GPU paths produce numerically equivalent results to single-GPU.
    Regressions here often indicate DDP rank-related bugs.
    """

    ATOL = 1e-4
    RTOL = 1e-4

    @pytest.fixture(scope="class")
    def seqs(self):
        # 2 sequences to allow even distribution across 2 GPUs
        return [_random_dna(131_072, seed=i) for i in range(2)]

    def _predict_on_devices(self, model, seqs, devices):
        from grelu.data.dataset import SeqDataset
        dataset = SeqDataset(seqs=seqs, seq_len=131_072)
        return model.predict_on_dataset(
            dataset,
            devices=devices,
            num_workers=0,
            batch_size=1,
        )

    def test_single_vs_multi_gpu_rna(self, seqs):
        model = _load_ag_model("rna_seq")
        try:
            p1 = self._predict_on_devices(model, seqs, devices=[0])
            p2 = self._predict_on_devices(model, seqs, devices=[0, 1])
        finally:
            _free_model(model)

        np.testing.assert_allclose(
            p1, p2, atol=self.ATOL, rtol=self.RTOL,
            err_msg="Multi-GPU RNA predictions differ from single-GPU baseline",
        )

    def test_single_vs_multi_gpu_cage(self, seqs):
        model = _load_ag_model("cage")
        try:
            p1 = self._predict_on_devices(model, seqs, devices=[0])
            p2 = self._predict_on_devices(model, seqs, devices=[0, 1])
        finally:
            _free_model(model)

        np.testing.assert_allclose(
            p1, p2, atol=self.ATOL, rtol=self.RTOL,
            err_msg="Multi-GPU CAGE predictions differ from single-GPU baseline",
        )

    def test_multi_gpu_output_shape_unchanged(self, seqs):
        """Batch dimension should not be split/shrunk by DDP in final output."""
        model = _load_ag_model("rna_seq")
        n_tracks = _track_count("rna_seq")
        try:
            preds = self._predict_on_devices(model, seqs, devices=[0, 1])
        finally:
            _free_model(model)

        assert preds.shape == (len(seqs), n_tracks, 1024)

    def test_multi_gpu_predictions_are_finite(self, seqs):
        model = _load_ag_model("rna_seq")
        try:
            preds = self._predict_on_devices(model, seqs, devices=[0, 1])
        finally:
            _free_model(model)

        assert np.all(np.isfinite(preds)), "Multi-GPU predictions contain NaN or Inf"

    def test_multi_gpu_memory_released_on_both_devices(self, seqs):
        """Both GPU contexts should be cleared after multi-GPU inference."""
        torch.cuda.synchronize()
        mem0_before = torch.cuda.memory_allocated(0)
        mem1_before = torch.cuda.memory_allocated(1)

        model = _load_ag_model("rna_seq")
        self._predict_on_devices(model, seqs, devices=[0, 1])
        _free_model(model)

        torch.cuda.synchronize()
        leak0 = (torch.cuda.memory_allocated(0) - mem0_before) / 1024 ** 2
        leak1 = (torch.cuda.memory_allocated(1) - mem1_before) / 1024 ** 2

        assert leak0 < 50, f"GPU 0 leaked {leak0:.1f} MB after multi-GPU cleanup"
        assert leak1 < 50, f"GPU 1 leaked {leak1:.1f} MB after multi-GPU cleanup"
