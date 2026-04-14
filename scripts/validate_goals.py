"""
Validation script — calls GreluTutorialApp methods directly.
No logic is reimplemented here.

Goals:
  G1. Multi-GPU inference runs without error
  G2. Single-GPU vs multi-GPU ISM results are numerically consistent
  G3. Specificity scores match Tutorial 1 ground truth; ISM CSVs are valid
  G4. Dataset parameters match the fixed specification
"""

import os, sys, shutil
import numpy as np
import pandas as pd

# Import the actual app — no reimplementation
from tutorial_1_ism_modular import GreluTutorialApp, ISM_RESULTS_DIR

# ── helpers ─────────────────────────────────────────────────────────────────
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, bool(ok)))
    return bool(ok)

def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")

# Ground-truth values (Tutorial 1)
GT_BORZOI_SPEC = 1.617   # Tutorial 1 external ground truth
GT_AG_SPEC     = 1.739   # empirical value from current code (no external Tutorial 1 ref)
SPEC_TOL       = 0.05
ISM_ATOL       = 1e-3   # max abs diff single vs multi GPU

# ════════════════════════════════════════════════════════════════════════════
section("G4 — Dataset specification")
# ════════════════════════════════════════════════════════════════════════════

app = GreluTutorialApp(gene="SRSF11", genome="hg38", devices="0", num_workers=1)
app.setup()

check("Chromosome = chr1", app.chrom == "chr1", app.chrom)
check("Borzoi input = 524288 bp",
      len(app.input_seqs) == 524288, len(app.input_seqs))
check("AlphaGenome input = 131072 bp",
      len(app.ag_seqs) == 131072, len(app.ag_seqs))
check("ISM window width = 200 bp",
      app.ism_region['end'] - app.ism_region['start'] == 200,
      f"{app.ism_region['start']}–{app.ism_region['end']}")

# Check track counts through the actual transform builders
app.setup_borzoi()
b_trans = app.get_borzoi_transform()
n_b_on  = len(b_trans.on_transform.tasks)
n_b_off = len(b_trans.off_transform.tasks)
check("Borzoi brain RNA tracks = 5",  n_b_on  == 5,  n_b_on)
check("Borzoi liver RNA tracks = 24", n_b_off == 24, n_b_off)
app._cleanup()

app.setup_ag_rna()
ag_trans = app.get_ag_transform(app.ag_rna)
n_ag_on  = len(ag_trans.on_transform.tasks)
n_ag_off = len(ag_trans.off_transform.tasks)
check("AlphaGenome brain RNA tracks = 2",      n_ag_on  == 2, n_ag_on)
check("AlphaGenome polyA+ liver tracks = 4",   n_ag_off == 4, n_ag_off)
app._cleanup()

# ════════════════════════════════════════════════════════════════════════════
section("G3 — Specificity scores vs Tutorial 1 ground truth")
# ════════════════════════════════════════════════════════════════════════════

# run_inference() stores scores in app.borzoi_plot_data["spec"] and app.ag_rna_plot_data["spec"]
app.run_inference()

b_spec = app.borzoi_plot_data["spec"]
ag_spec = app.ag_rna_plot_data["spec"]
print(f"  Borzoi specificity    = {b_spec:.4f}  (target {GT_BORZOI_SPEC} ±{SPEC_TOL})")
print(f"  AlphaGenome specificity = {ag_spec:.4f}  (target {GT_AG_SPEC} ±{SPEC_TOL})")
check("Borzoi specificity ≈ 1.617 (Tutorial 1 ground truth)",
      abs(b_spec - GT_BORZOI_SPEC) <= SPEC_TOL, f"{b_spec:.4f}")
check("AlphaGenome specificity > 1.0 (brain > liver, biological validation)",
      ag_spec > 1.0, f"{ag_spec:.4f}")
check(f"AlphaGenome specificity ≈ {GT_AG_SPEC} (current-code reference)",
      abs(ag_spec - GT_AG_SPEC) <= SPEC_TOL, f"{ag_spec:.4f}")

# ISM CSV validity (files already present from prior runs)
for model_name, fname in [("Borzoi", "borzoi_ism.csv"), ("AlphaGenome", "alphagenome_ism.csv")]:
    fpath = os.path.join(ISM_RESULTS_DIR, fname)
    if check(f"{model_name} ISM CSV exists", os.path.exists(fpath), fpath):
        df = pd.read_csv(fpath, index_col=0)
        check(f"{model_name} ISM shape = (4, 200)",
              df.shape == (4, 200), str(df.shape))
        max_abs = float(df.abs().values.max())
        check(f"{model_name} ISM has signal (max|log2FC| > 0.01)",
              max_abs > 0.01, f"{max_abs:.4f}")
        check(f"{model_name} ISM row index = A/C/G/T",
              list(df.index) == ["A","C","G","T"], list(df.index))

# ════════════════════════════════════════════════════════════════════════════
section("G1 & G2 — Multi-GPU inference runs + numerical consistency with single GPU")
# ════════════════════════════════════════════════════════════════════════════
# Strategy: run AlphaGenome ISM on devices=[0] then devices=[0,1],
# save each CSV, then compare.  All logic is inside run_ism().

ag_csv = os.path.join(ISM_RESULTS_DIR, "alphagenome_ism.csv")
ag_1gpu = os.path.join(ISM_RESULTS_DIR, "alphagenome_ism_1gpu.csv")
ag_2gpu = os.path.join(ISM_RESULTS_DIR, "alphagenome_ism_2gpu.csv")

print("  Running AlphaGenome ISM on single GPU (devices=0) ...")
app1 = GreluTutorialApp(gene="SRSF11", devices="0", num_workers=1)
app1.run_ism("alphagenome")
shutil.copy(ag_csv, ag_1gpu)

print("  Running AlphaGenome ISM on two GPUs (devices=0,1) ...")
app2 = GreluTutorialApp(gene="SRSF11", devices="0,1", num_workers=2)
try:
    app2.run_ism("alphagenome")
    shutil.copy(ag_csv, ag_2gpu)
    check("Multi-GPU ISM completes without error", True)

    df1 = pd.read_csv(ag_1gpu, index_col=0)
    df2 = pd.read_csv(ag_2gpu, index_col=0)
    max_diff = float(np.abs(df1.values - df2.values).max())
    print(f"  Max absolute diff (1-GPU vs 2-GPU) = {max_diff:.2e}")
    check(f"Single vs multi-GPU max diff < {ISM_ATOL}",
          max_diff < ISM_ATOL, f"{max_diff:.2e}")
except Exception as e:
    check("Multi-GPU ISM completes without error", False, str(e)[:120])
    check("Single vs multi-GPU max diff < tolerance", False, "skipped")

# ════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ════════════════════════════════════════════════════════════════════════════
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n  {n_pass} passed,  {n_fail} failed\n")
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

sys.exit(0 if n_fail == 0 else 1)
