"""
Tutorial 1 Comparison: Borzoi vs AlphaGenome
=============================================
Replicates Tutorial 1 (inference) using both Borzoi and AlphaGenome on the
same genomic locus (chr1 around SRSF11). Compares:
  - Brain CAGE predictions
  - Brain RNA-seq predictions
  - SRSF11 brain-vs-liver specificity score

Run: source activate.sh && python tutorial_1_borzoi_vs_alphagenome.py
Output files: comparison_cage.png, comparison_rna.png, comparison_specificity.png
"""

import math
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

import grelu.data.preprocess
import grelu.io.genome
import grelu.resources
import grelu.sequence.format
import grelu.visualize

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch"
    "/snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920"
    "/model_fold_0.safetensors"
)
AG_META_PATH = (
    "src/alphagenome_pytorch/src/alphagenome_pytorch"
    "/data/track_metadata_human.parquet"
)
GENOME = "hg38"
CHROM = "chr1"
DEVICE = 0

# Tutorial-1 exact coordinates
BORZOI_INPUT_START = 69993520

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — Borzoi (Tutorial 1 exact replication)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 1 — Borzoi inference (tutorial coordinates)")
print("═" * 60)

borzoi = grelu.resources.load_model(
    repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
)

tasks_b = pd.DataFrame(borzoi.data_params["tasks"])
borzoi_input_len = borzoi.data_params["train"]["seq_len"]          # 524288
borzoi_bin_size  = borzoi.data_params["train"]["bin_size"]         # 32
borzoi_crop_len  = borzoi.model_params["crop_len"]                 # 5120

borzoi_input_end = BORZOI_INPUT_START + borzoi_input_len           # 70517808
borzoi_center    = (BORZOI_INPUT_START + borzoi_input_end) // 2   # 70255664

borzoi_input_intervals = pd.DataFrame({
    "chrom": [CHROM],
    "start": [BORZOI_INPUT_START],
    "end":   [borzoi_input_end],
    "strand":["+"],
})

print(f"Input  : {CHROM}:{BORZOI_INPUT_START}-{borzoi_input_end} ({borzoi_input_len} bp)")

input_seqs = grelu.sequence.format.convert_input_type(
    borzoi_input_intervals, output_type="strings", genome=GENOME
)

print("Running Borzoi inference …")
borzoi_preds = borzoi.predict_on_seqs(input_seqs, device=DEVICE)
print(f"Predictions shape : {borzoi_preds.shape}  (batch, tasks, bins)")

borzoi_out_intervals = borzoi.input_intervals_to_output_intervals(borzoi_input_intervals)
borzoi_out_start = int(borzoi_out_intervals.start[0])
borzoi_out_end   = int(borzoi_out_intervals.end[0])
print(f"Output : {CHROM}:{borzoi_out_start}-{borzoi_out_end} ({borzoi_out_end - borzoi_out_start} bp)")

# ---- Pick brain CAGE & RNA-seq tracks ----
cage_brain_b = tasks_b[
    (tasks_b.assay == "CAGE") & tasks_b["sample"].str.contains("brain", case=False, na=False)
].head(2)
rna_brain_b  = tasks_b[
    (tasks_b.assay == "RNA")  & tasks_b["sample"].str.contains("brain", case=False, na=False)
].head(2)
rna_liver_b  = tasks_b[
    (tasks_b.assay == "RNA")  & tasks_b["sample"].str.contains("liver", case=False, na=False)
].head(2)

borzoi_cage_idx  = cage_brain_b.index.tolist()
borzoi_rna_brain_idx  = rna_brain_b.index.tolist()
borzoi_rna_liver_idx  = rna_liver_b.index.tolist()

print(f"Brain CAGE tasks  : {borzoi_cage_idx} — {cage_brain_b['name'].tolist()}")
print(f"Brain RNA tasks   : {borzoi_rna_brain_idx} — {rna_brain_b['name'].tolist()}")
print(f"Liver RNA tasks   : {borzoi_rna_liver_idx} — {rna_liver_b['name'].tolist()}")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — AlphaGenome inference
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 2 — AlphaGenome inference")
print("═" * 60)

assert os.path.exists(WEIGHTS_PATH), f"Weights not found: {WEIGHTS_PATH}"
print(f"Weights : {WEIGHTS_PATH}")

from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.model import AlphaGenome

ag_model = AlphaGenome.from_pretrained(
    WEIGHTS_PATH,
    dtype_policy=DtypePolicy.mixed_precision(),
    device=f"cuda:{DEVICE}",
)
ag_model.eval()
print("AlphaGenome model loaded.")

# AlphaGenome input: 131072 bp centred on the Borzoi input centre
AG_INPUT_LEN = 131072
ag_start = borzoi_center - AG_INPUT_LEN // 2   # 70190128
ag_end   = ag_start + AG_INPUT_LEN             # 70321200
print(f"Input  : {CHROM}:{ag_start}-{ag_end} ({AG_INPUT_LEN} bp)")
print(f"Output : same region (no cropping), 1024 bins × 128 bp")

ag_intervals = pd.DataFrame({
    "chrom": [CHROM], "start": [ag_start], "end": [ag_end], "strand": ["+"],
})
ag_seqs = grelu.sequence.format.convert_input_type(
    ag_intervals, output_type="strings", genome=GENOME
)

# One-hot → (1, L, 4) for AlphaGenome
ag_onehot = grelu.sequence.format.convert_input_type(ag_seqs, output_type="one_hot")
# ag_onehot may already be a tensor from grelu; use clone().detach() to avoid the
# "copy construct from tensor" UserWarning
if isinstance(ag_onehot, torch.Tensor):
    ag_tensor = ag_onehot.clone().detach().float().cuda(DEVICE)
else:
    ag_tensor = torch.tensor(np.array(ag_onehot), dtype=torch.float32).cuda(DEVICE)
ag_tensor = ag_tensor.transpose(1, 2)   # (1, L, 4)
print(f"Input tensor : {ag_tensor.shape}")

organism_idx = torch.zeros(1, dtype=torch.long, device=f"cuda:{DEVICE}")

print("Running AlphaGenome inference …")
with torch.no_grad():
    ag_outputs = ag_model.predict(
        ag_tensor, organism_idx,
        channels_last=False,    # → (B, C, L) format
        resolutions=(128,),     # skip expensive 1 bp decoder
    )
print("Inference done.")

# Check output shapes
for key in ["cage", "rna_seq"]:
    t = ag_outputs[key][128]
    print(f"  {key}[128] : {t.shape}")

# ---- Pick matching brain CAGE & RNA-seq tracks via track metadata ----
ag_meta = pd.read_parquet(AG_META_PATH)

cage_meta_h  = ag_meta[ag_meta.output_type == "cage"]
rna_meta_h   = ag_meta[ag_meta.output_type == "rna_seq"]

brain_cage_ag = cage_meta_h[
    cage_meta_h.biosample_name.str.contains("brain", case=False, na=False) &
    ~cage_meta_h.biosample_name.str.contains("vasculature", case=False, na=False)
].head(2)
brain_rna_ag  = rna_meta_h[
    rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)
].head(2)
liver_rna_ag  = rna_meta_h[
    rna_meta_h.biosample_name.str.contains("^liver$", case=False, na=False, regex=True) &
    (rna_meta_h.assay_title == "polyA plus RNA-seq")
].head(2)

ag_cage_idx       = brain_cage_ag.track_index.tolist()
ag_rna_brain_idx  = brain_rna_ag.track_index.tolist()
ag_rna_liver_idx  = liver_rna_ag.track_index.tolist()

print(f"Brain CAGE tracks : {ag_cage_idx} — {brain_cage_ag.track_name.tolist()}")
print(f"Brain RNA tracks  : {ag_rna_brain_idx} — {brain_rna_ag.track_name.tolist()}")
print(f"Liver RNA tracks  : {ag_rna_liver_idx} — {liver_rna_ag.track_name.tolist()}")

# Extract relevant track arrays (CPU numpy)
ag_cage_preds      = ag_outputs["cage"][128][0, ag_cage_idx, :].cpu().numpy()   # (2,1024)
ag_rna_brain_preds = ag_outputs["rna_seq"][128][0, ag_rna_brain_idx, :].cpu().numpy()
ag_rna_liver_preds = ag_outputs["rna_seq"][128][0, ag_rna_liver_idx, :].cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Gene / exon annotations
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 3 — Genomic annotations")
print("═" * 60)

exons = grelu.io.genome.read_gtf(GENOME, features="exon")

# Borzoi annotations (output region)
borzoi_out_ref = pd.DataFrame({
    "chrom": [CHROM], "start": [borzoi_out_start], "end": [borzoi_out_end]
})
borzoi_exons = grelu.data.preprocess.filter_overlapping(
    exons, ref_intervals=borzoi_out_ref, method="any"
)
borzoi_exons = grelu.data.preprocess.clip_intervals(
    borzoi_exons, start=borzoi_out_start, end=borzoi_out_end
)
borzoi_genes = grelu.data.preprocess.merge_intervals_by_column(
    borzoi_exons, group_col="gene_name"
)

# AlphaGenome annotations (full input/output region)
ag_ref = pd.DataFrame({
    "chrom": [CHROM], "start": [ag_start], "end": [ag_end]
})
ag_exons = grelu.data.preprocess.filter_overlapping(
    exons, ref_intervals=ag_ref, method="any"
)
ag_exons = grelu.data.preprocess.clip_intervals(ag_exons, start=ag_start, end=ag_end)
ag_genes  = grelu.data.preprocess.merge_intervals_by_column(
    ag_exons, group_col="gene_name"
)

print(f"Borzoi output region genes : {borzoi_genes.gene_name.tolist()}")
print(f"AlphaGenome region genes   : {ag_genes.gene_name.tolist()}")

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — Comparison plots
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 4 — Plots")
print("═" * 60)

# Borzoi CAGE track predictions for brain
borzoi_cage_preds = borzoi_preds[0, borzoi_cage_idx, :]   # (2, 6144)
borzoi_cage_names = (
    tasks_b.name[borzoi_cage_idx] + " " + tasks_b.description[borzoi_cage_idx]
).tolist()
ag_cage_track_names = brain_cage_ag.track_name.tolist()

borzoi_rna_brain_preds = borzoi_preds[0, borzoi_rna_brain_idx, :]
borzoi_rna_names = (
    tasks_b.name[borzoi_rna_brain_idx] + " " + tasks_b.description[borzoi_rna_brain_idx]
).tolist()
ag_rna_brain_names = brain_rna_ag.track_name.tolist()

# ---- Figure 1: Brain CAGE (Borzoi | AlphaGenome) ----
fig, axes = plt.subplots(2, 2, figsize=(18, 6), constrained_layout=True)
fig.suptitle("Brain CAGE Predictions — Borzoi (left) vs AlphaGenome (right)", fontsize=13)

for row in range(2):
    # Borzoi
    ax = axes[row, 0]
    ax.fill_between(
        range(len(borzoi_cage_preds[row])),
        borzoi_cage_preds[row],
        color="steelblue", alpha=0.7
    )
    ax.set_title(f"Borzoi — {borzoi_cage_names[row]}", fontsize=9)
    ax.set_xlabel(f"{CHROM}:{borzoi_out_start}–{borzoi_out_end}  (32 bp bins)")
    ax.set_ylabel("Predicted CAGE signal")
    ax.set_xlim(0, borzoi_cage_preds.shape[1])

    # AlphaGenome
    ax = axes[row, 1]
    ax.fill_between(
        range(len(ag_cage_preds[row])),
        ag_cage_preds[row],
        color="darkorange", alpha=0.7
    )
    ax.set_title(f"AlphaGenome — {ag_cage_track_names[row]}", fontsize=9)
    ax.set_xlabel(f"{CHROM}:{ag_start}–{ag_end}  (128 bp bins)")
    ax.set_ylabel("Predicted CAGE signal")
    ax.set_xlim(0, ag_cage_preds.shape[1])

plt.savefig("comparison_cage.png", dpi=150)
print("Saved comparison_cage.png")
plt.close()

# ---- Figure 2: Brain RNA-seq (Borzoi | AlphaGenome) ----
fig, axes = plt.subplots(2, 2, figsize=(18, 6), constrained_layout=True)
fig.suptitle("Brain RNA-seq Predictions — Borzoi (left) vs AlphaGenome (right)", fontsize=13)

for row in range(2):
    ax = axes[row, 0]
    ax.fill_between(
        range(len(borzoi_rna_brain_preds[row])),
        borzoi_rna_brain_preds[row],
        color="steelblue", alpha=0.7
    )
    ax.set_title(f"Borzoi — {borzoi_rna_names[row]}", fontsize=9)
    ax.set_xlabel(f"{CHROM}:{borzoi_out_start}–{borzoi_out_end}  (32 bp bins)")
    ax.set_ylabel("Predicted RNA-seq signal")
    ax.set_xlim(0, borzoi_rna_brain_preds.shape[1])

    ax = axes[row, 1]
    ax.fill_between(
        range(len(ag_rna_brain_preds[row])),
        ag_rna_brain_preds[row],
        color="darkorange", alpha=0.7
    )
    ax.set_title(f"AlphaGenome — {ag_rna_brain_names[row]}", fontsize=9)
    ax.set_xlabel(f"{CHROM}:{ag_start}–{ag_end}  (128 bp bins)")
    ax.set_ylabel("Predicted RNA-seq signal")
    ax.set_xlim(0, ag_rna_brain_preds.shape[1])

plt.savefig("comparison_rna.png", dpi=150)
print("Saved comparison_rna.png")
plt.close()

# ---- Figure 3: Overlap region side-by-side (same genomic x-axis) ----
# Borzoi bins that fall inside the AlphaGenome window [ag_start, ag_end]
borzoi_overlap_start_bin = math.ceil((ag_start - borzoi_out_start) / borzoi_bin_size)
borzoi_overlap_end_bin   = math.floor((ag_end   - borzoi_out_start) / borzoi_bin_size)
print(f"\nOverlap Borzoi bins : {borzoi_overlap_start_bin} – {borzoi_overlap_end_bin}")

# Genomic coords for the overlap bins
n_borzoi_overlap  = borzoi_overlap_end_bin - borzoi_overlap_start_bin
n_ag_overlap      = ag_cage_preds.shape[1]   # all 1024 bins

borzoi_cage_overlap = borzoi_cage_preds[:, borzoi_overlap_start_bin:borzoi_overlap_end_bin]
ag_cage_overlap     = ag_cage_preds

# Borzoi bin edges in genomic coords
borzoi_xcoords = np.arange(n_borzoi_overlap) * borzoi_bin_size + ag_start
ag_xcoords     = np.arange(n_ag_overlap) * 128 + ag_start

fig, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
fig.suptitle(
    f"Overlap region  {CHROM}:{ag_start}–{ag_end}\n"
    "Borzoi (blue) vs AlphaGenome (orange)",
    fontsize=12
)

track_pairs = [
    ("CAGE track 1",    borzoi_cage_overlap[0],         ag_cage_overlap[0]),
    ("CAGE track 2",    borzoi_cage_overlap[1],         ag_cage_overlap[1]),
    ("RNA-seq track 1", borzoi_rna_brain_preds[0, borzoi_overlap_start_bin:borzoi_overlap_end_bin],
                        ag_rna_brain_preds[0]),
    ("RNA-seq track 2", borzoi_rna_brain_preds[1, borzoi_overlap_start_bin:borzoi_overlap_end_bin],
                        ag_rna_brain_preds[1]),
]

for ax, (title, b_vals, ag_vals) in zip(axes, track_pairs):
    ax2 = ax.twinx()
    ax.fill_between(borzoi_xcoords, b_vals,  color="steelblue",  alpha=0.6, label="Borzoi (left y)")
    ax2.fill_between(ag_xcoords,    ag_vals, color="darkorange", alpha=0.6, label="AlphaGenome (right y)")
    ax.set_ylabel("Borzoi signal",     color="steelblue")
    ax2.set_ylabel("AlphaGenome signal", color="darkorange")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title(title)
    ax.set_xlim(ag_start, ag_end)
    # Legend
    lines = [
        plt.Line2D([0], [0], color="steelblue",  lw=4, label="Borzoi"),
        plt.Line2D([0], [0], color="darkorange", lw=4, label="AlphaGenome"),
    ]
    ax.legend(handles=lines, loc="upper right", fontsize=8)

axes[-1].set_xlabel(f"Genomic coordinate ({CHROM})")
plt.savefig("comparison_overlap.png", dpi=150)
print("Saved comparison_overlap.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — SRSF11 brain-vs-liver specificity score
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SECTION 5 — SRSF11 brain-vs-liver RNA-seq specificity score")
print("═" * 60)

srsf11_exons = exons[exons.gene_name == "SRSF11"].copy()
print(f"SRSF11 exons: {len(srsf11_exons)}")
print(f"  Range: {srsf11_exons.start.min()} – {srsf11_exons.end.max()}")

# ---- Borzoi: map SRSF11 exons to 32 bp bins ----
srsf11_exon_bins_b = borzoi.input_intervals_to_output_bins(
    srsf11_exons, start_pos=BORZOI_INPUT_START
)
selected_bins_b = set()
for row in srsf11_exon_bins_b.itertuples():
    selected_bins_b.update(range(row.start, row.end))
selected_bins_b = sorted(selected_bins_b)
print(f"\nBorzoi SRSF11 bins : {len(selected_bins_b)}")

borzoi_rna_liver_preds = borzoi_preds[0, borzoi_rna_liver_idx, :]
brain_liver_b = tasks_b[
    (tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("liver", case=False, na=False)
].head(2)
borzoi_rna_liver_idx2 = brain_liver_b.index.tolist()
borzoi_rna_liver_preds_full = borzoi_preds[0, borzoi_rna_liver_idx2, :]

brain_signal_b = borzoi_rna_brain_preds[:, selected_bins_b].mean()
liver_signal_b = borzoi_rna_liver_preds_full[:, selected_bins_b].mean()
borzoi_specificity = brain_signal_b / (liver_signal_b + 1e-10)
print(f"  Brain RNA-seq signal (mean over exons) : {brain_signal_b:.4f}")
print(f"  Liver RNA-seq signal (mean over exons) : {liver_signal_b:.4f}")
print(f"  Brain/Liver specificity ratio          : {borzoi_specificity:.4f}  (tutorial expects ~1.6x)")

# ---- AlphaGenome: map SRSF11 exons to 128 bp bins ----
# Only exons within the AlphaGenome window
srsf11_in_ag = srsf11_exons[
    (srsf11_exons.start < ag_end) & (srsf11_exons.end > ag_start)
].copy()
print(f"\nAlphaGenome SRSF11 exons in window: {len(srsf11_in_ag)}")

AG_BIN_SIZE = 128
selected_bins_ag = set()
for _, row in srsf11_in_ag.iterrows():
    start_bin = max(0, math.floor((row.start - ag_start) / AG_BIN_SIZE))
    end_bin   = min(1024, math.ceil((row.end   - ag_start) / AG_BIN_SIZE))
    selected_bins_ag.update(range(start_bin, end_bin))
selected_bins_ag = sorted(selected_bins_ag)
print(f"AlphaGenome SRSF11 bins : {len(selected_bins_ag)}")

if selected_bins_ag:
    brain_signal_ag = ag_rna_brain_preds[:, selected_bins_ag].mean()
    liver_signal_ag = ag_rna_liver_preds[:, selected_bins_ag].mean()
    ag_specificity  = brain_signal_ag / (liver_signal_ag + 1e-10)
    print(f"  Brain RNA-seq signal (mean over exons) : {brain_signal_ag:.4f}")
    print(f"  Liver RNA-seq signal (mean over exons) : {liver_signal_ag:.4f}")
    print(f"  Brain/Liver specificity ratio          : {ag_specificity:.4f}")
else:
    print("  WARNING: No SRSF11 exons overlap the AlphaGenome window — skipping ratio.")
    ag_specificity = None

# ---- Specificity comparison bar chart ----
scores = {"Borzoi": float(borzoi_specificity)}
if ag_specificity is not None:
    scores["AlphaGenome"] = float(ag_specificity)

fig, ax = plt.subplots(figsize=(5, 4))
colors = ["steelblue", "darkorange"]
bars = ax.bar(list(scores.keys()), list(scores.values()),
              color=colors[:len(scores)], edgecolor="black", width=0.5)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Equal (ratio=1)")
ax.set_ylabel("Brain RNA-seq / Liver RNA-seq\n(mean signal over SRSF11 exons)")
ax.set_title("SRSF11 Brain-vs-Liver Specificity Score\n(Tutorial target: ~1.6×)")
for bar, val in zip(bars, scores.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(list(scores.values())) * 1.3)
ax.legend()
plt.tight_layout()
plt.savefig("comparison_specificity.png", dpi=150)
print("\nSaved comparison_specificity.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)
print(f"Borzoi SRSF11 brain/liver specificity   : {borzoi_specificity:.2f}× (tutorial expects ~1.6×)")
if ag_specificity is not None:
    verdict = "✓ MATCHES or BEATS Borzoi" if ag_specificity >= borzoi_specificity * 0.9 else "✗ Below Borzoi"
    print(f"AlphaGenome SRSF11 brain/liver spec.    : {ag_specificity:.2f}×  {verdict}")
print()
print("Output files:")
for f in ["comparison_cage.png", "comparison_rna.png",
          "comparison_overlap.png", "comparison_specificity.png"]:
    exists = os.path.exists(f)
    print(f"  {'✓' if exists else '✗'}  {f}")
