import os
import argparse
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

import grelu.resources
import grelu.sequence.format
import grelu.interpret.score
import grelu.transforms.prediction_transforms
import grelu.io.genome
import grelu.io.bigwig
import grelu.data.preprocess
import grelu.visualize
from grelu.data.dataset import SeqDataset, BigWigSeqDataset
from grelu.lightning import LightningModel
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.metrics import pearson_r as _pearson_r

# --- 全局配置 ---
ISM_RESULTS_DIR = "ism_results"
os.makedirs(ISM_RESULTS_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/"
    "snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/model_fold_0.safetensors"
)
AG_META_PATH = "src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet"

# ── Architecture constants (biological / model-spec requirements) ─────────────
# Changing these constants changes the biological assay — update tests accordingly.
BORZOI_INPUT_LEN = 524_288   # Borzoi receptive field in bp
BORZOI_BIN_SIZE  = 32        # Borzoi output bin resolution in bp
AG_INPUT_LEN     = 131_072   # AlphaGenome receptive field in bp
AG_BIN_SIZE      = 128       # AlphaGenome output bin resolution in bp
AG_OUTPUT_BINS   = AG_INPUT_LEN // AG_BIN_SIZE   # 1024
ISM_HALF_WIDTH   = 100       # ±100 bp window around ISM centre


# ── Pure helper functions (module-level so tests can import directly) ─────────

def _parse_variant(v: str) -> tuple[str, int, str, str]:
    """Parse 'chr1_70355119_G_A' into (chrom, pos, ref, alt)."""
    parts = v.split("_")
    return parts[0], int(parts[1]), parts[2], parts[3]


def _stats(r_arr: np.ndarray) -> dict:
    """Summary statistics over a per-window Pearson R array; ignores NaN/Inf."""
    v = r_arr[np.isfinite(r_arr)]
    return {
        "mean_r":    float(np.mean(v))   if len(v) else float("nan"),
        "median_r":  float(np.median(v)) if len(v) else float("nan"),
        "std_r":     float(np.std(v))    if len(v) else float("nan"),
        "n_windows": int(len(v)),
    }


def _bin_obs(obs_raw: np.ndarray, bin_size: int, n_pred_bins: int) -> np.ndarray:
    """Bin base-resolution observations then centre-crop to match model output length.

    Args:
        obs_raw:     (N, seq_len) base-resolution signal array.
        bin_size:    Number of bases per output bin.
        n_pred_bins: Number of bins in the model prediction (crop target).

    Returns:
        (N, n_pred_bins) binned and cropped array.
    """
    n, total = obs_raw.shape
    obs = obs_raw.reshape(n, total // bin_size, bin_size).mean(axis=-1)
    n_obs_bins = obs.shape[1]
    if n_pred_bins != n_obs_bins:
        crop_start = (n_obs_bins - n_pred_bins) // 2
        obs = obs[:, crop_start : crop_start + n_pred_bins]
    return obs


class GreluTutorialApp:
    def __init__(self, gene="SRSF11", genome="hg38", devices="0,1,2,3", num_workers=1):
        self.gene = gene
        self.genome = genome
        
        if devices == "cpu":
            self.devices = "cpu"
            self.inference_device = "cpu"
        else:
            self.devices = [int(x) for x in devices.split(',')]
            self.inference_device = self.devices[0]
            
        self.num_workers = num_workers
        
        # 数据缓存
        self.exons = None
        self.input_seqs = None
        self.ag_seqs = None
        self.target_exons = None

    def _cleanup(self):
        """强制清理显存和内存引用"""
        print("Explicitly clearing GPU memory and garbage collecting...")
        if hasattr(self, 'borzoi'): del self.borzoi
        if hasattr(self, 'ag_rna'): del self.ag_rna
        if hasattr(self, 'ag_cage'): del self.ag_cage
        
        self.borzoi = None
        self.ag_rna = None
        self.ag_cage = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Cleanup complete.")

    def setup(self):
        if self.input_seqs is not None:
            return
            
        print(f"Setting up sequences for gene {self.gene}...")
        if self.exons is None:
            self.exons = grelu.io.genome.read_gtf(self.genome, features="exon")
        
        self.target_exons = self.exons[self.exons.gene_name == self.gene].copy()
        if len(self.target_exons) == 0:
            raise ValueError(f"Gene {self.gene} not found.")
            
        self.chrom = self.target_exons.chrom.iloc[0]
        self.ism_center = int(self.target_exons.start.min())

        self.borzoi_start_coord = self.ism_center - BORZOI_INPUT_LEN // 2

        self.ag_start_coord = self.ism_center - AG_INPUT_LEN // 2
        self.ag_end_coord   = self.ag_start_coord + AG_INPUT_LEN

        b_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.borzoi_start_coord], "end": [self.borzoi_start_coord + BORZOI_INPUT_LEN], "strand": ["+"]})
        self.input_seqs = grelu.sequence.format.convert_input_type(b_intervals, output_type="strings", genome=self.genome)[0]

        ag_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.ag_start_coord], "end": [self.ag_end_coord], "strand": ["+"]})
        self.ag_seqs = grelu.sequence.format.convert_input_type(ag_intervals, output_type="strings", genome=self.genome)[0]

        self.ism_region = {
            "start": self.ism_center - ISM_HALF_WIDTH,
            "end":   self.ism_center + ISM_HALF_WIDTH,
        }

    def setup_borzoi(self):
        print("Loading Borzoi model...")
        self.borzoi = grelu.resources.load_model(repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt")
        borzoi_input_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.borzoi_start_coord], "end": [self.borzoi_start_coord + BORZOI_INPUT_LEN], "strand": ["+"]})
        borzoi_out_intervals = self.borzoi.input_intervals_to_output_intervals(borzoi_input_intervals)
        self.borzoi_out_start = int(borzoi_out_intervals.start[0])
        self.borzoi_bin_size = BORZOI_BIN_SIZE

    def setup_ag_rna(self):
        print("Loading AlphaGenome RNA model...")
        ag_params = dict(weights_path=WEIGHTS_PATH, dtype_policy=DtypePolicy.mixed_precision(), resolution=128)
        self.ag_rna = LightningModel(
            model_params={"model_type": "AlphaGenomeModel", "output_key": "rna_seq", **ag_params},
            train_params={"task": "regression", "loss": "mse"},
        )
        self.ag_rna.data_params["train"] = {"seq_len": 131072, "bin_size": 128}
        self.ag_rna.model_params["crop_len"] = 0

    def setup_ag_cage(self):
        print("Loading AlphaGenome CAGE model...")
        ag_params = dict(weights_path=WEIGHTS_PATH, dtype_policy=DtypePolicy.mixed_precision(), resolution=128)
        self.ag_cage = LightningModel(
            model_params={"model_type": "AlphaGenomeModel", "output_key": "cage", **ag_params},
            train_params={"task": "regression", "loss": "mse"},
        )
        self.ag_cage.data_params["train"] = {"seq_len": 131072, "bin_size": 128}
        self.ag_cage.model_params["crop_len"] = 0

    def get_borzoi_transform(self):
        tasks_b = pd.DataFrame(self.borzoi.data_params["tasks"])
        b_on = tasks_b[(tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("brain", case=False, na=False)].index.tolist()
        b_off = tasks_b[(tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("liver", case=False, na=False)].index.tolist()
        b_bins = self.borzoi.input_intervals_to_output_bins(self.target_exons, start_pos=self.borzoi_start_coord)
        b_pos = sorted(list(set(sum([list(range(row.start, row.end)) for row in b_bins.itertuples()], []))))
        return grelu.transforms.prediction_transforms.Specificity(
            on_tasks=b_on, off_tasks=b_off, positions=b_pos, on_aggfunc="mean", off_aggfunc="mean", length_aggfunc="mean", compare_func="divide"
        )

    def get_ag_transform(self, model):
        ag_meta = pd.read_parquet(AG_META_PATH)
        rna_meta_h = ag_meta[ag_meta.output_type == "rna_seq"]
        ag_on = rna_meta_h[rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_index.tolist()
        ag_off = rna_meta_h[rna_meta_h.biosample_name.str.contains("liver", case=False, na=False) & (rna_meta_h.assay_title == "polyA plus RNA-seq")].track_index.tolist()
        ag_bins = model.input_intervals_to_output_bins(self.target_exons[(self.target_exons.start < self.ag_start_coord + 131072) & (self.target_exons.end > self.ag_start_coord)], start_pos=self.ag_start_coord)
        ag_pos = sorted(list(set(sum([list(range(max(0, row.start), min(1024, row.end))) for row in ag_bins.itertuples()], []))))
        return grelu.transforms.prediction_transforms.Specificity(
            on_tasks=ag_on, off_tasks=ag_off, positions=ag_pos, on_aggfunc="mean", off_aggfunc="mean", length_aggfunc="mean", compare_func="divide"
        )

    def run_inference(self):
        self.setup()
        
        # 1. Borzoi
        self.setup_borzoi()
        print(f"\nRunning Borzoi inference on {self.inference_device}...")
        borzoi_preds = self.borzoi.predict_on_seqs(self.input_seqs, device=self.inference_device)
        b_trans = self.get_borzoi_transform()
        borzoi_specificity = float(b_trans.compute(borzoi_preds).ravel()[0])
        
        tasks_b = pd.DataFrame(self.borzoi.data_params["tasks"])
        borzoi_cage_idx = tasks_b[(tasks_b.assay == "CAGE") & tasks_b["sample"].str.contains("brain", case=False, na=False)].head(2).index.tolist()
        borzoi_rna_brain_idx = tasks_b[(tasks_b.assay == "RNA") & tasks_b["sample"].str.contains("brain", case=False, na=False)].index.tolist()
        borzoi_cage_preds = borzoi_preds[0, borzoi_cage_idx, :]
        borzoi_rna_brain_preds = borzoi_preds[0, borzoi_rna_brain_idx, :]
        
        # 保存这些值用于绘图，然后清理模型
        self.borzoi_plot_data = {
            "cage": borzoi_cage_preds, "rna": borzoi_rna_brain_preds, "spec": borzoi_specificity,
            "cage_names": (tasks_b.name[borzoi_cage_idx] + " " + tasks_b.description[borzoi_cage_idx]).tolist(),
            "rna_names": (tasks_b.name[borzoi_rna_brain_idx] + " " + tasks_b.description[borzoi_rna_brain_idx]).tolist()
        }
        self._cleanup()

        # 2. AG RNA
        self.setup_ag_rna()
        print(f"\nRunning AlphaGenome RNA inference on {self.inference_device}...")
        ag_rna_preds = self.ag_rna.predict_on_seqs(self.ag_seqs, device=self.inference_device)
        ag_trans = self.get_ag_transform(self.ag_rna)
        ag_spec = float(ag_trans.compute(ag_rna_preds).ravel()[0])
        
        ag_meta = pd.read_parquet(AG_META_PATH)
        rna_meta_h = ag_meta[ag_meta.output_type == "rna_seq"]
        brain_rna_idx = rna_meta_h[rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_index.tolist()
        ag_rna_brain_preds_vals = ag_rna_preds[0, brain_rna_idx, :]
        
        self.ag_rna_plot_data = {
            "rna": ag_rna_brain_preds_vals, "spec": ag_spec, 
            "rna_names": rna_meta_h[rna_meta_h.biosample_name.str.contains("brain", case=False, na=False)].track_name.tolist()
        }
        self._cleanup()

        # 3. AG CAGE
        self.setup_ag_cage()
        print(f"\nRunning AlphaGenome CAGE inference on {self.inference_device}...")
        ag_cage_preds_full = self.ag_cage.predict_on_seqs(self.ag_seqs, device=self.inference_device)
        cage_meta_h = ag_meta[ag_meta.output_type == "cage"]
        brain_cage_idx = cage_meta_h[cage_meta_h.biosample_name.str.contains("brain", case=False, na=False) & ~cage_meta_h.biosample_name.str.contains("vasculature", case=False, na=False)].head(2).track_index.tolist()
        ag_cage_preds_vals = ag_cage_preds_full[0, brain_cage_idx, :]
        
        self.ag_cage_plot_data = {
            "cage": ag_cage_preds_vals,
            "cage_names": cage_meta_h[cage_meta_h.biosample_name.str.contains("brain", case=False, na=False) & ~cage_meta_h.biosample_name.str.contains("vasculature", case=False, na=False)].head(2).track_name.tolist()
        }
        self._cleanup()

        # 4. 绘图
        print("\nGenerating Comparison Plots...")
        # (绘图逻辑保持一致，使用刚才缓存的 data 对象)
        fig, axes = plt.subplots(2, 2, figsize=(18, 6), constrained_layout=True)
        for row in range(2):
            axes[row, 0].fill_between(range(self.borzoi_plot_data["cage"].shape[1]), self.borzoi_plot_data["cage"][row], color="steelblue", alpha=0.7)
            axes[row, 0].set_title(f"Borzoi — {self.borzoi_plot_data['cage_names'][row]}", fontsize=8)
            axes[row, 1].fill_between(range(self.ag_cage_plot_data["cage"].shape[1]), self.ag_cage_plot_data["cage"][row], color="darkorange", alpha=0.7)
            axes[row, 1].set_title(f"AlphaGenome — {self.ag_cage_plot_data['cage_names'][row]}", fontsize=8)
        plt.savefig("comparison_cage.png", dpi=150); plt.close()

        # Specificity
        scores = {"Borzoi": self.borzoi_plot_data["spec"], "AlphaGenome": self.ag_rna_plot_data["spec"]}
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(scores.keys()), list(scores.values()), color=["steelblue", "darkorange"], edgecolor="black", width=0.5)
        ax.axhline(y=1.0, color="gray", linestyle="--")
        ax.set_title(f"{self.gene} Brain-vs-Liver Specificity Ratio")
        plt.savefig("comparison_specificity.png", dpi=150); plt.close()
        print("Done. Inference results saved.")

    def run_ism(self, model_name):
        self.setup()
        self._cleanup() # 确保开始前绝对干净
        
        if model_name == "borzoi":
            self.setup_borzoi()
            trans = self.get_borzoi_transform()
            model_obj = self.borzoi
            seqs_obj = self.input_seqs
            s_pos = self.ism_region['start'] - self.borzoi_start_coord
            e_pos = self.ism_region['end'] - self.borzoi_start_coord
        else:
            self.setup_ag_rna()
            trans = self.get_ag_transform(self.ag_rna)
            model_obj = self.ag_rna
            seqs_obj = self.ag_seqs
            s_pos = self.ism_region['start'] - self.ag_start_coord
            e_pos = self.ism_region['end'] - self.ag_start_coord

        print(f"\nRunning {model_name} ISM on {self.devices}...")
        res = grelu.interpret.score.ISM_predict(
            seqs=seqs_obj, model=model_obj, prediction_transform=trans,
            devices=self.devices, num_workers=self.num_workers, batch_size=1,
            start_pos=s_pos, end_pos=e_pos, compare_func="log2FC", return_df=True
        )
        res.to_csv(os.path.join(ISM_RESULTS_DIR, f"{model_name}_ism.csv"))
        self._cleanup()

    # ── Genome-wide Pred-vs-Obs Pearson R ─────────────────────────────────────

    def _make_windows(
        self,
        chroms: list,
        seq_len: int,
        stride: int,
        max_windows: int | None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sliding-window intervals on the requested chromosomes."""
        sizes = grelu.io.genome.read_sizes(self.genome).set_index("chrom")["size"].to_dict()
        rows = []
        for chrom in chroms:
            chrom_len = sizes.get(chrom)
            if chrom_len is None:
                print(f"  [warn] {chrom} not found in chrom sizes — skipping")
                continue
            start = 0
            while start + seq_len <= chrom_len:
                rows.append({"chrom": chrom, "start": start,
                              "end": start + seq_len, "strand": "+"})
                start += stride
        df = pd.DataFrame(rows)
        print(f"  Generated {len(df)} windows across {chroms}")
        if max_windows is not None and len(df) > max_windows:
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(df), max_windows, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
            print(f"  Sub-sampled to {len(df)} windows (seed={seed})")
        return df

    def run_genome_wide_eval(
        self,
        model_name: str,
        bigwig_path: str,
        track_idx: int,
        output_key: str = "rna_seq",
        chroms: list | None = None,
        stride: int | None = None,
        max_windows: int = 300,
        batch_size: int = 4,
        seed: int = 42,
        min_obs_mean: float = 0.05,
        save_per_window: str | None = None,
    ) -> dict:
        """Pred-vs-Obs profile Pearson R on held-out chromosomes.

        Slides a window across *chroms*, runs model inference, reads the
        experimental BigWig signal, bins both to the same resolution, and
        computes per-window Pearson R.

        Args:
            model_name:       "borzoi" or "alphagenome".
            bigwig_path:      Path to the observed-signal BigWig file.
            track_idx:        Zero-based index of the output track to compare.
            output_key:       AlphaGenome head key (rna_seq, cage, …).
            chroms:           Chromosomes to evaluate (default: chr1, chr8, chr21).
            stride:           Window stride in bp (default: seq_len, no overlap).
            max_windows:      Cap on number of windows evaluated (0 = no cap).
            batch_size:       Inference batch size.
            seed:             Random seed for sub-sampling.
            save_per_window:  Optional CSV path for per-window Pearson R table.

        Returns:
            Dict with mean_r, median_r, std_r, n_windows, per_chrom.
        """
        self._cleanup()

        if chroms is None:
            chroms = ["chr1", "chr8", "chr21"]

        if model_name == "borzoi":
            print("Loading Borzoi model...")
            self.borzoi = grelu.resources.load_model(
                repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
            )
            model_obj = self.borzoi
            seq_len = 524_288
            bin_size = 32
        else:
            # Re-create the model with the requested output key so we can use
            # a different head (cage vs rna_seq) without touching setup_ag_rna.
            from alphagenome_pytorch.config import DtypePolicy as _DP
            ag_params = dict(
                weights_path=WEIGHTS_PATH,
                dtype_policy=_DP.mixed_precision(),
                resolution=128,
            )
            self.ag_rna = LightningModel(
                model_params={"model_type": "AlphaGenomeModel",
                              "output_key": output_key, **ag_params},
                train_params={"task": "regression", "loss": "mse"},
            )
            self.ag_rna.data_params["train"] = {"seq_len": 131072, "bin_size": 128}
            self.ag_rna.model_params["crop_len"] = 0
            model_obj = self.ag_rna
            seq_len = 131_072
            bin_size = 128

        if stride is None:
            stride = seq_len

        print(f"\nGenerating windows: chroms={chroms}, seq_len={seq_len:,}, stride={stride:,}")
        max_w = max_windows if max_windows > 0 else None
        windows = self._make_windows(chroms, seq_len, stride, max_w, seed)
        if len(windows) == 0:
            raise ValueError("No windows generated — check chromosome names.")

        # ── BigWigSeqDataset：序列 + 观测信号一次加载 ─────────────────────────
        print(f"Building BigWigSeqDataset (seq={seq_len}bp, bin={bin_size}bp) …")
        bw_files = [bigwig_path] if isinstance(bigwig_path, str) else bigwig_path
        dataset = BigWigSeqDataset(
            intervals=windows,
            bw_files=bw_files,
            seq_len=seq_len,
            genome=self.genome,
            bin_size=bin_size,
            label_aggfunc="mean",
        )
        # dataset.labels 是 base-resolution (N, 1, seq_len)
        # 需要手动 bin 到模型输出分辨率 (N, n_bins)，然后中心裁剪对齐预测长度
        obs_raw = dataset.labels[:, 0, :]  # (N, seq_len)

        # ── model predictions ─────────────────────────────────────────────────
        print(f"Running {model_name} inference on {len(windows)} windows (batch={batch_size}) …")
        preds_all = model_obj.predict_on_dataset(
            dataset,
            devices=self.devices,
            num_workers=self.num_workers,
            batch_size=batch_size,
        )  # (N, n_tracks, n_bins)

        self._cleanup()

        preds_track = preds_all[:, track_idx, :]  # (N, n_pred_bins)

        # ── bin + 中心裁剪（使用模块级 _bin_obs）─────────────────────────────
        obs = _bin_obs(obs_raw, bin_size=bin_size, n_pred_bins=preds_track.shape[1])

        # ── Pearson R per window ──────────────────────────────────────────────
        pred_t = torch.as_tensor(preds_track, dtype=torch.float32)
        obs_t  = torch.as_tensor(obs,          dtype=torch.float32)
        per_window_r = _pearson_r(pred_t, obs_t, dim=-1).numpy()  # (N,)

        windows_r = windows.copy()
        windows_r["pearson_r"] = per_window_r
        windows_r["obs_mean"]  = obs.mean(axis=-1)  # 每窗口观测信号均值

        all_valid = per_window_r[np.isfinite(per_window_r)]
        # 只保留有信号的窗口（obs_mean > min_obs_mean）
        signal_mask = np.isfinite(per_window_r) & (windows_r["obs_mean"].values > min_obs_mean)
        signal_r    = per_window_r[signal_mask]

        summary = {
            "model":      model_name,
            "output_key": output_key,
            "track_idx":  track_idx,
            "bigwig":     bigwig_path,
            "min_obs_mean_filter": min_obs_mean,
            # 全窗口（含空白区）
            "all_windows":    _stats(per_window_r),
            # 仅有信号窗口
            "signal_windows": _stats(signal_r),
            "n_nan": int(len(per_window_r) - len(all_valid)),
        }

        per_chrom = {}
        for chrom, grp in windows_r.groupby("chrom"):
            per_chrom[str(chrom)] = {
                "all":    _stats(grp["pearson_r"].values),
                "signal": _stats(grp.loc[grp["obs_mean"] > min_obs_mean, "pearson_r"].values),
            }
        summary["per_chrom"] = per_chrom

        # ── report ────────────────────────────────────────────────────────────
        aw = summary["all_windows"]
        sw = summary["signal_windows"]
        print(f"\n{'='*60}")
        print(f"  Model      : {model_name}  track_idx={track_idx}")
        print(f"  [全部窗口]  mean_r={aw['mean_r']:.4f}  median_r={aw['median_r']:.4f}  n={aw['n_windows']}")
        print(f"  [有信号窗口 obs_mean>{min_obs_mean}]  mean_r={sw['mean_r']:.4f}  median_r={sw['median_r']:.4f}  n={sw['n_windows']}")
        for chrom, st in sorted(per_chrom.items()):
            print(f"    {chrom:10s}  all mean_r={st['all']['mean_r']:.4f}  "
                  f"signal mean_r={st['signal']['mean_r']:.4f}  "
                  f"signal_n={st['signal']['n_windows']}")
        print(f"{'='*60}")

        if save_per_window:
            windows_r.to_csv(save_per_window, index=False)
            print(f"Per-window table → {save_per_window}")

        return summary

    def run_eqtl_auprc(
        self,
        cs_files: list,
        tissue: str = "brain",
        min_pip_causal: float = 0.5,
        min_cs_size: int = 5,
        max_loci: int = 200,
        model_names: list = None,
        output_prefix: str = "eqtl_auprc",
    ):
        """
        Evaluate AUPRC of variant effect scores against eQTL fine-mapping PIPs.

        For each credible set, variants are ranked by model |log2FC| score and
        AUPRC is computed using PIP >= min_pip_causal as the causal label.
        Reports mean AUPRC across loci and saves per-variant scores to CSV.

        Args:
            cs_files:        List of eQTL Catalogue credible_sets.tsv.gz file paths.
            tissue:          'brain' or 'blood' — selects relevant tracks.
            min_pip_causal:  PIP threshold for labeling a variant as causal.
            min_cs_size:     Minimum credible set size (filters noisy singletons).
            max_loci:        Cap on number of loci (randomly sampled for speed).
            model_names:     ['borzoi', 'alphagenome'] or a subset.
            output_prefix:   Prefix for output CSV / TXT files (in ISM_RESULTS_DIR).
        """
        from sklearn.metrics import average_precision_score
        from grelu.variant import predict_variant_effects
        from grelu.transforms.prediction_transforms import Aggregate

        if model_names is None:
            model_names = ["borzoi", "alphagenome"]

        # ── 1. 读取 + 过滤 credible sets ────────────────────────────────────
        print("\n[eQTL-AUPRC] Loading credible sets …")
        dfs = [pd.read_csv(f, sep="\t", compression="gzip") for f in cs_files]
        cs = pd.concat(dfs, ignore_index=True)
        print(f"  Total variants loaded : {len(cs)}")

        cs_max = cs.groupby(["gene_id", "cs_id"])["pip"].max().reset_index()
        cs_max.columns = ["gene_id", "cs_id", "max_pip"]
        cs_size = cs.groupby(["gene_id", "cs_id"])["pip"].count().reset_index()
        cs_size.columns = ["gene_id", "cs_id", "cs_size"]
        cs_info = cs_max.merge(cs_size)
        good = cs_info[
            (cs_info.max_pip >= min_pip_causal) & (cs_info.cs_size >= min_cs_size)
        ]
        print(f"  Loci after filtering  : {len(good)} "
              f"(max_pip>={min_pip_causal}, cs_size>={min_cs_size})")

        if len(good) > max_loci:
            good = good.sample(max_loci, random_state=42)
            print(f"  Sub-sampled to        : {max_loci} loci")

        cs_filtered = cs.merge(good[["gene_id", "cs_id"]], on=["gene_id", "cs_id"])
        print(f"  Variants to score     : {len(cs_filtered)}")

        # ── 2. 解析变异格式 chr1_70355119_G_A → chrom/pos/ref/alt ───────────
        parsed = cs_filtered["variant"].apply(_parse_variant)
        variants_df = pd.DataFrame(
            list(parsed), columns=["chrom", "pos", "ref", "alt"]
        )
        variants_df["gene_id"] = cs_filtered["gene_id"].values
        variants_df["cs_id"]   = cs_filtered["cs_id"].values
        variants_df["pip"]     = cs_filtered["pip"].values
        variants_df["causal"]  = (variants_df["pip"] >= min_pip_causal).astype(int)

        # 仅保留 SNPs（ref/alt 均为单碱基）
        snp_mask = variants_df["ref"].str.len().eq(1) & variants_df["alt"].str.len().eq(1)
        variants_df = variants_df[snp_mask].reset_index(drop=True)
        print(f"  SNPs only             : {len(variants_df)} "
              f"({snp_mask.mean()*100:.1f}% of variants)")

        # 过滤染色体末端附近的变异（seq_len 扩展后会越界）
        # 用最保守值（Borzoi = 最大 receptive field）
        chrom_sizes = grelu.io.genome.read_sizes("hg38").set_index("chrom")["size"].to_dict()
        half = BORZOI_INPUT_LEN // 2
        edge_mask = variants_df.apply(
            lambda r: (r.pos - half >= 0) and
                      (r.pos + half <= chrom_sizes.get(r.chrom, 0)),
            axis=1,
        )
        n_before = len(variants_df)
        variants_df = variants_df[edge_mask].reset_index(drop=True)
        print(f"  After edge filter     : {len(variants_df)} "
              f"(removed {n_before - len(variants_df)} near chrom ends)")

        # ── 3. 读取 track 元数据 ─────────────────────────────────────────────
        ag_meta = pd.read_parquet(AG_META_PATH)

        # ── 4. 逐模型打分（每模型在独立子进程中运行，保证 CUDA context 完整释放）──
        import tempfile, subprocess as _sp
        results = {}
        for mname in model_names:
            print(f"\n[eQTL-AUPRC] Scoring {mname.upper()} in subprocess ({len(variants_df)} SNPs) …")
            tmp_variants = tempfile.mktemp(suffix=".csv")
            tmp_scores   = tempfile.mktemp(suffix=".npy")
            variants_df.to_csv(tmp_variants, index=False)
            devices_str = (
                ",".join(str(d) for d in self.devices)
                if isinstance(self.devices, list) else str(self.devices)
            )
            cmd = [
                sys.executable, __file__,
                "--_eqtl_score_one",
                "--_eqtl_score_model",    mname,
                "--_eqtl_score_variants", tmp_variants,
                "--_eqtl_score_output",   tmp_scores,
                "--_eqtl_score_tissue",   tissue,
                "--devices",              devices_str,
                "--num_workers",          str(self.num_workers),
            ]
            _sp.run(cmd, check=True)
            results[mname] = np.load(tmp_scores)
            os.unlink(tmp_variants)
            os.unlink(tmp_scores)

        # ── 5. 计算 per-locus AUPRC ──────────────────────────────────────────
        print("\n[eQTL-AUPRC] Computing per-locus AUPRC …")
        output_rows = []
        auprc_rows  = []

        for mname, scores in results.items():
            variants_df[f"score_{mname}"] = scores

        for (gene_id, cs_id), grp in variants_df.groupby(["gene_id", "cs_id"]):
            row = {"gene_id": gene_id, "cs_id": cs_id,
                   "cs_size": len(grp), "n_causal": grp["causal"].sum()}
            if row["n_causal"] == 0 or row["n_causal"] == row["cs_size"]:
                continue  # 无法计算 AUPRC（全因果或全非因果）
            for mname in results:
                auprc = average_precision_score(grp["causal"], grp[f"score_{mname}"])
                row[f"auprc_{mname}"] = auprc
            auprc_rows.append(row)

        auprc_df = pd.DataFrame(auprc_rows)
        output_rows = variants_df.copy()

        # ── 6. 保存结果 ──────────────────────────────────────────────────────
        csv_variants = os.path.join(ISM_RESULTS_DIR, f"{output_prefix}_variants.csv")
        csv_auprc    = os.path.join(ISM_RESULTS_DIR, f"{output_prefix}_per_locus.csv")
        txt_report   = os.path.join(ISM_RESULTS_DIR, f"{output_prefix}_report.txt")

        output_rows.to_csv(csv_variants, index=False)
        auprc_df.to_csv(csv_auprc, index=False)

        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(txt_report, "w") as f:
            f.write(f"eQTL AUPRC 评估报告\n")
            f.write("=" * 68 + "\n")
            f.write(f"运行时间    : {ts}\n")
            f.write(f"组织        : {tissue}\n")
            f.write(f"因果阈值    : PIP >= {min_pip_causal}\n")
            f.write(f"位点总数    : {len(auprc_df)}\n")
            f.write(f"变异总数    : {len(output_rows)}\n\n")
            f.write("── 模型 AUPRC 汇总 ──────────────────────────────────────────\n")
            baseline = auprc_df["n_causal"].sum() / auprc_df["cs_size"].sum()
            f.write(f"  随机基线 AUPRC          : {baseline:.4f} "
                    f"（因果变异占比）\n")
            for mname in results:
                col = f"auprc_{mname}"
                if col in auprc_df.columns:
                    mean_a  = auprc_df[col].mean()
                    med_a   = auprc_df[col].median()
                    n_beats = (auprc_df[col] > baseline).sum()
                    f.write(f"\n  {mname.upper():15s}\n")
                    f.write(f"    mean AUPRC  : {mean_a:.4f}\n")
                    f.write(f"    median AUPRC: {med_a:.4f}\n")
                    f.write(f"    > baseline  : {n_beats}/{len(auprc_df)} 位点 "
                            f"({n_beats/len(auprc_df)*100:.1f}%)\n")
                    print(f"  {mname.upper():15s}  mean_AUPRC={mean_a:.4f}  "
                          f"median={med_a:.4f}  baseline={baseline:.4f}")
            f.write(f"\n── 输出文件 ──────────────────────────────────────────────────\n")
            f.write(f"  Per-variant scores : {csv_variants}\n")
            f.write(f"  Per-locus AUPRC    : {csv_auprc}\n")

        print(f"\n  报告 → {txt_report}")
        return auprc_df

    def plot_ism(self):
        # 绘图逻辑保持原样
        for m in ["borzoi", "alphagenome"]:
            p = os.path.join(ISM_RESULTS_DIR, f"{m}_ism.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, index_col=0)
                df.columns = [str(c).split('.')[0] for c in df.columns]
                grelu.visualize.plot_ISM(df, method="logo")
                plt.title(f"{m} Brain/Liver Ratio ISM Logo")
                plt.savefig(f"{m}_ism_logo.png", dpi=150); plt.close()

if __name__ == "__main__":
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--compute_ism", choices=["borzoi", "alphagenome", "both"])
    parser.add_argument("--plot_ism", action="store_true")
    parser.add_argument("--gene", default="SRSF11")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--num_workers", type=int, default=1)
    # Internal flag: marks a subprocess-spawned ISM run so it runs inline
    parser.add_argument("--_ism_subprocess", action="store_true", help=argparse.SUPPRESS)
    # Genome-wide Pred-vs-Obs evaluation
    parser.add_argument("--eval", choices=["borzoi", "alphagenome"],
                        help="Run genome-wide Pred-vs-Obs Pearson R evaluation.")
    parser.add_argument("--eval_bigwig", metavar="PATH",
                        help="Path to observed-signal BigWig for --eval.")
    parser.add_argument("--eval_track_idx", type=int, default=0, metavar="INT",
                        help="Zero-based output track index to compare against BigWig.")
    parser.add_argument("--eval_output_key", default="rna_seq",
                        help="AlphaGenome head key (rna_seq, cage, …). Default: rna_seq.")
    parser.add_argument("--eval_chroms", nargs="+", default=["chr1", "chr8", "chr21"],
                        metavar="CHROM")
    parser.add_argument("--eval_max_windows", type=int, default=300,
                        help="Max windows to evaluate (0=all). Default: 300.")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_output", default="eval_results.json",
                        help="JSON output path. Default: eval_results.json.")
    parser.add_argument("--eval_min_obs_mean", type=float, default=0.05,
                        help="只统计 obs_mean 超过此阈值的有信号窗口。Default: 0.5.")
    parser.add_argument("--eval_save_per_window", metavar="CSV",
                        help="Optional CSV path for per-window Pearson R table.")
    # eQTL AUPRC evaluation
    parser.add_argument("--eqtl_auprc", action="store_true",
                        help="Run eQTL fine-mapping AUPRC evaluation.")
    parser.add_argument("--eqtl_cs_files", nargs="+", metavar="PATH",
                        help="eQTL Catalogue credible_sets.tsv.gz file(s). "
                             "Defaults to all brain+blood files in "
                             "~/.cache/eqtl_finemapping/gtex_v8_susie_ge/.")
    parser.add_argument("--eqtl_tissue", default="brain",
                        help="Tissue keyword for track selection (default: brain).")
    parser.add_argument("--eqtl_min_pip", type=float, default=0.5,
                        help="PIP threshold for causal label (default: 0.5).")
    parser.add_argument("--eqtl_max_loci", type=int, default=200,
                        help="Max loci to evaluate (default: 200).")
    parser.add_argument("--eqtl_models", nargs="+", default=["borzoi", "alphagenome"],
                        choices=["borzoi", "alphagenome"],
                        help="Models to evaluate (default: both).")
    parser.add_argument("--eqtl_output_prefix", default="eqtl_auprc",
                        help="Output file prefix (default: eqtl_auprc).")
    # 隐藏子命令：由 run_eqtl_auprc 在独立子进程中调用，打完一个模型就退出
    parser.add_argument("--_eqtl_score_one",      action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_model",    default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_variants", default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_output",   default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_eqtl_score_tissue",   default="brain",     help=argparse.SUPPRESS)
    args = parser.parse_args()

    app = GreluTutorialApp(gene=args.gene, devices=args.devices, num_workers=args.num_workers)

    if args.inference:
        app.run_inference()

    if args.compute_ism:
        if args.inference and not args._ism_subprocess:
            # Inference has already dirtied the CUDA context on GPU 0.
            # Spawn a fresh process for ISM so it gets a clean context.
            print("\n[ISM] Inference was run in this process; isolating ISM in a "
                  "clean subprocess to avoid CUDA context conflicts...")
            cmd = [
                sys.executable, __file__,
                "--compute_ism", args.compute_ism,
                "--gene", args.gene,
                "--devices", args.devices,
                "--num_workers", str(args.num_workers),
                "--_ism_subprocess",
            ]
            subprocess.run(cmd, check=True)
        else:
            models_ran = []
            if args.compute_ism in ["borzoi", "both"]:
                app.run_ism("borzoi")
                models_ran.append("borzoi")
            if args.compute_ism in ["alphagenome", "both"]:
                app.run_ism("alphagenome")
                models_ran.append("alphagenome")

            # ── 写文本报告 ───────────────────────────────────────────────────
            import datetime
            import numpy as np
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for mname in models_ran:
                csv_p = os.path.join(ISM_RESULTS_DIR, f"{mname}_ism.csv")
                txt_p = os.path.join(ISM_RESULTS_DIR, f"{mname}_ism_report.txt")
                df = pd.read_csv(csv_p, index_col=0)
                vals = df.values.astype(float)
                col_max = np.max(np.abs(vals), axis=0)
                top10 = np.argsort(col_max)[::-1][:10]
                with open(txt_p, "w") as f:
                    f.write(f"ISM 结果报告 — {mname.upper()}\n")
                    f.write("=" * 68 + "\n")
                    f.write(f"运行时间       : {ts}\n")
                    f.write(f"基因           : {args.gene}\n")
                    f.write(f"GPU设备        : {args.devices}\n")
                    f.write(f"矩阵形状       : {df.shape[0]} 行 × {df.shape[1]} 列"
                            f"  [4碱基 × 位置]\n\n")
                    f.write("── 数值摘要 ──────────────────────────────────────────────\n")
                    f.write(f"  最大  log2FC : {vals.max():+.6f}\n")
                    f.write(f"  最小  log2FC : {vals.min():+.6f}\n")
                    f.write(f"  均值|log2FC| : {np.mean(np.abs(vals)):.6f}\n")
                    f.write(f"  非零元素比例 : {np.mean(vals != 0) * 100:.1f}%\n\n")
                    f.write("── 信号最强前10位置 ──────────────────────────────────────\n")
                    f.write(f"  {'位置(碱基)':>12}  {'max|log2FC|':>12}  最大突变碱基\n")
                    for i in top10:
                        col = df.columns[i]
                        best = df[col].abs().idxmax()
                        f.write(f"  {col:>12}  {col_max[i]:>12.6f}  {best}\n")
                    f.write("\n── 每突变碱基统计 ────────────────────────────────────────\n")
                    f.write(f"  {'碱基':>4}  {'最大':>10}  {'最小':>10}  {'均值':>10}  标准差\n")
                    for base in ["A", "C", "G", "T"]:
                        r = df.loc[base].values.astype(float)
                        f.write(f"  {base:>4}  {r.max():>+10.6f}  {r.min():>+10.6f}"
                                f"  {r.mean():>+10.6f}  {r.std():.6f}\n")
                    f.write("\n── 完整矩阵（TSV，行=碱基，列=序列位置，值=log2FC）─────\n")
                    f.write(df.to_csv(sep="\t"))
                print(f"  文本报告 → {txt_p}")

    if args.plot_ism:
        app.plot_ism()

    if args.eval:
        if not args.eval_bigwig:
            parser.error("--eval requires --eval_bigwig")
        import json
        summary = app.run_genome_wide_eval(
            model_name=args.eval,
            bigwig_path=args.eval_bigwig,
            track_idx=args.eval_track_idx,
            output_key=args.eval_output_key,
            chroms=args.eval_chroms,
            max_windows=args.eval_max_windows,
            batch_size=args.eval_batch_size,
            min_obs_mean=args.eval_min_obs_mean,
            save_per_window=args.eval_save_per_window,
        )
        out_path = args.eval_output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON → {out_path}")

    if args._eqtl_score_one:
        # ── 子进程：打完一个模型的分数就退出，CUDA context 随进程退出完全销毁 ──
        from grelu.variant import predict_variant_effects
        from grelu.transforms.prediction_transforms import Aggregate
        mname   = args._eqtl_score_model
        tissue  = args._eqtl_score_tissue
        vdf     = pd.read_csv(args._eqtl_score_variants)
        devices = [int(x) for x in args.devices.split(",")]

        if mname == "borzoi":
            print(f"[subprocess] Loading Borzoi …")
            model_obj = grelu.resources.load_model(
                repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
            )
            tasks_df  = pd.DataFrame(model_obj.data_params["tasks"])
            brain_idx = tasks_df[
                (tasks_df.assay == "RNA") &
                tasks_df["sample"].str.contains(tissue, case=False, na=False)
            ].index.tolist()
            bs = 2
        else:
            print(f"[subprocess] Loading AlphaGenome RNA …")
            ag_params = dict(
                weights_path=WEIGHTS_PATH,
                dtype_policy=DtypePolicy.mixed_precision(),
                resolution=128,
            )
            model_obj = LightningModel(
                model_params={"model_type": "AlphaGenomeModel",
                              "output_key": "rna_seq", **ag_params},
                train_params={"task": "regression", "loss": "mse"},
            )
            model_obj.data_params["train"] = {"seq_len": 131072, "bin_size": 128}
            model_obj.model_params["crop_len"] = 0
            ag_meta   = pd.read_parquet(AG_META_PATH)
            rna_meta  = ag_meta[ag_meta.output_type == "rna_seq"]
            brain_idx = rna_meta[
                rna_meta.biosample_name.str.contains(tissue, case=False, na=False)
            ].track_index.tolist()
            bs = 4

        print(f"[subprocess] {mname} {tissue} tracks: {len(brain_idx)}, "
              f"variants: {len(vdf)}, devices: {devices}, batch_size: {bs}")
        transform = Aggregate(tasks=brain_idx, length_aggfunc="mean", task_aggfunc="mean")
        odds = predict_variant_effects(
            variants=vdf,
            model=model_obj,
            devices=devices,
            num_workers=args.num_workers,
            batch_size=bs,
            genome="hg38",
            compare_func="log2FC",
            return_ad=False,
            prediction_transform=transform,
        )
        scores = np.abs(odds.squeeze())
        np.save(args._eqtl_score_output, scores)
        print(f"[subprocess] Scores saved → {args._eqtl_score_output}  shape={scores.shape}")
        sys.exit(0)

    if args.eqtl_auprc:
        import glob as _glob
        cs_files = args.eqtl_cs_files or sorted(
            _glob.glob(os.path.expanduser(
                "~/.cache/eqtl_finemapping/gtex_v8_susie_ge/*.tsv.gz"
            ))
        )
        if not cs_files:
            parser.error("--eqtl_auprc: no credible set files found. "
                         "Use --eqtl_cs_files or download data first.")
        print(f"[eQTL-AUPRC] Using {len(cs_files)} credible set file(s)")
        app.run_eqtl_auprc(
            cs_files=cs_files,
            tissue=args.eqtl_tissue,
            min_pip_causal=args.eqtl_min_pip,
            max_loci=args.eqtl_max_loci,
            model_names=args.eqtl_models,
            output_prefix=args.eqtl_output_prefix,
        )
