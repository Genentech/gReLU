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
        ism_center = self.target_exons.start.min()
        
        # Borzoi: 524kb input
        borzoi_input_len = 524288
        self.borzoi_start_coord = ism_center - borzoi_input_len // 2
        
        # AG: 131kb input
        ag_input_len = 131072
        self.ag_start_coord = ism_center - ag_input_len // 2
        self.ag_end_coord = self.ag_start_coord + ag_input_len
        
        b_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.borzoi_start_coord], "end": [self.borzoi_start_coord + borzoi_input_len], "strand": ["+"]})
        self.input_seqs = grelu.sequence.format.convert_input_type(b_intervals, output_type="strings", genome=self.genome)[0]
        
        ag_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.ag_start_coord], "end": [self.ag_end_coord], "strand": ["+"]})
        self.ag_seqs = grelu.sequence.format.convert_input_type(ag_intervals, output_type="strings", genome=self.genome)[0]
        
        self.ism_region = {'start': ism_center - 100, 'end': ism_center + 100}

    def setup_borzoi(self):
        print("Loading Borzoi model...")
        self.borzoi = grelu.resources.load_model(repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt")
        borzoi_input_intervals = pd.DataFrame({"chrom": [self.chrom], "start": [self.borzoi_start_coord], "end": [self.borzoi_start_coord + 524288], "strand": ["+"]})
        borzoi_out_intervals = self.borzoi.input_intervals_to_output_intervals(borzoi_input_intervals)
        self.borzoi_out_start = int(borzoi_out_intervals.start[0])
        self.borzoi_bin_size = 32

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
        # 需要手动 bin 到模型输出分辨率 (N, n_bins)
        obs_raw = dataset.labels[:, 0, :]  # (N, seq_len)
        n, total = obs_raw.shape
        obs = obs_raw.reshape(n, total // bin_size, bin_size).mean(axis=-1)  # (N, n_bins)

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

        # ── 对齐：Borzoi 等模型输出比输入短（中央裁剪），裁剪 obs 匹配 ──────
        n_pred_bins = preds_track.shape[1]
        n_obs_bins  = obs.shape[1]
        if n_pred_bins != n_obs_bins:
            crop_start = (n_obs_bins - n_pred_bins) // 2
            obs = obs[:, crop_start : crop_start + n_pred_bins]

        # ── Pearson R per window ──────────────────────────────────────────────
        pred_t = torch.as_tensor(preds_track, dtype=torch.float32)
        obs_t  = torch.as_tensor(obs,          dtype=torch.float32)
        per_window_r = _pearson_r(pred_t, obs_t, dim=-1).numpy()  # (N,)

        windows_r = windows.copy()
        windows_r["pearson_r"] = per_window_r
        windows_r["obs_mean"]  = obs.mean(axis=-1)  # 每窗口观测信号均值

        def _stats(r_arr: np.ndarray) -> dict:
            v = r_arr[np.isfinite(r_arr)]
            return {
                "mean_r":    float(np.mean(v))   if len(v) else float("nan"),
                "median_r":  float(np.median(v)) if len(v) else float("nan"),
                "std_r":     float(np.std(v))    if len(v) else float("nan"),
                "n_windows": int(len(v)),
            }

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
            if args.compute_ism in ["borzoi", "both"]:
                app.run_ism("borzoi")
            if args.compute_ism in ["alphagenome", "both"]:
                app.run_ism("alphagenome")

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
