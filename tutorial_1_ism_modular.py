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
import grelu.data.preprocess
import grelu.visualize
from grelu.lightning import LightningModel
from alphagenome_pytorch.config import DtypePolicy

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
