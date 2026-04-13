#!/usr/bin/env python3
"""
compare_pretrained_inference.py

This script directly uses the PRETRAINED AlphaGenome and Enformer models
to predict genomic tracks (like ATAC-seq) on a REAL biological DNA sequence.
We fetch the sequence of the human MYC oncogene promoter region via the UCSC API.
No training involved. We rely on gReLU's `predict_on_seqs` pipeline.
"""

import torch
import requests
import numpy as np
from huggingface_hub import hf_hub_download
from grelu.lightning import LightningModel
from grelu.model.models import EnformerPretrainedModel
from grelu.sequence.format import strings_to_one_hot

def fetch_ucsc_seq(chrom, start, end, genome="hg38"):
    """Fetch real genomic sequence from UCSC API."""
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome}&chrom={chrom}&start={start}&end={end}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['dna'].upper()
    else:
        raise Exception(f"Failed to fetch: {response.status_code}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f" Pretrained Inference Comparison: AlphaGenome vs Enformer")
    print(f" Dataset: REAL Human DNA (MYC Oncogene Promoter, hg38)")
    print(f"============================================================")
    
    # ---------------------------------------------------------
    # 1. Prepare AlphaGenome Pretrained Model
    # ---------------------------------------------------------
    print("\n[1/3] Downloading & Loading Pretrained AlphaGenome...")
    # Download the official Fold 0 weights from the PyTorch port HF repo
    ag_weights_path = hf_hub_download(
        repo_id="gtca/alphagenome_pytorch", 
        filename="model_fold_0.safetensors"
    )
    
    ag_params = {
        "model_type": "AlphaGenomeModel",
        "output_key": "atac",      # We evaluate ATAC-seq accessibility
        "resolution": 128,         # 128bp resolution
        "num_organisms": 2,
        "organism_index": 0,       # Human
        "weights_path": ag_weights_path,
        "dtype_policy": __import__('alphagenome_pytorch.config').config.DtypePolicy.mixed_precision() if device == "cuda" else None
    }
    
    # Instantiate via gReLU LightningWrapper
    ag_model = LightningModel(model_params=ag_params)
    ag_model.to(device)
    ag_model.eval()
    print("      AlphaGenome loaded successfully.")

    # ---------------------------------------------------------
    # 2. Prepare Enformer Pretrained Model
    # ---------------------------------------------------------
    print("\n[2/3] Downloading & Loading Pretrained Enformer...")
    # gReLU provides a specialized class that downloads and loads DeepMind's weights
    enf_base = EnformerPretrainedModel(n_tasks=5313, final_pool_func=None)
    enf_model = LightningModel(model_params={"model_type": "BaseModel", "embedding": enf_base.embedding, "head": enf_base.head})
    enf_model.to(device)
    enf_model.eval()
    print("      Enformer loaded successfully.")

    # ---------------------------------------------------------
    # 3. Real Biological Zero-Shot Prediction Task (MYC Promoter)
    # ---------------------------------------------------------
    print("\n[3/3] Fetching REAL Genomic Sequence (chr8: MYC Promoter)...")
    myc_tss = 127735434 # TSS of MYC transcript in hg38
    
    # AlphaGenome standard input is 131,072 bp
    ag_seq_len = 131072
    ag_start = myc_tss - (ag_seq_len // 2)
    ag_end = myc_tss + (ag_seq_len // 2)
    ag_seq_str = fetch_ucsc_seq("chr8", ag_start, ag_end)
    ag_seq = strings_to_one_hot([ag_seq_str]).float()
    
    # Enformer standard input is 393,216 bp
    enf_seq_len = 393216
    enf_start = myc_tss - (enf_seq_len // 2)
    enf_end = myc_tss + (enf_seq_len // 2)
    enf_seq_str = fetch_ucsc_seq("chr8", enf_start, enf_end)
    enf_seq = strings_to_one_hot([enf_seq_str]).float()

    with torch.no_grad():
        # Using gReLU's standardized inference call
        print("      Running AlphaGenome prediction...")
        ag_preds = ag_model(ag_seq.to(device)).cpu()
        
        print("      Running Enformer prediction...")
        enf_preds = enf_model(enf_seq.to(device)).cpu()

    print("\n============================================================")
    print(" Inference Results on MYC Promoter")
    print("============================================================")
    print(f"AlphaGenome output shape : {ag_preds.shape}")
    print(f"Enformer output shape    : {enf_preds.shape}")
    print(f"")
    print(f"Mean Signal (AlphaGenome ATAC tracks) : {ag_preds.mean().item():.4f}")
    print(f"Max Signal  (AlphaGenome ATAC tracks) : {ag_preds.max().item():.4f}")
    print(f"")
    print(f"Mean Signal (Enformer All tracks)     : {enf_preds.mean().item():.4f}")
    print(f"Max Signal  (Enformer All tracks)     : {enf_preds.max().item():.4f}")
    print("============================================================")
    print("[SUCCESS] Both pretrained models successfully executed inference on REAL DNA through the gReLU pipeline.")

if __name__ == "__main__":
    main()
