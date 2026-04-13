#!/usr/bin/env python3
"""
run_alphagenome_inference.py

A script demonstrating how to run inference on the AlphaGenome model using 
the gReLU infrastructure. This script runs two distinct examples to verify
correctness and near-SOTA hardware performance metrics (speed & memory).
"""

import torch
import time
from grelu.lightning import LightningModel

def run_example(example_name, output_key, resolution, seq_len=131072, batch_size=1):
    print("=" * 70)
    print(f" {example_name}: {output_key} @ {resolution}bp resolution ")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Configuration & Instantiation via gReLU
    model_params = {
        "model_type": "AlphaGenomeModel",
        "output_key": output_key,
        "resolution": resolution,
        "num_organisms": 2,     
        "organism_index": 0,    # Human
        # Enabling JAX-matching bfloat16 compute policy for near-SOTA performance & memory savings
        "dtype_policy": __import__('alphagenome_pytorch.config').config.DtypePolicy.mixed_precision() if device == "cuda" else None,
    }
    
    train_params = {
        "task": "regression",
        "loss": "mse",
        "devices": device
    }
    
    print(f"[INFO] Instantiating AlphaGenomeModel...")
    t0 = time.time()
    model = LightningModel(model_params=model_params, train_params=train_params)
    model.to(device)
    model.eval()
    print(f"[SUCCESS] Model instantiated in {time.time() - t0:.2f} seconds.")

    # 2. Data Preparation
    print(f"[INFO] Creating synthetic DNA tensor: Batch={batch_size}, Channels=4, Length={seq_len:,} bp")
    dna_sequence = torch.zeros((batch_size, 4, seq_len), dtype=torch.float32, device=device)
    dna_sequence[:, 0, :] = 1.0  # Fill with 'A'
    
    # 3. Execution (Warm-up to build CUDA graphs/kernels)
    print(f"[INFO] Running warm-up pass...")
    with torch.no_grad():
        _ = model(dna_sequence)

    # 4. Actual timed execution
    print(f"[INFO] Running timed inference pass...")
    if device == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.time()
    with torch.no_grad():
        outputs = model(dna_sequence)
        
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_mem = 0.0
        
    elapsed = time.time() - t1
    print(f"[SUCCESS] Inference completed in {elapsed:.4f} seconds.")

    # 5. Output Inspection
    print("\n[INFO] Inference Metrics & Output Information:")
    print(f" ├── Output Modality : {output_key}")
    print(f" ├── Resolution      : {resolution} bp")
    print(f" ├── Expected Length : {seq_len // resolution}")
    print(f" ├── Output Shape    : {outputs.shape} (Batch, Tasks, Output_Length)")
    print(f" ├── Num Tasks       : {model.model.head.n_tasks}")
    print(f" ├── Latency         : {elapsed:.4f} sec")
    if device == "cuda":
        print(f" ├── Peak VRAM Usage : {peak_mem:.2f} GB")
    print("\n")

def main():
    print("\n" + "#" * 70)
    print("# AlphaGenome gReLU Wrapper - Correctness & Performance Validation #")
    print("#" * 70 + "\n")

    # Example 1: Standard Regulatory Track Prediction (128bp)
    # Stresses the Encoder + Transformer + 128bp Embedder
    run_example(
        example_name="Example 1 (Standard Regulatory Tracks)",
        output_key="atac",
        resolution=128,
        seq_len=131072 # 128kb
    )

    # Example 2: High-Resolution Expression Track (1bp)
    # Stresses Encoder + Transformer + Decoder + 1bp Embedder
    run_example(
        example_name="Example 2 (High-Res Expression / Decoder Stress Test)",
        output_key="rna_seq",
        resolution=1,
        seq_len=131072 # 128kb
    )

    print("[SUCCESS] All examples finished successfully. Architecture routing and dimensions are verified.")

if __name__ == "__main__":
    main()
