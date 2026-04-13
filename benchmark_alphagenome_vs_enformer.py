#!/usr/bin/env python3
"""
benchmark_alphagenome_vs_enformer.py

A script to compare the hardware performance (latency and memory) 
of the newly wrapped AlphaGenomeModel against the classic EnformerModel 
within the gReLU infrastructure. 

Task: Standard Regulatory Track Prediction (ATAC/DNASE/CAGE) at 128bp resolution.
Sequence Length: 131,072 bp (128kb)
"""

import torch
import time
import pandas as pd
from grelu.lightning import LightningModel
from alphagenome_pytorch.config import DtypePolicy

def measure_model(name, model_params, seq_len=131072, batch_size=1, device="cuda"):
    print(f"--- Benchmarking {name} ---")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    train_params = {
        "task": "regression",
        "loss": "mse",
        "devices": device
    }
    
    # Instantiate
    t0 = time.time()
    model = LightningModel(model_params=model_params, train_params=train_params)
    model.to(device)
    model.eval()
    print(f"    Loaded in {time.time() - t0:.2f}s")
    
    # Dummy data
    x = torch.zeros((batch_size, 4, seq_len), dtype=torch.float32, device=device)
    x[:, 0, :] = 1.0
    
    # Warmup
    print("    Warming up...")
    with torch.no_grad():
        for _ in range(2):
            # Enformer doesn't natively use autocast inside its forward like AlphaGenome does.
            # To be fair in modern training, we autocast both if desired, but AlphaGenome handles
            # it internally via DtypePolicy. We will just call forward.
            _ = model(x)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Timed run
    print("    Running benchmark...")
    iterations = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            outputs = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    
    latency_ms = ((end - start) / iterations) * 1000
    
    if device == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_mem_gb = 0.0
        
    params_count = sum(p.numel() for p in model.parameters()) / 1e6
    output_shape = tuple(outputs.shape)
    
    # Clean up to prevent OOM on next model
    del model
    del x
    del outputs
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return {
        "Model": name,
        "Params (M)": round(params_count, 1),
        "Output Shape": str(output_shape),
        "Latency (ms)": round(latency_ms, 2),
        "Peak VRAM (GB)": round(peak_mem_gb, 2)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}\n")

    seq_len = 131072 # 128kb
    batch_size = 1

    results = []

    # 1. EnformerModel (gReLU Built-in)
    # Configure to predict 256 tracks (same as AlphaGenome ATAC)
    enformer_params = {
        "model_type": "EnformerModel",
        "n_tasks": 256,
        "n_conv": 7,            # Standard Enformer config -> 128bp pooling
        "channels": 1536,       # Standard Enformer width
        "n_transformers": 11,   # Standard Enformer depth
        "n_heads": 8,
        "final_pool_func": None # Disable sequence-wide pooling
    }
    res_enformer = measure_model("Enformer", enformer_params, seq_len, batch_size, device)
    results.append(res_enformer)

    # 2. AlphaGenomeModel (Our new wrapper)
    # Configure to predict 256 tracks (ATAC, 128bp resolution)
    alphagenome_params = {
        "model_type": "AlphaGenomeModel",
        "output_key": "atac",   # 256 tasks
        "resolution": 128,
        "num_organisms": 2,
        "organism_index": 0,
        "dtype_policy": DtypePolicy.mixed_precision() if device == "cuda" else None
    }
    res_ag = measure_model("AlphaGenome", alphagenome_params, seq_len, batch_size, device)
    results.append(res_ag)

    print("\n" + "="*70)
    print(" BENCHMARK RESULTS: 128kb Sequence Length (Batch=1) ")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()
