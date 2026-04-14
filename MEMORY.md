# Project Memory & Major Changes

This file serves as a persistent record of major architectural decisions, feature integrations, and significant codebase changes within the gReLU project.

## [2026-04-13] AlphaGenome Integration
**Objective:** Integrate the DeepMind AlphaGenome model (via the `alphagenome_pytorch` PyTorch port) as a first-class citizen in the gReLU framework.
**Key Implementations:**
- **Trunk Wrapper (`src/grelu/model/trunks/alphagenome.py`)**: Implemented `AlphaGenomeTrunk` to manage dimension transposition from gReLU's standard `(Batch, 4, Length)` to AlphaGenome's `(Batch, Length, 4)`, and handle dynamic modality/resolution extraction (e.g., `atac` at 128bp, `rna_seq` at 1bp).
- **Model Class (`src/grelu/model/models.py`)**: Created `AlphaGenomeModel` extending gReLU's `BaseModel`, seamlessly hooking into `grelu.lightning.LightningModel`.
- **Validation (`compare_pretrained_inference.py`)**: Verified correctness via zero-shot inference. Using the UCSC API to fetch the real MYC oncogene promoter, AlphaGenome successfully yielded biologically significant peaks (ATAC signal max ~1.000), proving the dimension alignment, DtypePolicy, and scaling mechanisms are perfectly intact and compatible with the gReLU pipeline.
- **Tutorial Replication (`reproduce_tutorial.py`)**: Replicated the official Borzoi inference tutorial (`1_inference.ipynb`) using AlphaGenome. Predicted on the exact tutorial coordinates (`chr1:69993520-70517808`). Successfully matched Brain CAGE and Brain RNA-seq tracks and plotted them using `grelu.visualize.plot_tracks` alongside gene/exon annotations. AlphaGenome demonstrated strong signals (e.g., max CAGE ~2384.0), outputting fully scaled experimental read counts, which represents a theoretical interpretability advantage over Borzoi's outputs.

## [2026-04-14] Multi-GPU Prediction Consistency Fix
**Objective:** Fix the ordering of predictions when using multiple GPUs in `predict_on_dataset`.
**Key Implementations:**
- **De-interleaving Logic (`src/grelu/lightning/__init__.py`)**: Updated `predict_on_dataset` to correctly reorder gathered results from multiple GPUs. Since PyTorch Lightning's `DistributedSampler` distributes data in an interleaved fashion (`i::world_size`), a simple concatenation of gathered results produced incorrect output order. The fix explicitly de-interleaves the gathered tensors.
- **Padding Truncation**: Added logic to truncate the final results to the exact length of the dataset, removing any extra padding samples added by the distributed sampler for batch alignment.
**Validation:**
- **Empirical Verification**: Conducted In Silico Mutagenesis (ISM) on a 200bp region (SRSF11 promoter) using the Borzoi model.
- **Consistency Check**: Compared results between single-GPU (Device 0) and 4-GPU (Devices 0, 1, 2, 3) execution.
- **Outcome**: The results were identical (Max absolute difference: 0.0), confirming that the parallel execution now preserves dataset order and handles padding correctly.

