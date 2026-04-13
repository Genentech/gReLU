# Project Memory & Major Changes

This file serves as a persistent record of major architectural decisions, feature integrations, and significant codebase changes within the gReLU project.

## [2026-04-13] AlphaGenome Integration
**Objective:** Integrate the DeepMind AlphaGenome model (via the `alphagenome_pytorch` PyTorch port) as a first-class citizen in the gReLU framework.
**Key Implementations:**
- **Trunk Wrapper (`src/grelu/model/trunks/alphagenome.py`)**: Implemented `AlphaGenomeTrunk` to manage dimension transposition from gReLU's standard `(Batch, 4, Length)` to AlphaGenome's `(Batch, Length, 4)`, and handle dynamic modality/resolution extraction (e.g., `atac` at 128bp, `rna_seq` at 1bp).
- **Model Class (`src/grelu/model/models.py`)**: Created `AlphaGenomeModel` extending gReLU's `BaseModel`, seamlessly hooking into `grelu.lightning.LightningModel`.
- **Validation (`compare_pretrained_inference.py`)**: Verified correctness via zero-shot inference. Using the UCSC API to fetch the real MYC oncogene promoter, AlphaGenome successfully yielded biologically significant peaks (ATAC signal max ~1.000), proving the dimension alignment, DtypePolicy, and scaling mechanisms are perfectly intact and compatible with the gReLU pipeline.
