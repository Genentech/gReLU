# gReLU Architecture Reference

This document describes the full module hierarchy, class relationships, and
call chains for the gReLU framework, including the AlphaGenome integration.

---

## 1. Package Layout

```
src/grelu/
├── lightning/           # LightningModel — the primary user-facing entry point
│   ├── __init__.py      #   LightningModel, LightningModelEnsemble
│   ├── losses.py        #   PoissonMultinomialLoss
│   └── metrics.py       #   MSE, PearsonCorrCoef, BestF1
│
├── model/               # Pure PyTorch model building blocks
│   ├── models.py        #   All complete model classes (BaseModel and subclasses)
│   ├── trunks/          #   Backbone (embedding) implementations
│   │   ├── __init__.py  #     ConvTrunk, DilatedConvTrunk, ConvGRUTrunk, ConvTransformerTrunk
│   │   ├── borzoi.py    #     BorzoiTrunk  (wraps pre-built Borzoi architecture)
│   │   ├── enformer.py  #     EnformerTrunk
│   │   ├── explainn.py  #     ExplaiNNTrunk
│   │   └── alphagenome.py #   AlphaGenomeTrunk  ← our integration
│   ├── heads.py         #   ConvHead, MLPHead
│   ├── blocks.py        #   ConvBlock, TransformerBlock, GRUBlock, ConvTower,
│   │                    #   TransformerTower, UnetBlock, UnetTower, Stem, …
│   ├── layers.py        #   Activation, Pool, Norm, Crop, Attention, FlashAttention, …
│   └── position.py      #   Positional encodings (sinusoidal, relative, RoPE)
│
├── data/
│   ├── dataset.py       #   All Dataset classes (sequence + label loaders)
│   ├── preprocess.py    #   Interval manipulation utilities
│   ├── augment.py       #   RC, shift, jitter augmentations
│   └── utils.py         #   Data helpers
│
├── transforms/
│   ├── prediction_transforms.py  # Aggregate, Specificity (post-model scoring)
│   ├── label_transforms.py       # Label normalization
│   └── seq_transforms.py         # Sequence penalty transforms
│
├── interpret/
│   ├── score.py         #   ISM_predict, get_attributions, get_attention_scores
│   ├── simulate.py      #   Marginalization, tiling, spacing experiments
│   ├── motifs.py        #   Motif scanning / TF-MoDISco wrappers
│   └── modisco.py       #   TF-MoDISco output parsing
│
├── resources/
│   └── __init__.py      #   list_models, load_model, download_dataset (HuggingFace Hub)
│
├── sequence/
│   ├── format.py        #   convert_input_type — the central sequence encoder
│   ├── mutate.py        #   SNP / indel helpers
│   ├── utils.py         #   reverse_complement, generate_random_sequences
│   └── metrics.py       #   GC content, entropy
│
├── io/
│   ├── genome.py        #   read_gtf, genomepy wrappers
│   ├── bigwig.py        #   BigWig read/write
│   ├── bed.py           #   BED/narrowPeak I/O
│   ├── fasta.py         #   FASTA I/O
│   └── motifs.py        #   MEME motif parsing
│
├── design.py            #   evolve (directed evolution), ledidi
├── variant.py           #   predict_variant_effects, marginalize_variants
├── visualize.py         #   plot_tracks, plot_ISM, plot_attention_matrix, …
└── utils.py             #   Miscellaneous helpers
```

---

## 2. Model Class Hierarchy

```
nn.Module
└── BaseModel  (grelu.model.models)
    │   embedding: nn.Module  (any Trunk)
    │   head:      nn.Module  (ConvHead | MLPHead | nn.Identity)
    │   forward(x: N×4×L) → N×T×L_out
    │
    ├── ConvModel             ConvTrunk          + ConvHead
    ├── DilatedConvModel      DilatedConvTrunk   + ConvHead
    ├── ConvGRUModel          ConvGRUTrunk       + ConvHead
    ├── ConvTransformerModel  ConvTransformerTrunk + ConvHead
    ├── ConvMLPModel          ConvTrunk          + MLPHead
    │
    ├── BorzoiModel           BorzoiTrunk        + ConvHead   (train from scratch)
    ├── BorzoiPretrainedModel BorzoiTrunk        + ConvHead   (loads pretrained)
    │
    ├── EnformerModel         EnformerTrunk      + ConvHead   (train from scratch)
    ├── EnformerPretrainedModel EnformerTrunk    + ConvHead   (loads pretrained)
    │
    └── AlphaGenomeModel      AlphaGenomeTrunk   + nn.Identity  ← our integration

nn.Module (separate hierarchy)
└── ExplaiNNModel  (grelu.model.models)
        ExplaiNNTrunk + linear
```

### Trunk implementations

| Trunk | Input shape | Output shape | Notes |
|-------|-------------|--------------|-------|
| `ConvTrunk` | `(N,4,L)` | `(N,C,L')` | Configurable depth/width |
| `DilatedConvTrunk` | `(N,4,L)` | `(N,C,L)` | Dilated convolutions, no pooling |
| `ConvGRUTrunk` | `(N,4,L)` | `(N,C,L')` | Conv stem + bidirectional GRU |
| `ConvTransformerTrunk` | `(N,4,L)` | `(N,C,L')` | Conv stem + Transformer tower |
| `BorzoiTrunk` | `(N,4,524288)` | `(N,1536,6144)` | U-Net + Transformer |
| `EnformerTrunk` | `(N,4,196608)` | `(N,3072,896)` | Conv + Transformer |
| `AlphaGenomeTrunk` | `(N,4,131072)` | `(N,C,1024)` | See §4 |

### Head implementations (`grelu.model.heads`)

| Head | Formula | `n_tasks` |
|------|---------|-----------|
| `ConvHead` | Conv1d(C→T, k=1) → optional pool | `n_tasks` param |
| `MLPHead` | Flatten → Linear stack | `n_tasks` param |
| `nn.Identity` | pass-through | set externally on `head.n_tasks` |

---

## 3. LightningModel — the central hub

`grelu.lightning.LightningModel` (inherits `pl.LightningModule`) wraps **any**
`BaseModel` and wires it to training, evaluation, and inference pipelines.

### Initialization chain

```
LightningModel(model_params, train_params)
    │
    ├── build_model()           # instantiates grelu.model.models.<model_type>(**kwargs)
    ├── initialize_loss()       # PoissonNLL | MSE | PoissonMultinomial | BCE | …
    ├── initialize_activation() # Softplus | Sigmoid | Exp | ReLU | Identity
    ├── initialize_metrics()    # Pearson | MSE | AUROC | AveragePrecision | …
    └── reset_transform()       # prediction_transform = nn.Identity (default)
```

### Key stored state

```python
model.model           # the underlying BaseModel
model.model_params    # dict: {'model_type', 'crop_len', …}
model.train_params    # dict: {'task', 'loss', 'lr', 'batch_size', …}
model.data_params     # dict: {'tasks': {...}, 'train': {'seq_len','bin_size',…}, …}
```

`data_params` is populated automatically during `train_on_dataset()`.
For externally loaded models (e.g. `load_model()`), it is deserialized from
the checkpoint.

### Inference chain

```
model.predict_on_seqs(seqs: List[str], device)
    │
    ├── strings_to_one_hot(seqs)           # grelu.sequence.format
    ├── tensor.to(device)
    ├── self.model.eval().to(device)
    ├── BaseModel.forward(x)
    │       ├── embedding(x)               # Trunk.forward(x: N×4×L)
    │       └── head(emb)                  # ConvHead | MLPHead | Identity
    └── .detach().cpu().numpy()
    → np.ndarray  (B, T, L_out)

model.predict_on_dataset(dataset, devices, batch_size, …)
    │
    ├── make_predict_loader(dataset)        # DataLoader from any Dataset subclass
    ├── pl.Trainer(accelerator, devices)
    ├── trainer.predict(self, dataloader)   # GPU-batched, calls predict_step
    │       └── predict_step → self.forward(batch)
    │               └── activation(BaseModel.forward(x))
    └── torch.concat → reshape → optional DataFrame
    → np.ndarray  (N, T, L_out)
```

### Coordinate helpers

```
model.input_intervals_to_output_intervals(intervals: pd.DataFrame)
    crop_coords = model_params["crop_len"] × data_params["train"]["bin_size"]
    output.start = intervals.start + crop_coords
    output.end   = intervals.end   − crop_coords

model.input_intervals_to_output_bins(intervals, start_pos)
    bin = (genomic_pos − start_pos) / bin_size − crop_len
```

For AlphaGenome: `crop_len = 0`, `bin_size = 128`.

### Training chain

```
model.train_on_dataset(train_ds, val_ds)
    │
    ├── finalize_model(train_ds)            # sets n_tasks, final_pool_func
    ├── populate_data_params(train_ds)      # writes seq_len, bin_size, tasks, …
    ├── make_train_loader / make_val_loader
    └── pl.Trainer.fit(self, train_dl, val_dl)
            ├── training_step → forward → loss → backward
            └── validation_step → forward → val_metrics
```

---

## 4. AlphaGenome Integration

### Call chain: gReLU → AlphaGenome

```
LightningModel.predict_on_seqs(seqs)               # grelu.lightning
    └── BaseModel.forward(x: N×4×L)                # AlphaGenomeModel
            └── AlphaGenomeTrunk.forward(x)         # grelu.model.trunks.alphagenome
                    │
                    ├── x = x.transpose(1,2)         # (N,4,L) → (N,L,4)
                    ├── organism_index = full(B, 0)  # human=0
                    │
                    ├─ [training]  AlphaGenome.forward(x, org, channels_last=False)
                    └─ [eval]      AlphaGenome.predict(x, org, channels_last=False)
                                        ├── autocast(bfloat16)
                                        ├── AlphaGenome.forward(…)
                                        └── _upcast_outputs → float32
                            │
                            └── out = outputs[output_key][resolution]
                                # e.g. "cage"[128] → (N, 640, 1024)
                    │
                    └── returns (N, C, L_out)
            └── nn.Identity head    # pass-through
    → np.ndarray  (N, T, 1024)
```

### AlphaGenomeTrunk parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_key` | `"atac"` | Modality: `atac/dnase/procap/cage/rna_seq/chip_tf/chip_histone/contact_maps` |
| `resolution` | `128` | Bin resolution: `1` or `128` bp |
| `organism_index` | `0` | `0`=human, `1`=mouse |
| `weights_path` | `None` | Path to `.pth` or `.safetensors` pretrained weights |
| `num_organisms` | `2` | Number of organisms |
| `gradient_checkpointing` | `False` | Trade compute for memory |

### Available output heads (human, fold 0)

| `output_key` | Tracks | Resolutions | `out_channels` |
|-------------|--------|-------------|----------------|
| `atac` | 256 | 1, 128 | 256 |
| `dnase` | 384 | 1, 128 | 384 |
| `procap` | 128 | 1, 128 | 128 |
| `cage` | 640 | 1, 128 | 640 |
| `rna_seq` | 768 | 1, 128 | 768 |
| `chip_tf` | 1664 | 128 only | 1664 |
| `chip_histone` | 1152 | 128 only | 1152 |
| `contact_maps` | 28 | pair S×S | 28 |

Track metadata:
`src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet`
— columns: `track_index, output_type, biosample_name, assay_title, strand, …`

### Pretrained weights

```
~/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/
  snapshots/b01c0ffa73e07c053491f3b5ea8bcf67d93b9920/
    model_fold_0.safetensors   # 920 MB, human fold 0
```

Usage via gReLU:
```python
from grelu.lightning import LightningModel

lm = LightningModel(model_params={
    "model_type":  "AlphaGenomeModel",
    "output_key":  "cage",         # or "atac", "rna_seq", …
    "resolution":  128,
    "weights_path": "~/.cache/…/model_fold_0.safetensors",
})
preds = lm.predict_on_seqs(seqs, device=0)   # → (N, 640, 1024)
```

---

## 5. Data Pipeline

### Dataset hierarchy (`grelu.data.dataset`)

```
torch.Dataset
├── SeqDataset                  # sequences only (no labels); used by predict_on_seqs
│
├── LabeledSeqDataset           # base for all sequence + label datasets
│   ├── DFSeqDataset            # labels from a pandas DataFrame column
│   ├── AnnDataSeqDataset       # labels from AnnData (scRNA / scATAC)
│   └── BigWigSeqDataset        # labels = aggregated BigWig coverage
│
├── VariantDataset              # ref/alt sequence pairs (predict_variant_effects)
├── VariantMarginalizeDataset   # ref + shuffled backgrounds (marginalize_variants)
├── PatternMarginalizeDataset   # motif marginalization
├── ISMDataset                  # all single-nucleotide substitutions (ISM_predict)
├── MotifScanDataset            # sequences with motif insertions
├── SpacingMarginalizeDataset   # motif spacing experiments
└── TilingShuffleDataset        # sliding-window shuffles (CREME / sufficiency test)
```

### Sequence format conversions (`grelu.sequence.format`)

```
Genomic intervals (pd.DataFrame  chrom/start/end)
        │  intervals_to_strings(genome=...)  — calls genomepy
        ▼
DNA strings (List[str])
        │  strings_to_indices
        ▼
Base indices (np.ndarray  N×L  int8  0–3)
        │  indices_to_one_hot
        ▼
One-hot tensor (torch.Tensor  N×4×L  float32)

convert_input_type(x, output_type, genome) handles any-to-any conversion.
```

---

## 6. Interpretation & Design Pipelines

### In-Silico Mutagenesis (`grelu.interpret.score.ISM_predict`)

```
ISM_predict(seqs, model, prediction_transform, start_pos, end_pos, …)
    ├── ISMDataset(seqs, …)                 # generates all N×4 substitutions
    ├── model.predict_on_dataset(ds)        # batched inference
    ├── prediction_transform.compute(preds) # optional score aggregation
    └── compare_func  (log2FC | subtract)
    → pd.DataFrame  (4 rows × n_positions cols)
```

### Gradient attribution (`grelu.interpret.score.get_attributions`)

```
get_attributions(seqs, model, target, …)
    ├── forward(one_hot) → loss.backward()
    └── attribution = one_hot × gradient
    → attribution tensor
```

### Directed Evolution (`grelu.design.evolve`)

```
evolve(seqs, model, prediction_transform, max_iter, …)
    └── for each iteration:
            ├── ISMDataset → model.predict_on_dataset
            ├── prediction_transform.compute → score each mutation
            └── accept best-scoring substitution
    → evolved sequences + per-iteration score history
```

### Variant Effect Prediction (`grelu.variant.predict_variant_effects`)

```
predict_variant_effects(variants, model, …)
    ├── variants_to_intervals + variant_to_seqs   # build ref/alt pairs
    ├── VariantDataset → model.predict_on_dataset
    └── compare ref vs alt predictions
    → score DataFrame  (one row per variant)
```

---

## 7. Transforms (`grelu.transforms.prediction_transforms`)

Post-model differentiable functions operating on `(N, T, L)` prediction tensors:

| Class | Purpose |
|-------|---------|
| `Aggregate` | Mean / sum over tasks and/or positions |
| `Specificity` | `on_tasks` signal / `off_tasks` signal; `compare_func` = `subtract` or `divide` |

Passed as `prediction_transform` to `ISM_predict`, `evolve`, `predict_variant_effects`.

---

## 8. Model Zoo (`grelu.resources`)

```
grelu.resources.load_model(repo_id, filename)
    ├── hf_hub_download(repo_id, filename)        # download .ckpt to HF cache
    └── LightningModel.load_from_checkpoint(path)
            ├── deserialize model_params, train_params, data_params
            └── build_model() → correct BaseModel subclass
```

Available models (HuggingFace `Genentech/` organization):

| `repo_id` | Architecture | Tasks |
|-----------|-------------|-------|
| `borzoi-model` | BorzoiPretrainedModel | 7611 human tracks @ 32 bp |
| `enformer-model` | EnformerPretrainedModel | 5313 human+mouse tracks @ 128 bp |
| `decima-model` | — | scRNA-seq, cell-type-specific |
| `human-atac-catlas-model` | EnformerPretrainedModel (fine-tuned) | 222 human ATAC cell types |
| `GM12878_dnase-model` | — | DNase-seq |
| `human-chromhmm-fullstack-model` | — | ChromHMM states |
| `human-mpra-gosai-2024-model` | — | MPRA activity |

---

## 9. Key Inter-file Dependency Graph

```
grelu.resources
    └── grelu.lightning.LightningModel.load_from_checkpoint
            └── grelu.model.models.*Model
                    ├── grelu.model.trunks.*Trunk
                    │       ├── grelu.model.blocks.*  (ConvTower, TransformerTower, …)
                    │       └── grelu.model.layers.*  (Activation, Pool, Crop, Attention)
                    └── grelu.model.heads.*  (ConvHead, MLPHead)

grelu.lightning.LightningModel.predict_on_seqs / predict_on_dataset
    ├── grelu.sequence.format.convert_input_type
    │       └── grelu.io.genome  (genomepy)  — for interval → string
    └── grelu.data.dataset.*Dataset
            ├── grelu.sequence.format
            ├── grelu.data.augment
            └── grelu.io.bigwig / .genome

grelu.interpret.score.ISM_predict
    ├── grelu.data.dataset.ISMDataset
    ├── grelu.lightning.LightningModel.predict_on_dataset
    └── grelu.transforms.prediction_transforms.*

grelu.design.evolve
    └── grelu.interpret.score.ISM_predict  (each iteration)

grelu.variant.predict_variant_effects
    ├── grelu.data.dataset.VariantDataset
    └── grelu.lightning.LightningModel.predict_on_dataset

grelu.interpret.simulate.*
    ├── grelu.data.dataset.{PatternMarginalize,Spacing,TilingShuffle}Dataset
    └── grelu.lightning.LightningModel.predict_on_dataset
```

---

## 10. AlphaGenome Submodule Internal Architecture

Location: `src/alphagenome_pytorch/`

```
alphagenome_pytorch/
├── model.py          # AlphaGenome: SequenceEncoder + TransformerTower + SequenceDecoder
│                     # public API: forward(), predict(), encode(), from_pretrained()
├── config.py         # DtypePolicy (full_float32 | mixed_precision bfloat16)
├── heads.py          # GenomeTracksHead, ContactMapsHead, predictions_scaling
├── attention.py      # RoPE, MHABlock, PairUpdateBlock, AttentionBiasBlock
├── convolutions.py   # StandardizedConv1d, DownResBlock, UpResBlock, Pool1d
├── embeddings.py     # OutputEmbedder, OutputPair
├── variant_scoring/  # VariantScorer utility class
└── data/
    ├── track_metadata_human.parquet   # 6126 rows: track names / tissues / assays
    └── track_metadata_mouse.parquet
```

### Internal forward pass

```
DNA  (B, 131072, 4)   NLC input
    │
    ▼  SequenceEncoder
    │   DnaEmbedder → 6× DownResBlock → Pool1d
    ▼
Trunk  (B, 1024, 1536)  @ 128 bp   +   intermediates (U-Net skip connections)
    │
    ▼  TransformerTower  (9 blocks)
    │   even blocks:  PairUpdateBlock  (updates pair activations in parallel)
    │   all blocks:   AttentionBiasBlock + MHA (RoPE) + MLP
    ▼
Trunk  (B, 1024, 1536)
    │
    ├── OutputEmbedder (128 bp)  → embeddings_128bp  (B, 1024, 3072)
    │       └── GenomeTracksHead (per assay) → 128 bp track tensors
    │
    └── SequenceDecoder  (U-Net upsampling via stored skip connections)
            ▼
        Decoded  (B, 131072, 768)  @ 1 bp
            └── OutputEmbedder (1 bp)  → embeddings_1bp  (B, 131072, 1536)
                    └── GenomeTracksHead (per assay) → 1 bp track tensors

Output dict (channels_last=False):
  { "atac":         {128: (B,256,1024),   1: (B,256,131072)},
    "dnase":        {128: (B,384,1024),   1: (B,384,131072)},
    "procap":       {128: (B,128,1024),   1: (B,128,131072)},
    "cage":         {128: (B,640,1024),   1: (B,640,131072)},
    "rna_seq":      {128: (B,768,1024),   1: (B,768,131072)},
    "chip_tf":      {128: (B,1664,1024)},
    "chip_histone": {128: (B,1152,1024)},
    "contact_maps": (B,28,64,64)           }
```

---

## 11. Benchmark: Borzoi vs AlphaGenome (Tutorial 1 Region)

Region: `chr1:70190128–70321200` (131 072 bp, centred on the Borzoi tutorial locus)

| | Borzoi | AlphaGenome |
|-|--------|-------------|
| Input length | 524 288 bp | 131 072 bp |
| Output resolution | 6144 bins × 32 bp | 1024 bins × 128 bp |
| SRSF11 brain/liver RNA specificity | **1.45×** | **2.86×** |
| Brain genes detected | ANKRD13C, LRRC40, SRSF11 | ANKRD13C, LRRC40, SRSF11 |
| Liver RNA assay matched | ENCODE RNA-seq | polyA plus RNA-seq (ENCODE) |

Comparison script: `tutorial_1_borzoi_vs_alphagenome.py`
