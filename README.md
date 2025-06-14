[![DOI](https://zenodo.org/badge/788591787.svg)](https://doi.org/10.5281/zenodo.15627611)

# gReLU

gReLU is a Python library to train, interpret, and apply deep learning models to DNA sequences. Code documentation is available [here](https://genentech.github.io/gReLU/).

![Flowchart](media/flowchart.jpg)

## Installation

To install from source:

```shell
git clone https://github.com/Genentech/gReLU.git
cd gReLU
pip install .
```

To install using pip:

```shell
pip install gReLU
```
Typical installation time including all dependencies is under 10 minutes.

To train or use transformer models containing flash attention layers, [flash-attn](https://github.com/Dao-AILab/flash-attention) needs to be installed first:
```shell
conda install -c conda-forge cudatoolkit-dev -y
pip install torch ninja
pip install flash-attn --no-build-isolation
pip install gReLU
```

## Contributing

See our [contribution guide](https://genentech.github.io/gReLU/contributing.html).

## Additional requirements

If you want to use genome annotation features through the function `grelu.io.genome.read_gtf`, you will need to install the following UCSC utilities: `genePredToBed`, `genePredToGtf`, `bedToGenePred`, `gtfToGenePred`, `gff3ToGenePred`.

If you want to create bigWig files through the function `grelu.data.preprocess.make_insertion_bigwig`, you will need to install the following UCSC utilities: `bedGraphToBigWig`.

UCSC utilities can be installed from `http://hgdownload.cse.ucsc.edu/admin/exe/`, for example using the following commands:

```shell
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /usr/bin/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/genePredToBed /usr/bin/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/genePredToGtf /usr/bin/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedToGenePred /usr/bin/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/gtfToGenePred /usr/bin/
rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/gff3ToGenePred /usr/bin/
```

or via bioconda:

```shell
conda install -y \
bioconda::ucsc-bedgraphtobigwig \
bioconda::ucsc-genepredtobed    \
bioconda::ucsc-genepredtogtf    \
bioconda::ucsc-bedtogenepred    \
bioconda::ucsc-gtftogenepred    \
bioconda::ucsc-gff3togenepred
```

If you want to create ATAC-seq coverage bigWig files using `grelu.data.preprocess.make_insertion_bigwig`, you will need to install bedtools. See https://bedtools.readthedocs.io/en/latest/content/installation.html for instructions.

## Citation

Please cite our preprint: https://www.biorxiv.org/content/10.1101/2024.09.18.613778v1
