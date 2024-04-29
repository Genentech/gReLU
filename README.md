# gReLU

gReLU is a python library to train, interpret, and apply deep learning models to DNA sequences. Code documentation is available at [http://go.gene.com/grelu](http://go.gene.com/grelu).

![Flowchart](media/flowchart.jpg)

## Installation

To install the package:

```shell
pip install \
    --trusted-host pypi.vida.science.roche.com \
    --extra-index-url https://pypi.vida.science.roche.com/simple/ \
    grelu
```

To upgrade the package:


```shell
pip install \
    --upgrade \
    --trusted-host pypi.vida.science.roche.com \
    --extra-index-url https://pypi.vida.science.roche.com/simple/ \
    grelu
```

To install from source:

```shell
git clone https://code.roche.com/braid-relu/grelu.git
cd grelu
pip install .
```

## Rosalind

`grelu` is available on `Rosalind` in managed environments.

```shell
ml spaces/gpy
ml gpyprd/gpy39 # available on Py 3.9 and 3.10 environments
```

We recommend using Volta GPUs on Rosalind to avoid CUDA issues. This can be achieved via the `--gres=gpu:volta:1` SLURM option in the command line or on the Advanced Slurm Options section of the Rosalind Portal.

For details on all possibilities, see [this Slack channel](https://gred.slack.com/archives/tool-grelu).

## Contributing

This project uses [pre-commit](https://pre-commit.com/). Please make sure to install it before making any changes:

```shell
pip install pre-commit
cd grelu
pre-commit install
```

It is a good idea to update the hooks to the latest version:

```shell
pre-commit autoupdate
```

## Additional requirements

If you want to use genome annotation features through the function `grelu.io.read_gtf`, you will need to install the following UCSC utilities: `genePredToBed`, `genePredToGtf`, `bedToGenePred`, `gtfToGenePred`, `gff3ToGenePred`.

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
conda install bioconda::ucsc-bedgraphtobigwig
conda install bioconda::ucsc-genepredtobed
conda install bioconda::ucsc-genepredtogtf
conda install bioconda::ucsc-bedtogenepred
conda install bioconda::ucsc-gtftogenepred
conda install bioconda::ucsc-gff3togenepred
```
