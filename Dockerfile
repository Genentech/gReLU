FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget vim gcc bzip2 ca-certificates libncurses5-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    zlib1g-dev \
    libssl-dev \
    make \
    g++ \
    git \
    rsync \
    openssh-client

# Install bedtools
RUN wget https://github.com/arq5x/bedtools2/releases/download/v2.30.0/bedtools.static.binary && \
    mv bedtools.static.binary bedtools && \
    chmod a+x bedtools && \
    mv bedtools /usr/bin

# Install UCSC utils
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedGraphToBigWig /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bigWigToBedGraph /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/genePredToBed /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/genePredToGtf /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/bedToGenePred /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/gtfToGenePred /usr/bin/
RUN rsync -aP rsync://hgdownload.soe.ucsc.edu/genome/admin/exe/linux.x86_64/gff3ToGenePred /usr/bin/


# Install python packages
RUN pip install flash-attn --no-build-isolation
RUN pip install cython setuptools jupyterlab pandas scikit-learn tables lxml html5lib
RUN pip install pytest pytest-cov pre-commit
RUN pip install black flake8 isort
RUN pip install captum==0.5.0 wandb tensorboard plotnine

RUN pip install bioframe biopython genomepy scanpy \
                pyjaspar pyBigWig pyfaidx pytabix
RUN pip install bpnet-lite>=0.5.7 ledidi enformer-pytorch genomepy statsmodels
RUN pip install pygenomeviz tangermeme >= 0.4.0

# Install modiscolite
RUN pip install modisco-lite@git+https://github.com/jmschrei/tfmodisco-lite.git

# Run jupyterlab
WORKDIR /
CMD jupyter lab --no-browser --allow-root --port 8891 --ip 0.0.0.0 --NotebookApp.token=''
