FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.2-cuda12.1.0


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
RUN pip install cython setuptools jupyterlab pandas scikit-learn tables lxml html5lib
RUN pip install pytest pytest-cov pre-commit
RUN pip install black flake8 isort
RUN pip install captum==0.5.0 wandb tensorboard plotnine

RUN pip install bioframe biopython genomepy scanpy \
                pyjaspar pymemesuite pyBigWig pyfaidx pytabix
RUN pip install bpnet-lite>=0.5.7 ledidi enformer-pytorch genomepy
RUN pip install pygenomeviz

# Install modiscolite
RUN pip install modisco-lite@git+https://github.com/jmschrei/tfmodisco-lite.git

# Install MEME suite
RUN wget https://meme-suite.org/meme/meme-software/5.5.1/meme-5.5.1.tar.gz && \
    tar -xvzf meme-5.5.1.tar.gz && \
    cd meme-5.5.1 && \
    ./configure --prefix=/usr --enable-build-libxml2 --enable-build-libxslt && \
    make && \
    make install

# Run jupyterlab
WORKDIR /
CMD jupyter lab --no-browser --allow-root --port 8891 --ip 0.0.0.0 --NotebookApp.token=''
