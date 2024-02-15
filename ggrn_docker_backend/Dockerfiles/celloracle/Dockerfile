# Using ubuntu 22 solves a GLIBCXX version error
FROM ubuntu:22.04

# Thanks to Fabio Rosado for the easy conda installation
# https://fabiorosado.dev/blog/install-conda-in-docker/
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml .
RUN conda env create -f environment.yml

# This is taken from the CO container at https://github.com/morris-lab/CellOracle/blob/master/docs/dockerfile/Dockerfile
RUN apt-get update \
 && apt-get install -y wget git nano gcc-4.9 libstdc++6 g++ libz-dev bedtools \
 && rm -rf /var/lib/apt/lists/*
RUN apt-get upgrade 

RUN cd \ 
 && git clone https://github.com/morris-lab/CellOracle.git

# from https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "scanpy_env", "/bin/bash", "-c"]

# This is simplified from the CO container at https://github.com/morris-lab/CellOracle/blob/master/docs/dockerfile/Dockerfile
RUN cd \
 && wget https://anaconda.org/bioconda/gimmemotifs/0.17.1/download/linux-64/gimmemotifs-0.17.1-py38h8ded8fe_1.tar.bz2 \
 && conda install --offline gimmemotifs-0.17.1-py38h8ded8fe_1.tar.bz2 \
 && rm gimmemotifs-0.17.1-py38h8ded8fe_1.tar.bz2 
RUN cd $HOME/CellOracle \
 && pip install . --default-timeout=100 \
 && cd  \
 && rm -r CellOracle

RUN cd ~ && git clone "https://github.com/ekernf01/load_networks"
RUN cd ~ && pip install -e load_networks/

# This python script applies CO, using the contents of a folder `to_from_docker` as input.
COPY ./train.py /train.py

# Run train.py in scanpy_env when the container starts
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scanpy_env", "python", "train.py"]
