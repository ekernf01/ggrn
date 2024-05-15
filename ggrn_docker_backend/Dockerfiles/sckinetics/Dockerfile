FROM ubuntu:22.04

# Thanks to Fabio Rosado for the easy conda installation
# https://fabiorosado.dev/blog/install-conda-in-docker/
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && apt-get install -y wget git \
    && rm -rf /var/lib/apt/lists/*
   
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

COPY sckinetics_requirements.yml .
RUN conda env create -f sckinetics_requirements.yml

# from https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sc_kinetics", "/bin/bash", "-c"]

# This helps us reformat some networks
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_networks"
RUN cd ~ && pip install -e pereggrn_networks/
RUN pip install --user git+https://github.com/KrishnaswamyLab/MAGIC.git#subdirectory=python

# We want to separate out the motif analysis from the dynamic modeling. 
# But here are the R packages needed for their chromatin analysis.
# RUN R CMD BATCH -e 'install.packages("BiocManager", "remotes")'
# RUN conda install R-curl # This doesn't work with pure R
# RUN R CMD BATCH -e 'remotes::install_github("YuLab-SMU/ggtree")'
# RUN R CMD BATCH -e 'BiocManager::install(c("httr", "BiocFileCache", "KEGGREST", "AnnotationDbi", "biomaRt", "GenomicFeatures", "rtracklayer", "TxDb.Mmusculus.UCSC.mm10.knownGene", "ChIPseeker"))'

# Run sckinetics
COPY sckinetics  sckinetics
COPY ./train.py /train.py

# Run train.py in sc_kinetics environment when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sc_kinetics", "python", "train.py"]