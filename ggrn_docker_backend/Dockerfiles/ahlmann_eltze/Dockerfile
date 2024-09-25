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

COPY environment.yml .
RUN conda env create -f environment.yml

# from https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "scanpy_env", "/bin/bash", "-c"]
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_perturbations"
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_networks"
RUN cd ~ && git clone "https://github.com/ekernf01/ggrn"
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn"
RUN cd ~ && pip install -e pereggrn_perturbations/
RUN cd ~ && pip install -e pereggrn_networks/
RUN cd ~ && pip install -e ggrn/
RUN cd ~ && pip install -e pereggrn/
COPY ./train.py /train.py


# Run train.py in sc_kinetics environment when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scanpy_env", "python", "train.py"]