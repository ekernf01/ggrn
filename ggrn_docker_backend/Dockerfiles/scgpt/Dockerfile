# scGPT is a PyTorch transformer. It needs CUDA at *runtime* to fine-tune/predict at any
# reasonable speed (it will fall back to CPU, but that is extremely slow). We start from an
# official CUDA + Python image and build a conda env from scratch, the same way template/ does.
#
# We deliberately do NOT install flash-attn (the `fa2` extra of scGPT). flash-attn requires a
# very specific GPU/CUDA/torch combination and is fragile to build. scGPT's TransformerGenerator
# supports `use_fast_transformer=False`, and scgpt.utils.load_pretrained transparently renames the
# flash-attn checkpoint keys to the vanilla nn.MultiheadAttention keys, so the pretrained
# "whole-human" weights load fine on the plain PyTorch backend. train.py sets use_fast_transformer=False.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities (mirrors template/Dockerfile).
RUN apt-get update \
    && apt-get install -y build-essential wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda (thanks to Fabio Rosado, as in template/Dockerfile).
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment (https://pythonspeed.com/articles/activate-conda-dockerfile/).
SHELL ["conda", "run", "-n", "scgpt_env", "/bin/bash", "-c"]

# scGPT + its perturbation-prediction dependencies.
#  - torch/torchtext are pinned to a mutually compatible pair that predates the torchtext.vocab
#    removal (scGPT imports torchtext.vocab.Vocab and torchtext._torchtext.Vocab).
#  - cell-gears<0.0.3 provides gears.PertData / gears.utils.create_cell_graph_dataset_for_prediction,
#    which scGPT's perturbation workflow is built on top of.
#  - torch-geometric supplies the DataLoader / Data graph objects gears uses.
#  - gdown is used at runtime by train.py to fetch the pretrained "whole-human" checkpoint folder
#    from Google Drive if it is not already baked in / mounted.
RUN pip install --no-cache-dir \
        torch==1.13.1+cu117 torchtext==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install --no-cache-dir torch-geometric==2.3.1
RUN pip install --no-cache-dir "cell-gears<0.0.3" "scvi-tools>=0.16.0,<1.0" "scanpy>=1.9.1,<2.0.0" gdown
RUN pip install --no-cache-dir scgpt

# Optionally pre-download the pretrained "whole-human" checkpoint at build time so the container
# can run without network access. This is a ~1.5GB Google Drive folder; comment out if you prefer
# to mount the weights (see README.md). If left commented, train.py downloads it on first run.
# RUN mkdir -p /scgpt_weights && \
#     gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y \
#           -O /scgpt_weights/scGPT_human

# Utility for reading/reshaping GGRN base networks (as template/ and celloracle/ do). scGPT does
# not use a base GRN, but we install it so the import in train.py never fails.
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_networks"
RUN cd ~ && pip install -e pereggrn_networks/

COPY ./train.py /train.py

# Run train.py in the scgpt_env environment when the container starts.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scgpt_env", "python", "train.py"]
