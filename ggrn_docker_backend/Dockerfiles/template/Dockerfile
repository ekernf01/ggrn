# from https://pythonspeed.com/articles/activate-conda-dockerfile/
FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "scanpy_env", "/bin/bash", "-c"]

# Demonstrate during the build that the environment is activated:
RUN echo "Make sure scanpy is installed:"
RUN python -c "import scanpy"

# This script will just predict 0's in the right shape of h5ad file. 
COPY ./train.py /train.py

# Run train.py in scanpy_env when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scanpy_env", "python", "train.py"]
