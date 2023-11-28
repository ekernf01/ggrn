FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "scanpy_env", "/bin/bash", "-c"]

# Install and test ggrn
RUN git clone https://github.com/ekernf01/ggrn
RUN pip install -e ggrn
RUN python -c "import ggrn.api as ggrn"

# Run train.py in scanpy_env when the container starts
COPY ./train.py /train.py
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scanpy_env", "python", "train.py"]
