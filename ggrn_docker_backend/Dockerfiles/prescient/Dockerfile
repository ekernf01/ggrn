FROM ubuntu:latest
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git
COPY hs_birth_msigdb_kegg.csv . 
COPY hs_death_msigdb_kegg.csv .
COPY estimate-growth-rates.py .
# Currently we need to use my fork of PRESCIENT in order to support h5ad input.
RUN pip install git+https://github.com/ekernf01/prescient
COPY ./train.py /train.py
ENTRYPOINT ["python3", "train.py"]
