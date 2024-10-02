FROM ubuntu:latest
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git
# Currently we need to use my fork of PRESCIENT in order to support h5ad input.
RUN pip install git+https://github.com/ekernf01/prescient@3c873425e96cbd894e877361fee2671a89ff2618  --break-system-packages
# COPY requirements.txt . 
# RUN pip install -r requirements.txt --break-system-packages
# COPY prescient prescient
# RUN pip install -e prescient --no-deps --break-system-packages
COPY human_birth.csv .
COPY human_death.csv .
COPY mouse_birth.csv .
COPY mouse_death.csv .
COPY zebrafish_birth.csv .
COPY zebrafish_death.csv .

COPY estimate-growth-rates.py .
COPY ./train.py /train.py
ENTRYPOINT ["python3", "train.py"]
