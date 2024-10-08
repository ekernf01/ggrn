FROM ubuntu:latest

# Install Python and other dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install packages
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install annoy
RUN /opt/venv/bin/pip install git+https://github.com/ekernf01/prescient@3c873425e96cbd894e877361fee2671a89ff2618

# Copy necessary files
COPY human_birth.csv .
COPY human_death.csv .
COPY mouse_birth.csv .
COPY mouse_death.csv .
COPY zebrafish_birth.csv .
COPY zebrafish_death.csv .

COPY estimate-growth-rates.py .
COPY ./train.py /train.py

# Set the entrypoint to use the virtual environment's Python
ENTRYPOINT ["/opt/venv/bin/python", "/train.py"]