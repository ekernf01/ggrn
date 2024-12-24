FROM ubuntu 
# julia:1.6.7-ubuntu

# We use anndata to read the input data, so we need python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py
RUN python3 --version && pip --version

# Need to read jon
RUN pip3 install anndata 

# # Install packages
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.6-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.6.6-linux-x86_64.tar.gz
RUN ln -s /julia-1.6.6/bin/julia /usr/local/bin/julia -f
RUN julia -e 'using Pkg; Pkg.add("Random")'
RUN julia -e 'using Pkg; Pkg.add("IterTools")'
RUN julia -e 'using Pkg; Pkg.add("Flux")'
RUN julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
RUN julia -e 'using Pkg; Pkg.add("DiffEqFlux")'
RUN julia -e 'using Pkg; Pkg.add("Lux")'
RUN julia -e 'using Pkg; Pkg.add("Optimisers")'
RUN julia -e 'using Pkg; Pkg.add("Zygote")'
RUN julia -e 'using Pkg; Pkg.add("OrdinaryDiffEq")'
RUN julia -e 'using Pkg; Pkg.add("ComponentArrays")'
RUN julia -e 'using Pkg; Pkg.add("Statistics")'
RUN julia -e 'using Pkg; Pkg.add("CSV")'
RUN julia -e 'using Pkg; Pkg.add("DataFrames")'

# # Check package installations
RUN julia -e 'using Distributed, CSV, DataFrames, Random, IterTools, Flux, DifferentialEquations'
RUN julia -e 'using DiffEqFlux, Lux, Optimisers, Zygote, OrdinaryDiffEq, ComponentArrays, Statistics'
RUN julia -e 'println("All dependencies are correctly installed!")'

# RUN git clone https://github.com/rawanelshobaky/RNAForecaster.jl
# RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/ekernf01/RNAForecaster.jl#patch-1")'
RUN julia -e 'using Pkg; Pkg.add("RNAForecaster")'
RUN julia -e 'using RNAForecaster'
# train the model
COPY ./train.jl /train.jl
COPY ./convert_output.py /convert_output.py
COPY ./convert_input.py /convert_input.py
COPY ./run.sh /run.sh
RUN chmod +x run.sh
ENTRYPOINT [ "bash", "run.sh" ]
