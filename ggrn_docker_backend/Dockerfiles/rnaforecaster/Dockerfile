FROM ubuntu:latest
# install julia
RUN curl -fsSL https://install.julialang.org | sh 
# This is some weird julia-specific terminal syntax I think? Need to test it. 
RUN ] add https://github.com/rossinerbe/RNAForecaster.jl 
COPY ./train.jl /train.jl
ENTRYPOINT ["julia", "train.jl"]
