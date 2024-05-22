FROM lfwa/dictys-cpu

# from https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the dictys environment:
SHELL ["conda", "run", "-n", "dictys", "/bin/bash", "-c"]
RUN pip install scanpy 
RUN pip install anndata
RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_networks"
RUN cd ~ && pip install -e pereggrn_networks/


COPY ./train.py /train.py
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dictys", "python", "train.py"]
