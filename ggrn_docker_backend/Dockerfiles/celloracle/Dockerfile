FROM kenjikamimoto126/celloracle_ubuntu:0.18.0

RUN cd ~ && git clone "https://github.com/ekernf01/pereggrn_networks"
RUN cd ~ && pip install -e pereggrn_networks/

# This python script applies CO, using the contents of a folder `to_from_docker` as input.
COPY ./train.py /train.py

# The CO container takes care of all the conda funny business. 
ENTRYPOINT ["python", "train.py"]