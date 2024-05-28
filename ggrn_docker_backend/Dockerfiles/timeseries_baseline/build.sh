#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_timeseries_baseline .

# We use this for easy interactive development but it should usually be commented out. 
# docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/Dockerfiles/timeseries_baseline/from_to_docker:/from_to_docker ggrn_docker_backend_timeseries_baseline

docker tag ggrn_docker_backend_timeseries_baseline ekernf01/ggrn_docker_backend_timeseries_baseline
docker login
docker push ekernf01/ggrn_docker_backend_timeseries_baseline
python test.py
