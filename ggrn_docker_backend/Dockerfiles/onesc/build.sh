#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_onesc .

# For interactive debugging
# docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/Dockerfiles/onesc/from_to_docker:/from_to_docker ggrn_docker_backend_onesc /bin/bash

docker tag ggrn_docker_backend_onesc ekernf01/ggrn_docker_backend_onesc
docker login
docker push ekernf01/ggrn_docker_backend_onesc
python test.py