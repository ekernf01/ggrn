#!/bin/bash
# Boilerplate to build this docker image

name=rnaforecaster

docker build -t ggrn_docker_backend_${name} .

# We use this for easy interactive development but it should usually be commented out. 
# docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/Dockerfiles/${name}/from_to_docker:/from_to_docker ggrn_docker_backend_${name}

docker tag ggrn_docker_backend_${name} ekernf01/ggrn_docker_backend_${name}
docker login
docker push ekernf01/ggrn_docker_backend_${name}
python test.py
