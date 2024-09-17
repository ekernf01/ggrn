#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_dictys .

# We use this for easy interactive development but it should usually be commented out. 
docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/Dockerfiles/dictys/from_to_docker:/from_to_docker ggrn_docker_backend_dictys

docker tag ggrn_docker_backend_dictys ekernf01/ggrn_docker_backend_dictys
docker login
docker push ekernf01/ggrn_docker_backend_dictys
python test.py
