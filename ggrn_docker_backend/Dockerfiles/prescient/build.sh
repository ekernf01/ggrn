#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_prescient .

# We use this for easy interactive development but it should usually be commented out. 
docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/from_to_docker:/from_to_docker ggrn_docker_backend_prescient

# docker tag ggrn_docker_backend_prescient ekernf01/ggrn_docker_backend_prescient
# docker login
# docker push ekernf01/ggrn_docker_backend_prescient
