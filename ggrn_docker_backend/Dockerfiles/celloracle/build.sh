#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_celloracle .

# We use this for easy interactive development but it should usually be commented out. 
# docker run --rm -it -v /home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn/ggrn_docker_backend/Dockerfiles/celloracle/from_to_docker:/from_to_docker ggrn_docker_backend_celloracle

docker tag ggrn_docker_backend_celloracle ekernf01/ggrn_docker_backend_celloracle
docker login
docker push ekernf01/ggrn_docker_backend_celloracle
python test.py