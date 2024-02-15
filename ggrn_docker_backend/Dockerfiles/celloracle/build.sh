#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_celloracle .
docker tag ggrn_docker_backend_celloracle ekernf01/ggrn_docker_backend_celloracle
docker login
docker push ekernf01/ggrn_docker_backend_celloracle
