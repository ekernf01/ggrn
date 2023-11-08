#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_ggrn .
docker tag ggrn_docker_backend_ggrn ekernf01/ggrn_docker_backend_ggrn
docker login
docker push ekernf01/ggrn_docker_backend_ggrn
