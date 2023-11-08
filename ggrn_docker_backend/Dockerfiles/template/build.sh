#!/bin/bash
# Boilerplate to build this docker image

docker build -t ggrn_docker_backend_template .
docker tag ggrn_docker_backend_template ekernf01/ggrn_docker_backend_template
docker login
docker push ekernf01/ggrn_docker_backend_template
